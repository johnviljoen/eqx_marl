"""
Template to fill out to create an RL environment using
Mujoco dynamics via BRAX.

Mujoco/BRAX handles all the dynamics, in this template we will show
you how to fill out everything else required, such as agentic actions
or rewards and set everything up :)
"""

# =================== #
# Imports and Configs #
# =================== #

import os
from datetime import datetime

import jax
import jax.numpy as jnp
import jax.random as jr
from eqx_marl.common import spaces # this is analagous to gym.Box and other "spaces"
import mujoco as mj # we use mj to change some foundational things in the simulation
from brax import math # this contains some useful things, like safe_norm and quaternion utils
from typing import Dict, Literal, Optional, Tuple, List, Union, NamedTuple

# Mujoco/BRAX requires our env inherits from PipelineEnv, and the state used is State
from brax.envs.base import PipelineEnv, State

# mjcf interprets Mujoco XML files for BRAX, html will render rollouts of BRAX States
from brax.io import mjcf, html

# ============================================== #
# UNCHANGED: Datastructures and Helper Functions #
# ============================================== #

class LogEnvState(NamedTuple):
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    step_in_episode: jnp.ndarray
    first_pipeline_state: State
    first_obs: jnp.ndarray
    truncation: jnp.ndarray

# ================= #
# Your Environment! #
# ================= #

class YourEnv(PipelineEnv):
    def __init__(
        self,

        # base env settings
        xml_path=os.path.join(os.path.dirname(__file__), 'assets/demo.xml'),
        backend='positional', # this is important for the ant
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        
        # parallel env settings
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
        # homogenisation_method: Optional[Literal["max", "concat"]] = None,
        
        # logging settings
        replace_info: bool = False,
        **kwargs
    ):
        
        # ============================= #
        # TODO: Mujoco/BRAX System Init #
        # ============================= #

        # brax provides utils to load xml file into mujoco
        sys = mjcf.load(xml_path)

        # n_frames are the number of simulation timesteps between new actions
        # in the interim we zero-order-hold the previous action
        n_frames = 5

        # in the case of the Ant the physics backed matters, and they reduce the physics timestep
        # from 0.01 to 0.005 and double n_frames to keep the interaction timestep the same.
        # I speculate that the increased fidelity of the timestep for these backends is required
        # for reliable, robust physics calculations
        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10

        # again more examples of using the "sys" to modify fundamental parts of the simulator.
        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mj.mjtSolver.mjSOL_NEWTON, # this determins how we solve contact forces
                'opt.disableflags': mj.mjtDisableBit.mjDSBL_EULERDAMP, # I believe this damping term stabilizes some dynamics
                'opt.iterations': 1, # number of optimization iterations for contact
                'opt.ls_iterations': 4, # number of line search iterations per optimization iteration for contact
            })

        # again more examples of using the "sys" to modify fundamental parts of the simulator
        if backend == 'positional':
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jnp.ones_like(sys.actuator.gear)
                )
            )

        # the kwargs are designed to pass options to the brax backend, in this case we
        # modify the kwargs n_frames according to the above code before passing it to super
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        # actually creating the BRAX system
        super().__init__(sys=sys, backend=backend, **kwargs)

        # various parameters for the Ant environment
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if self._use_contact_forces:
            raise NotImplementedError('use_contact_forces not implemented.')
        

        # ================================== #
        # TODO: Multi-Agent Environment Init #
        # ================================== #

        # UNCHANGED logging settings
        self.replace_info = replace_info
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.auto_reset = auto_reset

        if self._exclude_current_positions_from_observation:
            # TODO
            observation_size = sys.q_size() + sys.qd_size() - 2 # by design - match with get_obs function
        else: 
            observation_size = sys.q_size() + sys.qd_size()
        self.observation_space = spaces.Box(-jnp.inf, jnp.inf, shape=(observation_size,),)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(sys.act_size(),),)


    # ======================================== #
    # TODO: design observation function #
    # ======================================== #

    def get_obs(self, pipeline_state: State) -> jax.Array:

        # the pipeline_state is an attribute of env_state, which in turn is an attribute of state.
        # pipeline_state is what defines the physical simulation state, primarily through
        # pipeline_state.q and pipeline_state.qd which are the generalized positions and
        # velocities respectively. The rest of pipeline_state is read_only and useful
        # for creating expressive observations - as we can do in this function

        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jnp.concatenate([qpos] + [qvel])

    # ================================================= #
    # TODO: design pipeline_state random reset function #
    # ================================================= #    

    def get_random_pipeline_state(self, rng):

        # here you must decide how to randomly instantiate the generalized positions and
        # velocities q, and qd of the pipeline_state and then create the pipeline_state
        # to be used in the reset function and the automatic reset functionality later on.
        # I will explain the automatic reset later! fear not!

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        return pipeline_state

    def reset(self, rng: jr.PRNGKey) -> Tuple[Dict[str, jax.Array], State]:

        # =========================== #
        # TODO: Reset the environment #
        # =========================== #

        # TODO reset the environment as you see fit for your environment. You
        # should end up with a new env_state: State and obs
        # as the below example does. NOTE you should also include the exact same info
        # dict in the resulting env_state as the example does.

        pipeline_state = self.get_random_pipeline_state(rng); rng, _ = jr.split(rng)
        obs = self.get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_survive': zero,
            'reward_ctrl': zero,
            'reward_contact': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'forward_reward': zero,
        }

        # NOTE it is essential that info is created like this and added to env_state
        info = {
            "returned_episode_returns": jnp.zeros(1),
            "returned_episode_lengths": jnp.zeros(1),
            "returned_episode": jnp.zeros(1).astype(jnp.bool_)
        }

        env_state = State(pipeline_state, obs, reward, done, metrics, info)

        # ============================= #
        # UNCHANGED: log state wrapping #
        # ============================= #

        # NOTE we change the "first_pipeline_state" and "first_obs" at every
        # usage, therefore we generate a new pair here to be used as the first 
        # upon the next automatic reset in step - I will explain the automatic reset
        # later! fear not!

        new_first_pipeline_state = self.get_random_pipeline_state(rng)
        new_first_obs = self.get_obs(new_first_pipeline_state)

        # the struct we use to log the agent observations and the env state
        log_state = LogEnvState(
            env_state,
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            jnp.zeros((), jnp.int32), # the env step number in the current rollout
            new_first_pipeline_state,
            new_first_obs,
            jnp.array(0.)
        )

        return obs, log_state

    def step(
        self,
        rng: jr.PRNGKey, # this is not used in our deterministic env
        state: State, # this is the LogEnvState
        actions: jax.Array, # this is the agentic actions
    ) -> Tuple[
        Dict[str, jax.Array], State, Dict[str, float], Dict[str, bool], Dict
    ]:

        # We first ensure that states that were previously done (that already
        # have had their states reset) have their done flag reset
        state = state._replace(
            env_state=state.env_state.replace(
                done=jnp.zeros_like(state.env_state.done)
            )
        )

        # =============================================================== #
        # TODO: env_state step (reward, obs, state, metrics, done) #
        # =============================================================== #

        # here we calculate the reward, obs, state, metrics, and the done
        # NOTE the done is only for early termination (the end of episode termination
        # situation is handled later automatically - look through the remainder of this 
        # method to understand)

        # save the old pipeline_state to calculate some velocities
        pipeline_state0 = state.env_state.pipeline_state
        assert pipeline_state0 is not None

        # get the NEXT pipeline state yaaay
        next_pipeline_state = self.pipeline_step(state.env_state.pipeline_state, actions)  # type: ignore

        # environment specific calculations
        velocity = (next_pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range

        is_healthy = jnp.where(next_pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(next_pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(actions))
        contact_cost = 0.0

        obs = self.get_obs(next_pipeline_state)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        
        # finally we update the environment specific metrics that you have chosen to track
        state.env_state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=next_pipeline_state.x.pos[0, 0],
            y_position=next_pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(next_pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
        )

        # ==================================================================== #
        # UNCHANGED: Episode Wrapping Step (brax.envs.wrappers.EpisodeWrapper) #
        # ==================================================================== #

        step_in_episode = state.step_in_episode + self.action_repeat

        done = jnp.where(step_in_episode >= self.episode_length, jnp.ones_like(state.env_state.done), done)
        state = state._replace(
            truncation = jnp.where(
                step_in_episode >= jnp.array(self.episode_length), 1 - done, jnp.zeros_like(state.env_state.done)
            )
        )

        env_state = state.env_state.replace(
            pipeline_state=next_pipeline_state, 
            obs=obs, 
            reward=reward, 
            done=done
        )

        # You are done! Everything below this is unchanged for you (in the class at least)

        # ============================================================================== #
        # UNCHANGED: Auto Reset Wrapping Post-Step (brax.envs.wrappers.AutoResetWrapper) #
        # ============================================================================== #

        def where_done(x, y):
            done = env_state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)
        

        next_pipeline_state = jax.tree.map(
            where_done, state.first_pipeline_state, env_state.pipeline_state
        )

        obs = jax.tree.map(where_done, state.first_obs, obs)
        env_state = env_state.replace(pipeline_state=next_pipeline_state, obs=obs)

        # first_pipeline_state, rng = jax.lax.cond(
        #     done, 
        #     lambda: (self.get_random_pipeline_state(rng), jr.split(rng)[0]),
        #     lambda: (state.first_pipeline_state, rng)
        # )
        first_pipeline_state = self.get_random_pipeline_state(rng); rng, _ = jr.split(rng)
        first_obs = self.get_obs(first_pipeline_state)
        state = state._replace(
            first_pipeline_state=first_pipeline_state,
            first_obs=first_obs
        )

        if self.auto_reset is True:
            step_in_episode = jnp.where(env_state.done, jnp.zeros_like(step_in_episode), step_in_episode)
            state = state._replace(step_in_episode=step_in_episode)

        # ======================= #
        # UNCHANGED: Log wrapping #
        # ======================= #

        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = state._replace(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            step_in_episode=step_in_episode
        )

        info = env_state.info
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((1,), done).astype(jnp.bool_)

        return obs, state, reward, done, info

def generate_rollout(env, rng, jit=True, num_timesteps=100):
    obs, state = env.reset(rng=rng); rng, _rng = jr.split(rng)
    model = env.sys.mj_model # cpu mujoco model struct
    rollout = []
    if jit is True:
        env_step = jax.jit(env.step)
    else:
        env_step = env.step
    for i in range(num_timesteps):
        ctrl = jr.uniform(_rng, shape=(env.action_size), minval=-1, maxval=1); _rng, _ = jr.split(_rng)
        print(f"ctrl action chosen: {ctrl}")
        obs, state, reward, done, info = env_step(rng, state, ctrl)
        print(f"state.done: {done}")
        print(f"state.reward: {reward}")
        rollout.append(state.env_state.pipeline_state)
        print(f"step: {i}")
        print(f"info: {info}")
        ps = state.env_state.pipeline_state
        print(f"efc_force: {ps.efc_force[ps.contact.efc_address]}")
        # contact_forces = _get_contact_forces(model, state.env_state.pipeline_state.x)
        # print(f"contact_forces: {contact_forces}")
        if i == env.episode_length:
            print("THE PRIOR STATE.DONE SHOULD HAVE BEEN TRUE")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'data/rollouts/{current_datetime}.html'
    os.makedirs("data/rollouts", exist_ok=True)
    with open(save_name, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))
    pass

if __name__ == "__main__":

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # dynamically allocate memory like pytorch does
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["MUJOCO_GL"] = "egl"     # if you have NVIDIA + EGL drivers for rendering

    jax.config.update("jax_debug_nans", True)   # will error if a nan is detected
    jax.config.update("jax_log_compiles", True) # will print out recompilations
    jax.config.update('jax_default_matmul_precision', "highest") # sometimes certain contact dynamics need higher accuracy to prevent nans
    # this set of configs lets us cache some stuff to lower JIT times
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    # for contact forces
    from mujoco import mjx

    env = YourEnv(backend="mjx")
    obs, state = env.reset(jr.PRNGKey(0))

    # a single test step
    # action = {"agent_0": jnp.array([0.5, 0.5]), "agent_1": jnp.array([-0.5, -0.5])}
    # agent_obs, state, reward, done, info = env.step(jr.PRNGKey(0), state, action)

    # a test rollout
    generate_rollout(env, jr.PRNGKey(0), jit=True, num_timesteps=700)
