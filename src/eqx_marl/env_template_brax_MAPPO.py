"""
Template to fill out to create a multi-agent RL environment using
Mujoco dynamics via BRAX.

Mujoco/BRAX handles all the dynamics, in this template we will show
you how to fill out everything else required, such as agentic actions
or rewards and set everything up :)

There needs to be a couple changes to the environment if we are using MAPPO - 
specifically the critic will now (generally) recieve a global state as its 
observation, and the individual actors cannot. we add this as:

agent_obs["global_state"] 

to the output of steps agent_obs

NOTE: this algorithm outperforms IPPO which IS validated to be correct on the Ant4x2, but this
algorithm is NOT validated against a reference continuous time MAPPO.
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

# we must map each agent to their observations and actions in the whole environment
# This is a helper function to convert the ranges of observations
def listerize(ranges: List[Union[int, Tuple[int, int]]]) -> List[int]:
    """
    tuples mean that all global observations indexed from (0, 5) are included in that agents observation
    integers add that single global observation to the agent's observation
    in this case I am passing global observations to each agent for simplicity.
    Here is an example:

    ranges = {
        "agent_0": [(0,8)], # example: [(0, 5), 6, 7] == [0, 1, 2, 3, 4, 5, 6, 7]
        "agent_1": [(0,8)] # example: [(2, 5), 9, 10] == [2, 3, 4, 5, 9, 10]
    }
    agent_obs_mapping = {k: jnp.array(listerize(v)) for k, v in ranges.items()} # _agent_observation_mapping[env_name]
    """
    return [
        i
        for r in ranges
        for i in (range(r[0], r[1] + 1) if isinstance(r, tuple) else [r])
    ]

# ================= #
# Your Environment! #
# ================= #

class YourEnv(PipelineEnv):
    def __init__(
        self,

        # base env settings
        xml_path='src/eqx_marl/assets/demo.xml',
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
        
        # multi agent env settings
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
        homogenisation_method: Optional[Literal["max", "concat"]] = None,
        
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
        self.homogenisation_method = homogenisation_method
        
        # tuples mean that all global observations indexed from (0, 5) are included in that agents observation
        # integers add that single global observation to the agent's observation
        # in this case I am passing global observations to each agent for simplicity
        ranges = {
            # TODO fill this out, below is an example, also check out the listerize helper function above
            "agent_0": [(0, 5), 6, 7, 9, 11, (13, 18), 19, 20],
            "agent_1": [(0, 5), 7, 8, 9, 11, (13, 18), 21, 22],
            "agent_2": [(0, 5), 7, 9, 10, 11, (13, 18), 23, 24],
            "agent_3": [(0, 5), 7, 9, 11, 12, (13, 18), 25, 26],
        }
        self.agent_obs_mapping = {k: jnp.array(listerize(v)) for k, v in ranges.items()} # _agent_observation_mapping[env_name]

        # the agent action mapping is simpler, so we just use the indices of the actions
        self.agent_action_mapping = {
            # TODO
            "agent_0": jnp.array([0, 1]),
            "agent_1": jnp.array([2, 3]),
            "agent_2": jnp.array([4, 5]),
            "agent_3": jnp.array([6, 7]),
        }
        self.agents = list(self.agent_obs_mapping.keys())

        # ======================================================== #
        # UNCHANGED: Boilerplate That Doesn't Require Modification #
        # ======================================================== #
        
        # setup the obs and action spaces for each agent
        self.num_agents = len(self.agent_obs_mapping)
        obs_sizes = {
            agent: self.num_agents
            + max([o.size for o in self.agent_obs_mapping.values()])
            if homogenisation_method == "max"
            else self.env.observation_size
            if homogenisation_method == "concat"
            else obs.size
            for agent, obs in self.agent_obs_mapping.items()
        }
        act_sizes = {
            agent: max([a.size for a in self.agent_action_mapping.values()])
            if homogenisation_method == "max"
            else self.env.action_size
            if homogenisation_method == "concat"
            else act.size
            for agent, act in self.agent_action_mapping.items()
        }
        self.observation_spaces = {
            agent: spaces.Box(-jnp.inf, jnp.inf, shape=(obs_sizes[agent],),)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(-1.0, 1.0, shape=(act_sizes[agent],),)
            for agent in self.agents
        }

        # utility function to batchify floats originally placed in JaxMARLWrapper
        self._batchify_floats = lambda x: jnp.stack([x[a] for a in self.agents])

        # utility function to get obs, action spaces for each agent - required by the ppo algs
        self.observation_space = lambda agent: self.observation_spaces[agent]
        self.action_space = lambda agent: self.action_spaces[agent]

        # MAPPO specific!
        # need to figure out the global observation size for the critic
        test_pipeline_state = self.get_random_pipeline_state(jr.PRNGKey(0)) # random key doesnt matter as we just use this for shape info
        test_global_obs = self.get_global_obs(test_pipeline_state)
        self.global_observation_space = spaces.Box(-jnp.inf, jnp.inf, shape=test_global_obs.shape)

    # ======================================== #
    # TODO: design global observation function #
    # ======================================== #

    def get_global_obs(self, pipeline_state: State) -> jax.Array:

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

        # TODO reset the global environment as you see fit for your environment. You
        # should end up with a new env_state: State and agent_obs: Dict[str, jax.Array]
        # as the below example does. NOTE you should also include the exact same info
        # dict in the resulting env_state as the example does.

        pipeline_state = self.get_random_pipeline_state(rng); rng, _ = jr.split(rng)
        global_obs = self.get_global_obs(pipeline_state)

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
            "returned_episode_returns": jnp.zeros(self.num_agents),
            "returned_episode_lengths": jnp.zeros(self.num_agents),
            "returned_episode": jnp.zeros(self.num_agents).astype(jnp.bool_)
        }

        env_state = State(pipeline_state, global_obs, reward, done, metrics, info)

        agent_obs = self.map_global_obs_to_agents(global_obs)

        # ============================= #
        # UNCHANGED: log state wrapping #
        # ============================= #

        # NOTE we change the "first_pipeline_state" and "first_obs" at every
        # usage, therefore we generate a new pair here to be used as the first 
        # upon the next automatic reset in step - I will explain the automatic reset
        # later! fear not!

        new_first_pipeline_state = self.get_random_pipeline_state(rng)
        new_first_obs = self.get_global_obs(new_first_pipeline_state)

        # the struct we use to log the agent observations and the env state
        log_state = LogEnvState(
            env_state,
            jnp.zeros((self.num_agents,)),
            jnp.zeros((self.num_agents,)),
            jnp.zeros((self.num_agents,)),
            jnp.zeros((self.num_agents,)),
            jnp.zeros((), jnp.int32), # the env step number in the current rollout
            new_first_pipeline_state,
            new_first_obs,
            jnp.array(0.)
        )

        return agent_obs, log_state

    def step(
        self,
        rng: jr.PRNGKey, # this is not used in our deterministic env
        state: State, # this is the LogEnvState
        actions: Dict[str, jax.Array], # this is the agentic actions
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
        # TODO: global env_state step (reward, obs, state, metrics, done) #
        # =============================================================== #

        # here we calculate the global reward, obs, state, metrics, and the global_done
        # NOTE the global_done is only for early termination (the end of episode termination
        # situation is handled later automatically - look through the remainder of this 
        # method to understand)

        # we get the global actions
        global_action = self.map_agents_to_global_action(actions)

        # save the old pipeline_state to calculate some velocities
        pipeline_state0 = state.env_state.pipeline_state
        assert pipeline_state0 is not None

        # get the NEXT pipeline state yaaay
        next_pipeline_state = self.pipeline_step(state.env_state.pipeline_state, global_action)  # type: ignore

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
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(global_action))
        contact_cost = 0.0

        # finalise the global_obs, global_reward, and global_done (just for early termination)
        global_obs = self.get_global_obs(next_pipeline_state)
        global_reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        global_done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        
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

        # agent obs is formed AFTER we decide if the env is reset or not as this will
        # change the state which we observe

        # ==================================================================== #
        # UNCHANGED: Episode Wrapping Step (brax.envs.wrappers.EpisodeWrapper) #
        # ==================================================================== #

        step_in_episode = state.step_in_episode + self.action_repeat

        global_done = jnp.where(step_in_episode >= self.episode_length, jnp.ones_like(state.env_state.done), global_done)
        state = state._replace(
            truncation = jnp.where(
                step_in_episode >= jnp.array(self.episode_length), 1 - global_done, jnp.zeros_like(state.env_state.done)
            )
        )

        # ===================================== #
        # TODO: Agentic Rewards Dones Transform #
        # ===================================== #

        # default behaviour is just to give all agents same global reward
        reward = {agent: global_reward for agent in self.agents}
        reward["__all__"] = global_reward
        done = {agent: global_done.astype(jnp.bool_) for agent in self.agents}
        done["__all__"] = global_done.astype(jnp.bool_)

        # create new env_state here in ase global rewards or global dones rely on agentic things
        env_state = state.env_state.replace(
            pipeline_state=next_pipeline_state, 
            obs=global_obs, 
            reward=global_reward, 
            done=global_done
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

        global_obs = jax.tree.map(where_done, state.first_obs, global_obs)
        env_state = env_state.replace(pipeline_state=next_pipeline_state, obs=global_obs)
        agent_obs = self.map_global_obs_to_agents(global_obs)

        # first_pipeline_state, rng = jax.lax.cond(
        #     done["__all__"], 
        #     lambda: (self.get_random_pipeline_state(rng), jr.split(rng)[0]),
        #     lambda: (state.first_pipeline_state, rng)
        # )
        first_pipeline_state = self.get_random_pipeline_state(rng); rng, _ = jr.split(rng)
        first_obs = self.get_global_obs(first_pipeline_state)
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

        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = state._replace(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            step_in_episode=step_in_episode
        )

        info = env_state.info
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self.num_agents,), ep_done)

        return agent_obs, state, reward, done, info

    # ================================================================ #
    # UNCHANGED: mapping agent actions and obs to and from global actions and obs #
    # ================================================================ #

    def map_agents_to_global_action(
        self, agent_actions: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        global_action = jnp.zeros(self.action_size)
        for agent_name, action_indices in self.agent_action_mapping.items():
            if self.homogenisation_method == "max":
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name][: action_indices.size]
                )
            elif self.homogenisation_method == "concat":
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name][action_indices]
                )
            else:
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name]
                )
        return global_action

    def map_global_obs_to_agents(self, global_obs: jax.Array) -> Dict[str, jax.Array]:
        """Maps the global observation vector to the individual agent observations.
        Args:
            global_obs: The global observation vector.
        Returns:
            A dictionary mapping agent names to their observations. The mapping method
            is determined by the homogenisation_method parameter.
        """
        agent_obs = {}
        for agent_idx, (agent_name, obs_indices) in enumerate(
            self.agent_obs_mapping.items()
        ):
            if self.homogenisation_method == "max":
                # Vector with the agent idx one-hot encoded as the first num_agents
                # elements and then the agent's own observations (zero padded to
                # the size of the largest agent observation vector)
                agent_obs[agent_name] = (
                    jnp.zeros(
                        self.num_agents
                        + max([v.size for v in self.agent_obs_mapping.values()])
                    )
                    .at[agent_idx]
                    .set(1)
                    .at[agent_idx + 1 : agent_idx + 1 + obs_indices.size]
                    .set(global_obs[obs_indices])
                )
            elif self.homogenisation_method == "concat":
                # Zero vector except for the agent's own observations
                # (size of the global observation vector)
                agent_obs[agent_name] = (
                    jnp.zeros(global_obs.shape)
                    .at[obs_indices]
                    .set(global_obs[obs_indices])
                )
            else:
                # Just agent's own observations
                agent_obs[agent_name] = global_obs[obs_indices]

        # MAPPO difference:
        agent_obs["global_state"] = global_obs
        
        return agent_obs

def generate_rollout(env, rng, jit=True, num_timesteps=100):
    agent_obs, state = env.reset(rng=rng); rng, _rng = jr.split(rng)
    rollout = []
    if jit is True:
        env_step = jax.jit(env.step)
    else:
        env_step = env.step
    ctrl = jr.uniform(_rng, shape=(env.action_size), minval=-1, maxval=1)
    agent_ctrls = {"agent_0": ctrl[:2], "agent_1": ctrl[2:]}
    for i in range(num_timesteps):
        print(f"ctrl action chosen: {ctrl}")
        agent_obs, state, reward, done, info = env_step(rng, state, agent_ctrls)
        print(f"state.done: {done}")
        print(f"state.reward: {reward}")
        rollout.append(state.env_state.pipeline_state)
        print(f"step: {i}")
        print(f"info: {info}")
        if i == env.episode_length:
            print("THE PRIOR STATE.DONE SHOULD HAVE BEEN TRUE")
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'data/rollouts/{current_datetime}.html'
    os.makedirs("data/rollouts", exist_ok=True)
    with open(save_name, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))

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

    env = YourEnv()
    agent_obs, state = env.reset(jr.PRNGKey(0))

    # a single test step
    action = {"agent_0": jnp.array([0.5, 0.5]), "agent_1": jnp.array([-0.5, -0.5])}
    agent_obs, state, reward, done, info = env.step(jr.PRNGKey(0), state, action)

    # a test rollout
    generate_rollout(env, jr.PRNGKey(0), jit=True, num_timesteps=700)
