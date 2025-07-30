"""
Template to fill out to create a multi-agent RL environment using
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
import dataclasses as dc

import jax
import jax.numpy as jnp
import jax.random as jr
from eqx_marl.common import spaces # this is analagous to gym.Box and other "spaces"
import mujoco as mj # we use mj to change some foundational things in the simulation
from brax import math # this contains some useful things, like safe_norm and quaternion utils
from typing import Dict, Literal, Optional, Tuple, List, Union, NamedTuple

from waymax import env, config, dynamics, datatypes, dataloader, agents, visualization

from waymax.datatypes import observation_from_state, sdc_observation_from_state
from waymax.agents import actor_core
from waymax.datatypes import Action
from waymax.rewards import LinearCombinationReward

# --- Monkey-patch for jax.util.unzip2 for waymax ---
import jax.util

# Recreate the removed unzip2 function
def _unzip2(xs):
  if not xs:
    return (), ()
  return tuple(zip(*xs))

# Add the function back to the jax.util module
jax.util.unzip2 = _unzip2
# --- End of patch ---

# ============================================== #
# UNCHANGED: Datastructures and Helper Functions #
# ============================================== #

class LogEnvState(NamedTuple):
    env_state: jax.Array
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    step_in_episode: jax.Array
    first_pipeline_state: jax.Array
    first_obs: jax.Array
    truncation: jax.Array

class EnvState(NamedTuple):
    pipeline_state: jax.Array # not really an array lmao
    global_obs: jax.Array # not really an array lmao
    reward: jax.Array # not really an array lmao
    done: jax.Array
    metrics: Dict
    info: Dict

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

flatten_last_2_dim = lambda x: jnp.reshape(x, x.shape[:-2] + (-1,))


# ================= #
# Your Environment! #
# ================= #

# ================= #
# Waymax Adapted Environment! #
# ================= #

class IPPO_WaymaxEnv:
    def __init__(
        self,

        # base env settings
        batch_size=2,

        # multi agent env settings
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
        homogenisation_method: Optional[Literal["max", "concat"]] = None,

        # logging settings
        replace_info: bool = False,
        **kwargs
    ):

        self.batch_size = batch_size
        self.maximum_number_of_objects = 6
        self.env_config = config.DatasetConfig(
            path="gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000",
            max_num_objects=self.maximum_number_of_objects,
            batch_by_scenario=True,
            batch_dims=(batch_size,)
        )
        self.scenarios = dataloader.simulator_state_generator(config=self.env_config)
        self.dynamics_model = dynamics.InvertibleBicycleModel()
        action_spec = self.dynamics_model.action_spec()

        # Expect users to control all valid object in the scene.
        self.waymax_env = env.MultiAgentEnvironment(
            dynamics_model=self.dynamics_model,
            config=dc.replace(
                config.EnvironmentConfig(),
                max_num_objects=self.maximum_number_of_objects,
                controlled_object=config.ObjectType.VALID,
            ),
        )

        self.states = [self.waymax_env.reset(next(self.scenarios))]
        self.current_state = self.states[-1]
        self.timestep = 0
        self.done = False

        log_trajectory = self.current_state.log_trajectory
        self.num_agents = log_trajectory.num_objects
        self.num_timesteps = log_trajectory.num_timesteps
        self.observation_spaces = log_trajectory.xyz.shape

        self.agent_action_space = action_spec.shape[0] # per agent action_space

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
        scenario = next(self.scenarios)
        obs = flatten_last_2_dim(self.get_agent_obs(scenario))
        obs_size = obs.shape[-1]
        ranges = {"agent_"+str(i): [(0, obs_size-1)] for i in range(self.num_agents)}
        self.agent_obs_mapping = {k: jnp.array(listerize(v)) for k, v in ranges.items()} # _agent_observation_mapping[env_name]

        # the agent action mapping is simpler, so we just use the indices of the actions
        self.agent_action_mapping = {"agent_"+str(i): jnp.array(listerize([(i*(self.agent_action_space-1), (i+1)*(self.agent_action_space-1))])) for i in range(self.num_agents)}
        self.agents = list(self.agent_obs_mapping.keys())

        combination_config = config.LinearCombinationRewardConfig(
            {'overlap': -1.0, 'offroad': 1.0}
        )
        self.reward_func = (
            LinearCombinationReward(combination_config)
        )

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
            agent: spaces.Box(action_spec.minimum, action_spec.maximum, shape=action_spec.shape,)
            for agent in self.agents
        }

        # utility function to batchify floats originally placed in JaxMARLWrapper
        self._batchify_floats = lambda x: jnp.stack([x[a] for a in self.agents])

        # utility function to get obs, action spaces for each agent - required by the ppo algs
        self.observation_space = lambda agent: self.observation_spaces[agent]
        self.action_space = lambda agent: self.action_spaces[agent]

        self.action_size = self.maximum_number_of_objects * action_spec.shape[0]

    # ================================== #
    # TODO: design random reset function #
    # ================================== #
    
    def get_agent_obs(self, pipeline_state):
        global_obs_global_frame = observation_from_state(pipeline_state)
        global_obs_object_frame = observation_from_state(pipeline_state, coordinate_frame=config.CoordinateFrame.OBJECT)
        goal_xy = pipeline_state.log_trajectory.xy[:,:,-1]
        # goal_yaw = pipeline_state.log_trajectory.yaw[:,:,-1] # we dont care
        start_xy = pipeline_state.log_trajectory.xy[:,:,0]
        relative_goal_xy = (goal_xy - start_xy)[:,:,None,:] # reshape for concatenation
        
        valid = global_obs_object_frame.valid[:,:,None,None]
        valid = jnp.concatenate([valid]*relative_goal_xy.shape[-1], axis=-1)

        agent_obs = jnp.concatenate([
            relative_goal_xy, # relative position to goal
            global_obs_global_frame.trajectory.xy.squeeze(), # absolute positions
            global_obs_global_frame.trajectory.vel_xy.squeeze(), # absolute velocities
            global_obs_object_frame.trajectory.xy.squeeze(), # relative positions
            global_obs_object_frame.trajectory.vel_xy.squeeze(), # relative velocities
            global_obs_object_frame.roadgraph_static_points.xy.squeeze(), # relative positions to road
            valid, # still valid or not
        ], axis=2)
        return agent_obs

    # NOTE this is pre-vmapped
    def reset(self, rng: jr.PRNGKey) -> Tuple[Dict[str, jax.Array], jax.Array]:

        # =========================== #
        # TODO: Reset the environment #
        # =========================== #

        # TODO reset the global environment as you see fit for your environment. You
        # should end up with a new env_state: State and agent_obs: Dict[str, jax.Array]
        # as the below example does. NOTE you should also include the exact same info
        # dict in the resulting env_state as the example does.
        
        # next batch of states and global_obs
        pipeline_state = next(self.scenarios)
        agent_obs = self.get_agent_obs(pipeline_state)

        batching_shape = [self.batch_size, self.maximum_number_of_objects]
        reward = jnp.zeros(self.batch_size)
        done = jnp.zeros(self.batch_size)

        metrics = {}

        # NOTE it is essential that info is created like this and added to env_state
        info = {
            "returned_episode_returns": jnp.zeros(batching_shape),
            "returned_episode_lengths": jnp.zeros(batching_shape),
            "returned_episode": jnp.zeros(batching_shape).astype(jnp.bool_)
        }

        env_state = EnvState(pipeline_state, agent_obs, reward, done, metrics, info)

        # JOHN YOU ARE HERE

        # agent_obs = jax.vmap(self.map_global_obs_to_agents)(global_obs)

        # ============================= #
        # UNCHANGED: log state wrapping #
        # ============================= #

        # NOTE we change the "first_pipeline_state" and "first_obs" at every
        # usage, therefore we generate a new pair here to be used as the first
        # upon the next automatic reset in step - I will explain the automatic reset
        # later! fear not!

        new_first_pipeline_state = next(self.scenarios)
        new_first_obs = self.get_agent_obs(pipeline_state)

        # the struct we use to log the agent observations and the env state
        log_state = LogEnvState(
            env_state,
            jnp.zeros(batching_shape),
            jnp.zeros(batching_shape),
            jnp.zeros(batching_shape),
            jnp.zeros(batching_shape),
            jnp.zeros((self.batch_size), jnp.int32), # the env step number in the current rollout
            new_first_pipeline_state,
            new_first_obs,
            jnp.array((self.batch_size))
        )

        return agent_obs, log_state

    def step(
        self,
        rng: jr.PRNGKey, # this is not used in our deterministic env
        state: jax.Array, # this is the LogEnvState
        actions: Dict[str, jax.Array], # this is the agentic actions
    ) -> Tuple[
        Dict[str, jax.Array], jax.Array, Dict[str, float], Dict[str, bool], Dict
    ]:

        # We first ensure that states that were previously done (that already
        # have had their states reset) have their done flag reset
        state = state._replace(
            env_state=state.env_state._replace(
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
        # global_action = jax.vmap(self.map_agents_to_global_action)(actions)

        # save the old pipeline_state to calculate some velocities
        pipeline_state0 = state.env_state.pipeline_state
        assert pipeline_state0 is not None

        # get the NEXT pipeline state yaaay
        # next_pipeline_state = self.pipeline_step(state.env_state.pipeline_state, global_action)  # type: ignore

        # step the waymax env
        # I am so sorry - I am in a rush
        valid = state.env_state.global_obs[:,:,-1][...,None][:,:,0,:].astype(jnp.bool)
        waymax_action = Action(data=actions, valid=valid)
        next_pipeline_state = jax.vmap(self.waymax_env.step)(state.env_state.pipeline_state, waymax_action)

        # environment specific calculations
        # velocity = (next_pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        # forward_reward = velocity[0]

        # min_z, max_z = self._healthy_z_range

        # is_healthy = jnp.where(next_pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        # is_healthy = jnp.where(next_pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)

        agent_obs = self.get_agent_obs(next_pipeline_state)

        valid = agent_obs[:,:,-1,0]

        # if not healthy then we terminate later - with global done for that env
        is_healthy = jnp.all(valid != 0, axis=1).astype(jnp.int32)

        # finalise the global_obs, global_reward, and global_done (just for early termination)
        # global_obs = self.get_global_obs(next_pipeline_state)
        # global_reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        global_done = 1.0 - is_healthy # if self._terminate_when_unhealthy else 0.0

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
        agent_mask = jnp.ones(
            state.env_state.pipeline_state.sim_trajectory.shape[:2],
        )
        agent_reward = self.reward_func.compute(state.env_state.pipeline_state, waymax_action, agent_mask)
        global_reward = agent_reward.mean()
        # default behaviour is just to give all agents same global reward
        reward = {agent: agent_reward[:,i] for i, agent in enumerate(self.agents)}
        reward["__all__"] = global_reward
        done = {agent: global_done.astype(jnp.bool_) for agent in self.agents}
        done["__all__"] = global_done.astype(jnp.bool_)

        # create new env_state here in ase global rewards or global dones rely on agentic things
        env_state = state.env_state._replace(
            pipeline_state=next_pipeline_state,
            global_obs=agent_obs,
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

        global_obs = jax.tree.map(where_done, state.first_obs, agent_obs)
        env_state = env_state._replace(pipeline_state=next_pipeline_state, global_obs=global_obs)
        agent_obs = global_obs # self.map_global_obs_to_agents(global_obs)

        # first_pipeline_state, rng = jax.lax.cond(
        #     done["__all__"],
        #     lambda: (self.get_random_pipeline_state(rng), jr.split(rng)[0]),
        #     lambda: (state.first_pipeline_state, rng)
        # )
        first_pipeline_state = next(self.scenarios)# self.get_random_pipeline_state(rng); rng, _ = jr.split(rng)
        first_obs = self.get_agent_obs(first_pipeline_state)
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
        new_episode_return = state.episode_returns + self._batchify_floats(reward).T
        new_episode_length = state.episode_lengths + 1
        state = state._replace(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done[:,None]),
            episode_lengths=new_episode_length * (1 - ep_done[:,None]),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done[:,None])
            + new_episode_return * ep_done[:,None],
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done[:,None])
            + new_episode_length * ep_done[:,None],
            step_in_episode=step_in_episode
        )

        info = env_state.info
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self.batch_size, self.num_agents), ep_done[:,None])

        return agent_obs, state, reward, done, info

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["MUJOCO_GL"] = "egl"     # if you have NVIDIA + EGL drivers for rendering

    jax.config.update("jax_debug_nans", True)   # will error if a nan is detected
    jax.config.update("jax_log_compiles", True) # will print out recompilations
    jax.config.update('jax_default_matmul_precision', "highest") # sometimes certain contact dynamics need higher accuracy to prevent nans
    # this set of configs lets us cache some stuff to lower JIT times
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    import equinox as eqx

    env = IPPO_WaymaxEnv()
    rng = jr.PRNGKey(0)
    agent_obs, state = env.reset(rng); _rng, rng = jr.split(rng)

    # a single test step
    action = {
        "agent_0": jnp.array([0.5, 0.5]), 
        "agent_1": jnp.array([-0.5, -0.5]),
        "agent_2": jnp.array([-0.5, -0.5]),
        "agent_3": jnp.array([-0.5, -0.5]),
        "agent_4": jnp.array([-0.5, -0.5]),
        "agent_5": jnp.array([-0.5, -0.5])
    }

    from eqx_marl.common.models import ActorCritic

    network = ActorCritic(
        key=_rng,
        actor_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, env.action_space(env.agents[0]).shape[0]],
        critic_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, 1],
        actor_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 0.01],
        critic_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 1],
        activation=jax.nn.tanh,
    ); _rng, rng = jr.split(rng)

    action = eqx.filter_vmap(eqx.filter_vmap(network))(flatten_last_2_dim(agent_obs))[0]

    agent_obs, state, reward, done, info = env.step(_rng, state, action)

    # a test rollout
    generate_rollout(env, rng, jit=True, num_timesteps=700)
