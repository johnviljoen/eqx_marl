"""
During training many things can go wrong, but we can figure out what it is through
DEBUGGING... In creating this code I had to do plenty of debugging and had to build 
many many many things. I have left the most useful stuff in this file for future reference
"""

import os
from typing import NamedTuple
from datetime import datetime
import dataclasses as dc
from typing import Any
import jax
import jax.numpy as jnp
import jax.random as jr
from brax.io import html
from MARL_resources.policies import ActorCritic
import distreqx.distributions as dist
import equinox as eqx
from tqdm import tqdm

def find_nan_env_indices(state):
    """finds which envs in a vmapped state has a nan if any"""
    nan_envs = set()
    leaves = jax.tree_util.tree_leaves(state)
    for leaf in leaves:
        if isinstance(leaf, jnp.ndarray) and leaf.ndim > 0:
            nan_mask = jnp.any(jnp.isnan(leaf), axis=tuple(range(1, leaf.ndim)))
            indices = jnp.where(nan_mask)[0].tolist()
            nan_envs.update(indices)
    return sorted(nan_envs)


def extract_single_env_rollout(rollout, env_idx: int):
    """just for rendering one of the envs at the end"""
    single_rollout = []
    for state in rollout:
        # For each element in the PyTree, index with env_idx if possible.
        single_state = jax.tree_util.tree_map(
            lambda x: x[env_idx] if hasattr(x, "__getitem__") else x,
            state
        )
        single_rollout.append(single_state)
    return single_rollout


def _merge(target: Any, source: Any) -> Any:
    """Recursively copy matching fields/keys from `source` into `target`.

    • If both objects are dataclasses, we walk through the fields that exist
      in `target`, updating each when the same‑named field is present in
      `source`.  
    • If they are mappings (e.g. FrozenDict, dict) we update the values of
      the keys already in `target`.  
    • If they are sequences (tuple/list) we zip and merge element‑wise.  
    • For leaf values we simply return `source` (i.e. overwrite).
    """
    # ── dataclass branch ───────────────────────────────────────────────
    if dc.is_dataclass(target) and dc.is_dataclass(source):
        updates = {
            f.name: _merge(getattr(target, f.name), getattr(source, f.name))
            for f in dc.fields(target)
            if hasattr(source, f.name)                           # field exists in source
        }
        return dc.replace(target, **updates) if updates else target

    # ── mapping branch (works with FrozenDict or plain dict) ──────────
    if isinstance(target, dict) and isinstance(source, dict):
        return target.__class__(
            {k: _merge(target[k], source[k]) if k in source else target[k]
             for k in target}
        )

    # ── sequence branch ───────────────────────────────────────────────
    if (isinstance(target, (tuple, list))
            and isinstance(source, (tuple, list))):
        merged_elems = [_merge(t, s) for t, s in zip(target, source)]
        return target.__class__(merged_elems)

    # ── leaf value ────────────────────────────────────────────────────
    return source


def generate_and_validate_vmap_clp_rollout(env, rng, num_envs=100, jit=True, num_timesteps=1000):
    """
    This function will roll out a bunch of environments in parallel in closed loop,
    and detect and isolate environments in which NaNs occur for debugging purposes.
    """
    rng, _rng = jr.split(rng)
    agent_obs, state = jax.vmap(env.reset)(rng=jr.split(_rng, num_envs)); rng, _rng = jr.split(rng)
    rollout = []
    if jit is True:
        # the lambda form is for compatibility with vmapped dictionaries of actions
        env_step = jax.jit(jax.vmap(lambda key, state, action: env.step(key, state, action)))
    else:
        env_step = jax.vmap(lambda key, state, action: env.step(key, state, action))
    
    # ensemble of env.num_agents policies
    policy = jax.vmap(ActorCritic, in_axes=[0, None, None, None, None, None])(
        jr.split(_rng, env.num_agents),
        [env.observation_space(env.agents[0]).shape[0], 64, 64, env.action_space(env.agents[0]).shape[0]],
        [env.observation_space(env.agents[0]).shape[0], 64, 64, 1],
        [jnp.sqrt(2), jnp.sqrt(2), 0.01],
        [jnp.sqrt(2), jnp.sqrt(2), 1],
        jax.nn.tanh,
    ); rng, _rng = jr.split(rng)

    call_policy = jax.vmap(lambda x, policy: policy(x))
    vmap_call_policy = jax.vmap(call_policy, in_axes=[0, None])

    agents = sorted(agent_obs.keys())          # ['agent_0', 'agent_1', …]
    T, obs_dim = agent_obs[agents[0]].shape    # 100, 18
    num_agents = len(agents)                   # 4

    rollout = []
    for i in tqdm(range(num_timesteps)):
        obs = jnp.stack([agent_obs[a] for a in agents], axis=1)   # (100, 4, 18)
        mean, scale, value = vmap_call_policy(obs, policy)
        pi = eqx.filter_vmap(eqx.filter_vmap(dist.MultivariateNormalDiag))(mean, scale)
        act = pi.sample(_rng); rng, _rng = jr.split(rng)
        agent_acts = {a: act[:, i, :] for i, a in enumerate(agents)}
        agent_obs, state, reward, done, info = env_step(jr.split(_rng, num_envs), state, agent_acts); rng, _rng = jr.split(rng)
        isnan = find_nan_env_indices(state)
        if isnan:
            nan_rollout = extract_single_env_rollout(rollout, isnan[0]) # choose zeroth nan rollout
            # lets render the failure
            with open("test.html", 'w') as f:
                f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), nan_rollout[:]))
            pipeline_state0 = nan_rollout[-3]
            next_pipeline_state = nan_rollout[-2]

            print('fin')
        rollout.append(state.env_state.pipeline_state)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'data/rollouts/{current_datetime}.html'
    os.makedirs("data/rollouts", exist_ok=True)
    with open(save_name, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), extract_single_env_rollout(rollout, 0)))

    pass


def validate_eqx_against_flax_clp_rollout(eqx_env, flx_env, rng, jit=True, num_timesteps=1000):
    import numpy as np

    rng, _rng = jr.split(rng)
    eqx_agent_obs, eqx_state = eqx_env.reset(rng=_rng); rng, _rng = jr.split(rng)
    flx_agent_obs, flx_state = flx_env.reset(key=_rng); rng, _rng = jr.split(rng)

    # new_env_state = _merge(flx_state.env_state, eqx_state.env_state)
    flx_state = flx_state.replace(
        env_state=flx_state.env_state.replace(
            pipeline_state = eqx_state.env_state.pipeline_state
        )
    )

    if jit is True:
        # the lambda form is for compatibility with vmapped dictionaries of actions
        eqx_env_step = jax.jit(eqx_env.step)
        flx_env_step = jax.jit(flx_env.step)
    else:
        eqx_env_step = eqx_env.step
        flx_env_step = flx_env.step
    eqx_agents = sorted(eqx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    flx_agents = sorted(flx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    assert eqx_agents == flx_agents

    policy = jax.vmap(ActorCritic, in_axes=[0, None, None, None, None, None])(
        jr.split(_rng, eqx_env.num_agents),
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, eqx_env.action_space(eqx_env.agents[0]).shape[0]],
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, 1],
        [jnp.sqrt(2), jnp.sqrt(2), 0.01],
        [jnp.sqrt(2), jnp.sqrt(2), 1],
        jax.nn.tanh,
    ); rng, _rng = jr.split(rng)

    call_policy = jax.vmap(lambda x, policy: policy(x))

    eqx_rollout = []
    flx_rollout = []
    eqx_done = jnp.zeros(1, dtype=bool)
    flx_done = jnp.zeros(1, dtype=bool)
    eqx_lengths = []
    flx_lengths = []
    eqx_lens = jnp.zeros(1)
    flx_lens = jnp.zeros(1)
    eqx_sum_unhealthy = 0
    flx_sum_unhealthy = 0

    for i in tqdm(range(num_timesteps)):
        obs = jnp.stack([eqx_agent_obs[a] for a in eqx_agents], axis=1)   # (100, 4, 18)
        mean, scale, value = call_policy(obs.T, policy)
        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
        act = pi.sample(_rng); rng, _rng = jr.split(rng)
        agent_acts = {a: act[i, :] for i, a in enumerate(eqx_agents)} # use the same action to step the env
        
        # perform the step
        eqx_agent_obs, eqx_state, eqx_reward, eqx_done, eqx_info = eqx_env_step(_rng, eqx_state, agent_acts); rng, _rng = jr.split(rng)
        flx_agent_obs, flx_state, flx_reward, flx_done, flx_info = flx_env_step(_rng, flx_state, agent_acts); rng, _rng = jr.split(rng)
        eqx_rollout.append(eqx_state.env_state.pipeline_state)
        flx_rollout.append(flx_state.env_state.pipeline_state)

        # =========================================== #
        # ====== Record the length of rollouts ====== #
        # =========================================== #

        eqx_max_lens = eqx_lens[eqx_done["__all__"]]
        eqx_lengths.append(eqx_max_lens)

        # +1 to lens which havent terminated
        eqx_lens = np.where(~eqx_done["__all__"], eqx_lens+1, 0)       

        flx_max_lens = flx_lens[flx_done["__all__"]]
        flx_lengths.append(flx_max_lens)

        # +1 to lens which havent terminated
        flx_lens = np.where(~flx_done["__all__"], flx_lens+1, 0)


        # ============================================ #
        # == Determine if the systems are "healthy" == #
        # ============================================ #

        def global_done(pipeline_state, healthy_z_range):

            min_z, max_z = healthy_z_range
            is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
            is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
            return is_healthy
        
        assert eqx_env._healthy_z_range == flx_env._env.env._healthy_z_range

        eqx_is_healthy = global_done(eqx_state.env_state.pipeline_state, eqx_env._healthy_z_range)
        flx_is_healthy = global_done(flx_state.env_state.pipeline_state, flx_env._env.env._healthy_z_range)

        eqx_sum_unhealthy += (1 - eqx_is_healthy).sum()
        flx_sum_unhealthy += (1 - flx_is_healthy).sum()
        
    flx_lengths = np.concat(flx_lengths)
    eqx_lengths = np.concat(eqx_lengths)

    print(f"equinox mean length: {eqx_lengths.mean()}")
    print(f"flax mean length:    {flx_lengths.mean()}")
    print(f"equinox length std: {eqx_lengths.std()}")
    print(f"flax length std:    {flx_lengths.std()}")

    pass




def vmap_eqx_vs_flx(eqx_env, flx_env, rng, num_envs=100, jit=True, num_timesteps=1000):
    """
    This function will roll out a bunch of environments in parallel in closed loop,
    for both equinox and flax equivalent implementations. Then we will compare every part
    of the two environments to ensure they are equivalent.
    """
    import numpy as np

    rng, _rng = jr.split(rng)
    eqx_agent_obs, eqx_state = jax.vmap(eqx_env.reset)(rng=jr.split(_rng, num_envs)); rng, _rng = jr.split(rng)
    flx_agent_obs, flx_state = jax.vmap(flx_env.reset)(key=jr.split(_rng, num_envs)); rng, _rng = jr.split(rng)
    
    # new_env_state = _merge(flx_state.env_state, eqx_state.env_state)
    flx_state = flx_state.replace(
        env_state=flx_state.env_state.replace(
            pipeline_state = eqx_state.env_state.pipeline_state
        )
    )

    if jit is True:
        # the lambda form is for compatibility with vmapped dictionaries of actions
        eqx_env_step = jax.jit(jax.vmap(lambda key, state, action: eqx_env.step(key, state, action), axis_name="i"))
        flx_env_step = jax.jit(jax.vmap(lambda key, state, action: flx_env.step(key, state, action)))
    else:
        eqx_env_step = jax.vmap(lambda key, state, action: eqx_env.step(key, state, action))
        flx_env_step = jax.vmap(lambda key, state, action: flx_env.step(key, state, action))
    eqx_agents = sorted(eqx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    flx_agents = sorted(flx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    assert eqx_agents == flx_agents
        # ensemble of env.num_agents policies
    
    policy = jax.vmap(ActorCritic, in_axes=[0, None, None, None, None, None])(
        jr.split(_rng, eqx_env.num_agents),
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, eqx_env.action_space(eqx_env.agents[0]).shape[0]],
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, 1],
        [jnp.sqrt(2), jnp.sqrt(2), 0.01],
        [jnp.sqrt(2), jnp.sqrt(2), 1],
        jax.nn.tanh,
    ); rng, _rng = jr.split(rng)

    call_policy = jax.vmap(lambda x, policy: policy(x))
    vmap_call_policy = jax.vmap(call_policy, in_axes=[0, None])
    
    eqx_rollout = []
    flx_rollout = []
    eqx_done = jnp.zeros(num_envs, dtype=bool)
    flx_done = jnp.zeros(num_envs, dtype=bool)
    eqx_lengths = []
    flx_lengths = []
    eqx_lens = jnp.zeros([num_envs])
    flx_lens = jnp.zeros([num_envs])
    eqx_sum_unhealthy = 0
    flx_sum_unhealthy = 0
    for i in tqdm(range(num_timesteps)):
        obs = jnp.stack([eqx_agent_obs[a] for a in eqx_agents], axis=1)   # (100, 4, 18)
        mean, scale, value = vmap_call_policy(obs, policy)
        pi = eqx.filter_vmap(eqx.filter_vmap(dist.MultivariateNormalDiag))(mean, scale)
        act = pi.sample(_rng); rng, _rng = jr.split(rng)
        agent_acts = {a: act[:, i, :] for i, a in enumerate(eqx_agents)} # use the same action to step the env
       

        # perform the step
        eqx_agent_obs, eqx_state, eqx_reward, eqx_done, eqx_info = eqx_env_step(jr.split(_rng, num_envs), eqx_state, agent_acts); rng, _rng = jr.split(rng)
        flx_agent_obs, flx_state, flx_reward, flx_done, flx_info = flx_env_step(jr.split(_rng, num_envs), flx_state, agent_acts); rng, _rng = jr.split(rng)
        # if the envs reset then we want them both to reset to the same state
        if eqx_done:
            eqx_state = eqx_state.replace(
                env_state=eqx_state.env_state.replace(
                    pipeline_state = flx_state.env_state.info["first_pipeline_state"]
                )
            )

        delta_q = jnp.linalg.norm(eqx_state.env_state.pipeline_state.q - flx_state.env_state.pipeline_state.q)
        delta_qd = jnp.linalg.norm(eqx_state.env_state.pipeline_state.qd - flx_state.env_state.pipeline_state.qd)
        
        eqx_rollout.append(eqx_state.env_state.pipeline_state)
        flx_rollout.append(flx_state.env_state.pipeline_state)

        # rest flx state to eqx state
        # eqx_state = eqx_state.replace(
        #     env_state=eqx_state.env_state.replace(
        #         pipeline_state = flx_state.env_state.pipeline_state
        #     )
        # )

        # =========================================== #
        # ====== Record the length of rollouts ====== #
        # =========================================== #

        eqx_max_lens = eqx_lens[eqx_done["__all__"]]
        eqx_lengths.append(eqx_max_lens)

        # +1 to lens which havent terminated
        eqx_lens = np.where(~eqx_done["__all__"], eqx_lens+1, 0)       

        flx_max_lens = flx_lens[flx_done["__all__"]]
        flx_lengths.append(flx_max_lens)

        # +1 to lens which havent terminated
        flx_lens = np.where(~flx_done["__all__"], flx_lens+1, 0)

        # if eqx_done["__all__"] and not flx_done["__all__"]:
        #     pass

    flx_lengths = np.hstack(flx_lengths)
    eqx_lengths = np.hstack(eqx_lengths)

    print(f"equinox mean length: {eqx_lengths.mean()}")
    print(f"flax mean length:    {flx_lengths.mean()}")
    print(f"equinox length std: {eqx_lengths.std()}")
    print(f"flax length std:    {flx_lengths.std()}")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("data/rollouts", exist_ok=True)

    save_name = f'data/rollouts/eqx_{current_datetime}.html'
    with open(save_name, 'w') as f:
        f.write(html.render(eqx_env.sys.tree_replace({'opt.timestep': eqx_env.dt}), extract_single_env_rollout(eqx_rollout, 0)))
    save_name = f'data/rollouts/flx_{current_datetime}.html'
    with open(save_name, 'w') as f:
        f.write(html.render(flx_env.sys.tree_replace({'opt.timestep': eqx_env.dt}), extract_single_env_rollout(flx_rollout, 0)))

    pass


def eqx_to_flx_cpy(eqx_env, flx_env, rng, jit=True, num_timesteps=1000):
    import numpy as np

    rng, _rng = jr.split(rng)
    eqx_agent_obs, eqx_state = eqx_env.reset(rng=_rng); rng, _rng = jr.split(rng)
    flx_agent_obs, flx_state = flx_env.reset(key=_rng); rng, _rng = jr.split(rng)

    # new_env_state = _merge(flx_state.env_state, eqx_state.env_state)
    # new_info = flx_state.env_state.info
    # new_info["first_pipeline_state"] = eqx_state.first_pipeline_state
    # new_info["first_obs"] = eqx_state.first_obs
    flx_state = flx_state.replace(
        env_state=flx_state.env_state.replace(
            pipeline_state = eqx_state.env_state.pipeline_state,
            # info = new_info
        )
    )

    if jit is True:
        # the lambda form is for compatibility with vmapped dictionaries of actions
        eqx_env_step = jax.jit(eqx_env.step)
        flx_env_step = jax.jit(flx_env.step)
    else:
        eqx_env_step = eqx_env.step
        flx_env_step = flx_env.step
    eqx_agents = sorted(eqx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    flx_agents = sorted(flx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    assert eqx_agents == flx_agents

    policy = jax.vmap(ActorCritic, in_axes=[0, None, None, None, None, None])(
        jr.split(_rng, eqx_env.num_agents),
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, eqx_env.action_space(eqx_env.agents[0]).shape[0]],
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, 1],
        [jnp.sqrt(2), jnp.sqrt(2), 0.01],
        [jnp.sqrt(2), jnp.sqrt(2), 1],
        jax.nn.tanh,
    ); rng, _rng = jr.split(rng)

    call_policy = jax.vmap(lambda x, policy: policy(x))

    eqx_rollout = []
    flx_rollout = []
    eqx_done = jnp.zeros(1, dtype=bool)
    flx_done = jnp.zeros(1, dtype=bool)
    eqx_lengths = []
    flx_lengths = []
    eqx_lens = jnp.zeros(1)
    flx_lens = jnp.zeros(1)
    eqx_sum_unhealthy = 0
    flx_sum_unhealthy = 0

    for i in tqdm(range(num_timesteps)):
        obs = jnp.stack([eqx_agent_obs[a] for a in eqx_agents], axis=1)   # (100, 4, 18)
        mean, scale, value = call_policy(obs.T, policy)
        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
        act = pi.sample(_rng); rng, _rng = jr.split(rng)
        agent_acts = {a: act[i, :] for i, a in enumerate(eqx_agents)} # use the same action to step the env
        
        # perform the step
        prior_eqx_q, prior_eqx_qd = eqx_state.env_state.pipeline_state.q, eqx_state.env_state.pipeline_state.qd
        prior_flx_q, prior_flx_qd = flx_state.env_state.pipeline_state.q, flx_state.env_state.pipeline_state.qd
        prior_eqx_done = eqx_done
        prior_flx_done = flx_done
        eqx_agent_obs, eqx_state, eqx_reward, eqx_done, eqx_info = eqx_env_step(_rng, eqx_state, agent_acts); rng, _rng = jr.split(rng)
        flx_agent_obs, flx_state, flx_reward, flx_done, flx_info = flx_env_step(_rng, flx_state, agent_acts); rng, _rng = jr.split(rng)
        
        # if the envs reset then we want them both to reset to the same state
        # if (eqx_state.env_state.pipeline_state.q == eqx_state.first_pipeline_state.q).all() \
        #     and (eqx_state.env_state.pipeline_state.qd == eqx_state.first_pipeline_state.qd).all():
        #     eqx_state = eqx_state.replace(
        #         env_state=eqx_state.env_state.replace(
        #             pipeline_state = flx_state.env_state.info["first_pipeline_state"]
        #         )
        #     )

        if eqx_done["__all__"] or flx_done["__all__"]: # if either done reset both
            assert eqx_done["__all__"] == flx_done["__all__"]
            print("resetting both states")
            eqx_state = eqx_state.replace(
                env_state=eqx_state.env_state.replace(
                    pipeline_state = eqx_state.first_pipeline_state,
                    obs = eqx_state.first_obs
                )
            )
            flx_state = flx_state.replace(
                env_state=flx_state.env_state.replace(
                    pipeline_state = eqx_state.first_pipeline_state,
                    obs = eqx_state.first_obs
                )
            )
            flx_agent_obs = eqx_env.map_global_obs_to_agents(eqx_state.first_obs)
        eqx_rollout.append(eqx_state.env_state.pipeline_state)
        flx_rollout.append(flx_state.env_state.pipeline_state)

        # Print discrepancy between updated q and qd
        current_eqx_q, current_eqx_qd = eqx_state.env_state.pipeline_state.q, eqx_state.env_state.pipeline_state.qd
        current_flx_q, current_flx_qd = flx_state.env_state.pipeline_state.q, flx_state.env_state.pipeline_state.qd        
        delta_q = jnp.linalg.norm(eqx_state.env_state.pipeline_state.q - flx_state.env_state.pipeline_state.q)
        delta_qd = jnp.linalg.norm(eqx_state.env_state.pipeline_state.qd - flx_state.env_state.pipeline_state.qd)
        # print(f"delta in pipeline_state.q: {delta_q}")
        # print(f"delta in pipeline_state.qd: {delta_qd}")

        # print(f"eqx_done: {eqx_done}")
        # print(f"flx_done: {flx_done}")

        if delta_q >= 1e-8 or delta_qd >= 1e-8:
            print(f"WARNING: dynamics discrepancy detected:")
            print(f"prior_eqx_q: {prior_eqx_q}")
            print(f"prior_flx_q: {prior_flx_q}")
            print(f"prior_eqx_qd: {prior_eqx_qd}")
            print(f"prior_flx_qd: {prior_flx_qd}\n")

            print(f"current_eqx_q:  {current_eqx_q}")
            print(f"current_flx_q:  {current_flx_q}")
            print(f"current_eqx_qd: {current_eqx_qd}")
            print(f"current_flx_qd: {current_flx_qd}\n")

            print(f"delta_q: {delta_q}")
            print(f"delta_qd: {delta_qd}")
            pass

        # reset flx state to eqx state
        # eqx_state = eqx_state.replace(
        #     env_state=eqx_state.env_state.replace(
        #         pipeline_state = flx_state.env_state.pipeline_state
        #     )
        # )

        # =========================================== #
        # ====== Record the length of rollouts ====== #
        # =========================================== #

        eqx_max_lens = eqx_lens[eqx_done["__all__"]]
        eqx_lengths.append(eqx_max_lens)

        # +1 to lens which havent terminated
        eqx_lens = np.where(~eqx_done["__all__"], eqx_lens+1, 0)       

        flx_max_lens = flx_lens[flx_done["__all__"]]
        flx_lengths.append(flx_max_lens)

        # +1 to lens which havent terminated
        flx_lens = np.where(~flx_done["__all__"], flx_lens+1, 0)

        if eqx_done["__all__"] and not flx_done["__all__"]:
            pass

        # these lengths should MATCH the step_in_episode measure from both envs
        # print(f"eqx_state.returned_episode_lengths: {eqx_state.returned_episode_lengths[0]}")
        # print(f"flx_state.returned_episode_lengths: {flx_state.returned_episode_lengths[0]}")

        # ============================================ #
        # == Compare everything else == #
        # ============================================ #

        eqx_truncation = eqx_state.truncation
        eqx_episode_lengths = eqx_state.episode_lengths
        eqx_episode_returns = eqx_state.episode_returns
        eqx_returned_episode_lengths = eqx_state.returned_episode_lengths
        eqx_returned_episode_returns = eqx_state.returned_episode_returns
        eqx_step_in_episode = eqx_state.step_in_episode
        eqx_done
        eqx_agent_obs
        eqx_reward
        # optionally metrics
        eqx_returned_episode = eqx_state.env_state.info["returned_episode"]

        flx_truncation = flx_state.env_state.info["truncation"]
        flx_episode_lengths = flx_state.episode_lengths
        flx_episode_returns = flx_state.episode_returns
        flx_returned_episode_lengths = flx_state.returned_episode_lengths
        flx_returned_episode_returns = flx_state.returned_episode_returns
        flx_step_in_episode = flx_state.env_state.info["steps"]
        flx_done
        flx_agent_obs
        flx_reward
        # optionally metrics
        flx_returned_episode = flx_info["returned_episode"]

        tests = {}
        tests["trunction"] = eqx_truncation == flx_truncation
        tests["episode_lengths"] = (eqx_episode_lengths == flx_episode_lengths).all()
        tests["episode_returns"] = (eqx_episode_returns == flx_episode_returns).all()
        tests["returned_episode_lengths"] = (eqx_returned_episode_lengths == flx_returned_episode_lengths).all()
        tests["returned_episode_returns"] = (eqx_returned_episode_returns == flx_returned_episode_returns).all()
        tests["step_in_episode"] = eqx_step_in_episode == flx_step_in_episode
        tests["done"] = eqx_done["__all__"] == flx_done["__all__"]
        tests["obs"] = all([(eqx_agent_obs[key] == flx_agent_obs[key]).all() for key in eqx_agent_obs])
        tests["reward"] = all([(eqx_reward[key] == flx_reward[key]).all() for key in eqx_reward])
        tests["returned_episode"] = (eqx_returned_episode == flx_returned_episode).all()

        for k, v in tests.items():
            if v == False:
                print(f"test: {k} failed")
        
    flx_lengths = np.concat(flx_lengths)
    eqx_lengths = np.concat(eqx_lengths)

    print(f"equinox mean length: {eqx_lengths.mean()}")
    print(f"flax mean length:    {flx_lengths.mean()}")
    print(f"equinox length std: {eqx_lengths.std()}")
    print(f"flax length std:    {flx_lengths.std()}")

    save_name = f'eqx.html'
    with open(save_name, 'w') as f:
        f.write(html.render(eqx_env.sys.tree_replace({'opt.timestep': eqx_env.dt}), eqx_rollout))
    save_name = f'flx.html'
    with open(save_name, 'w') as f:
        f.write(html.render(flx_env.sys.tree_replace({'opt.timestep': eqx_env.dt}), flx_rollout))

    pass


def flx_to_eqx_cpy(eqx_env, flx_env, rng, jit=True, num_timesteps=1000):
    import numpy as np

    rng, _rng = jr.split(rng)
    eqx_agent_obs, eqx_state = eqx_env.reset(rng=_rng); rng, _rng = jr.split(rng)
    flx_agent_obs, flx_state = flx_env.reset(key=_rng); rng, _rng = jr.split(rng)

    # new_env_state = _merge(flx_state.env_state, eqx_state.env_state)
    # new_info = flx_state.env_state.info
    # new_info["first_pipeline_state"] = eqx_state.first_pipeline_state
    # new_info["first_obs"] = eqx_state.first_obs
    eqx_state = eqx_state.replace(
        env_state=eqx_state.env_state.replace(
            pipeline_state = flx_state.env_state.pipeline_state,
            # info = new_info
        )
    )

    if jit is True:
        # the lambda form is for compatibility with vmapped dictionaries of actions
        eqx_env_step = jax.jit(eqx_env.step)
        flx_env_step = jax.jit(flx_env.step)
    else:
        eqx_env_step = eqx_env.step
        flx_env_step = flx_env.step
    eqx_agents = sorted(eqx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    flx_agents = sorted(flx_agent_obs.keys())          # ['agent_0', 'agent_1', …]
    assert eqx_agents == flx_agents

    policy = jax.vmap(ActorCritic, in_axes=[0, None, None, None, None, None])(
        jr.split(_rng, eqx_env.num_agents),
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, eqx_env.action_space(eqx_env.agents[0]).shape[0]],
        [eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, 1],
        [jnp.sqrt(2), jnp.sqrt(2), 0.01],
        [jnp.sqrt(2), jnp.sqrt(2), 1],
        jax.nn.tanh,
    ); rng, _rng = jr.split(rng)

    call_policy = jax.vmap(lambda x, policy: policy(x))

    eqx_rollout = []
    flx_rollout = []
    eqx_done = jnp.zeros(1, dtype=bool)
    flx_done = jnp.zeros(1, dtype=bool)
    eqx_lengths = []
    flx_lengths = []
    eqx_lens = jnp.zeros(1)
    flx_lens = jnp.zeros(1)
    eqx_sum_unhealthy = 0
    flx_sum_unhealthy = 0

    for i in tqdm(range(num_timesteps)):
        obs = jnp.stack([eqx_agent_obs[a] for a in eqx_agents], axis=1)   # (100, 4, 18)
        mean, scale, value = call_policy(obs.T, policy)
        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
        act = pi.sample(_rng); rng, _rng = jr.split(rng)
        agent_acts = {a: act[i, :] for i, a in enumerate(eqx_agents)} # use the same action to step the env
        
        # perform the step
        prior_eqx_q, prior_eqx_qd = eqx_state.env_state.pipeline_state.q, eqx_state.env_state.pipeline_state.qd
        prior_flx_q, prior_flx_qd = flx_state.env_state.pipeline_state.q, flx_state.env_state.pipeline_state.qd
        prior_eqx_done = eqx_done
        prior_flx_done = flx_done
        eqx_agent_obs, eqx_state, eqx_reward, eqx_done, eqx_info = eqx_env_step(_rng, eqx_state, agent_acts); rng, _rng = jr.split(rng)
        flx_agent_obs, flx_state, flx_reward, flx_done, flx_info = flx_env_step(_rng, flx_state, agent_acts); rng, _rng = jr.split(rng)
        
        # if the envs reset then we want them both to reset to the same state
        # if (eqx_state.env_state.pipeline_state.q == eqx_state.first_pipeline_state.q).all() \
        #     and (eqx_state.env_state.pipeline_state.qd == eqx_state.first_pipeline_state.qd).all():
        #     eqx_state = eqx_state.replace(
        #         env_state=eqx_state.env_state.replace(
        #             pipeline_state = flx_state.env_state.info["first_pipeline_state"]
        #         )
        #     )

        if eqx_done["__all__"] or flx_done["__all__"]: # if either done reset both
            assert eqx_done["__all__"] == flx_done["__all__"]

            print("resetting both states")
            eqx_state = eqx_state.replace(
                env_state=eqx_state.env_state.replace(
                    pipeline_state = flx_state.env_state.info["first_pipeline_state"],
                    obs = flx_state.env_state.info["first_obs"]
                )
            )
            flx_state = flx_state.replace(
                env_state=flx_state.env_state.replace(
                    pipeline_state = flx_state.env_state.info["first_pipeline_state"],
                    obs = flx_state.env_state.info["first_obs"]
                )
            )
            eqx_agent_obs = eqx_env.map_global_obs_to_agents(flx_state.env_state.info["first_obs"])
        eqx_rollout.append(eqx_state.env_state.pipeline_state)
        flx_rollout.append(flx_state.env_state.pipeline_state)

        # Print discrepancy between updated q and qd
        current_eqx_q, current_eqx_qd = eqx_state.env_state.pipeline_state.q, eqx_state.env_state.pipeline_state.qd
        current_flx_q, current_flx_qd = flx_state.env_state.pipeline_state.q, flx_state.env_state.pipeline_state.qd        
        delta_q = jnp.linalg.norm(eqx_state.env_state.pipeline_state.q - flx_state.env_state.pipeline_state.q)
        delta_qd = jnp.linalg.norm(eqx_state.env_state.pipeline_state.qd - flx_state.env_state.pipeline_state.qd)
        # print(f"delta in pipeline_state.q: {delta_q}")
        # print(f"delta in pipeline_state.qd: {delta_qd}")

        # print(f"eqx_done: {eqx_done}")
        # print(f"flx_done: {flx_done}")

        if delta_q >= 1e-8 or delta_qd >= 1e-8:
            print(f"WARNING: dynamics discrepancy detected:")
            print(f"prior_eqx_q: {prior_eqx_q}")
            print(f"prior_flx_q: {prior_flx_q}")
            print(f"prior_eqx_qd: {prior_eqx_qd}")
            print(f"prior_flx_qd: {prior_flx_qd}\n")

            print(f"current_eqx_q:  {current_eqx_q}")
            print(f"current_flx_q:  {current_flx_q}")
            print(f"current_eqx_qd: {current_eqx_qd}")
            print(f"current_flx_qd: {current_flx_qd}\n")

            print(f"delta_q: {delta_q}")
            print(f"delta_qd: {delta_qd}")
            pass

        # reset flx state to eqx state
        # eqx_state = eqx_state.replace(
        #     env_state=eqx_state.env_state.replace(
        #         pipeline_state = flx_state.env_state.pipeline_state
        #     )
        # )

        # =========================================== #
        # ====== Record the length of rollouts ====== #
        # =========================================== #

        eqx_max_lens = eqx_lens[eqx_done["__all__"]]
        eqx_lengths.append(eqx_max_lens)

        # +1 to lens which havent terminated
        eqx_lens = np.where(~eqx_done["__all__"], eqx_lens+1, 0)       

        flx_max_lens = flx_lens[flx_done["__all__"]]
        flx_lengths.append(flx_max_lens)

        # +1 to lens which havent terminated
        flx_lens = np.where(~flx_done["__all__"], flx_lens+1, 0)

        if eqx_done["__all__"] and not flx_done["__all__"]:
            pass

        # these lengths should MATCH the step_in_episode measure from both envs
        # print(f"eqx_state.returned_episode_lengths: {eqx_state.returned_episode_lengths[0]}")
        # print(f"flx_state.returned_episode_lengths: {flx_state.returned_episode_lengths[0]}")

        # ============================================ #
        # == Compare everything else == #
        # ============================================ #

        eqx_truncation = eqx_state.truncation
        eqx_episode_lengths = eqx_state.episode_lengths
        eqx_episode_returns = eqx_state.episode_returns
        eqx_returned_episode_lengths = eqx_state.returned_episode_lengths
        eqx_returned_episode_returns = eqx_state.returned_episode_returns
        eqx_step_in_episode = eqx_state.step_in_episode
        eqx_done
        eqx_agent_obs
        eqx_reward
        # optionally metrics
        eqx_returned_episode = eqx_state.env_state.info["returned_episode"]

        flx_truncation = flx_state.env_state.info["truncation"]
        flx_episode_lengths = flx_state.episode_lengths
        flx_episode_returns = flx_state.episode_returns
        flx_returned_episode_lengths = flx_state.returned_episode_lengths
        flx_returned_episode_returns = flx_state.returned_episode_returns
        flx_step_in_episode = flx_state.env_state.info["steps"]
        flx_done
        flx_agent_obs
        flx_reward
        # optionally metrics
        flx_returned_episode = flx_info["returned_episode"]

        tests = {}
        tests["trunction"] = eqx_truncation == flx_truncation
        tests["episode_lengths"] = (eqx_episode_lengths == flx_episode_lengths).all()
        tests["episode_returns"] = (eqx_episode_returns == flx_episode_returns).all()
        tests["returned_episode_lengths"] = (eqx_returned_episode_lengths == flx_returned_episode_lengths).all()
        tests["returned_episode_returns"] = (eqx_returned_episode_returns == flx_returned_episode_returns).all()
        tests["step_in_episode"] = eqx_step_in_episode == flx_step_in_episode
        tests["done"] = eqx_done["__all__"] == flx_done["__all__"]
        tests["obs"] = all([(eqx_agent_obs[key] == flx_agent_obs[key]).all() for key in eqx_agent_obs])
        tests["reward"] = all([(eqx_reward[key] == flx_reward[key]).all() for key in eqx_reward])
        tests["returned_episode"] = (eqx_returned_episode == flx_returned_episode).all()

        for k, v in tests.items():
            if v == False:
                print(f"test: {k} failed")
        
    flx_lengths = np.concat(flx_lengths)
    eqx_lengths = np.concat(eqx_lengths)

    print(f"equinox mean length: {eqx_lengths.mean()}")
    print(f"flax mean length:    {flx_lengths.mean()}")
    print(f"equinox length std: {eqx_lengths.std()}")
    print(f"flax length std:    {flx_lengths.std()}")

    save_name = f'eqx.html'
    with open(save_name, 'w') as f:
        f.write(html.render(eqx_env.sys.tree_replace({'opt.timestep': eqx_env.dt}), eqx_rollout))
    save_name = f'flx.html'
    with open(save_name, 'w') as f:
        f.write(html.render(flx_env.sys.tree_replace({'opt.timestep': eqx_env.dt}), flx_rollout))

    pass

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    # print('max_dim', max_dim)
    def pad(z):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))], -1)
    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def imitate_env_step(eqx_env, flx_env, rng, config):
    from MARL_resources.eqx_utils import filter_scan
    config["NUM_ACTORS"] = eqx_env.num_agents * config["NUM_ENVS"]
    assert eqx_env.num_agents == flx_env.num_agents
    rng, _rng = jr.split(rng)
    network = ActorCritic(
            key=_rng,
            actor_layer_sizes=[eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, eqx_env.action_space(eqx_env.agents[0]).shape[0]],
            critic_layer_sizes=[eqx_env.observation_space(eqx_env.agents[0]).shape[0], 64, 64, 1],
            actor_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 0.01],
            critic_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 1],
            activation=jax.nn.tanh,
        )
    
    reset_rng = jr.split(rng, config["NUM_ENVS"])
    eqx_agent_obs, eqx_state = jax.vmap(eqx_env.reset)(reset_rng); rng, _rng = jr.split(rng)
    flx_agent_obs, flx_state = jax.vmap(flx_env.reset)(reset_rng); rng, _rng = jr.split(rng)
    eqx_agent_obs = flx_agent_obs
    eqx_state = eqx_state.replace(
        env_state=eqx_state.env_state.replace(
            pipeline_state = flx_state.env_state.pipeline_state,
            # info = new_info
        )
    )
    eqx_env_step = jax.jit(jax.vmap(eqx_env.step))
    flx_env_step = jax.jit(jax.vmap(flx_env.step))
    call_network = eqx.filter_jit(eqx.filter_vmap(network))
    original_first_q_flx = flx_state.env_state.info["first_pipeline_state"].q

    eqx_traj_batch = []
    flx_traj_batch = []
    for i in tqdm(range(config["NUM_STEPS"])):
        eqx_obs_batch = batchify(eqx_agent_obs, eqx_env.agents, config["NUM_ACTORS"])
        flx_obs_batch = batchify(flx_agent_obs, flx_env.agents, config["NUM_ACTORS"])
        
        # SELECT ACTION
        rng, _rng = jr.split(rng)
        mean, scale, value = call_network(flx_obs_batch)
        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
        pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
        action = pi.sample(_rng)
        log_prob = eqx.filter_vmap(pi_log_prob)(pi, action)
        env_act = unbatchify(action, flx_env.agents, config["NUM_ENVS"], flx_env.num_agents)

        # STEP ENVS
        rng, _rng = jr.split(rng)
        rng_step = jr.split(_rng, config["NUM_ENVS"])
        old_env_act = env_act
        old_eqx_state = eqx_state
        eqx_agent_obs, eqx_state, eqx_reward, eqx_done, eqx_info = eqx_env_step(
            rng_step, eqx_state, env_act,
        )
        old_flx_state = flx_state
        flx_agent_obs, flx_state, flx_reward, flx_done, flx_info = flx_env_step(
            rng_step, flx_state, env_act,
        )

        done_mask = jnp.logical_or(eqx_done["__all__"], flx_done["__all__"])   # shape (N,)
        discrepancy_mask = ~(eqx_done["__all__"].astype(jnp.int32) == flx_done["__all__"].astype(jnp.int32))

        # check that all done states have been reset already
        reset_state_delta = jnp.linalg.norm((eqx_state.env_state.pipeline_state.q - eqx_state.first_pipeline_state.q)[eqx_done["__all__"]])
        # assert reset_state_delta <= 1e-5

        # check all non-reset states are equivalent
        non_reset_state_delta = jnp.linalg.norm((eqx_state.env_state.pipeline_state.q - flx_state.env_state.pipeline_state.q)[~done_mask])
        # assert non_reset_state_delta <= 1e-3

        if flx_done["__all__"].any():
            print("INFO: some flax envs are resetting this step")

        if jnp.linalg.norm(flx_state.env_state.info["first_pipeline_state"].q - original_first_q_flx) >= 1e-5:
            print("WARNING: the flax first_pipeline_state has CHANGED AUTOMATICALLY")
            new_first_q_flx = flx_state.env_state.info["first_pipeline_state"].q
            # check that this updated first_q is what the reset states have changed to
            reset_changed = flx_state.env_state.pipeline_state.q[flx_done["__all__"]] == new_first_q_flx[flx_done["__all__"]]
            if reset_changed.all():
                print("WARNING: this altered first_pipeline_state has been used to reset done envs")

            # check if the updated first_q has been taken from anywhere else or its been randomly chosen
            for q in new_first_q_flx:
                # print(q)
                matches = jnp.any(q == flx_state.env_state.pipeline_state.q[~flx_done["__all__"]], axis=1)
                if any(matches):
                    print("WARNING: the new_first_q_flx was taken from the existing set")
            
            # check if all reset states were changed or just the ones used
            whole_reset = (new_first_q_flx[~flx_done["__all__"]] == original_first_q_flx[~flx_done["__all__"]]).all()
            if whole_reset:
                print("WARNING: the entire first_q_flx was changed")
            
            original_first_q_flx = new_first_q_flx

        if discrepancy_mask.any():
            pass

        # upon reset - reset the flax_env to the eqx one
        def conditional_replace(done_mask, reset_tree, live_tree):
            """Return a new pytree where leaves are
            reset if done_mask[i] == True, else unchanged."""
            
            def _select(reset_leaf, live_leaf):
                # Broadcast the 1-D mask onto this leaf’s full shape
                # e.g. (N,) -> (N,1,1,…) so jnp.where works.
                mask = done_mask
                while mask.ndim < live_leaf.ndim:
                    mask = mask[..., None]          # add trailing singleton axes
                return jnp.where(mask, reset_leaf, live_leaf)

            return jax.tree_util.tree_map(_select, reset_tree, live_tree)
        
        new_flax_pipeline_state = conditional_replace(flx_done["__all__"], flx_state.env_state.info["first_pipeline_state"], flx_state.env_state.pipeline_state) # jnp.where(flx_done["__all__"], eqx_state.first_pipeline_state[done_mask], flx_state.env_state.pipeline_state[done_mask])
        flx_state = flx_state.replace(
            env_state=flx_state.env_state.replace(
                pipeline_state = new_flax_pipeline_state,
                # info = new_info
            )
        )
        new_eqx_pipeline_state = conditional_replace(eqx_done["__all__"], flx_state.env_state.info["first_pipeline_state"], eqx_state.env_state.pipeline_state) # jnp.where(eqx_done, eqx_state.first_pipeline_state[done_mask], eqx_state.env_state.pipeline_state[done_mask])
        eqx_state = eqx_state.replace(
            env_state=eqx_state.env_state.replace(
                pipeline_state = new_eqx_pipeline_state,
                # info = new_info
            )
        )



        eqx_info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), eqx_info)
        eqx_transition = Transition(
            batchify(eqx_done, eqx_env.agents, config["NUM_ACTORS"]).squeeze(),
            action,
            value,
            batchify(eqx_reward, eqx_env.agents, config["NUM_ACTORS"]).squeeze(),
            log_prob,
            eqx_obs_batch,
            eqx_info,
        )
        eqx_traj_batch.append(eqx_transition)
        flx_info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), flx_info)
        flx_transition = Transition(
            batchify(flx_done, flx_env.agents, config["NUM_ACTORS"]).squeeze(),
            action,
            value,
            batchify(flx_reward, flx_env.agents, config["NUM_ACTORS"]).squeeze(),
            log_prob,
            flx_obs_batch,
            flx_info,
        )
        flx_traj_batch.append(flx_transition)

    def stack_transitions(traj_list, axis: int = 0):
        """Return a single Transition whose leaves are
        jnp.stack([...], axis=axis) over time."""
        return jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=axis),  # applied to every leaf
            *traj_list                            # unpack list -> positional args
        )

    eqx_traj_batch = stack_transitions(eqx_traj_batch)
    flx_traj_batch = stack_transitions(flx_traj_batch)

    eqx_returned_episode = eqx_traj_batch[6]["returned_episode"].sum() / eqx_traj_batch[6]["returned_episode"].size
    flx_returned_episode = flx_traj_batch[6]["returned_episode"].sum() / flx_traj_batch[6]["returned_episode"].size

    """
    JOHN you are here - the traj batch returned_episodes are definitely wrong on the equinox
    side and not the flax side - we don't know why without unwrapping this scan. Oh my
    what if the filter_scan is affecting the eqx environment... surely not
    """

    import numpy as np

    eqx_actions = eqx_traj_batch[1][:,12:16]
    eqx_obs = eqx_traj_batch[5][:,12:16]
    eqx_ret_ep = eqx_traj_batch[6]["returned_episode"].astype(jnp.int32)[:,12:16]
    eqx_dones = eqx_traj_batch[0][:,12:16].astype(jnp.int32)
    flx_actions = flx_traj_batch[1][:,12:16]
    flx_obs = flx_traj_batch[5][:,12:16]
    flx_ret_ep = flx_traj_batch[6]["returned_episode"].astype(jnp.int32)[:,12:16]
    flx_dones = flx_traj_batch[0][:,12:16].astype(jnp.int32)

    np.savetxt("eqx_returned_episode.csv", eqx_ret_ep, delimiter=",", fmt="%d")
    np.savetxt("flx_returned_episode.csv", flx_ret_ep, delimiter=",", fmt="%d")

    np.savetxt("eqx_dones.csv", eqx_dones, delimiter=",", fmt="%d")

    # traj_batch[6]["returned_episode"][:,12:16]

    # ret_ep = np.asarray(traj_batch[6]["returned_episode"].astype(jnp.int32))
    # np.savetxt("returned_episode.csv", ret_ep, delimiter=",", fmt="%d")
    pass

def test_optax_mappo():
    from MARL_resources.policies import ActorCritic
    from MARL_resources.IPPO_env_template import YourEnv

    import optax

    rng = jr.PRNGKey(0)

    env = YourEnv()

    network = ActorCritic(
        key=rng,
        actor_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, env.action_space(env.agents[0]).shape[0]],
        critic_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, 1],
        actor_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 0.01],
        critic_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 1],
        activation=jax.nn.tanh,
    )

    opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(1e-3, eps=1e-5)
    )

    # this is the opt_state for both actor and critic - everything in the actor_critic
    opt_state = opt.init(network)

    has_actor = hasattr(opt_state[1][0].mu, "actor_layers")
    has_log_std = hasattr(opt_state[1][0].mu, "log_std")
    has_critic = hasattr(opt_state[1][0].mu, "critic_layers")

    def actor_mask(model: ActorCritic):
        """True for actor weights + log_std, False everywhere else."""
        mask = jax.tree_util.tree_map(lambda _: False, model)          # start all-False

        # actor layers
        mask = eqx.tree_at(lambda m: m.actor_layers,
                        mask,
                        replace_fn=lambda sub: jax.tree_util.tree_map(lambda _: True, sub))

        # learnable log-σ  (single leaf, not a list like the layers)
        mask = eqx.tree_at(lambda m: m.log_std, mask, replace_fn=lambda _: True)
        return mask
    
    def critic_mask(model: ActorCritic):
        """True for critic weights, False everywhere else."""
        mask = jax.tree_util.tree_map(lambda _: False, model)
        mask = eqx.tree_at(lambda m: m.critic_layers,
                        mask,
                        replace_fn=lambda sub: jax.tree_util.tree_map(lambda _: True, sub))
        return mask
    
    actor_opt = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(1e-3, eps=1e-5)
    )
    actor_opt  = optax.masked(actor_opt,  actor_mask(network))


    def leaves_equal(a, b):
        return jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), a, b)
        )

    @eqx.filter_value_and_grad
    def dummy_loss(model, x):
        mean, scale, value = model(x)
        return jnp.square(mean).sum() + value.sum()

    x = jax.random.normal(jax.random.PRNGKey(0), (18))

    params0          = network
    opt_state_actor0 = actor_opt.init(params0)

    # one update
    loss, grads      = dummy_loss(network, x)
    upd, opt_state_actor1 = actor_opt.update(grads, opt_state_actor0, params0)
    params1 = optax.apply_updates(params0, upd)

    print("critic unchanged?",
        leaves_equal(params0.critic_layers, params1.critic_layers))
    # → True
    print("actor unchanged?",
        leaves_equal(params0.actor_layers,  params1.actor_layers))
    # → False  (will show False because actor weights did change)

    eqx.is_array

    pass

if __name__ == "__main__":
    test_optax_mappo()



    pass
