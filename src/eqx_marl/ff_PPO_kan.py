"""
ff_PPO_mk2 with Kolmogorov-Arnold networks (kaneqx) as actor and critic.

Differences from ff_PPO_mk2:
- KANActor / KANCritic instead of the MLP Actor / Critic.
- Running observation normalization (RunningMeanStd), carried through the scan.
- kaneqx's Adam (jax.example_libraries interface) with global-norm clipping instead of optax,
  so its moments can be transitioned across grid extensions.
- Training is split into stages keyed by PPO update index (config["GRID_STAGES"]). Each
  stage is one scan at a fixed grid size G; between stages the grids of both networks are
  extended on the last rollout's observations and the Adam state is transitioned.
- Optional same-size grid updates every config["GRID_UPDATE_EVERY"] updates inside the scan
  (knots follow the observation distribution, shapes unchanged, Adam state kept).
- Optional sparsification penalty config["LAMB"] on the actor, and a post-training
  attribution / pruning analysis.
"""

import os
import json
from datetime import datetime
from tensorboardX import SummaryWriter
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import distreqx.distributions as dist

from kaneqx import adam, transition, trainable_filter, reg, edge_forward_scales, node_scores, prune

from eqx_marl.common.models import KANActor, KANCritic
from eqx_marl.common.normalize import RunningMeanStd
from eqx_marl.common.eqx_utils import filter_scan, filter_cond


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # raw observations; normalize with the rms before feeding a network
    info: jnp.ndarray


class TrainState(NamedTuple):
    """One network: kaneqx AdamState (holds the trainable params) + the frozen remainder (grid, mask, statics)."""
    opt_state: object
    static: object


def clip_by_global_norm(grads, max_norm):
    leaves = jax.tree_util.tree_leaves(grads)
    g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def make_train(env, config, rng_init):

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f'data/eqx_ppo_kan_validation/{current_datetime}'
    os.makedirs(save_path, exist_ok=False)
    writer = SummaryWriter(log_dir=save_path)
    config["SAVE_PATH"] = save_path
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    config["NUM_ACTORS"] = 1 * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    step_size = linear_schedule if config["ANNEAL_LR"] else config["LR"]
    opt_init, opt_update, get_params = adam(step_size, eps=1e-5)

    stages = sorted((int(k), int(v)) for k, v in config["GRID_STAGES"].items())
    assert stages[0][0] == 0, "GRID_STAGES must start at update 0"

    # INIT NETWORK
    rng_actor, rng_critic = jr.split(rng_init)
    actor = KANActor(rng_actor, [obs_dim] + config["ACTOR_HIDDEN"] + [act_dim], k=config["K"], G=stages[0][1])
    critic = KANCritic(rng_critic, [obs_dim] + config["CRITIC_HIDDEN"] + [1], k=config["K"], G=stages[0][1])

    def make_train_state(net, opt_state=None):
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(opt_init(params) if opt_state is None else transition(opt_state, params), static)

    def net_of(ts):
        return eqx.combine(get_params(ts.opt_state), ts.static)

    def with_net(ts, net):
        """Put a network with unchanged shapes back into its TrainState (keeps Adam moments)."""
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(eqx.tree_at(lambda s: s.params, ts.opt_state, params), static)

    def train(rng):

        # INIT ENV
        rng, _rng = jr.split(rng)
        reset_rng = jr.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.jit(jax.vmap(env.reset))(reset_rng)

        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            train_states, env_state, last_obs, update_count, opt_step, rms, grid_obs, rng = runner_state
            actor_ts, critic_ts = train_states
            actor, critic = net_of(actor_ts), net_of(critic_ts)

            obs_batch = rms.normalize(last_obs)
            rng, _rng = jr.split(rng)
            mean, scale = eqx.filter_vmap(actor)(obs_batch)
            value = eqx.filter_vmap(critic)(obs_batch)

            pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
            pi_log_prob = lambda d, a: d.log_prob(a)
            action = pi.sample(_rng)
            log_prob = eqx.filter_vmap(pi_log_prob)(pi, action)

            # STEP ENV
            rng, _rng = jr.split(rng)
            rng_step = jr.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(rng_step, env_state, action)

            info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
            transition_ = Transition(done, action, value, reward, log_prob, last_obs, info)
            runner_state = (train_states, env_state, obsv, update_count, opt_step, rms, grid_obs, rng)
            return runner_state, transition_

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition_):
                gae, next_value = gae_and_next_value
                done, value, reward = transition_.done, transition_.value, transition_.reward
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=8,
            )
            return advantages, advantages + traj_batch.value

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            runner_state, traj_batch = filter_scan(_env_step, runner_state, None, config["NUM_STEPS"])
            train_states, env_state, last_obs, update_count, opt_step, rms, grid_obs, rng = runner_state
            actor_ts, critic_ts = train_states

            # OBSERVATION NORMALIZATION: update from this rollout, then use for the whole update
            rms = rms.update(traj_batch.obs)
            # A fixed-size subsample of this rollout's observations, for grid updates and extensions
            rng, _rng = jr.split(rng)
            flat_obs = traj_batch.obs.reshape(-1, obs_dim)
            grid_obs = flat_obs[jr.choice(_rng, flat_obs.shape[0], (config["GRID_UPDATE_SAMPLES"],), replace=False)]

            # SAME-SIZE GRID UPDATE: knots follow the data, shapes unchanged, Adam moments kept
            if config["GRID_UPDATE_EVERY"] > 0:
                do_update = (update_count % config["GRID_UPDATE_EVERY"]) == 0

                def _refresh(actor_ts, critic_ts, rms):
                    x = rms.normalize(grid_obs)
                    actor, critic = net_of(actor_ts), net_of(critic_ts)
                    actor = actor.update_grids(x, actor.kan.layers[0].G)
                    critic = critic.update_grids(x, critic.kan.layers[0].G)
                    return with_net(actor_ts, actor), with_net(critic_ts, critic)

                actor_ts, critic_ts = filter_cond(
                    do_update, _refresh, lambda a, c, r: (a, c), actor_ts, critic_ts, rms
                )

            # CALCULATE ADVANTAGE
            last_val = eqx.filter_vmap(net_of(critic_ts))(rms.normalize(last_obs))
            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(carry, batch_info):
                    actor_ts, critic_ts, opt_step = carry
                    traj_batch, advantages, targets = batch_info
                    obs_n = rms.normalize(traj_batch.obs)

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        actor = eqx.combine(actor_params, actor_ts.static)
                        mean, scale = eqx.filter_vmap(actor)(obs_n)
                        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
                        pi_log_prob = lambda d, a: d.log_prob(a)
                        log_prob = eqx.filter_vmap(pi_log_prob)(pi, traj_batch.action)

                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        pi_entropy = lambda d: d.entropy()
                        entropy = eqx.filter_vmap(pi_entropy)(pi).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = loss_actor - config["ENT_COEF"] * entropy
                        if config["LAMB"] > 0:
                            # pykan's sparsification penalty on the actor's edge activation scales
                            _, stats = actor.kan.forward_with_stats(obs_n)
                            total_loss = total_loss + config["LAMB"] * reg(
                                actor.kan, edge_forward_scales(stats), config["LAMB_L1"], config["LAMB_ENTROPY"]
                            )
                        return total_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        critic = eqx.combine(critic_params, critic_ts.static)
                        value = eqx.filter_vmap(critic)(obs_n)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        return config["VF_COEF"] * value_loss, value_loss

                    actor_loss, actor_grads = eqx.filter_value_and_grad(_actor_loss_fn, has_aux=True)(
                        get_params(actor_ts.opt_state), traj_batch, advantages
                    )
                    critic_loss, critic_grads = eqx.filter_value_and_grad(_critic_loss_fn, has_aux=True)(
                        get_params(critic_ts.opt_state), traj_batch, targets
                    )

                    actor_grads = clip_by_global_norm(actor_grads, config["MAX_GRAD_NORM"])
                    critic_grads = clip_by_global_norm(critic_grads, config["MAX_GRAD_NORM"])
                    actor_ts = TrainState(opt_update(opt_step, actor_grads, actor_ts.opt_state), actor_ts.static)
                    critic_ts = TrainState(opt_update(opt_step, critic_grads, critic_ts.opt_state), critic_ts.static)

                    loss_info = {
                        "total_loss": actor_loss[0] + critic_loss[0],
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }
                    return (actor_ts, critic_ts, opt_step + 1), loss_info

                carry, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jr.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                permutation = jr.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch
                )
                carry, loss_info = filter_scan(_update_minbatch, carry, minibatches)
                return (carry, traj_batch, advantages, targets, rng), loss_info

            update_state = ((actor_ts, critic_ts, opt_step), traj_batch, advantages, targets, rng)
            update_state, loss_info = filter_scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            (actor_ts, critic_ts, opt_step), _, _, _, rng = update_state

            metric = traj_batch.info
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            def callback(metric):
                step = metric["update_step"]
                for key, value in metric.items():
                    writer.add_scalar(key, value, step)

            update_count = update_count + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_count
            metric["env_step"] = update_count * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["grid_size"] = jnp.asarray(net_of(actor_ts).kan.layers[0].G)
            metric = {**metric, **loss_info}
            jax.experimental.io_callback(callback, None, metric)

            runner_state = ((actor_ts, critic_ts), env_state, last_obs, update_count, opt_step, rms, grid_obs, rng)
            return runner_state, metric

        @eqx.filter_jit
        def run_stage(runner_state, n_updates):
            return filter_scan(_update_step, runner_state, None, n_updates)

        rng, _rng = jr.split(rng)
        runner_state = (
            (make_train_state(actor), make_train_state(critic)),
            env_state, obsv, jnp.array(0), jnp.array(0),
            RunningMeanStd((obs_dim,)), jnp.zeros((config["GRID_UPDATE_SAMPLES"], obs_dim)), _rng,
        )

        # STAGES: one scan per grid size; extend grids + transition Adam in between
        boundaries = [start for start, _ in stages] + [config["NUM_UPDATES"]]
        metrics = []
        for (start, G), end in zip(stages, boundaries[1:]):
            if start > 0:
                train_states, env_state, last_obs, update_count, opt_step, rms, grid_obs, rng = runner_state
                actor_ts, critic_ts = train_states
                x = rms.normalize(grid_obs)
                actor_ext = net_of(actor_ts).update_grids(x, G)
                critic_ext = net_of(critic_ts).update_grids(x, G)
                train_states = (make_train_state(actor_ext, actor_ts.opt_state), make_train_state(critic_ext, critic_ts.opt_state))
                runner_state = (train_states, env_state, last_obs, update_count, opt_step, rms, grid_obs, rng)
                print(f"INFO: update {start}: grids extended to G={G}, Adam state transitioned")
            if end > start:
                runner_state, metric = run_stage(runner_state, end - start)
                metrics.append(metric)

        metric = jax.tree.map(lambda *xs: jnp.concatenate(xs), *metrics)
        return {"runner_state": runner_state, "metrics": metric}

    return train


def evaluate(net_fn, env, rng, num_envs=64, num_steps=1000):
    """Mean undiscounted return of the deterministic policy net_fn(obs) -> action over one episode per env."""
    rng, _rng = jr.split(rng)
    obs, env_state = jax.vmap(env.reset)(jr.split(_rng, num_envs))

    def step(carry, _):
        obs, env_state, ret, alive, rng = carry
        rng, _rng = jr.split(rng)
        obs, env_state, reward, done, _ = jax.vmap(env.step)(jr.split(_rng, num_envs), env_state, net_fn(obs))
        ret = ret + reward * alive
        alive = alive * (1 - done)
        return (obs, env_state, ret, alive, rng), None

    (_, _, ret, _, _), _ = jax.lax.scan(step, (obs, env_state, jnp.zeros(num_envs), jnp.ones(num_envs), rng), None, num_steps)
    return ret.mean()


if __name__ == "__main__":

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    from eqx_marl.env_template_brax_PPO import YourEnv

    config = {
        "LR": 1e-3,
        "NUM_ENVS": 64,
        "NUM_STEPS": 300,
        "TOTAL_TIMESTEPS": 1e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 2e-6,
        "VF_COEF": 4.5,
        "MAX_GRAD_NORM": 0.5,
        "SEED": 0,
        "ANNEAL_LR": True,
        "DEVICE": 0,
        # KAN
        "ACTOR_HIDDEN": [32, 32],
        "CRITIC_HIDDEN": [32, 32],
        "K": 3,
        "GRID_STAGES": {0: 3, 200: 5, 350: 10},  # PPO update index -> grid size G
        "GRID_UPDATE_EVERY": 10,                  # same-size grid updates inside a stage; 0 disables
        "GRID_UPDATE_SAMPLES": 2048,
        "LAMB": 0.0,                              # sparsification penalty weight on the actor
        "LAMB_L1": 1.0,
        "LAMB_ENTROPY": 2.0,
        # post-training pruning analysis
        "NODE_TH": 1e-2,
        "EDGE_TH": 3e-2,
    }

    rng = jr.PRNGKey(config["SEED"])
    rng, _rng = jr.split(rng)
    env = YourEnv(backend="mjx")  # same backend as ff_PPO_mk2, which needs mjx for contact forces
    config["ENV_NAME"] = env.__class__.__name__
    train = make_train(env, config, _rng)
    out = train(rng)
    print("INFO: training complete")

    # save the final networks and observation normalizer; reload with
    # eqx.tree_deserialise_leaves(path, (KANActor(..., G=final_G), KANCritic(..., G=final_G), RunningMeanStd((obs_dim,))))
    train_states, _, _, _, _, rms, grid_obs, rng = out["runner_state"]
    actor_ts, critic_ts = train_states
    eqx.tree_serialise_leaves(
        os.path.join(config["SAVE_PATH"], "checkpoint.eqx"),
        (eqx.combine(actor_ts.opt_state.params, actor_ts.static), eqx.combine(critic_ts.opt_state.params, critic_ts.static), rms),
    )
    print(f"INFO: saved checkpoint to {config['SAVE_PATH']}")

    # ================================ #
    # attribution and pruning analysis #
    # ================================ #
    train_states, _, _, _, _, rms, grid_obs, rng = out["runner_state"]
    actor_ts, _ = train_states
    actor = eqx.combine(actor_ts.opt_state.params, actor_ts.static)
    x = rms.normalize(grid_obs)

    _, stats = actor.kan.forward_with_stats(x)
    scores = node_scores(stats)
    print("actor input attribution (obs index: score):")
    for i, s in sorted(enumerate(np.asarray(scores[0])), key=lambda t: -t[1]):
        print(f"  {i:2d}: {s:.3f}")
    print("hidden node scores:", [np.round(np.asarray(s), 3).tolist() for s in scores[1:-1]])

    pruned_kan = prune(actor.kan, x, node_th=config["NODE_TH"], edge_th=config["EDGE_TH"])
    pruned = eqx.tree_at(lambda a: a.kan, actor, pruned_kan)
    print(f"actor width {actor.kan.width} -> pruned {pruned_kan.width}, "
          f"active edges {[int(l.mask.sum()) for l in pruned_kan.layers]}")

    rng, r1, r2 = jr.split(rng, 3)
    policy = lambda net: (lambda obs: eqx.filter_vmap(net)(rms.normalize(obs))[0])
    ret_full = evaluate(policy(actor), env, r1)
    ret_pruned = evaluate(policy(pruned), env, r2)
    print(f"deterministic return: full {float(ret_full):.1f}, pruned {float(ret_pruned):.1f}")
