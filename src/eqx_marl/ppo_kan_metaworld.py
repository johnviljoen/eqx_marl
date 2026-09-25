"""
PPO with KAN actor and critic on Continual World (Meta-World v3) single tasks, mirroring
ff_PPO_kan.py (the Ant) as closely as a CPU gym environment allows:

- Same losses, GAE, clipping, Adam-with-transition, grid stages keyed by PPO update index,
  same-size grid updates every GRID_UPDATE_EVERY updates, running observation normalization.
- Rollouts are collected in a Python loop over VecEnv (auto-reset copies of the metaworld env)
  with a jitted policy step; the whole PPO update (normalizer update, GAE, epochs, minibatches)
  is one jitted function. Truncation at the 200-step horizon counts as done, as brax's did.
- Actions are Gaussian samples (unbounded, as on the Ant); they are clipped to [-1, 1] only
  when stepping the env, the log-prob uses the unclipped sample.

    python -m eqx_marl.ppo_kan_metaworld --task push-v3 --seed 0            # Ant hyperparameters
    python -m eqx_marl.ppo_kan_metaworld --task push-v3 --seed 0 --smoke    # 5 tiny updates

Saves data/eqx_ppo_kan/<task>/<timestamp>/{config.json, run_config.json, checkpoint.eqx,
events.out.tfevents.*, <task>_deterministic.mp4, HANDOFF.md}; checkpoint = (actor, critic, rms).
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import distreqx.distributions as dist
from tensorboardX import SummaryWriter

from kaneqx import adam, transition, trainable_filter
from eqx_marl.common.models import KANActor, KANCritic, Actor, Critic
from eqx_marl.common.normalize import RunningMeanStd
from eqx_marl.common.eqx_utils import filter_scan
from eqx_marl.common.metaworld_env import make_env, VecEnv, HORIZON, OBS_DIM, ACT_DIM


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray  # raw observations; normalize with the rms before feeding a network


class TrainState(NamedTuple):
    opt_state: object
    static: object


def clip_by_global_norm(grads, max_norm):
    leaves = jax.tree_util.tree_leaves(grads)
    g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (g_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def _is_kan(net):
    return hasattr(net, "kan")


def _grid_size(*nets):
    return next((n.kan.layers[0].G for n in nets if _is_kan(n)), 0)


def make_train(config, writer):
    config["NUM_UPDATES"] = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    assert config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"] == batch_size

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    step_size = linear_schedule if config["ANNEAL_LR"] else config["LR"]
    opt_init, opt_update, get_params = adam(step_size, eps=1e-5)
    stages = sorted((int(k), int(v)) for k, v in config["GRID_STAGES"].items())
    assert stages[0][0] == 0, "GRID_STAGES must start at update 0"

    def make_train_state(net, opt_state=None):
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(opt_init(params) if opt_state is None else transition(opt_state, params), static)

    def net_of(ts):
        return eqx.combine(get_params(ts.opt_state), ts.static)

    def with_net(ts, net):
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(eqx.tree_at(lambda s: s.params, ts.opt_state, params), static)

    @eqx.filter_jit
    def policy_step(actor, critic, rms, obs, key):
        x = rms.normalize(obs)
        mean, scale = eqx.filter_vmap(actor)(x)
        value = eqx.filter_vmap(critic)(x)
        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
        action = pi.sample(key)
        log_prob = eqx.filter_vmap(lambda d, a: d.log_prob(a))(pi, action)
        return action, log_prob, value

    @eqx.filter_jit
    def det_action(actor, rms, obs):
        return jnp.clip(actor(rms.normalize(obs))[0], -1.0, 1.0)

    @eqx.filter_jit
    def refresh_grids(actor_ts, critic_ts, rms, grid_obs, G):
        """Grid update (same size or extension) on a rollout subsample; MLP components untouched."""
        x = rms.normalize(grid_obs)
        upd = lambda net: net.update_grids(x, G) if _is_kan(net) else net
        return upd(net_of(actor_ts)), upd(net_of(critic_ts))

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition_):
            gae, next_value = gae_and_next_value
            done, value, reward = transition_.done, transition_.value, transition_.reward
            delta = reward + config["GAMMA"] * next_value * (1 - done) - value
            gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=8)
        return advantages, advantages + traj_batch.value

    @eqx.filter_jit
    def update(actor_ts, critic_ts, opt_step, rms, traj_batch, last_obs, rng):
        """One PPO update on a rollout (NUM_STEPS, NUM_ENVS, ...): as _update_step of ff_PPO_kan.py after the env scan."""
        last_val = eqx.filter_vmap(net_of(critic_ts))(rms.normalize(last_obs))
        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state, unused):
            def _update_minbatch(carry, batch_info):
                actor_ts, critic_ts, opt_step = carry
                traj_batch, advantages, targets = batch_info
                obs_n = rms.normalize(traj_batch.obs)

                def _actor_loss_fn(actor_params, traj_batch, gae):
                    actor = eqx.combine(actor_params, actor_ts.static)
                    mean, scale = eqx.filter_vmap(actor)(obs_n)
                    pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
                    log_prob = eqx.filter_vmap(lambda d, a: d.log_prob(a))(pi, traj_batch.action)
                    logratio = log_prob - traj_batch.log_prob
                    ratio = jnp.exp(logratio)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy = eqx.filter_vmap(lambda d: d.entropy())(pi).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                    return loss_actor - config["ENT_COEF"] * entropy, (loss_actor, entropy, approx_kl, clip_frac)

                def _critic_loss_fn(critic_params, traj_batch, targets):
                    critic = eqx.combine(critic_params, critic_ts.static)
                    value = eqx.filter_vmap(critic)(obs_n)
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    return config["VF_COEF"] * value_loss, value_loss

                actor_loss, actor_grads = eqx.filter_value_and_grad(_actor_loss_fn, has_aux=True)(get_params(actor_ts.opt_state), traj_batch, advantages)
                critic_loss, critic_grads = eqx.filter_value_and_grad(_critic_loss_fn, has_aux=True)(get_params(critic_ts.opt_state), traj_batch, targets)
                actor_grads = clip_by_global_norm(actor_grads, config["MAX_GRAD_NORM"])
                critic_grads = clip_by_global_norm(critic_grads, config["MAX_GRAD_NORM"])
                actor_ts = TrainState(opt_update(opt_step, actor_grads, actor_ts.opt_state), actor_ts.static)
                critic_ts = TrainState(opt_update(opt_step, critic_grads, critic_ts.opt_state), critic_ts.static)
                loss_info = {"actor_loss": actor_loss[1][0], "value_loss": critic_loss[1], "entropy": actor_loss[1][1],
                             "approx_kl": actor_loss[1][2], "clip_frac": actor_loss[1][3]}
                return (actor_ts, critic_ts, opt_step + 1), loss_info

            carry, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jr.split(rng)
            permutation = jr.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch)
            carry, loss_info = filter_scan(_update_minbatch, carry, minibatches)
            return (carry, traj_batch, advantages, targets, rng), loss_info

        update_state = ((actor_ts, critic_ts, opt_step), traj_batch, advantages, targets, rng)
        update_state, loss_info = filter_scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
        (actor_ts, critic_ts, opt_step), _, _, _, _ = update_state
        loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
        loss_info["value_mean"] = jnp.mean(traj_batch.value); loss_info["target_mean"] = jnp.mean(targets)
        return actor_ts, critic_ts, opt_step, loss_info

    def train(rng):
        rng, ra, rc = jr.split(rng, 3)
        if config["ACTOR_NET"] == "kan":
            actor = KANActor(ra, [OBS_DIM] + config["ACTOR_HIDDEN"] + [ACT_DIM], k=config["K"], G=stages[0][1])
        else:
            actor = Actor(ra, [OBS_DIM] + config["MLP_HIDDEN"] + [ACT_DIM])
        if config["CRITIC_NET"] == "kan":
            critic = KANCritic(rc, [OBS_DIM] + config["CRITIC_HIDDEN"] + [1], k=config["K"], G=stages[0][1])
        else:
            critic = Critic(rc, [OBS_DIM] + config["MLP_HIDDEN"] + [1])
        actor_ts, critic_ts = make_train_state(actor), make_train_state(critic)
        opt_step = jnp.array(0)
        rms = RunningMeanStd((OBS_DIM,))
        extensions = {start: G for start, G in stages if start > 0}

        vec = VecEnv(config["TASK"], config["NUM_ENVS"], seed=config["SEED"])
        test_env = make_env(config["TASK"], seed=config["SEED"] + 100)
        rng_np = np.random.default_rng(config["SEED"])

        def evaluate(actor, rms, n_eps):
            rets, succs = [], []
            for _ in range(n_eps):
                o, _ = test_env.reset(); ret = 0.0
                for _ in range(HORIZON):
                    o, r, term, trunc, info = test_env.step(np.asarray(det_action(actor, rms, jnp.asarray(o, jnp.float32)))); ret += r
                    if term or trunc:
                        break
                rets.append(ret); succs.append(info["episode_success"])
            return float(np.mean(rets)), float(np.mean(succs))

        obs = vec.reset()
        ep_ret = np.zeros(config["NUM_ENVS"]); ep_returns, ep_successes = [], []
        t0 = time.time()
        print(f"INFO: {config['NUM_UPDATES']} PPO updates of {batch_size} steps; grid extensions at {extensions}; same-size grid update every {config['GRID_UPDATE_EVERY']} updates")
        for u in range(config["NUM_UPDATES"]):
            # ROLLOUT (Python loop over the vectorized metaworld env)
            actor, critic = net_of(actor_ts), net_of(critic_ts)
            buf = {k: [] for k in Transition._fields}
            for _ in range(config["NUM_STEPS"]):
                rng, k = jr.split(rng)
                a, logp, v = policy_step(actor, critic, rms, jnp.asarray(obs), k)
                a = np.asarray(a)
                o2, r, d, succ = vec.step(np.clip(a, -1.0, 1.0))
                buf["done"].append(d); buf["action"].append(a); buf["value"].append(np.asarray(v))
                buf["reward"].append(r); buf["log_prob"].append(np.asarray(logp)); buf["obs"].append(obs)
                ep_ret += r
                for i in np.flatnonzero(d):
                    ep_returns.append(ep_ret[i]); ep_successes.append(bool(succ[i])); ep_ret[i] = 0.0
                obs = o2
            traj = Transition(**{k: jnp.asarray(np.stack(v)) for k, v in buf.items()})

            # NORMALIZER from this rollout, then a subsample of its observations for the grids
            rms = rms.update(traj.obs)
            flat = np.asarray(traj.obs).reshape(-1, OBS_DIM)
            grid_obs = jnp.asarray(flat[rng_np.choice(flat.shape[0], config["GRID_UPDATE_SAMPLES"], replace=False)])
            if u in extensions:
                a_ext, c_ext = refresh_grids(actor_ts, critic_ts, rms, grid_obs, extensions[u])
                actor_ts, critic_ts = make_train_state(a_ext, actor_ts.opt_state), make_train_state(c_ext, critic_ts.opt_state)
                print(f"INFO: update {u}: grids extended to G={extensions[u]}, Adam state transitioned")
            elif config["GRID_UPDATE_EVERY"] > 0 and u > 0 and u % config["GRID_UPDATE_EVERY"] == 0:
                a_new, c_new = refresh_grids(actor_ts, critic_ts, rms, grid_obs, _grid_size(net_of(actor_ts), net_of(critic_ts)))
                actor_ts, critic_ts = with_net(actor_ts, a_new), with_net(critic_ts, c_new)

            rng, k = jr.split(rng)
            actor_ts, critic_ts, opt_step, loss_info = update(actor_ts, critic_ts, opt_step, rms, traj, jnp.asarray(obs), k)
            for key in ("actor_loss", "value_loss"):
                if not bool(jnp.isfinite(loss_info[key])):
                    raise FloatingPointError(f"non-finite {key} at update {u}")

            # LOGGING
            env_step = (u + 1) * batch_size
            if ep_returns:
                writer.add_scalar("train/return", float(np.mean(ep_returns[-100:])), env_step)
                writer.add_scalar("train/success", float(np.mean(ep_successes[-100:])), env_step)
            for key, v in loss_info.items():
                writer.add_scalar(f"loss/{key}", float(v), env_step)
            writer.add_scalar("grid_size", _grid_size(net_of(actor_ts), net_of(critic_ts)), env_step)
            if (u + 1) % config["EVAL_EVERY"] == 0 or u + 1 == config["NUM_UPDATES"]:
                ret_d, suc_d = evaluate(net_of(actor_ts), rms, config["TEST_EPS"])
                writer.add_scalar("test/deterministic/return", ret_d, env_step); writer.add_scalar("test/deterministic/success", suc_d, env_step)
                print(f"update {u+1:4d} step {env_step:9d} | train ret {np.mean(ep_returns[-100:]) if ep_returns else 0:7.1f} succ {np.mean(ep_successes[-100:]) if ep_successes else 0:.2f} "
                      f"| det ret {ret_d:7.1f} succ {suc_d:.2f} | value {float(loss_info['value_mean']):.1f} targ {float(loss_info['target_mean']):.1f} "
                      f"ent {float(loss_info['entropy']):.2f} kl {float(loss_info['approx_kl']):.4f} G {_grid_size(net_of(actor_ts), net_of(critic_ts))} | {env_step/(time.time()-t0):.0f} steps/s")
        return net_of(actor_ts), net_of(critic_ts), rms, evaluate

    return train


def render_episode(config, actor, rms, path):
    import imageio
    env = make_env(config["TASK"], seed=config["SEED"] + 200, render_mode="rgb_array", camera_name=config["CAMERA"])
    act = eqx.filter_jit(lambda a, r, x: jnp.clip(a(r.normalize(x))[0], -1.0, 1.0))
    o, _ = env.reset(); frames, ret = [], 0.0
    for _ in range(HORIZON):
        frames.append(np.flipud(env.render()))
        o, r, term, trunc, info = env.step(np.asarray(act(actor, rms, jnp.asarray(o, jnp.float32)))); ret += r
        if term or trunc:
            break
    imageio.mimwrite(path, frames, fps=20, codec="libx264", quality=8)
    return ret, info["episode_success"]


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="push-v3"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=float, default=1e7); p.add_argument("--tag", default="")
    p.add_argument("--actor", choices=["kan", "mlp"], default="kan"); p.add_argument("--critic", choices=["kan", "mlp"], default="kan")
    p.add_argument("--smoke", action="store_true", help="5 tiny updates through both grid extensions")
    args = p.parse_args()

    config = {
        "TASK": args.task, "SEED": args.seed,
        # PPO, exactly as ff_PPO_kan.py on the Ant
        "LR": 1e-3, "NUM_ENVS": 64, "NUM_STEPS": 300, "TOTAL_TIMESTEPS": int(args.steps),
        "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2,
        "ENT_COEF": 2e-6, "VF_COEF": 4.5, "MAX_GRAD_NORM": 0.5, "ANNEAL_LR": True,
        # KAN, as on the Ant
        "ACTOR_HIDDEN": [32, 32], "CRITIC_HIDDEN": [32, 32], "K": 3,
        "GRID_STAGES": {"0": 3, "200": 5, "350": 10}, "GRID_UPDATE_EVERY": 10, "GRID_UPDATE_SAMPLES": 2048,
        "ACTOR_NET": args.actor, "CRITIC_NET": args.critic, "MLP_HIDDEN": [64, 64],
        # evaluation
        "EVAL_EVERY": 10, "TEST_EPS": 10, "FINAL_TEST_EPS": 50,
        "CAMERA": "corner2", "ENV_NAME": "metaworld-3.1.1 " + args.task + " (Continual World wrappers: horizon 200, random_init_all)",
    }
    if args.smoke:
        config.update(NUM_ENVS=8, NUM_STEPS=40, TOTAL_TIMESTEPS=8 * 40 * 5, GRID_STAGES={"0": 3, "2": 5, "4": 10},
                      GRID_UPDATE_EVERY=1, GRID_UPDATE_SAMPLES=64, EVAL_EVERY=2, TEST_EPS=1, FINAL_TEST_EPS=2)
    hybrid = "" if (args.actor, args.critic) == ("kan", "kan") else f"_{args.actor}actor_{args.critic}critic"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + hybrid + (f"_{args.tag}" if args.tag else "") + ("_smoke" if args.smoke else "")
    save_path = f"data/eqx_ppo_kan/{args.task}/{stamp}"
    os.makedirs(save_path, exist_ok=False)
    config["SAVE_PATH"] = save_path
    json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)
    writer = SummaryWriter(log_dir=save_path)

    t0 = time.time()
    actor, critic, rms, evaluate = make_train(config, writer)(jr.PRNGKey(args.seed))
    train_time = time.time() - t0
    print(f"INFO: training complete in {train_time/60:.1f} min")

    eqx.tree_serialise_leaves(os.path.join(save_path, "checkpoint.eqx"), (actor, critic, rms))
    ret_d, suc_d = evaluate(actor, rms, config["FINAL_TEST_EPS"])
    print(f"final over {config['FINAL_TEST_EPS']} episodes: deterministic return {ret_d:.1f} success {suc_d:.2f}")
    mp4 = os.path.join(save_path, f"{args.task}_deterministic.mp4")
    ret_v, suc_v = render_episode(config, actor, rms, mp4)
    config.update(FINAL_DET_RETURN=ret_d, FINAL_DET_SUCCESS=suc_d, TRAIN_MINUTES=train_time / 60)
    json.dump(config, open(os.path.join(save_path, "run_config.json"), "w"), indent=4)
    G = _grid_size(actor, critic)
    actor_tmpl = (f"KANActor(key, [{OBS_DIM},{','.join(map(str, config['ACTOR_HIDDEN']))},{ACT_DIM}], k={config['K']}, G={G})" if config["ACTOR_NET"] == "kan"
                  else f"Actor(key, [{OBS_DIM},{','.join(map(str, config['MLP_HIDDEN']))},{ACT_DIM}])")
    critic_tmpl = (f"KANCritic(key, [{OBS_DIM},{','.join(map(str, config['CRITIC_HIDDEN']))},1], k={config['K']}, G={G})" if config["CRITIC_NET"] == "kan"
                   else f"Critic(key, [{OBS_DIM},{','.join(map(str, config['MLP_HIDDEN']))},1])")
    with open(os.path.join(save_path, "HANDOFF.md"), "w") as f:
        f.write(f"""# PPO on {args.task} (Continual World, metaworld 3.1.1 v3 task): actor {config['ACTOR_NET'].upper()}, critic {config['CRITIC_NET'].upper()}

- Run directory: `{save_path}`; checkpoint.eqx = (actor, critic, RunningMeanStd) via eqx.tree_serialise_leaves.
- Reload: templates `{actor_tmpl}`, `{critic_tmpl}`, `RunningMeanStd(({OBS_DIM},))`.
- Deterministic policy: `clip(actor(rms.normalize(obs))[0], -1, 1)`. Obs are the 39-dim metaworld v3 observations, normalized only at the network input.
- Env: `eqx_marl.common.metaworld_env.make_env("{args.task}", seed)` with the benchmark wrappers (horizon {HORIZON}, random_init_all, success = any step with info['success']).
- Training: {config['TOTAL_TIMESTEPS']} env steps = {config['NUM_UPDATES']} PPO updates of {config['NUM_ENVS']}x{config['NUM_STEPS']} steps, hyperparameters of ff_PPO_kan.py (Ant), grid stages {config['GRID_STAGES']} by update index, same-size grid update every {config['GRID_UPDATE_EVERY']} updates, seed {args.seed}, {train_time/60:.1f} min.
- Final ({config['FINAL_TEST_EPS']} episodes): deterministic return {ret_d:.1f}, success {suc_d:.2f}. Rendered episode `{os.path.basename(mp4)}`: return {ret_v:.1f}, success {suc_v}.
- Code: `src/eqx_marl/ppo_kan_metaworld.py`, `common/models.py` (KANActor/KANCritic, Actor/Critic), `common/metaworld_env.py`. Conda env `mappo`.
""")
    print(f"INFO: saved to {save_path}")
