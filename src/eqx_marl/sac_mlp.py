"""
SAC with the Continual World benchmark's MLP actor and twin MLP critics on Meta-World v3
single tasks. Baseline for sac_kan.py: the SAC core below (replay, backup, losses, Adam,
alpha tuning, polyak targets, scanned 50-step updates, evaluation) is the KAN file's with the
KAN-only machinery removed: no grid stages / refits / Adam transitions and no observation
normalizer, since the benchmark uses neither.

Hyperparameters mirror continualworld/sac/sac.py and the run_single.py defaults: 1e6 env
steps, replay 1e6, batch 128, lr 1e-3, gamma 0.99, polyak 0.995, auto alpha with target
entropy -act_dim, 1e4 uniform start steps, updates start after 1e3 steps, every 50 env steps
do 50 gradient steps, hidden [256, 256, 256, 256], leaky ReLU, layer norm on, no gradient
clipping, evaluation every 2e4 steps with 10 stochastic + 1 deterministic episodes.

    python -m eqx_marl.sac_mlp --task push-v3 --seed 0

Saves data/eqx_sac_mlp/<task>/<timestamp>/{config.json, run_config.json, checkpoint.eqx,
events.out.tfevents.*, <task>_deterministic.mp4, HANDOFF.md}. Reload with
    eqx.tree_deserialise_leaves(path, (MLPSACActor(key, 39, 4), MLPQ(key, 39, 4), MLPQ(key, 39, 4)))
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import NamedTuple, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from tensorboardX import SummaryWriter

from kaneqx import adam, trainable_filter
from eqx_marl.common.models_sac import MLPSACActor, MLPQ
from eqx_marl.common.eqx_utils import filter_scan
from eqx_marl.common.metaworld_env import make_env, HORIZON, OBS_DIM, ACT_DIM


class TrainState(NamedTuple):
    opt_state: Any   # kaneqx AdamState over the trainable partition
    static: Any      # the non-trainable partition (static fields)


class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = np.zeros((size, obs_dim), np.float32); self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32); self.rew = np.zeros(size, np.float32); self.done = np.zeros(size, np.float32)
        self.size, self.ptr, self.n = size, 0, 0

    def store(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, rng, batch_size, n=None):
        """One batch (batch_size, ...) or, with n, a stack of n batches (n, batch_size, ...)."""
        idx = rng.integers(0, self.n, batch_size if n is None else (n, batch_size))
        return dict(obs=self.obs[idx], act=self.act[idx], rew=self.rew[idx], next_obs=self.next_obs[idx], done=self.done[idx])


def _global_norm(grads):
    return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array))))


def make_train(config):
    opt_init, opt_update, get_params = adam(config["LR"])
    alpha_init, alpha_update, alpha_params = adam(config["LR"])
    target_entropy = -float(ACT_DIM)
    gamma, polyak = config["GAMMA"], config["POLYAK"]

    def make_ts(net):
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(opt_init(params), static)

    def net_of(ts):
        return eqx.combine(get_params(ts.opt_state), ts.static)

    def _update_one(carry, batch, step, key):
        actor_ts, q1_ts, q2_ts, q1_t, q2_t, log_alpha, alpha_state = carry
        obs, next_obs = batch["obs"], batch["next_obs"]
        act, rew, done = batch["act"], batch["rew"], batch["done"]
        k1, k2 = jr.split(key)
        keys1, keys2 = jr.split(k1, obs.shape[0]), jr.split(k2, obs.shape[0])
        alpha = jnp.exp(log_alpha)
        actor = net_of(actor_ts)

        # Target: entropy-regularised clipped double-Q backup with the current policy's next action
        pi_next, logp_next = eqx.filter_vmap(actor.sample)(next_obs, keys2)
        q_next = jnp.minimum(eqx.filter_vmap(q1_t)(next_obs, pi_next), eqx.filter_vmap(q2_t)(next_obs, pi_next))
        backup = rew + gamma * (1.0 - done) * (q_next - alpha * logp_next)

        def critic_loss(params):
            q1 = eqx.combine(params[0], q1_ts.static); q2 = eqx.combine(params[1], q2_ts.static)
            q1_loss = 0.5 * jnp.mean((backup - eqx.filter_vmap(q1)(obs, act)) ** 2)
            q2_loss = 0.5 * jnp.mean((backup - eqx.filter_vmap(q2)(obs, act)) ** 2)
            return q1_loss + q2_loss, (q1_loss, q2_loss)

        (v_loss, (q1_loss, q2_loss)), c_grads = eqx.filter_value_and_grad(critic_loss, has_aux=True)(
            (get_params(q1_ts.opt_state), get_params(q2_ts.opt_state)))

        def actor_loss(params):
            a = eqx.combine(params, actor_ts.static)
            pi, logp = eqx.filter_vmap(a.sample)(obs, keys1)
            q1, q2 = net_of(q1_ts), net_of(q2_ts)
            min_q = jnp.minimum(eqx.filter_vmap(q1)(obs, pi), eqx.filter_vmap(q2)(obs, pi))
            return jnp.mean(alpha * logp - min_q), (logp, jnp.mean(min_q))

        (pi_loss, (logp, q_pi)), a_grads = eqx.filter_value_and_grad(actor_loss, has_aux=True)(get_params(actor_ts.opt_state))
        alpha_loss, al_grad = jax.value_and_grad(lambda la: -jnp.mean(la * jax.lax.stop_gradient(logp + target_entropy)))(log_alpha)

        actor_ts = TrainState(opt_update(step, a_grads, actor_ts.opt_state), actor_ts.static)
        q1_ts = TrainState(opt_update(step, c_grads[0], q1_ts.opt_state), q1_ts.static)
        q2_ts = TrainState(opt_update(step, c_grads[1], q2_ts.opt_state), q2_ts.static)
        alpha_state = alpha_update(step, al_grad, alpha_state)
        # Polyak targets over the trainable leaves; static fields are shared with the live critics
        q1_t = jax.tree.map(lambda t, n: polyak * t + (1 - polyak) * n, q1_t, net_of(q1_ts))
        q2_t = jax.tree.map(lambda t, n: polyak * t + (1 - polyak) * n, q2_t, net_of(q2_ts))
        metrics = dict(pi_loss=pi_loss, q1_loss=q1_loss, q2_loss=q2_loss, alpha=alpha, logp_pi=jnp.mean(logp),
                       q1=jnp.mean(eqx.filter_vmap(net_of(q1_ts))(obs, act)),
                       # diagnostics: regression target, Q at policy actions, gradient norms
                       target_q=jnp.mean(backup), q_pi=q_pi,
                       actor_grad_norm=_global_norm(a_grads), critic_grad_norm=_global_norm(c_grads))
        return (actor_ts, q1_ts, q2_ts, q1_t, q2_t, alpha_params(alpha_state), alpha_state), metrics

    @eqx.filter_jit
    def update_chunk(carry, batches, step0, key):
        """UPDATE_EVERY gradient steps in one jitted scan: batches are stacked (n, B, ...)."""
        n = batches["rew"].shape[0]
        xs = (batches, step0 + jnp.arange(n), jr.split(key, n))
        carry, metrics = filter_scan(lambda c, x: _update_one(c, x[0], x[1], x[2]), carry, xs)
        return carry, jax.tree.map(lambda m: m[-1], metrics)

    def train(rng, writer, save_path):
        rng_np = np.random.default_rng(config["SEED"])
        env = make_env(config["TASK"], seed=config["SEED"])
        test_env = make_env(config["TASK"], seed=config["SEED"] + 100)
        rng, ra, r1, r2 = jr.split(rng, 4)
        actor = MLPSACActor(ra, OBS_DIM, ACT_DIM, config["ACTOR_HIDDEN"], config["USE_LAYER_NORM"])
        q1 = MLPQ(r1, OBS_DIM, ACT_DIM, config["CRITIC_HIDDEN"], config["USE_LAYER_NORM"])
        q2 = MLPQ(r2, OBS_DIM, ACT_DIM, config["CRITIC_HIDDEN"], config["USE_LAYER_NORM"])
        actor_ts, q1_ts, q2_ts = make_ts(actor), make_ts(q1), make_ts(q2)
        q1_t, q2_t = q1, q2
        log_alpha = jnp.array(0.0); alpha_state = alpha_init(log_alpha)
        buf = ReplayBuffer(config["REPLAY_SIZE"], OBS_DIM, ACT_DIM)

        ue = config["UPDATE_EVERY"]
        total_grad_steps = (config["STEPS"] - config["UPDATE_AFTER"]) // ue * ue
        print(f"INFO: {total_grad_steps} gradient steps in chunks of {ue}")

        sample_act = eqx.filter_jit(lambda a, x, k: a.sample(x, k)[0])
        det_act = eqx.filter_jit(lambda a, x: a.deterministic(x))

        def evaluate(actor, deterministic, n_eps):
            nonlocal rng
            rets, succs = [], []
            for _ in range(n_eps):
                o, _ = test_env.reset(); ret = 0.0
                for _ in range(HORIZON):
                    x = jnp.asarray(o, jnp.float32)
                    if deterministic:
                        a = det_act(actor, x)
                    else:
                        rng, k = jr.split(rng); a = sample_act(actor, x, k)
                    o, r, term, trunc, info = test_env.step(np.asarray(a)); ret += r
                    if term or trunc:
                        break
                rets.append(ret); succs.append(info["episode_success"])
            return float(np.mean(rets)), float(np.mean(succs))

        carry = (actor_ts, q1_ts, q2_ts, q1_t, q2_t, log_alpha, alpha_state)
        obs, _ = env.reset(); ep_ret, ep_len, ep_succ = 0.0, 0, False
        grad_step, t0 = 0, time.time()
        ep_returns, ep_successes = [], []
        m = {k: jnp.nan for k in ("pi_loss", "q1_loss", "alpha", "q1", "target_q", "actor_grad_norm", "critic_grad_norm")}
        for t in range(config["STEPS"]):
            if t < config["START_STEPS"]:
                a = rng_np.uniform(-1, 1, ACT_DIM).astype(np.float32)
            else:
                rng, k = jr.split(rng)
                a = np.asarray(sample_act(net_of(carry[0]), jnp.asarray(obs, jnp.float32), k))
            o2, r, term, trunc, info = env.step(a)
            ep_ret += r; ep_len += 1; ep_succ = ep_succ or info["success"]
            buf.store(obs, a, r, o2, float(term))  # truncation at the horizon is not a terminal state
            obs = o2
            if term or trunc:
                ep_returns.append(ep_ret); ep_successes.append(ep_succ)
                obs, _ = env.reset(); ep_ret, ep_len, ep_succ = 0.0, 0, False

            if t >= config["UPDATE_AFTER"] and (t + 1) % ue == 0:
                batches = {k: jnp.asarray(v) for k, v in buf.sample(rng_np, config["BATCH_SIZE"], n=ue).items()}
                rng, k = jr.split(rng)
                carry, m = update_chunk(carry, batches, jnp.asarray(grad_step), k)
                grad_step += ue
                if not bool(jnp.isfinite(m["pi_loss"])) or not bool(jnp.isfinite(m["q1_loss"])):
                    raise FloatingPointError(f"non-finite loss at step {t}: {m}")

            if (t + 1) % config["LOG_EVERY"] == 0:
                actor = net_of(carry[0])
                ret_s, suc_s = evaluate(actor, False, config["TEST_EPS_STOCHASTIC"])
                ret_d, suc_d = evaluate(actor, True, config["TEST_EPS_DETERMINISTIC"])
                writer.add_scalar("test/stochastic/return", ret_s, t + 1); writer.add_scalar("test/stochastic/success", suc_s, t + 1)
                writer.add_scalar("test/deterministic/return", ret_d, t + 1); writer.add_scalar("test/deterministic/success", suc_d, t + 1)
                if ep_returns:
                    writer.add_scalar("train/return", float(np.mean(ep_returns[-20:])), t + 1)
                    writer.add_scalar("train/success", float(np.mean(ep_successes[-20:])), t + 1)
                for key, v in m.items():
                    writer.add_scalar(f"loss/{key}", float(v), t + 1)
                print(f"step {t+1:8d} | train ret {np.mean(ep_returns[-20:]) if ep_returns else 0:7.1f} succ {np.mean(ep_successes[-20:]) if ep_successes else 0:.2f} "
                      f"| test stoch ret {ret_s:7.1f} succ {suc_s:.2f} | det ret {ret_d:7.1f} succ {suc_d:.2f} "
                      f"| alpha {float(m['alpha']):.3f} q1 {float(m['q1']):.1f} targ {float(m['target_q']):.1f} "
                      f"gn a {float(m['actor_grad_norm']):.2f} c {float(m['critic_grad_norm']):.2f} | {(t+1)/(time.time()-t0):.0f} steps/s")

        actor_ts, q1_ts, q2_ts = carry[:3]
        return net_of(actor_ts), net_of(q1_ts), net_of(q2_ts), evaluate

    return train


def render_episode(config, actor, path):
    import imageio
    env = make_env(config["TASK"], seed=config["SEED"] + 200, render_mode="rgb_array", camera_name=config["CAMERA"])
    det_act = eqx.filter_jit(lambda a, x: a.deterministic(x))
    o, _ = env.reset(); frames, ret = [], 0.0
    for _ in range(HORIZON):
        frames.append(np.flipud(env.render()))  # metaworld's offscreen corner cameras come out upside down
        o, r, term, trunc, info = env.step(np.asarray(det_act(actor, jnp.asarray(o, jnp.float32)))); ret += r
        if term or trunc:
            break
    imageio.mimwrite(path, frames, fps=20, codec="libx264", quality=8)
    return ret, info["episode_success"]


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="push-v3"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=float, default=1e6); p.add_argument("--tag", default="")
    p.add_argument("--final_eps", type=int, default=50)
    args = p.parse_args()

    config = {
        "TASK": args.task, "SEED": args.seed, "STEPS": int(args.steps),
        # SAC, as in continualworld/sac/sac.py
        "REPLAY_SIZE": int(1e6), "BATCH_SIZE": 128, "LR": 1e-3, "GAMMA": 0.99, "POLYAK": 0.995,
        "START_STEPS": 10_000, "UPDATE_AFTER": 1000, "UPDATE_EVERY": 50,
        "LOG_EVERY": 20_000, "TEST_EPS_STOCHASTIC": 10, "TEST_EPS_DETERMINISTIC": 1, "FINAL_TEST_EPS": args.final_eps,
        # networks, as in run_single.py defaults
        "ACTOR_HIDDEN": [256, 256, 256, 256], "CRITIC_HIDDEN": [256, 256, 256, 256], "ACTIVATION": "lrelu", "USE_LAYER_NORM": True,
        "CAMERA": "corner2", "ENV_NAME": "metaworld-3.1.1 " + args.task + " (Continual World wrappers: horizon 200, random_init_all)",
    }
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args.tag}" if args.tag else "")
    save_path = f"data/eqx_sac_mlp/{args.task}/{stamp}"
    os.makedirs(save_path, exist_ok=False)
    config["SAVE_PATH"] = save_path
    json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)
    writer = SummaryWriter(log_dir=save_path)

    t0 = time.time()
    actor, q1, q2, evaluate = make_train(config)(jr.PRNGKey(args.seed), writer, save_path)
    train_time = time.time() - t0
    print(f"INFO: training complete in {train_time/60:.1f} min")

    eqx.tree_serialise_leaves(os.path.join(save_path, "checkpoint.eqx"), (actor, q1, q2))
    ret_d, suc_d = evaluate(actor, True, config["FINAL_TEST_EPS"])
    ret_s, suc_s = evaluate(actor, False, config["FINAL_TEST_EPS"])
    print(f"final over {config['FINAL_TEST_EPS']} episodes: deterministic return {ret_d:.1f} success {suc_d:.2f} | stochastic return {ret_s:.1f} success {suc_s:.2f}")
    mp4 = os.path.join(save_path, f"{args.task}_deterministic.mp4")
    ret_v, suc_v = render_episode(config, actor, mp4)
    config.update(FINAL_DET_RETURN=ret_d, FINAL_DET_SUCCESS=suc_d, FINAL_STOCH_RETURN=ret_s, FINAL_STOCH_SUCCESS=suc_s, TRAIN_MINUTES=train_time / 60)
    json.dump(config, open(os.path.join(save_path, "run_config.json"), "w"), indent=4)
    hid = ",".join(map(str, config["ACTOR_HIDDEN"]))
    with open(os.path.join(save_path, "HANDOFF.md"), "w") as f:
        f.write(f"""# MLP SAC on {args.task} (Continual World, metaworld 3.1.1 v3 task)

- Run directory: `{save_path}`; checkpoint.eqx = (MLPSACActor, MLPQ, MLPQ) via eqx.tree_serialise_leaves.
- Reload: templates `MLPSACActor(key, {OBS_DIM}, {ACT_DIM}, [{hid}], use_layer_norm={config['USE_LAYER_NORM']})`,
  `MLPQ(key, {OBS_DIM}, {ACT_DIM}, [{hid}], use_layer_norm={config['USE_LAYER_NORM']})` twice. No observation normalizer: the nets take the raw 39-dim v3 observation.
- Deterministic policy: `actor.deterministic(obs)` (tanh of the mean head).
- Env: `eqx_marl.common.metaworld_env.make_env("{args.task}", seed)` with the benchmark wrappers (horizon {HORIZON}, random_init_all, success = any step with info['success']).
- Training: {config['STEPS']} env steps, SAC hyperparameters of continualworld/sac/sac.py with the run_single.py network defaults (hidden [{hid}], leaky ReLU, layer norm), seed {args.seed}, {train_time/60:.1f} min.
- Final ({config['FINAL_TEST_EPS']} episodes): deterministic return {ret_d:.1f}, success {suc_d:.2f}; stochastic return {ret_s:.1f}, success {suc_s:.2f}. Rendered episode `{os.path.basename(mp4)}`: return {ret_v:.1f}, success {suc_v}.
- Code: `src/eqx_marl/sac_mlp.py` (same SAC core as `sac_kan.py`), `common/models_sac.py`, `common/metaworld_env.py`. Conda env `mappo` (metaworld installed with --no-deps on mujoco 3.2.7).
""")
    print(f"INFO: saved to {save_path}")
