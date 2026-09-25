"""
SAC with a KAN actor and twin KAN critics on Continual World (Meta-World v3) single tasks.

Hyperparameters mirror continualworld/sac/sac.py: 1e6 env steps, replay 1e6, batch 128,
lr 1e-3, gamma 0.99, polyak 0.995, auto alpha with target entropy -act_dim, 1e4 uniform
start steps, updates start after 1e3 steps, every 50 env steps do 50 gradient steps,
evaluation every 2e4 steps with 10 stochastic + 1 deterministic episodes.

KAN-specific, as in ff_PPO_kan.py: running observation normalization (replay holds raw obs,
normalized at sample time), grid stages keyed by gradient-step fraction with Adam moment
transition (kaneqx.transition), and periodic same-size grid refits on replay samples.

    python -m eqx_marl.sac_kan --task push-v3 --seed 0            # full run
    python -m eqx_marl.sac_kan --task push-v3 --seed 0 --steps 2e5 # smoke run

Saves data/eqx_sac_kan/<task>/<timestamp>/{config.json, run_config.json, checkpoint.eqx,
events.out.tfevents.*, <task>_deterministic.mp4, HANDOFF.md}. Reload with
    eqx.tree_deserialise_leaves(path, (KANSACActor(...G=10), KANQ(...G=10), KANQ(...G=10), RunningMeanStd((39,))))
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

from kaneqx import adam, transition, trainable_filter
from eqx_marl.common.models_sac import KANSACActor, KANQ, MLPSACActor, MLPQ
from eqx_marl.common.normalize import RunningMeanStd
from eqx_marl.common.eqx_utils import filter_scan
from eqx_marl.common.metaworld_env import make_env, HORIZON, OBS_DIM, ACT_DIM


def _is_kan(net):
    return hasattr(net, "kan")


def _grid_size(*nets):
    """G of the first KAN among nets, or 0 when the run has no KAN."""
    return next((n.kan.layers[0].G for n in nets if _is_kan(n)), 0)


def _global_norm(grads):
    return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(eqx.filter(grads, eqx.is_array))))


class TrainState(NamedTuple):
    opt_state: Any   # kaneqx AdamState over the trainable partition
    static: Any      # the non-trainable partition (grid, mask, static fields)


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


def make_train(config):
    stages = sorted((float(k), int(v)) for k, v in config["GRID_STAGES"].items())
    assert stages[0][0] == 0.0, "GRID_STAGES must start at fraction 0"
    opt_init, opt_update, get_params = adam(config["LR"])
    alpha_init, alpha_update, alpha_params = adam(config["LR"])
    target_entropy = -float(ACT_DIM)
    gamma, polyak = config["GAMMA"], config["POLYAK"]

    def make_ts(net, opt_state=None):
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(opt_init(params) if opt_state is None else transition(opt_state, params), static)

    def net_of(ts):
        return eqx.combine(get_params(ts.opt_state), ts.static)

    def with_net(ts, net):
        params, static = eqx.partition(net, trainable_filter(net))
        return TrainState(eqx.tree_at(lambda s: s.params, ts.opt_state, params), static)

    def _update_one(carry, rms, batch, step, key):
        actor_ts, q1_ts, q2_ts, q1_t, q2_t, log_alpha, alpha_state = carry
        obs, next_obs = rms.normalize(batch["obs"]), rms.normalize(batch["next_obs"])
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
        # Polyak targets: trainable leaves move, grid/mask are shared with the live critics by construction
        q1_t = jax.tree.map(lambda t, n: polyak * t + (1 - polyak) * n, q1_t, net_of(q1_ts))
        q2_t = jax.tree.map(lambda t, n: polyak * t + (1 - polyak) * n, q2_t, net_of(q2_ts))
        metrics = dict(pi_loss=pi_loss, q1_loss=q1_loss, q2_loss=q2_loss, alpha=alpha, logp_pi=jnp.mean(logp),
                       q1=jnp.mean(eqx.filter_vmap(net_of(q1_ts))(obs, act)),
                       # diagnostics (as in sac_mlp.py): regression target, Q at policy actions, gradient norms
                       target_q=jnp.mean(backup), q_pi=q_pi,
                       actor_grad_norm=_global_norm(a_grads), critic_grad_norm=_global_norm(c_grads))
        return (actor_ts, q1_ts, q2_ts, q1_t, q2_t, alpha_params(alpha_state), alpha_state), metrics

    @eqx.filter_jit
    def update_chunk(carry, rms, batches, step0, key):
        """UPDATE_EVERY gradient steps in one jitted scan: batches are stacked (n, B, ...)."""
        n = batches["rew"].shape[0]
        xs = (batches, step0 + jnp.arange(n), jr.split(key, n))
        carry, metrics = filter_scan(lambda c, x: _update_one(c, rms, x[0], x[1], x[2]), carry, xs)
        return carry, jax.tree.map(lambda m: m[-1], metrics)

    @eqx.filter_jit
    def refit_grids(actor_ts, q1_ts, q2_ts, q1_t, q2_t, rms, obs, act, G):
        """Same-size (or extended) grid update on replay samples; returns new nets (Adam handled by caller)."""
        x = rms.normalize(obs); xa = jnp.concatenate([x, act], axis=1)
        upd = lambda net, inp: net.update_grids(inp, G) if _is_kan(net) else net   # MLP components are left alone
        actor = upd(net_of(actor_ts), x)
        q1, q2 = upd(net_of(q1_ts), xa), upd(net_of(q2_ts), xa)
        q1_t, q2_t = upd(q1_t, xa), upd(q2_t, xa)
        return actor, q1, q2, q1_t, q2_t

    def train(rng, writer, save_path):
        rng_np = np.random.default_rng(config["SEED"])
        env = make_env(config["TASK"], seed=config["SEED"])
        test_env = make_env(config["TASK"], seed=config["SEED"] + 100)
        rng, ra, r1, r2 = jr.split(rng, 4)
        G0 = stages[0][1]
        if config["ACTOR_NET"] == "kan":
            actor = KANSACActor(ra, [OBS_DIM] + config["ACTOR_HIDDEN"] + [ACT_DIM], k=config["K"], G=G0)
        else:   # the benchmark MLP of sac_mlp.py, fed the same normalized observations
            actor = MLPSACActor(ra, OBS_DIM, ACT_DIM, config["MLP_HIDDEN"], config["MLP_LAYER_NORM"])
        if config["CRITIC_NET"] == "kan":
            q1 = KANQ(r1, [OBS_DIM + ACT_DIM] + config["CRITIC_HIDDEN"] + [1], k=config["K"], G=G0)
            q2 = KANQ(r2, [OBS_DIM + ACT_DIM] + config["CRITIC_HIDDEN"] + [1], k=config["K"], G=G0)
        else:
            q1 = MLPQ(r1, OBS_DIM, ACT_DIM, config["MLP_HIDDEN"], config["MLP_LAYER_NORM"])
            q2 = MLPQ(r2, OBS_DIM, ACT_DIM, config["MLP_HIDDEN"], config["MLP_LAYER_NORM"])
        actor_ts, q1_ts, q2_ts = make_ts(actor), make_ts(q1), make_ts(q2)
        q1_t, q2_t = q1, q2
        log_alpha = jnp.array(0.0); alpha_state = alpha_init(log_alpha)
        rms = RunningMeanStd((OBS_DIM,))
        buf = ReplayBuffer(config["REPLAY_SIZE"], OBS_DIM, ACT_DIM)

        ue = config["UPDATE_EVERY"]
        total_grad_steps = (config["STEPS"] - config["UPDATE_AFTER"]) // ue * ue
        boundaries = {int(round(f * total_grad_steps / ue)) * ue: G for f, G in stages if f > 0}   # aligned to chunks
        refit_every = max(ue, int(round(config["GRID_REFIT_FRAC"] * total_grad_steps / ue)) * ue)
        print(f"INFO: {total_grad_steps} gradient steps in chunks of {ue}; grid extensions at {boundaries}; same-size refit every {refit_every}")

        sample_act = eqx.filter_jit(lambda a, x, k: a.sample(x, k)[0])
        det_act = eqx.filter_jit(lambda a, x: a.deterministic(x))

        def evaluate(actor, deterministic, n_eps):
            rets, succs = [], []
            for _ in range(n_eps):
                o, _ = test_env.reset(); ret = 0.0
                for _ in range(HORIZON):
                    x = rms.normalize(jnp.asarray(o, jnp.float32))
                    nonlocal rng
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
        grad_step, t0, obs_chunk = 0, time.time(), []
        ep_returns, ep_successes = [], []
        m = {k: jnp.nan for k in ("pi_loss", "q1_loss", "alpha", "q1", "target_q", "actor_grad_norm", "critic_grad_norm")}
        for t in range(config["STEPS"]):
            if t < config["START_STEPS"]:
                a = rng_np.uniform(-1, 1, ACT_DIM).astype(np.float32)
            else:
                rng, k = jr.split(rng)
                a = np.asarray(sample_act(net_of(carry[0]), rms.normalize(jnp.asarray(obs, jnp.float32)), k))
            o2, r, term, trunc, info = env.step(a)
            ep_ret += r; ep_len += 1; ep_succ = ep_succ or info["success"]
            buf.store(obs, a, r, o2, float(term))  # truncation at the horizon is not a terminal state
            obs_chunk.append(obs); obs = o2
            if term or trunc:
                ep_returns.append(ep_ret); ep_successes.append(ep_succ)
                obs, _ = env.reset(); ep_ret, ep_len, ep_succ = 0.0, 0, False

            if t >= config["UPDATE_AFTER"] and (t + 1) % ue == 0:
                rms = rms.update(jnp.asarray(np.stack(obs_chunk), jnp.float32)); obs_chunk = []
                if grad_step in boundaries or (grad_step > 0 and grad_step % refit_every == 0):
                    actor_ts, q1_ts, q2_ts, q1_t, q2_t, log_alpha, alpha_state = carry
                    G = boundaries.get(grad_step, _grid_size(net_of(actor_ts), net_of(q1_ts)))
                    gb = buf.sample(rng_np, config["GRID_UPDATE_SAMPLES"])
                    actor_n, q1_n, q2_n, q1_t, q2_t = refit_grids(actor_ts, q1_ts, q2_ts, q1_t, q2_t, rms, jnp.asarray(gb["obs"]), jnp.asarray(gb["act"]), G)
                    if grad_step in boundaries:  # extension: shapes change, transition Adam moments
                        actor_ts, q1_ts, q2_ts = make_ts(actor_n, actor_ts.opt_state), make_ts(q1_n, q1_ts.opt_state), make_ts(q2_n, q2_ts.opt_state)
                        print(f"INFO: grad step {grad_step}: grids extended to G={G}, Adam state transitioned")
                    else:                        # same size: knots move, Adam moments kept
                        actor_ts, q1_ts, q2_ts = with_net(actor_ts, actor_n), with_net(q1_ts, q1_n), with_net(q2_ts, q2_n)
                    carry = (actor_ts, q1_ts, q2_ts, q1_t, q2_t, log_alpha, alpha_state)
                batches = {k: jnp.asarray(v) for k, v in buf.sample(rng_np, config["BATCH_SIZE"], n=ue).items()}
                rng, k = jr.split(rng)
                carry, m = update_chunk(carry, rms, batches, jnp.asarray(grad_step), k)
                grad_step += ue
                if not bool(jnp.isfinite(m["pi_loss"])) or not bool(jnp.isfinite(m["q1_loss"])):
                    raise FloatingPointError(f"non-finite loss at step {t}: {m}")

            if (t + 1) % config["LOG_EVERY"] == 0:
                actor_ts = carry[0]; actor = net_of(actor_ts)
                ret_s, suc_s = evaluate(actor, False, config["TEST_EPS_STOCHASTIC"])
                ret_d, suc_d = evaluate(actor, True, config["TEST_EPS_DETERMINISTIC"])
                writer.add_scalar("test/stochastic/return", ret_s, t + 1); writer.add_scalar("test/stochastic/success", suc_s, t + 1)
                writer.add_scalar("test/deterministic/return", ret_d, t + 1); writer.add_scalar("test/deterministic/success", suc_d, t + 1)
                if ep_returns:
                    writer.add_scalar("train/return", float(np.mean(ep_returns[-20:])), t + 1)
                    writer.add_scalar("train/success", float(np.mean(ep_successes[-20:])), t + 1)
                for key, v in m.items():
                    writer.add_scalar(f"loss/{key}", float(v), t + 1)
                writer.add_scalar("grid_size", _grid_size(actor, net_of(carry[1])), t + 1)
                print(f"step {t+1:8d} | train ret {np.mean(ep_returns[-20:]) if ep_returns else 0:7.1f} succ {np.mean(ep_successes[-20:]) if ep_successes else 0:.2f} "
                      f"| test stoch ret {ret_s:7.1f} succ {suc_s:.2f} | det ret {ret_d:7.1f} succ {suc_d:.2f} "
                      f"| alpha {float(m['alpha']):.3f} q1 {float(m['q1']):.1f} targ {float(m['target_q']):.1f} "
                      f"gn a {float(m['actor_grad_norm']):.2f} c {float(m['critic_grad_norm']):.2f} G {_grid_size(actor, net_of(carry[1]))} | {(t+1)/(time.time()-t0):.0f} steps/s")

        actor_ts, q1_ts, q2_ts = carry[:3]
        return net_of(actor_ts), net_of(q1_ts), net_of(q2_ts), rms, evaluate

    return train


def render_episode(config, actor, rms, path):
    import imageio
    env = make_env(config["TASK"], seed=config["SEED"] + 200, render_mode="rgb_array", camera_name=config["CAMERA"])
    det_act = eqx.filter_jit(lambda a, x: a.deterministic(x))
    o, _ = env.reset(); frames, ret = [], 0.0
    for _ in range(HORIZON):
        frames.append(np.flipud(env.render()))  # metaworld's offscreen corner cameras come out upside down
        o, r, term, trunc, info = env.step(np.asarray(det_act(actor, rms.normalize(jnp.asarray(o, jnp.float32))))); ret += r
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
    p.add_argument("--actor", choices=["kan", "mlp"], default="kan"); p.add_argument("--critic", choices=["kan", "mlp"], default="kan")
    args = p.parse_args()

    config = {
        "TASK": args.task, "SEED": args.seed, "STEPS": int(args.steps),
        # SAC, as in continualworld/sac/sac.py
        "REPLAY_SIZE": int(1e6), "BATCH_SIZE": 128, "LR": 1e-3, "GAMMA": 0.99, "POLYAK": 0.995,
        "START_STEPS": 10_000, "UPDATE_AFTER": 1000, "UPDATE_EVERY": 50,
        "LOG_EVERY": 20_000, "TEST_EPS_STOCHASTIC": 10, "TEST_EPS_DETERMINISTIC": 1, "FINAL_TEST_EPS": 50,
        # KAN
        "ACTOR_HIDDEN": [32, 32], "CRITIC_HIDDEN": [32, 32], "K": 3,
        "GRID_STAGES": {"0": 3, "0.38": 5, "0.67": 10},   # by fraction of gradient steps, as the Ant's 0/200/350 of 520 updates
        "GRID_REFIT_FRAC": 0.02, "GRID_UPDATE_SAMPLES": 2048,
        # hybrid diagnostics: either side can be the benchmark MLP (run_single.py defaults, as in sac_mlp.py)
        "ACTOR_NET": args.actor, "CRITIC_NET": args.critic, "MLP_HIDDEN": [256, 256, 256, 256], "MLP_LAYER_NORM": True,
        "CAMERA": "corner2", "ENV_NAME": "metaworld-3.1.1 " + args.task + " (Continual World wrappers: horizon 200, random_init_all)",
    }
    hybrid = "" if (args.actor, args.critic) == ("kan", "kan") else f"_{args.actor}actor_{args.critic}critic"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + hybrid + (f"_{args.tag}" if args.tag else "")
    save_path = f"data/eqx_sac_kan/{args.task}/{stamp}"
    os.makedirs(save_path, exist_ok=False)
    config["SAVE_PATH"] = save_path
    json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)
    writer = SummaryWriter(log_dir=save_path)

    t0 = time.time()
    actor, q1, q2, rms, evaluate = make_train(config)(jr.PRNGKey(args.seed), writer, save_path)
    train_time = time.time() - t0
    print(f"INFO: training complete in {train_time/60:.1f} min")

    eqx.tree_serialise_leaves(os.path.join(save_path, "checkpoint.eqx"), (actor, q1, q2, rms))
    ret_d, suc_d = evaluate(actor, True, config["FINAL_TEST_EPS"])
    ret_s, suc_s = evaluate(actor, False, config["FINAL_TEST_EPS"])
    print(f"final over {config['FINAL_TEST_EPS']} episodes: deterministic return {ret_d:.1f} success {suc_d:.2f} | stochastic return {ret_s:.1f} success {suc_s:.2f}")
    mp4 = os.path.join(save_path, f"{args.task}_deterministic.mp4")
    ret_v, suc_v = render_episode(config, actor, rms, mp4)
    config.update(FINAL_DET_RETURN=ret_d, FINAL_DET_SUCCESS=suc_d, FINAL_STOCH_RETURN=ret_s, FINAL_STOCH_SUCCESS=suc_s, TRAIN_MINUTES=train_time / 60)
    json.dump(config, open(os.path.join(save_path, "run_config.json"), "w"), indent=4)
    G = _grid_size(actor, q1)
    mlp_h = ",".join(map(str, config["MLP_HIDDEN"]))
    actor_tmpl = (f"KANSACActor(key, [{OBS_DIM},{','.join(map(str, config['ACTOR_HIDDEN']))},{ACT_DIM}], k={config['K']}, G={G})" if config["ACTOR_NET"] == "kan"
                  else f"MLPSACActor(key, {OBS_DIM}, {ACT_DIM}, [{mlp_h}], use_layer_norm={config['MLP_LAYER_NORM']})")
    critic_tmpl = (f"KANQ(key, [{OBS_DIM + ACT_DIM},{','.join(map(str, config['CRITIC_HIDDEN']))},1], k={config['K']}, G={G})" if config["CRITIC_NET"] == "kan"
                   else f"MLPQ(key, {OBS_DIM}, {ACT_DIM}, [{mlp_h}], use_layer_norm={config['MLP_LAYER_NORM']})")
    with open(os.path.join(save_path, "HANDOFF.md"), "w") as f:
        f.write(f"""# SAC on {args.task} (Continual World, metaworld 3.1.1 v3 task): actor {config['ACTOR_NET'].upper()}, critics {config['CRITIC_NET'].upper()}

- Run directory: `{save_path}`; checkpoint.eqx = (actor, q1, q2, RunningMeanStd) via eqx.tree_serialise_leaves.
- Reload: templates `{actor_tmpl}`, `{critic_tmpl}` twice, `RunningMeanStd(({OBS_DIM},))`.
- Deterministic policy: `actor.deterministic(rms.normalize(obs))` (tanh of the mean head). Both network types take the normalized observation. Obs are the 39-dim metaworld v3 observations, raw in the replay, normalized only at the network input.
- Env: `eqx_marl.common.metaworld_env.make_env("{args.task}", seed)` with the benchmark wrappers (horizon {HORIZON}, random_init_all, success = any step with info['success']).
- Training: {config['STEPS']} env steps, SAC hyperparameters of continualworld/sac/sac.py, grid stages {config['GRID_STAGES']} by fraction of gradient steps, same-size grid refits every {config['GRID_REFIT_FRAC']:.0%}, seed {args.seed}, {train_time/60:.1f} min.
- Final ({config['FINAL_TEST_EPS']} episodes): deterministic return {ret_d:.1f}, success {suc_d:.2f}; stochastic return {ret_s:.1f}, success {suc_s:.2f}. Rendered episode `{os.path.basename(mp4)}`: return {ret_v:.1f}, success {suc_v}.
- Code: `src/eqx_marl/sac_kan.py`, `common/models_sac.py`, `common/metaworld_env.py`; network library kaneqx. Conda env `mappo` (metaworld installed with --no-deps on mujoco 3.2.7).
""")
    print(f"INFO: saved to {save_path}")
