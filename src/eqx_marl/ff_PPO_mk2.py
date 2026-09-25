"""
Based on JaxMARL IPPO - adapted back to a clean PPO, then adapted to AHAC
according to open source code base + paper

This was then adapted from the AHAC back to PPO and outperforms the original
PPO on the mjx ant so I am keeping it around. We keep the Actor and Critic
seperate here.
"""

import os
import json
from datetime import datetime
from tensorboardX import SummaryWriter
from typing import NamedTuple

import mujoco as mj # lets us extract contact forces through mjx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import distreqx.distributions as dist

from eqx_marl.common.models import Actor, Critic
from eqx_marl.common.eqx_utils import filter_scan

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    cfs: jnp.ndarray # contact forces

def make_train(env, config, rng_init):

    # create a directory for saving the model and logs
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f'data/eqx_ppo_mk2_validation/{current_datetime}'
    os.makedirs(save_path, exist_ok=False)
    writer = SummaryWriter(log_dir=save_path)
    config["SAVE_PATH"] = save_path
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    config["NUM_ACTORS"] = 1 * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # INIT NETWORK
    actor_network = Actor(
        key=rng_init,
        actor_layer_sizes=[env.observation_space.shape[0], 64, 64, env.action_space.shape[0]],
        actor_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 0.01],
        activation=jax.nn.tanh,
    )

    critic_network = Critic(
        key=rng_init,
        critic_layer_sizes=[env.observation_space.shape[0], 64, 64, 1],
        critic_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 1],
        activation=jax.nn.tanh,
    )

    if config["ANNEAL_LR"]:
        actor_opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
        critic_opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        actor_opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
            optax.adam(config["LR"], eps=1e-5)
        )
        critic_opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
            optax.adam(config["LR"], eps=1e-5)
        )

    actor_opt_state = actor_opt.init(actor_network)
    critic_opt_state = critic_opt.init(critic_network)

    def train(rng):

        # INIT ENV
        rng, _rng = jr.split(rng)
        reset_rng = jr.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            train_states, env_state, last_obs, update_count, rng = runner_state
            actor_network, _, critic_network, _ = train_states

            obs_batch = last_obs # batchify(last_obs, env.agents, config["NUM_ACTORS"])
            # SELECT ACTION
            rng, _rng = jr.split(rng)
            mean, scale = eqx.filter_vmap(actor_network)(obs_batch)
            value = eqx.filter_vmap(critic_network)(last_obs) # this should be changed if we want global obs for value and local for actors

            pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
            pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
            action = pi.sample(_rng)
            log_prob = eqx.filter_vmap(pi_log_prob)(pi, action)

            env_act = action # unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

            # STEP ENV
            rng, _rng = jr.split(rng)
            rng_step = jr.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, env_state, env_act,
            )

            # CONTACT
            ps = env_state.env_state.pipeline_state
            force_pyramid = ps.efc_force[:, ps.contact.efc_address]

            info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
            transition = Transition(
                done, # batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                action,
                value,
                reward, # batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                log_prob,
                obs_batch,
                info,
                force_pyramid
            )
            runner_state = (train_states, env_state, obsv, update_count, rng)
            return runner_state, transition

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=8,
            )
            return advantages, advantages + traj_batch.value

        # TRAIN LOOP
        def _update_step(runner_state, unused):

            runner_state, traj_batch = filter_scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, update_count, rng = runner_state
            _, _, critic_network, _ = train_states

            last_obs_batch = last_obs # batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_val = eqx.filter_vmap(critic_network)(last_obs_batch)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_network, actor_opt_state, critic_network, critic_opt_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_network, traj_batch, gae):
                        # RERUN NETWORK
                        mean, scale = eqx.filter_vmap(actor_network)(traj_batch.obs)
                        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
                        pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
                        log_prob = eqx.filter_vmap(pi_log_prob)(pi, traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()

                        pi_entropy = lambda d: d.entropy()
                        entropy = eqx.filter_vmap(pi_entropy)(pi).mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_network, traj_batch, targets):
                        # RERUN NETWORK
                        value = eqx.filter_vmap(critic_network)(traj_batch.obs)
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = eqx.filter_value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_network, traj_batch, advantages
                    )
                    critic_grad_fn = eqx.filter_value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_network, traj_batch, targets
                    )

                    actor_updates, actor_opt_state = actor_opt.update(actor_grads, actor_opt_state)
                    actor_network = eqx.apply_updates(actor_network, actor_updates)
                    
                    critic_updates, critic_opt_state = critic_opt.update(critic_grads, critic_opt_state)
                    critic_network = eqx.apply_updates(critic_network, critic_updates)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_network, actor_opt_state, critic_network, critic_opt_state), loss_info
                train_states, traj_batch, advantages, targets, rng = update_state


                rng, _rng = jr.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jr.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_states, loss_info = filter_scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (train_states, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (train_states, traj_batch, advantages, targets, rng)

            update_state, loss_info = filter_scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]

            metric = traj_batch.info
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = loss_info
            rng = update_state[-1]

            def callback(metric):
                step = metric["update_step"]
                for key, value in metric.items():
                    if key != "loss":
                        writer.add_scalar(key, value, step)
                    else:
                        for k, v in metric["loss"].items():
                            writer.add_scalar(k, v, step)

            update_count = update_count + 1
            r0 = {"ratio0": loss_info["ratio"].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_count
            metric["env_step"] = update_count * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **loss_info, **r0}
            jax.experimental.io_callback(callback, None, metric)
            
            runner_state = (train_states, env_state, last_obs, update_count, rng)
            return runner_state, metric

        rng, _rng = jr.split(rng)
        runner_state = (
            (
                actor_network,
                actor_opt_state,
                critic_network,
                critic_opt_state,
            ),
            env_state, obsv, jnp.array(0), _rng
        )
        runner_state, metric = filter_scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

if __name__ == "__main__":

    # ============= #
    # example usage #
    # ============= #

    # some setup, feel free to change
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # dynamically allocate memory like pytorch does
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["MUJOCO_GL"] = "egl"     # if you have NVIDIA + EGL drivers for rendering
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_log_compiles", True)
    jax.config.update('jax_default_matmul_precision', "highest")
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    # we can use the same PPO env here
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
        "DISABLE_JIT": False,
    }

    rng = jr.PRNGKey(config["SEED"])
    rng, _rng = jr.split(rng)
    env = YourEnv() #backend="mjx") # need mjx for contact forces
    config["ENV_NAME"] = env.__class__.__name__
    train = make_train(env, config, _rng)
    train_jit = jax.jit(train, device=jax.devices()[config["DEVICE"]])
    out = train_jit(rng)

    print("INFO: training complete")

    # save the final networks; reload with eqx.tree_deserialise_leaves(path, (Actor(...), Critic(...)))
    actor_network, _, critic_network, _ = out["runner_state"][0]
    eqx.tree_serialise_leaves(os.path.join(config["SAVE_PATH"], "checkpoint.eqx"), (actor_network, critic_network))
    print(f"INFO: saved checkpoint to {config['SAVE_PATH']}")
