""" 
Based on JaxMARL IPPO we have an equinox implementation of IPPO
"""

import os
import json
from datetime import datetime
from tensorboardX import SummaryWriter
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import distreqx.distributions as dist

from eqx_marl.common.models import ActorCritic
from eqx_marl.common.eqx_utils import filter_scan

from env_template_waymax_IPPO import flatten_last_2_dim

flatten_first_2_dim = lambda x: jnp.reshape(x, (-1,) + x.shape[2:])

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))], -1)
    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(env, config, rng_init):

    # create a directory for saving the model and logs
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f'data/eqx_ippo_waymax/{current_datetime}'
    os.makedirs(save_path, exist_ok=False)
    writer = SummaryWriter(log_dir=save_path)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    network = ActorCritic(
        key=rng_init,
        actor_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, env.action_space(env.agents[0]).shape[0]],
        critic_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, 1],
        actor_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 0.01],
        critic_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 1],
        activation=jax.nn.tanh,
    )

    if config["ANNEAL_LR"]:
        opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        opt = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), 
            optax.adam(config["LR"], eps=1e-5)
        )

    opt_state = opt.init(network)

    def train(rng):        

        # INIT ENV
        rng, _rng = jr.split(rng)
        reset_rng = jr.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                network, opt_state, env_state, last_obs, update_count, rng = runner_state
                obs_batch = flatten_last_2_dim(last_obs) # batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # SELECT ACTION
                rng, _rng = jr.split(rng)
                mean, scale, value = eqx.filter_vmap(eqx.filter_vmap(network))(obs_batch)
                
                pi = eqx.filter_vmap(eqx.filter_vmap(dist.MultivariateNormalDiag))(mean, scale)
                pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
                action = pi.sample(_rng)
                log_prob = eqx.filter_vmap(eqx.filter_vmap(pi_log_prob))(pi, action)
                
                # env_act = {a: action[:,i] for i, a in enumerate(env.agents)} # unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jr.split(rng)
                rng_step = jr.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action,
                )

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    flatten_first_2_dim(action),
                    flatten_first_2_dim(value),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    flatten_first_2_dim(log_prob),
                    flatten_first_2_dim(obs_batch),
                    info,
                )
                runner_state = (network, opt_state, env_state, obsv, update_count, rng)
                return runner_state, transition

            runner_state, traj_batch = filter_scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            # CALCULATE ADVANTAGE
            network, opt_state, env_state, last_obs, update_count, rng = runner_state
            
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            _, _, last_val = eqx.filter_vmap(network)(last_obs_batch)

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

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(network, traj_batch, gae, targets):
                        # RERUN NETWORK
                        mean, scale, value = eqx.filter_vmap(network)(traj_batch.obs)
                        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
                        pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
                        log_prob = eqx.filter_vmap(pi_log_prob)(pi, traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    network, opt_state = train_state
                    grad_fn = eqx.filter_value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(network, traj_batch, advantages, targets)
                    updates, opt_state = opt.update(grads, opt_state)
                    network = eqx.apply_updates(network, updates)

                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }
                    return (network, opt_state), loss_info
                network, opt_state, traj_batch, advantages, targets, rng = update_state

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
                (network, opt_state), loss_info = filter_scan(
                    _update_minbatch, (network, opt_state), minibatches
                )
                update_state = (network, opt_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric):
                step = metric["update_step"]
                for key, value in metric.items():
                    if key != "update_step":
                        writer.add_scalar(key, value, step)

            update_state = (network, opt_state, traj_batch, advantages, targets, rng)

            update_state, loss_info = filter_scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            network, opt_state = update_state[0], update_state[1]

            metric = traj_batch.info
            rng = update_state[-1]

            update_count = update_count + 1
            r0 = {"ratio_0": loss_info["ratio"][0,0].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_count
            metric["env_step"] = update_count * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **loss_info, **r0}
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (network, opt_state, env_state, last_obs, update_count, rng)
            return runner_state, metric

        rng, _rng = jr.split(rng)
        runner_state = (network, opt_state, env_state, obsv, jnp.array(0), _rng)
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

    from env_template_waymax_IPPO import IPPO_WaymaxEnv

    config = {
        "LR": 1e-3,
        # "NUM_ENVS": 64,
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
    env = IPPO_WaymaxEnv()
    config["NUM_ENVS"] = env.batch_size
    config["ENV_NAME"] = env.__class__.__name__
    train = make_train(env, config, _rng)
    train_jit = jax.jit(train, device=jax.devices()[config["DEVICE"]])
    out = train_jit(rng)
    
    print("INFO: training complete")