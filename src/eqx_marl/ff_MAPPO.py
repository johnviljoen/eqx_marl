"""
Based on the MAPPO implementations from JaxMARL and MAVA, but continuous obs, act
spaces
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

from eqx_marl.common.models import Actor, Critic
from eqx_marl.common.eqx_utils import filter_scan

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_state: jnp.ndarray
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
    save_path = f'data/eqx_mappo_validation/{current_datetime}'
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
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # INIT NETWORK
    actor_network = Actor(
        key=rng_init,
        actor_layer_sizes=[env.observation_space(env.agents[0]).shape[0], 64, 64, env.action_space(env.agents[0]).shape[0]],
        actor_kernel_init=[jnp.sqrt(2), jnp.sqrt(2), 0.01],
        activation=jax.nn.tanh,
    )

    critic_network = Critic(
        key=rng_init,
        critic_layer_sizes=[env.global_observation_space.shape[0], 64, 64, 1],
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
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, rng = runner_state
                actor_network, _, critic_network, _ = train_states

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                mean, scale = eqx.filter_vmap(actor_network)(obs_batch)

                pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
                action = pi.sample(_rng)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # LOG PROB
                pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
                log_prob = eqx.filter_vmap(pi_log_prob)(pi, action)

                # VALUE
                value = eqx.filter_vmap(critic_network)(last_obs["global_state"]) # this should be changed if we want global obs for value and local for actors

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    jnp.tile(value, env.num_agents).squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    jnp.tile(last_obs["global_state"], (env.num_agents, 1)), # world_state,
                    info,
                    # avail_actions, # only for discrete time systems
                )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, transition

            runner_state, traj_batch = filter_scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, rng = runner_state
            _, _, critic_network, _ = train_states

            last_val = eqx.filter_vmap(critic_network)(last_obs["global_state"])
            last_val = jnp.tile(last_val, env.num_agents).squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
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
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_network, actor_opt_state, critic_network, critic_opt_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_network, traj_batch, gae):
                        # RERUN NETWORK
                        mean, scale = eqx.filter_vmap(eqx.filter_vmap(actor_network))(traj_batch.obs)

                        pi = eqx.filter_vmap(eqx.filter_vmap(dist.MultivariateNormalDiag))(mean, scale)

                        # TODO validate this is the correct number of filter_vmaps john
                        pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
                        log_prob = eqx.filter_vmap(eqx.filter_vmap(pi_log_prob))(pi, traj_batch.action)

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
                        entropy = eqx.filter_vmap(eqx.filter_vmap(pi_entropy))(pi).mean()

                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                    
                    def _critic_loss_fn(critic_network, traj_batch, targets):
                        # RERUN NETWORK
                        value = eqx.filter_vmap(eqx.filter_vmap(critic_network))(traj_batch.global_state) # this should be changed if we want global obs for value and local for actors

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

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                
                batch = (
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = filter_scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info


            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )


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
                step = metric["update_steps"]
                for key, value in metric.items():
                    if key != "loss":
                        writer.add_scalar(key, value, step)
                    else:
                        for k, v in metric["loss"].items():
                            writer.add_scalar(k, v, step)

            update_steps = update_steps + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (
                actor_network,
                actor_opt_state,
                critic_network,
                critic_opt_state,
            ),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = filter_scan(
            _update_step, (runner_state, jnp.array(0)), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def generate_and_render_clp_rollout(env, actor_network, rng, num_timesteps=1000):
    
    from tqdm import tqdm
    from brax.io import html

    rng, _rng = jr.split(rng)
    agent_obs, state = env.reset(rng=_rng); rng, _rng = jr.split(rng)
    rollout = []
    env_step = jax.jit(env.step)
    actor_network = jax.jit(jax.vmap(actor_network))

    agents = sorted(agent_obs.keys())          # ['agent_0', 'agent_1', …]
    agents.remove("global_state")

    rollout = []
    for i in tqdm(range(num_timesteps)):
        obs = jnp.stack([agent_obs[a] for a in agents], axis=1)   # (100, 4, 18)
        mean, scale = actor_network(obs.T)
        pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(mean, scale)
        act = pi.sample(_rng); rng, _rng = jr.split(rng)
        agent_acts = {a: act[i, :] for i, a in enumerate(agents)}
        agent_obs, state, reward, done, info = env_step(_rng, state, agent_acts); rng, _rng = jr.split(rng)
        rollout.append(state.env_state.pipeline_state)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'data/rollouts/{current_datetime}.html'
    os.makedirs("data/rollouts", exist_ok=True)
    with open(save_name, 'w') as f:
        f.write(html.render(env.sys.tree_replace({'opt.timestep': env.dt}), rollout))

    pass

if __name__ == "__main__":

    # these was the config for hanabi - probably need something different for our env
    # config = {
    #     "LR": 5.0e-4,
    #     "NUM_ENVS": 1024 ,
    #     "NUM_STEPS": 128 ,
    #     "TOTAL_TIMESTEPS": 1e10,
    #     "FC_DIM_SIZE": 128,
    #     "UPDATE_EPOCHS": 4,
    #     "NUM_MINIBATCHES": 4,
    #     "GAMMA": 0.99,
    #     "GAE_LAMBDA": 0.95,
    #     "CLIP_EPS": 0.2,
    #     "SCALE_CLIP_EPS": False,
    #     "ENT_COEF": 0.01,
    #     "VF_COEF": 0.5,
    #     "MAX_GRAD_NORM": 0.5,
    #     "SEED": 30,
    #     "NUM_SEEDS": 2,
    #     "ENV_KWARGS": {},
    #     "ANNEAL_LR": True
    # }

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # dynamically allocate memory like pytorch does
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["MUJOCO_GL"] = "egl"     # if you have NVIDIA + EGL drivers for rendering

    # jax.config.update("jax_disable_jit", True)

    # ant env IPPO successful config - also MAPPO successful
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
        "SCALE_CLIP_EPS": False, # new to MAPPO
        "ENT_COEF": 2e-6,
        "VF_COEF": 4.5,
        "MAX_GRAD_NORM": 0.5,
        "SEED": 0,
        "ANNEAL_LR": True,
        "DEVICE": 0,
        "DISABLE_JIT": False,
    }

    rng = jax.random.PRNGKey(config["SEED"])
    rng, _rng = jr.split(rng)

    from eqx_marl.env_template_brax_MAPPO import YourEnv

    env = YourEnv()

    train_jit = jax.jit(make_train(env, config, _rng)); rng, _rng = jr.split(rng)
    out = train_jit(_rng); rng, _rng = jr.split(rng)

    runner_state = out["runner_state"]
    actor_network = runner_state[0][0][0]

    generate_and_render_clp_rollout(env, actor_network, _rng)

    pass

