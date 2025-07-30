"""
In this file we have a simple ActorCritic for continuous observation action spaces.
It has a policy that outputs (mean, scale) and a value function that outputs value.
NOTE: therefore the actual sampling of this action distribution is done outside of this
object. A simple example is in the if __name__ == "__main__" below.
"""

import os
import numpy as np
from typing import Tuple, Callable, List

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class ActorCritic(eqx.Module):
    """
    This is a feed forward actor critic combined module. This is useful for IPPO
    where we modify both the actor and critic simultaneously and therefore can
    group the actor and critic together here and use optax to get an optimization
    state across the entire eqx.Module
    """
    
    # Learned variables
    actor_layers: Tuple[eqx.nn.Linear, ...]
    critic_layers: Tuple[eqx.nn.Linear, ...]
    log_std: jax.Array

    # Static parameters
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key: jr.PRNGKey,
        actor_layer_sizes: List[int] = [6, 64, 64, 2],
        critic_layer_sizes: List[int] = [6, 64, 64, 1],
        actor_kernel_init: List[float] = [np.sqrt(2), np.sqrt(2), 0.01],
        critic_kernel_init: List[float] = [np.sqrt(2), np.sqrt(2), 1.0],
        activation: Callable = jax.nn.relu,
    ):
        self.activation = activation
        actor_key, critic_key = jr.split(key)

        # —— actor network ——
        actor_keys = jr.split(actor_key, num=len(actor_layer_sizes))
        self.actor_layers = []
        
        for i, (in_f, out_f) in enumerate(
            zip(actor_layer_sizes[:-1], actor_layer_sizes[1:])
        ):
            layer = eqx.nn.Linear(in_f, out_f, key=actor_keys[i])

            # (Re‑)initialise using orthogonal(scale) + constant(0) bias
            wkey, _ = jr.split(actor_keys[i])
            weight = jax.nn.initializers.orthogonal(actor_kernel_init[i])(
                wkey, (out_f, in_f), jnp.float32
            )
            bias = jnp.zeros((out_f,), dtype=jnp.float32)

            # Update the layer – eqx Modules are frozen, so use object.__setattr__
            object.__setattr__(layer, "weight", weight)
            object.__setattr__(layer, "bias", bias)

            self.actor_layers.append(layer)

        # —— critic network ——
        critic_keys = jr.split(critic_key, len(critic_layer_sizes) - 1)
        self.critic_layers = []

        for i, (in_f, out_f) in enumerate(
            zip(critic_layer_sizes[:-1], critic_layer_sizes[1:])
        ):
            layer = eqx.nn.Linear(in_f, out_f, key=critic_keys[i])

            wkey, _ = jr.split(critic_keys[i])
            weight = jax.nn.initializers.orthogonal(critic_kernel_init[i])(
                wkey, (out_f, in_f), jnp.float32
            )
            bias = jnp.zeros((out_f,), dtype=jnp.float32)

            object.__setattr__(layer, "weight", weight)
            object.__setattr__(layer, "bias", bias)

            self.critic_layers.append(layer)

        # —— learnable log‑std parameter ——
        self.log_std = jnp.zeros((actor_layer_sizes[-1],))  # broadcasted over batch at runtime

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Returns (policy_dist, value_estimate) for an input batch `x`."""

        # —— actor ——
        h = x
        for layer in self.actor_layers[:-1]:
            h = self.activation(layer(h))
        actor_mean = self.actor_layers[-1](h)                        # (B, action_dim)
        actor_scale = jnp.exp(self.log_std)

        # —— critic ——
        h = x
        for layer in self.critic_layers[:-1]:
            h = self.activation(layer(h))
        value = jnp.squeeze(self.critic_layers[-1](h), axis=-1)      # (B,)

        return actor_mean, actor_scale, value


class Actor(eqx.Module):
    """
    This is an independent actor module without any critic. This is useful for
    systems that require asynchronous actor, critic updates. Therefore we can use
    an optax optimizer to update just the actor independently of the critic easily
    if they are separated. This is useful for MAPPO.
    """

    # Learned variables
    actor_layers: Tuple[eqx.nn.Linear, ...]
    log_std: jax.Array

    # Static parameters
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key: jr.PRNGKey,
        actor_layer_sizes: List[int] = [6, 64, 64, 2],
        actor_kernel_init: List[float] = [np.sqrt(2), np.sqrt(2), 0.01],
        activation: Callable = jax.nn.relu,
    ):
        self.activation = activation

        # —— actor network ——
        actor_keys = jr.split(key, num=len(actor_layer_sizes))
        self.actor_layers = []
        
        for i, (in_f, out_f) in enumerate(
            zip(actor_layer_sizes[:-1], actor_layer_sizes[1:])
        ):
            layer = eqx.nn.Linear(in_f, out_f, key=actor_keys[i])

            # (Re‑)initialise using orthogonal(scale) + constant(0) bias
            wkey, _ = jr.split(actor_keys[i])
            weight = jax.nn.initializers.orthogonal(actor_kernel_init[i])(
                wkey, (out_f, in_f), jnp.float32
            )
            bias = jnp.zeros((out_f,), dtype=jnp.float32)

            # Update the layer – eqx Modules are frozen, so use object.__setattr__
            object.__setattr__(layer, "weight", weight)
            object.__setattr__(layer, "bias", bias)

            self.actor_layers.append(layer)

        # —— learnable log‑std parameter ——
        self.log_std = jnp.zeros((actor_layer_sizes[-1],))  # broadcasted over batch at runtime

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns (policy_dist, value_estimate) for an input batch `x`."""

        # —— actor ——
        h = x
        for layer in self.actor_layers[:-1]:
            h = self.activation(layer(h))
        actor_mean = self.actor_layers[-1](h)                        # (B, action_dim)
        actor_scale = jnp.exp(self.log_std)

        return actor_mean, actor_scale



class Critic(eqx.Module):
    """
    This is an independent critic module without any actor. This is useful for
    systems that require asynchronous actor, critic updates. Therefore we can use
    an optax optimizer to update just the critic independently of the actor easily
    if they are separated. This is useful for MAPPO.
    """
    # Learned variables
    critic_layers: Tuple[eqx.nn.Linear, ...]

    # Static parameters
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        key: jr.PRNGKey,
        critic_layer_sizes: List[int] = [6, 64, 64, 1],
        critic_kernel_init: List[float] = [np.sqrt(2), np.sqrt(2), 1.0],
        activation: Callable = jax.nn.relu,
    ):
        self.activation = activation

        # —— critic network ——
        critic_keys = jr.split(key, len(critic_layer_sizes) - 1)
        self.critic_layers = []

        for i, (in_f, out_f) in enumerate(
            zip(critic_layer_sizes[:-1], critic_layer_sizes[1:])
        ):
            layer = eqx.nn.Linear(in_f, out_f, key=critic_keys[i])

            wkey, _ = jr.split(critic_keys[i])
            weight = jax.nn.initializers.orthogonal(critic_kernel_init[i])(
                wkey, (out_f, in_f), jnp.float32
            )
            bias = jnp.zeros((out_f,), dtype=jnp.float32)

            object.__setattr__(layer, "weight", weight)
            object.__setattr__(layer, "bias", bias)

            self.critic_layers.append(layer)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Returns (policy_dist, value_estimate) for an input batch `x`."""

        # —— critic ——
        h = x
        for layer in self.critic_layers[:-1]:
            h = self.activation(layer(h))
        value = jnp.squeeze(self.critic_layers[-1](h), axis=-1)      # (B,)

        return value



if __name__ == "__main__":

    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

    # temporary for testing
    from flax import linen as nn
    from flax.core import FrozenDict
    from flax.linen.initializers import constant, orthogonal
    from typing import Sequence
    import distrax
    import distreqx.distributions as dist

    def _flax_to_eqx(flax_params: FrozenDict, eqx_model: eqx.Module) -> eqx.Module:
        """
        Returns a *new* Equinox model whose Linear layers contain the (transposed)
        weights & biases from the Flax `Dense` layers.  All other parameters
        (log_std, etc.) are copied verbatim.
        """
        dense_kernels, dense_biases = [], []
        for lyr in ["Dense_0", "Dense_1", "Dense_2"]:
            dense_kernels.append(flax_params["params"][lyr]["kernel"])
            dense_biases.append(flax_params["params"][lyr]["bias"])

        # Flax kernel   : (in , out)
        # Equinox weight: (out, in )  → need transpose
        new_actor_layers, new_critic_layers = [], []
        for (layer, k, b) in zip(eqx_model.actor_layers, dense_kernels, dense_biases):
            layer = eqx.tree_at(
                lambda l: (l.weight, l.bias),
                layer,
                (k.T, b),
            )
            new_actor_layers.append(layer)

        dense_kernels, dense_biases = [], []
        for lyr in ["Dense_3", "Dense_4", "Dense_5"]:
            dense_kernels.append(flax_params["params"][lyr]["kernel"])
            dense_biases.append(flax_params["params"][lyr]["bias"])

        # same trick for critic
        for (layer, k, b) in zip(eqx_model.critic_layers, dense_kernels, dense_biases):
            layer = eqx.tree_at(
                lambda l: (l.weight, l.bias),
                layer,
                (k.T, b),
            )
            new_critic_layers.append(layer)

        # copy log_std too
        new_model = eqx.tree_at(
            lambda m: (m.actor_layers, m.critic_layers, m.log_std),
            eqx_model,
            (tuple(new_actor_layers), tuple(new_critic_layers),
            flax_params["params"]["log_std"]),
        )
        return new_model


    class ActorCriticFlax(nn.Module):
        action_dim: Sequence[int]
        activation: str = "tanh"

        @nn.compact
        def __call__(self, x):
            if self.activation == "relu":
                activation = nn.relu
            else:
                activation = nn.tanh
            actor_mean = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            actor_mean = activation(actor_mean)
            actor_mean = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(actor_mean)
            actor_mean = activation(actor_mean)
            actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

            critic = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(x)
            critic = activation(critic)
            critic = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
            )(critic)
            critic = activation(critic)
            critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
                critic
            )

            return pi, jnp.squeeze(critic, axis=-1)


    # Little example of how to inference the actor critic
    obs_dim = 24
    action_dim = 6
    key = jax.random.PRNGKey(0)

    model = ActorCritic(
        key=key,
        actor_layer_sizes=[obs_dim, 64, 64, action_dim],
        critic_layer_sizes=[obs_dim, 64, 64, 1],
        actor_kernel_init=[np.sqrt(2), np.sqrt(2), 0.01],
        critic_kernel_init=[np.sqrt(2), np.sqrt(2), 1],
        activation=jax.nn.relu    
    )

    dummy_obs = jnp.zeros((32, obs_dim))
    actor_mean, actor_scale, value = jax.vmap(model)(dummy_obs)        # policy.log_prob(…), value.shape == (32,)
    
    # this is how we then apply the distribution over the top of this
    actor_pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(
        actor_mean,
        actor_scale,                        # (action_dim,)
    )
    # we need these functions to allow for vmapping of the log_prob and entropies
    eqx_pi_log_prob = lambda d, a: d.log_prob(a)  # helper for filter_vmap
    eqx_pi_entropy = lambda d: d.entropy()        # helper for filter_vmap

    # ====================================== #
    # ====== validation against flax ======= #
    # ====================================== #

    # ---- keys & dummy batch ----------------------------------------------------
    BATCH  = 128
    OBS_DIM, ACT_DIM = 6, 2
    rng     = jr.PRNGKey(0)
    rng, k1, k2, k3 = jr.split(rng, 4)
    obs     = jr.normal(k1, (BATCH, OBS_DIM))

    # ---- Flax ------------------------------------------------------------------
    flax_model = ActorCriticFlax(action_dim=ACT_DIM, activation="tanh")
    flax_vars  = flax_model.init(k2, obs)                  # {'params': …}
    flax_pi, flax_val = flax_model.apply(flax_vars, obs)

    # ---- Equinox ---------------------------------------------------------------
    equinox_model = ActorCritic(                       # ← your Equinox class
        key=k3,
        actor_layer_sizes=[OBS_DIM, 64, 64, ACT_DIM],
        critic_layer_sizes=[OBS_DIM, 64, 64, 1],
        actor_kernel_init=[np.sqrt(2), np.sqrt(2), 0.01],
        critic_kernel_init=[np.sqrt(2), np.sqrt(2), 1],
        activation=jax.nn.tanh,
    )

    # (optionally) overwrite weights so the two start identical
    equinox_model = _flax_to_eqx(flax_vars, equinox_model)

    eqx_mean, eqx_scale, eqx_val = eqx.filter_vmap(equinox_model)(obs)
    
    eqx_pi = eqx.filter_vmap(dist.MultivariateNormalDiag)(
        eqx_mean,
        eqx_scale,                        
    )
    eqx_pi_log_prob = lambda d, a: d.log_prob(a)
    eqx_pi_entropy = lambda d: d.entropy()
    eqx_samp  = eqx_pi.sample(key)                      # works
    ent = eqx.filter_vmap(eqx_pi_entropy)(eqx_pi)       # works
    lp = eqx.filter_vmap(eqx_pi_log_prob)(eqx_pi, eqx_samp) # works

    # ent = eqx_pi.entropy() # evaluates, incorrect values, correct shape
    # lp = eqx_pi.log_prob(eqx_samp) # evaluates, incorrect values (all same), correct shape
    
    # ---- probe -----------------------------------------------------------------
    probe_key = jr.PRNGKey(42)
    actions   = flax_pi.sample(seed=probe_key)             # shape (BATCH, ACT_DIM)

    flax_lp   = flax_pi.log_prob(actions)
    eqx_lp    = eqx.filter_vmap(eqx_pi_log_prob)(eqx_pi, actions)

    print("mean|Δ log‑prob| :", jnp.mean(jnp.abs(flax_lp - eqx_lp)))
    print("mean|Δ entropy| :", jnp.mean(jnp.abs(flax_pi.entropy() - eqx.filter_vmap(eqx_pi_entropy)(eqx_pi))))
    print("mean|Δ value  | :", jnp.mean(jnp.abs(flax_val - eqx_val)))

    # extra: sample comparison ----------------------------------------------------
    flax_samp = flax_pi.sample(seed=probe_key)
    eqx_samp  = eqx_pi.sample(probe_key)

    print("mean|Δ sample | :", jnp.mean(jnp.abs(flax_samp - eqx_samp)))


