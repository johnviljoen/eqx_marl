"""
KAN actor and Q-function for SAC, built on kaneqx. Same conventions as KANActor / KANCritic
in models.py: per-sample __call__, batch with vmap, update_grids(x, G) for grid changes.

Actor: squashed Gaussian as in continualworld/sac/models.py (tanh on the sample, log-std
clipped to [-20, 2], log-prob corrected with the numerically stable softplus form).
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from kaneqx import KAN
from eqx_marl.common.models import _scale_last_layer

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def _gaussian_logp(x, mu, log_std):
    return jnp.sum(-0.5 * (((x - mu) / (jnp.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)))


class KANSACActor(eqx.Module):
    """obs (obs_dim,) -> (mu, log_std), each (act_dim,); sample(obs, key) -> (action, logp) after tanh."""

    kan: KAN
    act_dim: int = eqx.field(static=True)

    def __init__(self, key, layer_sizes: List[int], k: int = 3, G: int = 3,
                 grid_range: Tuple[float, float] = (-3.0, 3.0), last_layer_scale: float = 0.01, **kan_kwargs):
        self.act_dim = layer_sizes[-1]
        kan = KAN(layer_sizes[:-1] + [2 * self.act_dim], key, k=k, G=G, grid_range=grid_range, **kan_kwargs)
        self.kan = _scale_last_layer(kan, last_layer_scale)

    def __call__(self, x):
        out = self.kan(x)
        mu, log_std = out[: self.act_dim], out[self.act_dim:]
        return mu, jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, x, key):
        mu, log_std = self(x)
        pi = mu + jax.random.normal(key, mu.shape) * jnp.exp(log_std)
        logp = _gaussian_logp(pi, mu, log_std)
        logp = logp - jnp.sum(2.0 * (jnp.log(2.0) - pi - jax.nn.softplus(-2.0 * pi)))  # tanh correction
        return jnp.tanh(pi), logp

    def deterministic(self, x):
        return jnp.tanh(self(x)[0])

    def update_grids(self, x, G_new: int) -> "KANSACActor":
        """x: (batch, obs_dim) normalized observations."""
        return eqx.tree_at(lambda a: a.kan, self, self.kan.update_grids(x, G_new))


class KANQ(eqx.Module):
    """Q(obs, act): KAN on the concatenation, (obs_dim + act_dim,) -> scalar."""

    kan: KAN

    def __init__(self, key, layer_sizes: List[int], k: int = 3, G: int = 3,
                 grid_range: Tuple[float, float] = (-3.0, 3.0), last_layer_scale: float = 1.0, **kan_kwargs):
        kan = KAN(layer_sizes, key, k=k, G=G, grid_range=grid_range, **kan_kwargs)
        self.kan = _scale_last_layer(kan, last_layer_scale)

    def __call__(self, x, a):
        return self.kan(jnp.concatenate([x, a]))[0]

    def update_grids(self, xa, G_new: int) -> "KANQ":
        """xa: (batch, obs_dim + act_dim) normalized observations concatenated with actions."""
        return eqx.tree_at(lambda q: q.kan, self, self.kan.update_grids(xa, G_new))


# ----------------------------------------------------------------------------------------
# MLP actor and Q-function mirroring continualworld/sac/models.py with the run_single.py
# defaults: hidden (256, 256, 256, 256), leaky ReLU (slope 0.2 as tf.nn.leaky_relu), layer norm
# on = Dense -> LayerNorm -> tanh for the first block, Dense -> leaky ReLU for the rest, then a
# linear head. Keras init: glorot-uniform weights, zero biases. Same call/sample/deterministic
# interface as the KAN classes above so sac_mlp.py and sac_kan.py share the SAC core.
# ----------------------------------------------------------------------------------------

import jax.random as jr


def _glorot_linear(key, n_in: int, n_out: int) -> eqx.nn.Linear:
    lin = eqx.nn.Linear(n_in, n_out, key=key)
    lim = jnp.sqrt(6.0 / (n_in + n_out))
    w = jr.uniform(key, (n_out, n_in), minval=-lim, maxval=lim)
    return eqx.tree_at(lambda l: (l.weight, l.bias), lin, (w, jnp.zeros(n_out)))


class MLPCore(eqx.Module):
    layers: list
    norm: eqx.nn.LayerNorm | None
    slope: float = eqx.field(static=True)

    def __init__(self, key, n_in: int, hidden: List[int], use_layer_norm: bool = True, slope: float = 0.2):
        sizes = [n_in] + list(hidden)
        self.layers = [_glorot_linear(k, a, b) for k, a, b in zip(jr.split(key, len(hidden)), sizes[:-1], sizes[1:])]
        self.norm = eqx.nn.LayerNorm(hidden[0], eps=1e-3) if use_layer_norm else None  # keras LayerNormalization eps
        self.slope = slope

    def __call__(self, x):
        x = self.layers[0](x)
        x = jnp.tanh(self.norm(x)) if self.norm is not None else jax.nn.leaky_relu(x, self.slope)
        for lin in self.layers[1:]:
            x = jax.nn.leaky_relu(lin(x), self.slope)
        return x


class MLPSACActor(eqx.Module):
    """obs (obs_dim,) -> (mu, log_std); separate linear heads as in the benchmark's MlpActor."""

    core: MLPCore
    head_mu: eqx.nn.Linear
    head_log_std: eqx.nn.Linear
    act_dim: int = eqx.field(static=True)

    def __init__(self, key, obs_dim: int, act_dim: int, hidden: List[int] = (256, 256, 256, 256), use_layer_norm: bool = True):
        kc, km, ks = jr.split(key, 3)
        self.act_dim = act_dim
        self.core = MLPCore(kc, obs_dim, hidden, use_layer_norm)
        self.head_mu = _glorot_linear(km, hidden[-1], act_dim)
        self.head_log_std = _glorot_linear(ks, hidden[-1], act_dim)

    def __call__(self, x):
        h = self.core(x)
        return self.head_mu(h), jnp.clip(self.head_log_std(h), LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, x, key):
        mu, log_std = self(x)
        pi = mu + jax.random.normal(key, mu.shape) * jnp.exp(log_std)
        logp = _gaussian_logp(pi, mu, log_std)
        logp = logp - jnp.sum(2.0 * (jnp.log(2.0) - pi - jax.nn.softplus(-2.0 * pi)))
        return jnp.tanh(pi), logp

    def deterministic(self, x):
        return jnp.tanh(self(x)[0])


class MLPQ(eqx.Module):
    """Q(obs, act): MLP core on the concatenation, linear head to a scalar (the benchmark's MlpCritic)."""

    core: MLPCore
    head: eqx.nn.Linear

    def __init__(self, key, obs_dim: int, act_dim: int, hidden: List[int] = (256, 256, 256, 256), use_layer_norm: bool = True):
        kc, kh = jr.split(key)
        self.core = MLPCore(kc, obs_dim + act_dim, hidden, use_layer_norm)
        self.head = _glorot_linear(kh, hidden[-1], 1)

    def __call__(self, x, a):
        return self.head(self.core(jnp.concatenate([x, a])))[0]


if __name__ == "__main__":
    import jax.random as jr
    key = jr.PRNGKey(0)
    actor = KANSACActor(key, [39, 32, 32, 4], G=3)
    q = KANQ(key, [43, 32, 32, 1], G=3)
    x = jr.normal(jr.PRNGKey(1), (64, 39))
    a, logp = eqx.filter_vmap(actor.sample)(x, jr.split(jr.PRNGKey(2), 64))
    assert a.shape == (64, 4) and logp.shape == (64,) and bool(jnp.all(jnp.abs(a) < 1))
    assert bool(jnp.all(jnp.abs(eqx.filter_vmap(actor.deterministic)(x)) < 0.1)), "scaled last layer: near-zero initial mean"
    # tanh correction check against the exact form log(1 - tanh(pi)^2)
    mu, ls = actor(x[0]); pi = mu + 0.3
    exact = _gaussian_logp(pi, mu, ls) - jnp.sum(jnp.log(1 - jnp.tanh(pi) ** 2 + 1e-8))
    stable = _gaussian_logp(pi, mu, ls) - jnp.sum(2.0 * (jnp.log(2.0) - pi - jax.nn.softplus(-2.0 * pi)))
    assert jnp.allclose(exact, stable, atol=1e-4), (exact, stable)
    qv = eqx.filter_vmap(q)(x, a); assert qv.shape == (64,)
    actor2 = actor.update_grids(x, 5); q2 = q.update_grids(jnp.concatenate([x, a], 1), 5)
    assert actor2.kan.layers[0].G == 5 and q2.kan.layers[0].G == 5
    ma = MLPSACActor(key, 39, 4); mq = MLPQ(key, 39, 4)
    from kaneqx import trainable_filter
    n_a = sum(p.size for p in jax.tree.leaves(eqx.filter(ma, trainable_filter(ma))))
    n_q = sum(p.size for p in jax.tree.leaves(eqx.filter(mq, trainable_filter(mq))))
    assert n_a == 39*256+256 + 2*256 + 3*(256*256+256) + 2*(256*4+4), n_a   # layer-norm gain/bias included
    assert n_q == 43*256+256 + 2*256 + 3*(256*256+256) + 256+1, n_q
    a, logp = eqx.filter_vmap(ma.sample)(x, jr.split(jr.PRNGKey(2), 64))
    assert a.shape == (64, 4) and logp.shape == (64,) and eqx.filter_vmap(mq)(x, a).shape == (64,)
    assert bool(jnp.all(jnp.abs(ma.head_mu.bias) == 0)) and bool(jnp.all(ma.core.norm.weight == 1))
    print("models_sac ok (KAN + MLP)")

