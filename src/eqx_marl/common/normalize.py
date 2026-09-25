"""
Running mean / variance of observations (Welford, parallel form), as an immutable
eqx.Module so it can live in a scan carry.

    rms = RunningMeanStd((obs_dim,))
    rms = rms.update(obs_batch)          # obs_batch: (N, obs_dim)
    x = rms.normalize(obs)
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class RunningMeanStd(eqx.Module):
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = jnp.asarray(epsilon)

    def update(self, batch) -> "RunningMeanStd":
        batch = batch.reshape(-1, *self.mean.shape)
        b_mean = batch.mean(axis=0)
        b_var = batch.var(axis=0)
        b_count = batch.shape[0]

        delta = b_mean - self.mean
        tot = self.count + b_count
        mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        var = (m_a + m_b + jnp.square(delta) * self.count * b_count / tot) / tot
        return eqx.tree_at(lambda r: (r.mean, r.var, r.count), self, (mean, var, tot))

    def normalize(self, x, clip: float = 10.0):
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -clip, clip)


if __name__ == "__main__":
    key = jax.random.key(0)
    data = 3.0 + 2.0 * jax.random.normal(key, (5000, 4))
    rms = RunningMeanStd((4,))
    for chunk in jnp.split(data, 10):
        rms = rms.update(chunk)
    assert jnp.allclose(rms.mean, data.mean(0), atol=1e-2)
    assert jnp.allclose(rms.var, data.var(0), atol=1e-1)
    z = rms.normalize(data)
    assert jnp.allclose(z.mean(0), 0.0, atol=1e-2) and jnp.allclose(z.std(0), 1.0, atol=1e-2)
    print("normalize ok")
