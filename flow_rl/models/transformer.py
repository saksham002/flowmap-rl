"""Shared transformer pieces: pre-LN block + Fourier time embedding."""

import math

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray


class TransformerBlock(eqx.Module):
    """Pre-LN transformer encoder block (self-attention + MLP)."""

    ln1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ln2: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, hidden_size: int, heads: int, ff_mult: int, *, key: PRNGKeyArray):
        k_attn, k1, k2 = jax.random.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(hidden_size)
        self.attn = eqx.nn.MultiheadAttention(num_heads = heads, query_size = hidden_size, key = k_attn)
        self.ln2 = eqx.nn.LayerNorm(hidden_size)
        self.fc1 = eqx.nn.Linear(hidden_size, ff_mult * hidden_size, key = k1)
        self.fc2 = eqx.nn.Linear(ff_mult * hidden_size, hidden_size, key = k2)

    def __call__(self, x: Array) -> Array:
        h = jax.vmap(self.ln1)(x)
        x = x + self.attn(h, h, h)
        h = jax.vmap(self.ln2)(x)
        h = jax.vmap(self.fc2)(jax.nn.gelu(jax.vmap(self.fc1)(h)))
        return x + h


class FourierTimeEmbedding(eqx.Module):
    """Cosine Fourier-basis embedding of flow time t.

    Copied from floq's `utils/networks.py` (categorical_floq):
        times_embed = jnp.arange(1, dim + 1) * jnp.pi * times
        times_embed = jnp.cos(times_embed)
    (Their line `jnp.cos(times)` is an obvious typo for `jnp.cos(times_embed)`.)
    `dim` is set to the transformer hidden size, so the result is used directly
    as a separate time token. No learnable parameters.
    """

    dim: int

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, t: Array) -> Array:
        ks = jnp.arange(1, self.dim + 1, dtype = jnp.float32)
        return jnp.cos(ks * jnp.pi * t)   # (dim,)


class SinusoidalTimeEmbedding(eqx.Module):
    """Smooth continuous-time embedding with BOUNDED frequencies (standard diffusion/DiT timestep
    embedding). For MeanFlow, where u is differentiated w.r.t. t (the JVP), the embedding's d/dt must be
    O(1): geometric freqs in (1/max_period, 1] over t in [0,1] give |d/dt| ~ O(1). FourierTimeEmbedding
    (freqs up to dim*pi) instead has |d/dt| ~ 1500, which blows up the MeanFlow target. No parameters."""

    dim: int = eqx.field(static = True)            # static: hyperparams, NOT pytree leaves (so checkpoints
    max_period: float = eqx.field(static = True)   # saved before this module was added still deserialize)

    def __init__(self, dim: int, max_period: float = 10000.0):
        self.dim = dim
        self.max_period = max_period

    def __call__(self, t: Array) -> Array:
        half = self.dim // 2
        freqs = jnp.exp(-math.log(self.max_period) * jnp.arange(half, dtype = jnp.float32) / half)
        args = t * freqs                                            # t in [0,1], freqs <= 1 -> smooth
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)])
        if self.dim % 2:
            emb = jnp.concatenate([emb, jnp.zeros((1,), jnp.float32)])
        return emb                                                  # (dim,)
