"""Sinusoidal positional embeddings (ported from jflow_match)."""

import math
import numpy as np
import jax.numpy as jnp
from jaxtyping import Array


def get_1d_sincos_pos_embed(embed_dim: int, length: int, max_period: int = 10000) -> Array:
    """1D sinusoidal positional embeddings of shape (length, embed_dim)."""
    positions = jnp.arange(length, dtype = jnp.float32)
    half = embed_dim // 2
    freqs = jnp.exp(-math.log(max_period) * jnp.arange(0, half, dtype = jnp.float32) / half)
    args = positions[:, None] * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis = -1)
    if embed_dim % 2 == 1:
        embedding = jnp.concatenate([embedding, jnp.zeros((length, 1))], axis = -1)
    return embedding


def get_2d_sincos_pos_embed(embed_dim: int, height: int, width: int) -> np.ndarray:
    """2D sinusoidal positional embeddings of shape (height*width, embed_dim)."""
    grid_h = np.arange(height, dtype = np.float32)
    grid_w = np.arange(width, dtype = np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis = 0)
    grid = grid.reshape([2, 1, height, width])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis = 1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype = np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis = 1)
