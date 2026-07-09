"""Transformer critic Q(s, a_chunk) -> scalar, with LayerNorm (pre-LN)."""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from flow_rl.models.pos_emb import get_1d_sincos_pos_embed
from flow_rl.models.transformer import TransformerBlock


class CriticHead(eqx.Module):
    """
    Q(s, a_{t:t+h}) over an action chunk.

    Tokens = obs tokens + action-chunk tokens + a learned readout (CLS) token,
    with learned type embeddings; CLS output -> LayerNorm -> Linear -> scalar Q.
    """

    action_proj: eqx.nn.Linear
    action_pos_embed: Array
    type_embed: Array          # (3, hidden): obs / action / cls
    cls_token: Array           # (hidden,)
    blocks: list
    final_ln: eqx.nn.LayerNorm
    head: eqx.nn.Linear

    horizon: int
    act_dim: int
    hidden_size: int
    sigmoid_out: bool

    def __init__(
        self,
        horizon: int = 5,
        act_dim: int = 2,
        hidden_size: int = 512,
        depth: int = 6,
        heads: int = 8,
        ff_mult: int = 4,
        sigmoid_out: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.horizon = horizon
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.sigmoid_out = sigmoid_out

        k_act, k_type, k_cls, k_head, k_blocks = jax.random.split(key, 5)
        self.action_proj = eqx.nn.Linear(act_dim, hidden_size, key = k_act)
        self.action_pos_embed = get_1d_sincos_pos_embed(hidden_size, horizon)
        self.type_embed = 0.02 * jax.random.normal(k_type, (3, hidden_size))
        self.cls_token = 0.02 * jax.random.normal(k_cls, (hidden_size,))
        block_keys = jax.random.split(k_blocks, depth)
        self.blocks = [
            TransformerBlock(hidden_size, heads, ff_mult, key = block_keys[i]) for i in range(depth)
        ]
        self.final_ln = eqx.nn.LayerNorm(hidden_size)
        self.head = eqx.nn.Linear(hidden_size, 1, key = k_head)

    def __call__(self, obs_tokens: Array, action_chunk: Array) -> Array:
        """
        Args:
            obs_tokens: (num_obs_tokens, hidden_size)
            action_chunk: (horizon, act_dim)
        Returns:
            q: scalar
        """
        act_tokens = jax.vmap(self.action_proj)(action_chunk) + self.action_pos_embed
        obs_tokens = obs_tokens + self.type_embed[0]
        act_tokens = act_tokens + self.type_embed[1]
        cls = (self.cls_token + self.type_embed[2])[None, :]

        x = jnp.concatenate([obs_tokens, act_tokens, cls], axis = 0)
        for block in self.blocks:
            x = block(x)
        q = self.head(self.final_ln(x[-1])).squeeze()
        return jax.nn.sigmoid(q) if self.sigmoid_out else q   # sigmoid => Q in (0,1) (sparse reward); else raw
