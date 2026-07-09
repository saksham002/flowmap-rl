"""
Simple-transformer flow policy (no MMDiT).

Tokens = observation tokens (modified ResNet vision + state, from ObsEncoder)
+ a separate Fourier time token (flow mode) + action-chunk tokens; a pre-LN
transformer reads out the action positions.

  - velocity(obs_tokens, x_t, t): flow-matching velocity over the chunk.
  - regress(obs_tokens):          deterministic action chunk (L2-regression BC).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from flow_rl.models.transformer import TransformerBlock, FourierTimeEmbedding, SinusoidalTimeEmbedding
from flow_rl.models.pos_emb import get_1d_sincos_pos_embed


class TransformerPolicy(eqx.Module):
    time_embed: FourierTimeEmbedding
    mf_time_embed: SinusoidalTimeEmbedding   # OFQL/MeanFlow: smooth bounded-freq time embed (u is differentiated w.r.t. t)
    action_in_proj: eqx.nn.Linear      # flow: embed noisy action x_t
    action_query: Array                # regression: learned action queries (h, hidden)
    action_pos_embed: Array            # (h, hidden)
    type_embed: Array                  # (3, hidden): obs / time / action
    blocks: list
    final_ln: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    log_std_head: eqx.nn.Linear    # predict_dev: per-dim log-std of the Gaussian velocity
    r_type_embed: Array            # OFQL: type embedding for the 2nd (target-step r) time token; unused by velocity()
    # F2D2 divergence head (shares the transformer blocks); None unless use_divergence -> old ckpts add zero leaves
    div_cls: Array | None
    div_final_ln: eqx.nn.LayerNorm | None
    div_head: eqx.nn.Linear | None

    use_divergence: bool = eqx.field(static = True)
    horizon: int
    act_dim: int
    hidden_size: int

    def __init__(
        self,
        horizon: int = 5,
        act_dim: int = 2,
        hidden_size: int = 512,
        depth: int = 5,
        heads: int = 8,
        ff_mult: int = 4,
        use_divergence: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.horizon = horizon
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.use_divergence = use_divergence

        k_in, k_q, k_type, k_head, k_blocks, k_logstd = jax.random.split(key, 6)
        self.time_embed = FourierTimeEmbedding(hidden_size)   # dim = hidden (floq cosine basis)
        self.mf_time_embed = SinusoidalTimeEmbedding(hidden_size)   # smooth time embed for velocity_avg (OFQL)
        self.action_in_proj = eqx.nn.Linear(act_dim, hidden_size, key = k_in)
        self.action_query = 0.02 * jax.random.normal(k_q, (horizon, hidden_size))
        self.action_pos_embed = get_1d_sincos_pos_embed(hidden_size, horizon)
        self.type_embed = 0.02 * jax.random.normal(k_type, (3, hidden_size))
        block_keys = jax.random.split(k_blocks, depth)
        self.blocks = [TransformerBlock(hidden_size, heads, ff_mult, key = block_keys[i]) for i in range(depth)]
        self.final_ln = eqx.nn.LayerNorm(hidden_size)
        self.head = eqx.nn.Linear(hidden_size, act_dim, key = k_head)
        self.log_std_head = eqx.nn.Linear(hidden_size, act_dim, key = k_logstd)
        # fold_in (not a 7-way split) keeps k_in..k_logstd byte-identical -> velocity()/regress() unchanged
        self.r_type_embed = 0.02 * jax.random.normal(jax.random.fold_in(key, 7), (hidden_size,))
        # divergence head keyed via fold_in(8/10) -> does not disturb the split above (velocity byte-identical)
        if use_divergence:
            self.div_cls = 0.02 * jax.random.normal(jax.random.fold_in(key, 8), (hidden_size,))
            self.div_final_ln = eqx.nn.LayerNorm(hidden_size)
            self.div_head = eqx.nn.Linear(hidden_size, 1, key = jax.random.fold_in(key, 10))
        else:
            self.div_cls = None
            self.div_final_ln = None
            self.div_head = None

    def _features(self, tokens: Array) -> Array:
        for block in self.blocks:
            tokens = block(tokens)
        act_out = tokens[-self.horizon:]
        return jax.vmap(self.final_ln)(act_out)   # (h, hidden)

    def _readout(self, tokens: Array) -> Array:
        return jax.vmap(self.head)(self._features(tokens))   # (h, act_dim)

    def _flow_tokens(self, obs_tokens: Array, x_t: Array, t: Array) -> Array:
        time_tok = (self.time_embed(t) + self.type_embed[1])[None, :]                 # (1, hidden)
        act_tok = jax.vmap(self.action_in_proj)(x_t) + self.action_pos_embed + self.type_embed[2]
        obs = obs_tokens + self.type_embed[0]
        return jnp.concatenate([obs, time_tok, act_tok], axis = 0)

    def velocity(self, obs_tokens: Array, x_t: Array, t: Array) -> Array:
        return self._readout(self._flow_tokens(obs_tokens, x_t, t))

    def _flow_tokens_rt(self, obs_tokens: Array, x_t: Array, r: Array, t: Array) -> Array:
        """OFQL flow-map tokens: obs + a (current time t) token + a (target step r) token + action tokens."""
        t_tok = (self.mf_time_embed(t) + self.type_embed[1])[None, :]                  # (1, hidden) smooth time embed
        r_tok = (self.mf_time_embed(r) + self.r_type_embed)[None, :]                   # (1, hidden) (bounded d/dt for the JVP)
        act_tok = jax.vmap(self.action_in_proj)(x_t) + self.action_pos_embed + self.type_embed[2]
        obs = obs_tokens + self.type_embed[0]
        return jnp.concatenate([obs, t_tok, r_tok, act_tok], axis = 0)                 # action tokens stay LAST

    def velocity_avg(self, obs_tokens: Array, x_t: Array, r: Array, t: Array) -> Array:
        """OFQL: average velocity u(x_t, r, t) over the chunk (MeanFlow flow map)."""
        return self._readout(self._flow_tokens_rt(obs_tokens, x_t, r, t))

    def divergence(self, obs_tokens: Array, x_t: Array, r: Array, t: Array) -> Array:
        """F2D2 divergence head: scalar cumulative-divergence D(x_t, r, t) over the chunk.

        Same flow tokens as velocity_avg + a learned CLS token last, through the SHARED blocks, CLS ->
        LayerNorm -> Linear -> scalar (mirrors CriticHead). Separate forward from velocity_avg (shares block
        weights only). log q(a) = log N(e) + divergence_rescale * D(e, 0, 1) after one endpoint eval."""
        tokens = self._flow_tokens_rt(obs_tokens, x_t, r, t)
        x = jnp.concatenate([tokens, self.div_cls[None, :]], axis = 0)      # CLS token last
        for block in self.blocks:
            x = block(x)
        return self.div_head(self.div_final_ln(x[-1])).squeeze()

    def velocity_dev(self, obs_tokens: Array, x_t: Array, t: Array) -> tuple[Array, Array]:
        """predict_dev: (mean, log_std) of the Gaussian velocity v=a-z over the chunk."""
        feat = self._features(self._flow_tokens(obs_tokens, x_t, t))
        return jax.vmap(self.head)(feat), jax.vmap(self.log_std_head)(feat)

    def regress(self, obs_tokens: Array) -> Array:
        act_tok = self.action_query + self.action_pos_embed + self.type_embed[2]
        obs = obs_tokens + self.type_embed[0]
        x = jnp.concatenate([obs, act_tok], axis = 0)
        return self._readout(x)
