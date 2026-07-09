"""
Top-level model: shared/separate obs encoder + flow policy head + critic head.

The observation encoder is the component shared (or not) between policy and
critic for experiment 2. Stop-gradient routing on the encoder output controls
which objective trains the shared backbone:

  variant                  shared  stopgrad_critic  stopgrad_policy
  ---------------------------------------------------------------
  separate                 no      -                -
  shared_no_stopgrad       yes     no               no
  shared_stopgrad_critic   yes     yes              no    (only actor/FM trains psi)
  shared_stopgrad_policy   yes     no               yes   (only critic trains psi)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from flow_rl.models.encoder import ObsEncoder
from flow_rl.models.policy import TransformerPolicy
from flow_rl.models.critic import CriticHead

VARIANTS = ("separate", "shared_no_stopgrad", "shared_stopgrad_critic", "shared_stopgrad_policy")


class FlowRLModel(eqx.Module):
    policy_encoder: ObsEncoder
    critic_encoder: ObsEncoder | None
    policy: TransformerPolicy
    critics: list                       # ensemble of CriticHead (clipped double-Q)

    shared: bool
    stopgrad_critic: bool
    stopgrad_policy: bool
    horizon: int
    act_dim: int

    def __init__(
        self,
        variant: str = "separate",
        state_dim: int = 2,
        encoder_num: int = 1,
        hidden_size: int = 512,
        img_size: int = 96,
        rgb_encoder_model: str = "resnet-18",
        pretrained: bool = True,
        horizon: int = 5,
        act_dim: int = 2,
        policy_depth: int = 5,
        critic_depth: int = 5,
        num_critics: int = 2,
        heads: int = 8,
        ff_mult: int = 4,
        use_image: bool = True,
        critic_sigmoid: bool = True,
        use_divergence: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        assert variant in VARIANTS, f"unknown variant {variant}"
        self.shared = variant != "separate"
        self.stopgrad_critic = variant == "shared_stopgrad_critic"
        self.stopgrad_policy = variant == "shared_stopgrad_policy"
        self.horizon = horizon
        self.act_dim = act_dim

        k_penc, k_cenc, k_pol, k_crit = jax.random.split(key, 4)

        enc_kwargs = dict(
            state_dim = state_dim,
            encoder_num = encoder_num,
            hidden_size = hidden_size,
            img_size = img_size,
            rgb_encoder_model = rgb_encoder_model,
            pretrained = pretrained,
            use_image = use_image,
        )
        self.policy_encoder = ObsEncoder(**enc_kwargs, key = k_penc)
        self.critic_encoder = None if self.shared else ObsEncoder(**enc_kwargs, key = k_cenc)

        self.policy = TransformerPolicy(
            horizon = horizon, act_dim = act_dim, hidden_size = hidden_size,
            depth = policy_depth, heads = heads, ff_mult = ff_mult,
            use_divergence = use_divergence, key = k_pol,
        )
        critic_keys = jax.random.split(k_crit, num_critics)
        self.critics = [
            CriticHead(
                horizon = horizon, act_dim = act_dim, hidden_size = hidden_size,
                depth = critic_depth, heads = heads, ff_mult = ff_mult, sigmoid_out = critic_sigmoid, key = ck,
            )
            for ck in critic_keys
        ]

    # --- encoder forwards with stop-gradient routing ---
    def policy_tokens(self, img: Array, state: Array, key: PRNGKeyArray) -> Array:
        t = self.policy_encoder(img, state, key)
        return jax.lax.stop_gradient(t) if self.stopgrad_policy else t

    def critic_tokens(self, img: Array, state: Array, key: PRNGKeyArray) -> Array:
        enc = self.policy_encoder if self.critic_encoder is None else self.critic_encoder
        t = enc(img, state, key)
        return jax.lax.stop_gradient(t) if self.stopgrad_critic else t

    # --- head forwards ---
    def velocity(self, tokens_p: Array, x_t: Array, t: Array) -> Array:
        return self.policy.velocity(tokens_p, x_t, t)

    def velocity_avg(self, tokens_p: Array, x_t: Array, r: Array, t: Array) -> Array:
        return self.policy.velocity_avg(tokens_p, x_t, r, t)

    def divergence(self, tokens_p: Array, x_t: Array, r: Array, t: Array) -> Array:
        return self.policy.divergence(tokens_p, x_t, r, t)

    def velocity_dev(self, tokens_p: Array, x_t: Array, t: Array):
        return self.policy.velocity_dev(tokens_p, x_t, t)

    def regress(self, tokens_p: Array) -> Array:
        return self.policy.regress(tokens_p)

    def q(self, tokens_c: Array, action_chunk: Array) -> Array:
        return jnp.stack([c(tokens_c, action_chunk) for c in self.critics])   # (num_critics,)
