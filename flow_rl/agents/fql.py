"""
FQL agent: one-step (or few-step) flow + Q-max, BC-regularized by flow matching.

Single combined loss minimized by one optimizer over the whole model:
  total = alpha * L_FM - Q(s, a_hat)            [policy / actor]
          + (Q(s, a_data) - y)^2                [critic / TD]

Gradient separation is enforced within the loss: critic params are detached in
the actor's Q term, the TD target y is fully stop-gradiented, and the shared
encoder's gradient flow follows FlowRLModel.{policy_tokens, critic_tokens}.

This class also covers the BC-only (use_q=False) and L2-regression
(policy_type="regression") configs.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array

from flow_rl.flow import interpolate, sample_action as flow_sample
from flow_rl.agents.base import BaseAgent


class FQLAgent(BaseAgent):

    def bc_loss(self, model, tokens_p, action, keys) -> dict:
        cfg = self.cfg
        h, act_dim = cfg.horizon, cfg.act_dim

        # --- L2-regression BC (no flow) ---
        if cfg.policy_type == "regression":
            a_pred = model.regress(tokens_p)
            return {"l2": jnp.mean((a_pred - action) ** 2)}

        # --- flow-matching BC loss (regularization toward data policy) ---
        z_fm = jax.random.normal(keys[1], (h, act_dim))
        # straight-flow: target a-z at fixed t=0 (one-step path v(z,0)); else standard flow matching t~U[0,max_t]
        t = jnp.array(0.0) if cfg.agent.straight_flow else jax.random.uniform(
            keys[2], (), minval = 0.0, maxval = cfg.agent.max_t)
        x_t, v_target = interpolate(action, z_fm, t)
        v_pred = model.velocity(tokens_p, x_t, t)
        l_fm = jnp.mean((v_pred - v_target) ** 2)
        return {"l_fm": l_fm}

    def sample_action(self, model, tokens_p, z_key, eps_key) -> Array:
        cfg = self.cfg
        h, act_dim = cfg.horizon, cfg.act_dim
        z = jax.random.normal(z_key, (h, act_dim))
        velocity_fn = lambda x, tt: model.velocity(tokens_p, x, tt)
        return jnp.clip(flow_sample(velocity_fn, z, steps = cfg.agent.inner_flow_steps), -1.0, 1.0)  # FQL clips

    def predict_action(self, model, tokens_p, z_key, eps_key, steps) -> Array:
        cfg = self.cfg
        if cfg.policy_type == "regression":
            return jnp.clip(model.regress(tokens_p), -1.0, 1.0)
        z = jax.random.normal(z_key, (cfg.horizon, cfg.act_dim))
        velocity_fn = lambda x, tt: model.velocity(tokens_p, x, tt)
        return jnp.clip(flow_sample(velocity_fn, z, steps = steps), -1.0, 1.0)
