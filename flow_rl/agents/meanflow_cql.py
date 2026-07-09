"""
MeanFlow-CQL agent: the MeanFlow-F2D2 policy/divergence objective (arXiv:2512.02636 Sec. 3.2,
Eq 3.11-3.13, as in agents/meanflow.py) + double-Q critic + CQL conservative penalty, with the divergence
head supplying log q for the importance-sampled logsumexp. SINGLE stage; REQUIRES the meanflow BC run's
final checkpoint as the warm start (cfg.init_ckpt). Structural mirror of agents/shortcut_cql.py.

Convention = flow.py (t=0 noise, t=1 data); current time t <= target s; GROUND-TRUTH tangent v = a - e;
endpoint a = clip(e + u(e,0,1), -1, 1);  log q(a) = log N(e) + rescale * D(e,0,1)  [D predicts -avg-div/rescale;
divergence_rescale MUST match the meanflow BC run; adaptive L2 (norm_eps, norm_p) on L_mf ONLY, L_div-mf raw].

  L_mf     = adaptive( || u_th(x_t,t,s) - sg( (s-t)*du/dt + v ) ||^2 )      [weight alpha on t==s, 1 on t<s]
  L_div-mf = ( D_th(x_t,t,s) - sg( (s-t)*dD/dt - hutch_div(u_th(x_t,t,t)) ) )^2       [RAW; weight 1]

Both targets fully sg'd (noise on the target side). NO teacher checkpoint and NO actor-side EMA: the
divergence anchor is the STUDENT's own diagonal under sg (Eq 3.12, reference "self"); the EMA target
network is used ONLY for the critic TD backup. D is trained only by L_div-mf; the CQL proposals and their
log q are stop-gradiented, so the penalty never trains the policy map or D.

  total = alpha*L_mf(t==s) + L_mf(t<s) + L_div-mf - E[mean_i Q_i(s, a_hat)] + L_TD + L_CQL
  dual:  max_{alpha'>=0} alpha' * (mean gap - cql_budget)                (CQLMixin, unchanged)
"""

import math
import jax
import jax.numpy as jnp
from jaxtyping import Array

from flow_rl.flow import sample_action as flow_sample
from flow_rl.agents.base import BaseAgent, _detach_critics
from flow_rl.agents.cql import CQLMixin


class MeanFlowCQLAgent(CQLMixin, BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.agent.use_q, "MeanFlowCQLAgent is the RL stage (use_q=True)"
        assert cfg.agent.use_divergence, "CQL log q comes from the divergence head"
        assert cfg.init_ckpt, "init_ckpt (the meanflow BC run's final checkpoint) is required"

    def _sample_t_s(self, key):
        """Logit-normal pair ordered t <= s (MeanFlow time sampling), fraction flow_ratio exactly t == s."""
        a = self.cfg.agent
        k_t0, k_t1, k_coin = jax.random.split(key, 3)
        if a.time_dist == "logitnormal":
            t0 = jax.nn.sigmoid(a.t_mean + a.t_std * jax.random.normal(k_t0, ()))
            t1 = jax.nn.sigmoid(a.t_mean + a.t_std * jax.random.normal(k_t1, ()))
        else:
            t0 = jax.random.uniform(k_t0, ())
            t1 = jax.random.uniform(k_t1, ())
        t = jnp.minimum(t0, t1).astype(jnp.float32)
        s = jnp.maximum(t0, t1).astype(jnp.float32)
        s = jnp.where(jax.random.uniform(k_coin, ()) < a.flow_ratio, t, s).astype(jnp.float32)
        return t, s

    # ---------- CQL proposal block: one-step endpoint + divergence-head log q ----------
    def _policy_logp_block(self, model, toks, key):
        """N meanflow-endpoint proposals + F2D2 divergence-head log q. Returns (a, logp, log_base, logdet),
        all stop-gradiented (D is trained only by its own loss, never by the CQL penalty)."""
        cfg = self.cfg
        h, act_dim, N = cfg.horizon, cfg.act_dim, cfg.agent.cql_n_actions
        d = h * act_dim
        rescale = cfg.agent.divergence_rescale
        k_z, _ = jax.random.split(key)                            # keep the (k_z, k_e) split shape as the base block
        e = jax.random.normal(k_z, (N, h, act_dim))
        zero, one = jnp.float32(0.0), jnp.float32(1.0)

        def endpoint_logp(ee):
            u = model.velocity_avg(toks, ee, zero, one)
            a_hat = ee + u                                        # noise -> data endpoint (pre-clip)
            D = model.divergence(toks, ee, zero, one)
            log_base = -0.5 * jnp.sum(ee ** 2) - 0.5 * d * math.log(2.0 * math.pi)   # log N(e; 0, I)
            logp = log_base + rescale * D                        # log q(a) = log N(e) + rescale * D(e,0,1)
            logdet = log_base - logp                             # = -rescale*D (keeps CQL logdet diagnostics)
            return a_hat, logp, log_base, logdet

        a_hat, logp, log_base, logdet = jax.vmap(endpoint_logp)(e)
        a_hat = jax.lax.stop_gradient(jnp.clip(a_hat, -1.0, 1.0))  # clip for Q eval (matches inference)
        return a_hat, jax.lax.stop_gradient(logp), jax.lax.stop_gradient(log_base), jax.lax.stop_gradient(logdet)

    # ---------- MeanFlow-F2D2 BC + divergence terms (one (t, s) draw; single code path) ----------
    def _bc_div_terms(self, model, tokens_p, action, keys) -> dict:
        cfg = self.cfg
        a = cfg.agent
        h, act_dim = cfg.horizon, cfg.act_dim
        k_e, k_ts = jax.random.split(keys[1])
        e = jax.random.normal(k_e, (h, act_dim))
        t, s = self._sample_t_s(k_ts)
        is_diag = (t == s)                                         # exact (bitwise copy in the sampler)
        x_t = (1.0 - t) * e + t * action
        v = action - e                                             # GROUND-TRUTH conditional velocity

        def adaptive(err):
            if a.adaptive_weight:
                return err / jax.lax.stop_gradient((err + a.norm_eps) ** a.norm_p)
            return err

        # L_mf (Eq 3.11): total derivative at the CURRENT end, JVP tangents (v, 1), target fully sg'd
        u_fn = lambda x, tt: model.velocity_avg(tokens_p, x, tt, s)
        u_pred, du_dt = jax.jvp(u_fn, (x_t, t), (v, jnp.ones_like(t)))
        u_tgt = jax.lax.stop_gradient((s - t) * du_dt + v)
        l_mf = adaptive(jnp.sum((u_pred - u_tgt) ** 2))

        # L_div-mf (Eq 3.12): div anchor = the STUDENT's own diagonal under sg (no teacher / no EMA); the head
        # predicts -div/rescale, so log q = log N + rescale*D matches the meanflow BC run's convention
        _, vjp_fn = jax.vjp(lambda yy: model.velocity_avg(tokens_p, yy, t, t), x_t)
        def hutch(kk):
            eps = jax.random.normal(kk, (h, act_dim))
            return jnp.sum(vjp_fn(eps)[0] * eps)
        div_t = jnp.mean(jax.vmap(hutch)(jax.random.split(keys[11], a.div_hutch_samples)))
        D_fn = lambda x, tt: model.divergence(tokens_p, x, tt, s)
        D_pred, dD_dt = jax.jvp(D_fn, (x_t, t), (v, jnp.ones_like(t)))
        D_tgt = jax.lax.stop_gradient((s - t) * dD_dt - div_t / a.divergence_rescale)   # -div/rescale anchor
        l_div = (D_pred - D_tgt) ** 2                              # RAW (adaptive weighting on L_mf only)

        fd = is_diag.astype(jnp.float32)
        return {"l_mf": l_mf, "l_div": l_div,
                "l_u_tan": fd * l_mf, "l_u_id": (1.0 - fd) * l_mf,
                "l_D_tan": fd * l_div, "l_D_id": (1.0 - fd) * l_div,
                "div_tgt_raw": fd * (-div_t), "D_pred": D_pred, "is_diag": fd}

    # ---------- full per-example skeleton (BaseAgent's, with the meanflow BC/div terms inlined) ----------
    def _per_example_loss(self, model, target, batch, key) -> dict:
        cfg = self.cfg
        h = cfg.horizon
        keys = jax.random.split(key, 12)   # same slot layout as BaseAgent

        state = batch["obs_state"]
        img = batch["obs_img"] if cfg.obs_type == "image" else None
        action = batch["action"]

        tokens_p = model.policy_tokens(img, state, keys[0])
        out = self._bc_div_terms(model, tokens_p, action, keys)

        tokens_c = model.critic_tokens(img, state, keys[3])

        # --- actor Q-max: one-step endpoint, MEAN over detached critics ---
        a_hat = self.sample_action(model, tokens_p, keys[4], keys[9])
        det_critics = _detach_critics(model)
        tokens_c_sg = jax.lax.stop_gradient(tokens_c)
        q_pi = jnp.mean(jnp.stack([c(tokens_c_sg, a_hat) for c in det_critics]))

        # --- critic TD: chunked backup, clipped double-Q (min over target critics; EMA used ONLY here) ---
        q_data = model.q(tokens_c, action)
        next_state = batch["next_obs_state"]
        next_img = batch["next_obs_img"] if cfg.obs_type == "image" else None
        on_tokens_p_next = model.policy_tokens(next_img, next_state, keys[5])
        a_next = self.sample_action(model, on_tokens_p_next, keys[6], keys[10])
        tgt_tokens_c = target.critic_tokens(next_img, next_state, keys[7])
        q_next = jnp.min(target.q(tgt_tokens_c, a_next))

        ste = batch["steps_to_end"]
        reward = jnp.where(ste < h, cfg.agent.gamma ** ste, 0.0)
        bootstrap = (cfg.agent.gamma ** h) * (1.0 - batch["done"])
        y = jax.lax.stop_gradient(reward + bootstrap * q_next)
        l_td = jnp.sum((q_data - y) ** 2)

        out.update({"q_pi": q_pi, "l_td": l_td, "q_data": jnp.mean(q_data)})
        out.update(self.extra_critic_term(model, tokens_p, on_tokens_p_next, tokens_c, q_data, keys[8]))
        return out

    # ---------- loss aggregation (BaseAgent's, with alpha*L_mf + L_div-mf; alpha outside the div term) ----------
    def make_loss(self):
        cfg = self.cfg

        def loss_fn(model, target, batch, key, alpha_prime):
            keys = jax.random.split(key, batch["action"].shape[0])
            per = jax.vmap(lambda b, k: self._per_example_loss(model, target, b, k))(batch, keys)
            l_mf = jnp.mean(per["l_mf"])                 # full L_mf batch-mean (logging only)
            l_mf_diag = jnp.mean(per["l_u_tan"])         # L_mf on t==s points (weight alpha), batch-mean
            l_mf_off = jnp.mean(per["l_u_id"])           # L_mf on t<s points (weight 1), batch-mean
            l_div = jnp.mean(per["l_div"])

            q_pi, q_data = per["q_pi"], per["q_data"]
            if cfg.agent.normalize_q_loss:
                denom = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_pi))) + 1e-6
                q_term = jnp.mean(q_pi) / denom
            else:
                q_term = jnp.mean(q_pi)
            l_td = jnp.mean(per["l_td"])

            penalty, gap, extra_metrics = self.aggregate_extra(per, alpha_prime)
            critic_loss = l_td + penalty
            actor_loss = cfg.agent.alpha * l_mf_diag + l_mf_off + l_div - q_term
            total = actor_loss + critic_loss
            n_diag = jnp.maximum(jnp.sum(per["is_diag"]), 1.0)
            n_off = jnp.maximum(jnp.sum(1.0 - per["is_diag"]), 1.0)
            metrics = {
                "loss/total": total,
                "actor/loss": actor_loss,
                "actor/BC_flow_loss": l_mf,          # the MeanFlow identity term (weight alpha)
                "actor/div_loss": l_div,
                "loss/u_tangent": jnp.sum(per["l_u_tan"]) / n_diag,
                "loss/u_identity": jnp.sum(per["l_u_id"]) / n_off,
                "loss/D_tangent": jnp.sum(per["l_D_tan"]) / n_diag,
                "loss/D_identity": jnp.sum(per["l_D_id"]) / n_off,
                # divergence/: raw UNSCALED tangent target -div(u_th(x_t,t,t)) + the head's outputs
                "divergence/target_raw": jnp.sum(per["div_tgt_raw"]) / n_diag,
                "divergence/target_raw_abs": jnp.sum(jnp.abs(per["div_tgt_raw"])) / n_diag,
                "divergence/target_scaled": jnp.sum(per["div_tgt_raw"]) / n_diag / cfg.agent.divergence_rescale,
                "divergence/D_pred_mean": jnp.mean(per["D_pred"]),
                "divergence/D_pred_absmean": jnp.mean(jnp.abs(per["D_pred"])),
                "actor/Q_loss": -q_term,
                "actor/Q": jnp.mean(q_pi),
                "actor/Q_max": jnp.max(q_pi),
                "critic/loss": critic_loss,
                "critic/td_loss": l_td,
                "critic/Q": jnp.mean(q_data),
                "critic/Q_max": jnp.max(q_data),
            }
            metrics.update(extra_metrics)
            return total, (metrics, gap)

        return loss_fn

    # ---------- sampling ----------
    def _endpoint(self, model, tokens_p, z_key) -> Array:
        """One-step flow map noise -> data: a = clip(e + u(e, 0, 1), -1, 1), single network eval."""
        cfg = self.cfg
        e = jax.random.normal(z_key, (cfg.horizon, cfg.act_dim))
        u = model.velocity_avg(tokens_p, e, jnp.asarray(0.0, jnp.float32), jnp.asarray(1.0, jnp.float32))
        return jnp.clip(e + u, -1.0, 1.0)

    def sample_action(self, model, tokens_p, z_key, eps_key) -> Array:
        return self._endpoint(model, tokens_p, z_key)

    def predict_action(self, model, tokens_p, z_key, eps_key, steps) -> Array:
        if steps == 1:
            return self._endpoint(model, tokens_p, z_key)
        e = jax.random.normal(z_key, (self.cfg.horizon, self.cfg.act_dim))
        vel_fn = lambda x, tt: model.velocity_avg(tokens_p, x, tt, tt)   # k-step Euler on the diagonal field
        return jnp.clip(flow_sample(vel_fn, e, steps = steps), -1.0, 1.0)

    def bc_loss(self, model, tokens_p, action, keys) -> dict:
        raise NotImplementedError("MeanFlowCQLAgent overrides _per_example_loss/make_loss directly")
