"""
Shortcut-CQL agent: the shortcut-model policy (paper arXiv:2512.02636 Sec. 3.2 instantiation) + double-Q
critic + CQL conservative penalty, with the F2D2 divergence head supplying log q for the importance-sampled
logsumexp. SINGLE STAGE -- requires a BC-trained checkpoint (the staged shortcut run's final model) as both
the warm start (cfg.init_ckpt) and the frozen VM-SC teacher (cfg.agent.teacher_ckpt).

Per example, ONE (t, s) draw with t == s exactly diag_fraction (=0.5, reference) of the time; midpoint
r = (t+s)/2. Convention = flow.py (t=0 noise, t=1 data); endpoint a = clip(e + u(e,0,1), -1, 1);
log q(a) = log N(e) + rescale * D(e,0,1)   [paper sign: D ~ -div/rescale].

  L_VMSC (t==s):  || u_th(x_t,t,t) - sg( v_phi(x_t,t) ) ||^2            [FROZEN BC teacher; weight alpha]
                  distill_f2d2=False (default): target = GROUND-TRUTH v* = a - e instead of v_phi (no teacher
                  anywhere; teacher_ckpt unused). init_ckpt (a u + D BC checkpoint) is required either way.
  L_u-sc (t< s):  || u_th(x_t,t,s) - 1/2*sg( u_th(x_t,t,r) + u_th(x_r,r,s) ) ||^2          [weight 1]
  L_div  (t==s):  ( D_th(x_t,t,t) + sg(hutch_div u_th(x_t,t)) / rescale )^2                [weight 1]
         (t< s):  ( D_th(x_t,t,s) - 1/2*sg( D_th(x_t,t,r) + D_th(x_r,r,s) ) )^2
with x_r = sg(x_t + (r-t)*u_th(x_t,t,r)). NO actor-side target networks (matches the reference, which has
none): compositions and the div anchor use sg(STUDENT) (the reference's "self" source); v_phi (the FROZEN
BC teacher checkpoint, the reference's external diagonal teacher) appears ONLY in the tangent VM-SC term;
the EMA target network is used ONLY for the critic TD backup. D is trained only by L_div; the CQL proposals
and their log q are stop-gradiented, so the penalty never trains the policy map or D.

  total = alpha*L_VMSC + L_u-sc - E[mean_i Q_i(s, a_hat)] + L_div + L_TD + L_CQL
  dual:  max_{alpha'>=0} alpha' * (mean gap - cql_budget)                (CQLMixin, unchanged)
"""

import math
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array

from flow_rl.flow import sample_action as flow_sample
from flow_rl.agents.base import BaseAgent, make_model, _detach_critics
from flow_rl.agents.cql import CQLMixin


class ShortcutCQLAgent(CQLMixin, BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.agent.use_q, "ShortcutCQLAgent is the RL stage (use_q=True)"
        assert cfg.agent.use_divergence, "CQL log q comes from the divergence head"
        assert cfg.init_ckpt, "init_ckpt (a u + D BC checkpoint) is required as the warm start"
        if cfg.agent.distill_f2d2:
            assert cfg.agent.teacher_ckpt, ("distill_f2d2=True: teacher_ckpt (the staged BC run's final "
                                            "checkpoint) is required; pass init_ckpt=<same> for the warm start")
            # frozen VM-SC teacher v_phi = the BC checkpoint's diagonal (div head not needed on the teacher
            # side, but the ckpt contains one -- load with a div-full template so the leaves line up)
            tmpl = make_model(cfg, jax.random.PRNGKey(0), use_divergence = True)
            self.teacher = eqx.tree_deserialise_leaves(cfg.agent.teacher_ckpt, tmpl)
        else:
            self.teacher = None                                    # VM-SC target = ground-truth v* = a - e

    def _sample_t_s(self, key):
        """UNIFORM times ordered t <= s, fraction diag_fraction exactly diagonal (reference sampling)."""
        k_t0, k_t1, k_coin = jax.random.split(key, 3)
        t0, t1 = jax.random.uniform(k_t0, ()), jax.random.uniform(k_t1, ())
        t = jnp.minimum(t0, t1).astype(jnp.float32)
        s = jnp.maximum(t0, t1).astype(jnp.float32)
        is_diag = jax.random.uniform(k_coin, ()) < self.cfg.agent.diag_fraction
        s = jnp.where(is_diag, t, s).astype(jnp.float32)
        return t, s

    # ---------- CQL proposal block: one-step endpoint + divergence-head log q ----------
    def _policy_logp_block(self, model, toks, key):
        """N shortcut-endpoint proposals + F2D2 divergence-head log q. Returns (a, logp, log_base, logdet),
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

    # ---------- BC + divergence terms (one (t,s) draw; tangent vs semigroup split) ----------
    def _bc_div_terms(self, model, tokens_p, t_toks, action, keys) -> dict:
        cfg = self.cfg
        a = cfg.agent
        h, act_dim = cfg.horizon, cfg.act_dim
        k_e, k_ts = jax.random.split(keys[1])
        e = jax.random.normal(k_e, (h, act_dim))
        t, s = self._sample_t_s(k_ts)
        r = 0.5 * (t + s)                                          # MIDPOINT (paper r <- (t+s)/2)
        is_diag = (t == s)                                         # exact (bitwise copy in the sampler)
        x_t = (1.0 - t) * e + t * action

        # velocity: tangent (frozen v_phi) vs semigroup (sg(STUDENT) compositions -- reference "self")
        u_pred = model.velocity_avg(tokens_p, x_t, t, s)
        u_tr = model.velocity_avg(tokens_p, x_t, t, r)
        x_r = jax.lax.stop_gradient(x_t + (r - t) * u_tr)          # midpoint of the (sg'd) student map
        u_rs = model.velocity_avg(tokens_p, x_r, r, s)
        u_sc_tgt = jax.lax.stop_gradient(0.5 * (u_tr + u_rs))
        if a.distill_f2d2:
            u_tan_tgt = jax.lax.stop_gradient(self.teacher.velocity_avg(t_toks, x_t, t, t))   # v_phi(x_t, t)
        else:
            u_tan_tgt = action - e                                 # GROUND-TRUTH v* (plain FM tangent)
        err_u = jnp.sum((u_pred - jnp.where(is_diag, u_tan_tgt, u_sc_tgt)) ** 2)
        l_vmsc = jnp.where(is_diag, err_u, 0.0)                    # weighted alpha in make_loss
        l_usc = jnp.where(is_diag, 0.0, err_u)                     # weight 1

        # divergence head: tangent -sg(div u_th)/rescale vs semigroup sg(STUDENT)-D compositions
        D_pred = model.divergence(tokens_p, x_t, t, s)
        _, vjp_fn = jax.vjp(lambda yy: model.velocity_avg(tokens_p, yy, t, t), x_t)
        def hutch(kk):
            eps = jax.random.normal(kk, (h, act_dim))
            return jnp.sum(vjp_fn(eps)[0] * eps)
        div_t = jnp.mean(jax.vmap(hutch)(jax.random.split(keys[11], a.div_hutch_samples)))
        D_tan_tgt = -jax.lax.stop_gradient(div_t) / a.divergence_rescale
        D_tr = model.divergence(tokens_p, x_t, t, r)
        D_rs = model.divergence(tokens_p, x_r, r, s)
        D_sc_tgt = jax.lax.stop_gradient(0.5 * (D_tr + D_rs))
        l_div = (D_pred - jnp.where(is_diag, D_tan_tgt, D_sc_tgt)) ** 2
        fd = is_diag.astype(jnp.float32)
        return {"l_vmsc": l_vmsc, "l_usc": l_usc, "l_div": l_div,
                "l_D_tan": fd * l_div, "l_D_sc": (1.0 - fd) * l_div,
                "div_tgt_raw": fd * (-div_t), "D_pred": D_pred, "is_diag": fd}

    # ---------- full per-example skeleton (BaseAgent's, with the shortcut BC/div terms inlined) ----------
    def _per_example_loss(self, model, target, batch, key) -> dict:
        cfg = self.cfg
        h = cfg.horizon
        keys = jax.random.split(key, 12)   # same slot layout as BaseAgent

        state = batch["obs_state"]
        img = batch["obs_img"] if cfg.obs_type == "image" else None
        action = batch["action"]

        tokens_p = model.policy_tokens(img, state, keys[0])
        t_toks = (self.teacher.policy_tokens(img, state, keys[0])  # frozen v_phi's own encoder (tangent only)
                  if cfg.agent.distill_f2d2 else None)

        out = self._bc_div_terms(model, tokens_p, t_toks, action, keys)

        tokens_c = model.critic_tokens(img, state, keys[3])

        # --- actor Q-max: one-step endpoint, MEAN over detached critics ---
        a_hat = self.sample_action(model, tokens_p, keys[4], keys[9])
        det_critics = _detach_critics(model)
        tokens_c_sg = jax.lax.stop_gradient(tokens_c)
        q_pi = jnp.mean(jnp.stack([c(tokens_c_sg, a_hat) for c in det_critics]))

        # --- critic TD: chunked backup, clipped double-Q (min over target critics) ---
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

    # ---------- loss aggregation (BaseAgent's, with alpha*L_VMSC + L_u-sc split) ----------
    def make_loss(self):
        cfg = self.cfg

        def loss_fn(model, target, batch, key, alpha_prime):
            keys = jax.random.split(key, batch["action"].shape[0])
            per = jax.vmap(lambda b, k: self._per_example_loss(model, target, b, k))(batch, keys)
            l_vmsc = jnp.mean(per["l_vmsc"])
            l_usc = jnp.mean(per["l_usc"])
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
            actor_loss = cfg.agent.alpha * l_vmsc + l_usc - q_term + l_div
            total = actor_loss + critic_loss
            # conditional means over the tangent (t==s) / semigroup (t<s) sub-batches for debugging
            n_diag = jnp.maximum(jnp.sum(per["is_diag"]), 1.0)
            n_off = jnp.maximum(jnp.sum(1.0 - per["is_diag"]), 1.0)
            metrics = {
                "loss/total": total,
                "actor/loss": actor_loss,
                "actor/BC_flow_loss": l_vmsc,        # tangent VM-SC distill term (weight alpha)
                "actor/usc_loss": l_usc,             # semigroup self-consistency term (weight 1)
                "actor/div_loss": l_div,
                "loss/u_tangent": jnp.sum(per["l_vmsc"]) / n_diag,
                "loss/u_semigroup": jnp.sum(per["l_usc"]) / n_off,
                "loss/D_tangent": jnp.sum(per["l_D_tan"]) / n_diag,
                "loss/D_semigroup": jnp.sum(per["l_D_sc"]) / n_off,
                # divergence/: raw UNSCALED tangent target -div(u_EMA(x_t,t,t)) + the head's outputs
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
        raise NotImplementedError("ShortcutCQLAgent overrides _per_example_loss/make_loss directly")
