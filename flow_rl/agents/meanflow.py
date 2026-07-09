"""
Staged MeanFlow-F2D2 BC agent: the paper's MeanFlow instantiation (arXiv:2512.02636 Sec. 3.2, Eq 3.11-3.13),
BOTH PHASES IN ONE RUN (mirrors agents/shortcut.py's single-run machinery). Convention = flow.py (t=0 noise,
t=1 data; x_t = (1-t)*e + t*a); current time t <= target time s; GROUND-TRUTH tangent v = a - e.

  L_mf     = || u_th(x_t,t,s) - sg( (s-t)*du/dt + v ) ||^2          du/dt = JVP tangents (v, 1) on (x_t, t)
  L_div-mf = ( D_th(x_t,t,s) - sg( (s-t)*dD/dt - hutch_div(u_th(x_t,t,t))/rescale ) )^2   [same tangents]

Both targets are FULLY stop-gradiented (noise on the target side, like the validated OFQL u loss); the
divergence anchor is the STUDENT's own diagonal under sg (Eq 3.12; no teacher / no actor-side EMA -- the
identity at t==s reduces to flow matching and D = -div). No semigroup/midpoint terms: the MeanFlow
identity covers self-consistency, so this is 2 losses vs the shortcut agent's 4.

self_tangent=True: in phase 2 ONLY, every occurrence of the stochastic v (JVP tangents of BOTH losses and
the additive u-target term) is replaced by u_EMA(x_t,t,t) -- the tau-EMA of the student (TrainState.target,
updated every step across BOTH phases) -- deterministic targets; on the diagonal L_mf regresses the student
to the EMA diagonal (slow-moving anchor) instead of vanishing against itself.

frozen_teacher=True (mf14): same phase-2 tangent substitution, but the source u_phi = TrainState.target is a
SNAPSHOT of the student frozen exactly at step p1 (shortcut.py's boundary-snapshot machinery, no EMA), and
the Hutchinson divergence anchor is ALSO taken from u_phi (shortcut-recipe-faithful: deterministic AND
data-anchored, vs self_tangent's tracking EMA whose fully self-referential phase 2 ran away -- mf13).

Phases (lax.switch on the optimizer step; phase_steps = (p1, p2)):
  Phase 1 (1..p1):      L_mf only (data-anchored, trains from scratch).
  Phase 2 (p1+1..end):  L_mf + div_coef * L_div-mf (warm start by continuation; D head starts training
                        once u is competent -- the paper's "teacher availability" staging, nothing more).
Times: logit-normal pair ordered t <= s (MeanFlow convention; t_mean sign FLIPPED vs the reversed-convention
legacy reversed-convention agent since time is relabeled), fraction flow_ratio exactly diagonal t == s.
log q recovery: log q(a) = log N(e) + rescale * D(e,0,1) (D predicts -avg-div/divergence_rescale
-- adaptive L2 weighting (norm_eps, norm_p) applies to L_mf ONLY; L_div-mf is raw);  endpoint a = clip(e + u(e,0,1), -1, 1).
Eval predictors: steps == 1 -> one-step endpoint; steps > 1 -> k-step Euler on the diagonal field.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array

from flow_rl.flow import sample_action as flow_sample
from flow_rl.agents.base import BaseAgent, TrainState, _ema_update


class MeanFlowAgent(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert not cfg.agent.use_q, "MeanFlowAgent is BC-only (use_q=False)"
        assert cfg.agent.use_divergence, "phase 2 trains the divergence head"
        assert cfg.total_steps == sum(cfg.agent.phase_steps), \
            f"total_steps ({cfg.total_steps}) must equal sum(phase_steps) ({sum(cfg.agent.phase_steps)})"
        assert not (cfg.agent.self_tangent and cfg.agent.frozen_teacher), \
            "self_tangent (EMA) and frozen_teacher (phase-1 snapshot) are mutually exclusive"

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

    def _mf_terms(self, model, target, tokens_p, t_toks, action, keys, with_div):
        """One example's L_mf (+ optional L_div-mf) on a single (t, s) draw. Single code path: at t == s
        the sg'd targets reduce to v (flow matching) and -div (tangent condition) automatically."""
        cfg = self.cfg
        a = cfg.agent
        h, act_dim = cfg.horizon, cfg.act_dim
        k_e, k_ts = jax.random.split(keys[1])
        e = jax.random.normal(k_e, (h, act_dim))
        t, s = self._sample_t_s(k_ts)
        is_diag = (t == s)                                         # exact (bitwise copy in the sampler)
        x_t = (1.0 - t) * e + t * action
        v = action - e                                             # GROUND-TRUTH conditional velocity
        if with_div and (a.self_tangent or a.frozen_teacher):      # phase 2: v -> EMA / frozen-snapshot diagonal
            v = target.velocity_avg(t_toks, x_t, t, t)             # no grads (target isn't differentiated)

        # L_mf: total derivative along the trajectory at the CURRENT end (JVP tangents (v, 1); s closed over)
        u_fn = lambda x, tt: model.velocity_avg(tokens_p, x, tt, s)
        u_pred, du_dt = jax.jvp(u_fn, (x_t, t), (v, jnp.ones_like(t)))
        u_tgt = jax.lax.stop_gradient((s - t) * du_dt + v)         # sg the WHOLE target (Eq 3.11)
        err_u = jnp.sum((u_pred - u_tgt) ** 2)
        if a.adaptive_weight:                                      # MeanFlow adaptive L2 (validated OFQL recipe)
            l_u = err_u / jax.lax.stop_gradient((err_u + a.norm_eps) ** a.norm_p)
        else:
            l_u = err_u
        fd = is_diag.astype(jnp.float32)
        out = {"l_fm": l_u, "l_div": 0.0, "l_u_tan": fd * l_u, "l_u_id": (1.0 - fd) * l_u,
               "l_D_tan": 0.0, "l_D_id": 0.0, "div_tgt_raw": 0.0, "D_pred": 0.0, "is_diag": fd}
        if not with_div:
            return out

        # L_div-mf: div anchor = the STUDENT's own diagonal under sg (Eq 3.12; no teacher / no EMA),
        # or the FROZEN phase-1 snapshot u_phi when frozen_teacher (shortcut-recipe anchors)
        anchor_net, anchor_toks = (target, t_toks) if a.frozen_teacher else (model, tokens_p)
        _, vjp_fn = jax.vjp(lambda yy: anchor_net.velocity_avg(anchor_toks, yy, t, t), x_t)
        def hutch(kk):
            eps = jax.random.normal(kk, (h, act_dim))
            return jnp.sum(vjp_fn(eps)[0] * eps)
        div_t = jnp.mean(jax.vmap(hutch)(jax.random.split(keys[11], a.div_hutch_samples)))
        D_fn = lambda x, tt: model.divergence(tokens_p, x, tt, s)
        D_pred, dD_dt = jax.jvp(D_fn, (x_t, t), (v, jnp.ones_like(t)))
        D_tgt = jax.lax.stop_gradient((s - t) * dD_dt - div_t / a.divergence_rescale)   # -div/rescale anchor
        l_D = (D_pred - D_tgt) ** 2                                # RAW squared loss (adaptive weighting is
                                                                   # applied to L_mf ONLY, per user direction)
        out.update({"l_div": l_D, "l_D_tan": fd * l_D, "l_D_id": (1.0 - fd) * l_D,
                    "div_tgt_raw": fd * (-div_t), "D_pred": D_pred})
        return out

    def _per_example_staged(self, model, target, batch, key, step) -> dict:
        cfg = self.cfg
        keys = jax.random.split(key, 12)   # [0] tokens, [1] draw, [11] hutch (same slots as BaseAgent)
        state = batch["obs_state"]
        img = batch["obs_img"] if cfg.obs_type == "image" else None
        action = batch["action"]
        tokens_p = model.policy_tokens(img, state, keys[0])
        t_toks = target.policy_tokens(img, state, keys[0])        # EMA model's own encoder (self_tangent)
        p1, _ = cfg.agent.phase_steps
        phase = (step > p1).astype(jnp.int32)
        return jax.lax.switch(phase,
                              [lambda ks: self._mf_terms(model, target, tokens_p, t_toks, action, ks, with_div = False),
                               lambda ks: self._mf_terms(model, target, tokens_p, t_toks, action, ks, with_div = True)],
                              keys)

    # ---------- self-contained staged loss + train step (base framework untouched) ----------
    def make_loss(self):
        cfg = self.cfg
        p1, _ = cfg.agent.phase_steps

        def loss_fn(model, target, batch, key, step):
            keys = jax.random.split(key, batch["action"].shape[0])
            per = jax.vmap(lambda b, k: self._per_example_staged(model, target, b, k, step))(batch, keys)
            l_fm = jnp.mean(per["l_fm"])
            l_div = jnp.mean(per["l_div"])
            total = l_fm + cfg.agent.div_coef * l_div
            n_diag = jnp.maximum(jnp.sum(per["is_diag"]), 1.0)
            n_off = jnp.maximum(jnp.sum(1.0 - per["is_diag"]), 1.0)
            metrics = {
                "loss/total": total,
                "actor/BC_flow_loss": l_fm,
                "actor/div_loss": l_div,                   # nonzero only in phase 2
                "loss/u_tangent": jnp.sum(per["l_u_tan"]) / n_diag,
                "loss/u_identity": jnp.sum(per["l_u_id"]) / n_off,
                "loss/D_tangent": jnp.sum(per["l_D_tan"]) / n_diag,
                "loss/D_identity": jnp.sum(per["l_D_id"]) / n_off,
                # divergence/: raw UNSCALED tangent target -div(u_th(x_t,t,t)) + the head's outputs
                "divergence/target_raw": jnp.sum(per["div_tgt_raw"]) / n_diag,
                "divergence/target_raw_abs": jnp.sum(jnp.abs(per["div_tgt_raw"])) / n_diag,
                "divergence/D_pred_mean": jnp.mean(per["D_pred"]),
                "divergence/D_pred_absmean": jnp.mean(jnp.abs(per["D_pred"])),
                "train/phase": 1 + (step > p1),
            }
            return total, metrics

        return loss_fn

    def make_train_step(self):
        cfg = self.cfg
        optimizer = self.optimizer
        loss_fn = self.make_loss()
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux = True)
        p1, _ = cfg.agent.phase_steps

        @eqx.filter_jit
        def train_step(state: TrainState, batch, key):
            # 1-based current step from the optimizer's update count (adam + schedule counts are identical)
            step = optax.tree_utils.tree_get_all_with_path(state.opt_state, "count")[0][1] + 1
            (_, metrics), grads = grad_fn(state.model, state.target, batch, key, step)
            params = eqx.filter(state.model, eqx.is_inexact_array)
            gnorm = optax.global_norm(grads)                 # pre-clip gradient norm
            updates, opt_state = optimizer.update(grads, state.opt_state, params)
            model = eqx.apply_updates(state.model, updates)
            if cfg.agent.frozen_teacher:
                # teacher u_phi = SNAPSHOT of the student frozen at the end of phase 1 (shortcut.py machinery)
                snap = (step == p1)
                m_arr = eqx.filter(model, eqx.is_inexact_array)
                t_arr = eqx.filter(state.target, eqx.is_inexact_array)
                new_t = jax.tree_util.tree_map(lambda m, t: jnp.where(snap, m, t), m_arr, t_arr)
                target = eqx.combine(new_t, state.target)
            else:
                target = _ema_update(state.target, model, cfg.agent.tau)   # phase-2 tangent source when self_tangent
            pnorm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
            metrics = {**metrics, "params/grad_norm": gnorm, "params/param_norm": pnorm}
            return TrainState(model = model, target = target, opt_state = opt_state,
                              log_alpha_prime = state.log_alpha_prime,
                              dual_opt_state = state.dual_opt_state), metrics

        return train_step

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
        raise NotImplementedError("MeanflowStagedAgent uses its own staged make_loss/make_train_step")
