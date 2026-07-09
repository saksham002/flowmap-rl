"""
Staged Shortcut-Distill-F2D2 agent: ALL THREE PHASES IN ONE RUN (paper arXiv:2512.02636 §3.3; reference
Keely-Ai/JAX-F2D2 losses.py, stopgrad_type=convex; toy-validated by scratch/toy_shortcut_staged.py,
log-q oracle corr 0.997).

Losses = the paper's SHORTCUT instantiation (Sec. 3.2): one (t, s) draw per example with t == s exactly
diag_fraction (=0.5, reference) of the time; TANGENT losses on the diagonal, SEMIGROUP (midpoint two-half-jump
composition, r = (t+s)/2) losses off it. Tangent targets come from the frozen snapshot teacher (reference
diag_teacher_source="external"); semigroup compositions come from sg(student) (reference "self"):
  L_u  (t==s): || u(x_t,t,t) - sg(v_phi(x_t,t)) ||^2
       (t< s): || u(x_t,t,s) - 1/2*sg( u(x_t,t,r) + u(x_r,r,s) ) ||^2,   x_r = sg(x_t + (r-t)*u(x_t,t,r))
  L_D  (t==s): ( D(x_t,t,t) + sg(hutch_div v_phi(x_t,t)) / rescale )^2      [paper: D ~ -div/rescale]
       (t< s): ( D(x_t,t,s) - 1/2*sg( D(x_t,t,r) + D(x_r,r,s) ) )^2

distill_f2d2=False (default): NO teacher in any phase -- 2 phases (p1, p2): phase 1 trains the u map
directly (diagonal tangent target = GROUND-TRUTH v* = a - e, i.e. plain FM, + the same sg(self) midpoint
semigroup off-diagonal); phase 2 adds the D losses with the tangent anchor = the STUDENT's own diagonal
(hutch_div u_th(x_t,t,t), sg'd). TrainState.target is never snapshotted (no teacher).

distill_f2d2=True: the paper recipe below.
Phases switch at cfg.agent.phase_steps = (p1, p2, p3) boundaries inside the jitted loss (lax.switch on the
optimizer step); the frozen teacher lives in TrainState.target, which this agent SNAPSHOTS from the student
exactly at the phase boundaries instead of EMA-updating (reference: teacher.load_path = previous stage's ckpt):
  Phase 1 (steps 1..p1):        v_phi flow matching on the velocity_avg DIAGONAL (sinusoidal time embeddings,
                                no ckpt loading): u(x_t,t,t) <- v* = a - e, t ~ U[0,1].  [target unused]
  Phase 2 (p1+1..p1+p2):        L_u; teacher := snapshot(end of phase 1) = v_phi.
  Phase 3 (p1+p2+1..total):     L_u + L_D; teacher := snapshot(end of phase 2).
LR: set lr_schedule="sqrt_decay" with lr_decay_start = p1+p2 (constant 1e-4 through phases 1-2, reference
sqrt decay inside phase 3). The phase-1 checkpoint cache = <ckpt_dir>/<exp_name>/model_<p1>.eqx.

Convention = flow.py / FQL (t=0 noise, t=1 data; x_t = (1-t)*e + t*a): map Phi_{t,s}(x) = x + (s-t)*u(x,t,s)
jumps SOURCE t -> TARGET s toward data; one-step endpoint a = clip(e + u(e,0,1), -1, 1).
log q recovery (paper sign, D ~ -div/rescale): log q(a) = log N(e) + rescale * D(e, 0, 1).
Eval predictors: steps == 1 -> the one-step map endpoint (meaningful from phase 2; garbage during phase 1);
steps > 1 -> k-step Euler integration of the diagonal field (tracks v_phi quality in every phase).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array

from flow_rl.flow import sample_action as flow_sample
from flow_rl.agents.base import BaseAgent, TrainState


class ShortcutAgent(BaseAgent):

    def __init__(self, cfg):
        super().__init__(cfg)
        assert not cfg.agent.use_q, "ShortcutAgent is BC-only (use_q=False)"
        assert cfg.agent.use_divergence, "staged pipeline trains the divergence head in the final phase"
        assert cfg.total_steps == sum(cfg.agent.phase_steps), \
            f"total_steps ({cfg.total_steps}) must equal sum(phase_steps) ({sum(cfg.agent.phase_steps)})"
        n_expected = 3 if cfg.agent.distill_f2d2 else 2
        assert len(cfg.agent.phase_steps) == n_expected, \
            f"distill_f2d2={cfg.agent.distill_f2d2} needs {n_expected} phases, got {cfg.agent.phase_steps}"

    def _sample_t_s(self, key):
        """UNIFORM times ordered t <= s (t = source, s = target), fraction diag_fraction exactly diagonal
        (reference _sample_diagonal/_sample_triangle + diag_fraction = 0.5)."""
        k_t0, k_t1, k_coin = jax.random.split(key, 3)
        t0, t1 = jax.random.uniform(k_t0, ()), jax.random.uniform(k_t1, ())
        t = jnp.minimum(t0, t1).astype(jnp.float32)
        s = jnp.maximum(t0, t1).astype(jnp.float32)
        is_diag = jax.random.uniform(k_coin, ()) < self.cfg.agent.diag_fraction
        s = jnp.where(is_diag, t, s).astype(jnp.float32)
        return t, s

    def _per_example_staged(self, model, teacher, batch, key, step) -> dict:
        """One example's loss for the CURRENT phase (lax.switch on the scalar step; only that branch runs)."""
        cfg = self.cfg
        a = cfg.agent
        h, act_dim = cfg.horizon, cfg.act_dim
        keys = jax.random.split(key, 12)   # [0] tokens, [1] main draw, [11] hutch (same slots as BaseAgent)

        state = batch["obs_state"]
        img = batch["obs_img"] if cfg.obs_type == "image" else None
        action = batch["action"]
        tokens_p = model.policy_tokens(img, state, keys[0])
        t_toks = teacher.policy_tokens(img, state, keys[0])        # frozen phase-boundary teacher's own encoder
        if a.distill_f2d2:
            p1, p2, _ = a.phase_steps
            phase = (step > p1).astype(jnp.int32) + (step > p1 + p2).astype(jnp.int32)
        else:
            p1, _ = a.phase_steps
            phase = (step > p1).astype(jnp.int32)                  # 2 phases: u map -> + D head

        def v_phi(x, tt):                                          # teacher's instantaneous (diagonal) velocity
            return teacher.velocity_avg(t_toks, x, tt, tt)

        # every branch returns this exact structure (lax.switch requirement); diagnostics zero when n/a
        def _out(l_fm, l_div, l_u_tan=0.0, l_u_sc=0.0, l_D_tan=0.0, l_D_sc=0.0,
                 div_tgt_raw=0.0, D_pred=0.0, is_diag=0.0):
            return {"l_fm": l_fm, "l_div": l_div, "l_u_tan": l_u_tan, "l_u_sc": l_u_sc,
                    "l_D_tan": l_D_tan, "l_D_sc": l_D_sc, "div_tgt_raw": div_tgt_raw,
                    "D_pred": D_pred, "is_diag": is_diag}

        def fm_loss(ks):                                           # phase 1: train v_phi itself (plain FM)
            k_e, k_t = jax.random.split(ks[1])
            e = jax.random.normal(k_e, (h, act_dim))
            t = jax.random.uniform(k_t, ()).astype(jnp.float32)
            x_t = (1.0 - t) * e + t * action                       # interpolant (t=0 noise convention)
            l = jnp.mean((model.velocity_avg(tokens_p, x_t, t, t) - (action - e)) ** 2)
            return _out(l, 0.0, l_u_tan = l, is_diag = 1.0)

        def _draw_and_u_terms(ks):
            """Shared phase-2/3 draw + the paper's tangent/semigroup velocity loss (one (t,s), 50% diag)."""
            k_e, k_ts = jax.random.split(ks[1])
            e = jax.random.normal(k_e, (h, act_dim))
            t, s = self._sample_t_s(k_ts)
            r = 0.5 * (t + s)                                      # MIDPOINT (paper r <- (t+s)/2)
            is_diag = (t == s)                                     # exact (bitwise copy in the sampler)
            x_t = (1.0 - t) * e + t * action
            u_pred = model.velocity_avg(tokens_p, x_t, t, s)       # t==s -> the diagonal prediction
            # semigroup composition target: mean of the two sg(SELF) half-jumps (reference "self" offdiag)
            u_tr = model.velocity_avg(tokens_p, x_t, t, r)
            x_r = jax.lax.stop_gradient(x_t + (r - t) * u_tr)      # midpoint of the (sg'd) student map
            u_rs = model.velocity_avg(tokens_p, x_r, r, s)
            u_sc_tgt = jax.lax.stop_gradient(0.5 * (u_tr + u_rs))
            if a.distill_f2d2:
                u_tan_tgt = jax.lax.stop_gradient(v_phi(x_t, t))   # tangent target: frozen snapshot teacher
            else:
                u_tan_tgt = action - e                             # tangent target: GROUND-TRUTH v* (plain FM)
            l_u = jnp.sum((u_pred - jnp.where(is_diag, u_tan_tgt, u_sc_tgt)) ** 2)
            return l_u, x_t, x_r, t, s, r, is_diag

        def distill_loss(ks):                                      # phase 2: tangent + semigroup (velocity)
            l_u, x_t, x_r, t, s, r, is_diag = _draw_and_u_terms(ks)
            fd = is_diag.astype(jnp.float32)
            return _out(l_u, 0.0, l_u_tan = fd * l_u, l_u_sc = (1.0 - fd) * l_u, is_diag = fd)

        def f2d2_loss(ks):                                         # final phase: + the same split for D
            l_u, x_t, x_r, t, s, r, is_diag = _draw_and_u_terms(ks)
            D_pred = model.divergence(tokens_p, x_t, t, s)
            # tangent: D(x,t,t) <- -sg(hutch_div u(x_t,t))/rescale  (paper: || D + div(u) ||^2); div field =
            # frozen teacher v_phi when distill_f2d2, else the STUDENT's own diagonal (sg'd either way)
            div_field = v_phi if a.distill_f2d2 else (lambda yy, tt: model.velocity_avg(tokens_p, yy, tt, tt))
            _, vjp_fn = jax.vjp(lambda yy: div_field(yy, t), x_t)
            def hutch(kk):
                eps = jax.random.normal(kk, (h, act_dim))
                return jnp.sum(vjp_fn(eps)[0] * eps)
            div_t = jnp.mean(jax.vmap(hutch)(jax.random.split(ks[11], a.div_hutch_samples)))
            D_tan_tgt = -jax.lax.stop_gradient(div_t) / a.divergence_rescale
            # semigroup: mean of the two sg(SELF) half-jump D's along the same midpoint
            D_tr = model.divergence(tokens_p, x_t, t, r)
            D_rs = model.divergence(tokens_p, x_r, r, s)
            D_sc_tgt = jax.lax.stop_gradient(0.5 * (D_tr + D_rs))
            l_D = (D_pred - jnp.where(is_diag, D_tan_tgt, D_sc_tgt)) ** 2
            fd = is_diag.astype(jnp.float32)
            return _out(l_u, l_D, l_u_tan = fd * l_u, l_u_sc = (1.0 - fd) * l_u,
                        l_D_tan = fd * l_D, l_D_sc = (1.0 - fd) * l_D,
                        div_tgt_raw = fd * (-div_t), D_pred = D_pred, is_diag = fd)

        branches = [fm_loss, distill_loss, f2d2_loss] if a.distill_f2d2 else [distill_loss, f2d2_loss]
        return jax.lax.switch(phase, branches, keys)

    # ---------- self-contained staged loss + train step (base framework untouched) ----------
    def make_loss(self):
        cfg = self.cfg
        ps = cfg.agent.phase_steps
        bounds = [sum(ps[:i + 1]) for i in range(len(ps) - 1)]     # phase-switch boundaries (2 or 1 of them)

        def loss_fn(model, teacher, batch, key, step):
            keys = jax.random.split(key, batch["action"].shape[0])
            per = jax.vmap(lambda b, k: self._per_example_staged(model, teacher, b, k, step))(batch, keys)
            l_fm = jnp.mean(per["l_fm"])
            l_div = jnp.mean(per["l_div"])
            total = l_fm + cfg.agent.div_coef * l_div
            # conditional means over the tangent (t==s) / semigroup (t<s) sub-batches for debugging
            n_diag = jnp.maximum(jnp.sum(per["is_diag"]), 1.0)
            n_off = jnp.maximum(jnp.sum(1.0 - per["is_diag"]), 1.0)
            metrics = {
                "loss/total": total,
                "actor/BC_flow_loss": l_fm,                # phase 1: FM; phases 2-3: tangent+semigroup velocity loss
                "actor/div_loss": l_div,                   # nonzero only in phase 3
                "loss/u_tangent": jnp.sum(per["l_u_tan"]) / n_diag,
                "loss/u_semigroup": jnp.sum(per["l_u_sc"]) / n_off,
                "loss/D_tangent": jnp.sum(per["l_D_tan"]) / n_diag,
                "loss/D_semigroup": jnp.sum(per["l_D_sc"]) / n_off,
                # divergence/: raw UNSCALED tangent target -div(u_teacher(x_t,t,t)) + the head's outputs
                "divergence/target_raw": jnp.sum(per["div_tgt_raw"]) / n_diag,
                "divergence/target_raw_abs": jnp.sum(jnp.abs(per["div_tgt_raw"])) / n_diag,
                "divergence/target_scaled": jnp.sum(per["div_tgt_raw"]) / n_diag / cfg.agent.divergence_rescale,
                "divergence/D_pred_mean": jnp.mean(per["D_pred"]),
                "divergence/D_pred_absmean": jnp.mean(jnp.abs(per["D_pred"])),
                "train/phase": 1 + sum((step > b).astype(jnp.int32) for b in bounds),
            }
            return total, metrics

        return loss_fn

    def make_train_step(self):
        cfg = self.cfg
        optimizer = self.optimizer
        loss_fn = self.make_loss()
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux = True)
        ps = cfg.agent.phase_steps
        bounds = [sum(ps[:i + 1]) for i in range(len(ps) - 1)]

        @eqx.filter_jit
        def train_step(state: TrainState, batch, key):
            # 1-based current step from the optimizer's update count (adam + schedule counts are identical)
            step = optax.tree_utils.tree_get_all_with_path(state.opt_state, "count")[0][1] + 1
            (_, metrics), grads = grad_fn(state.model, state.target, batch, key, step)
            params = eqx.filter(state.model, eqx.is_inexact_array)
            gnorm = optax.global_norm(grads)                 # pre-clip gradient norm
            updates, opt_state = optimizer.update(grads, state.opt_state, params)
            model = eqx.apply_updates(state.model, updates)

            # teacher = phase-boundary SNAPSHOT of the student (NOT an EMA): freeze v_phi at the end of
            # phase 1, the distilled map at the end of phase 2 (reference teacher.load_path = prior stage).
            # distill_f2d2=False: the teacher is never consulted -- snapshots disabled.
            if cfg.agent.distill_f2d2:
                snap = jnp.logical_or(step == bounds[0], step == bounds[1])
            else:
                snap = jnp.asarray(False)
            m_arr = eqx.filter(model, eqx.is_inexact_array)
            t_arr = eqx.filter(state.target, eqx.is_inexact_array)
            new_t = jax.tree_util.tree_map(lambda m, t: jnp.where(snap, m, t), m_arr, t_arr)
            target = eqx.combine(new_t, state.target)

            pnorm = optax.global_norm(m_arr)
            metrics = {**metrics, "params/grad_norm": gnorm, "params/param_norm": pnorm}
            return TrainState(model = model, target = target, opt_state = opt_state,
                              log_alpha_prime = state.log_alpha_prime,
                              dual_opt_state = state.dual_opt_state), metrics

        return train_step

    # ---------- sampling ----------
    def _endpoint(self, model, tokens_p, z_key) -> Array:
        """One-step flow map noise -> data: a = clip(e + u(e, s=0, t=1), -1, 1), single network eval."""
        cfg = self.cfg
        e = jax.random.normal(z_key, (cfg.horizon, cfg.act_dim))
        u = model.velocity_avg(tokens_p, e, jnp.asarray(0.0, jnp.float32), jnp.asarray(1.0, jnp.float32))
        return jnp.clip(e + u, -1.0, 1.0)

    def _euler_diag(self, model, tokens_p, z_key, steps) -> Array:
        """k-step Euler integration of the diagonal field u(x, t, t) (tracks v_phi in every phase)."""
        cfg = self.cfg
        e = jax.random.normal(z_key, (cfg.horizon, cfg.act_dim))
        vel_fn = lambda x, tt: model.velocity_avg(tokens_p, x, tt, tt)
        return jnp.clip(flow_sample(vel_fn, e, steps = steps), -1.0, 1.0)

    def sample_action(self, model, tokens_p, z_key, eps_key) -> Array:
        return self._endpoint(model, tokens_p, z_key)

    def predict_action(self, model, tokens_p, z_key, eps_key, steps) -> Array:
        if steps == 1:                                             # the one-step map (the deliverable)
            return self._endpoint(model, tokens_p, z_key)
        return self._euler_diag(model, tokens_p, z_key, steps)     # k-step diagonal Euler (v_phi quality)

    def bc_loss(self, model, tokens_p, action, keys) -> dict:
        raise NotImplementedError("ShortcutAgent uses its own staged make_loss/make_train_step")
