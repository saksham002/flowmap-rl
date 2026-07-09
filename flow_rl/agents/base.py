"""
Shared agent machinery: model/optimizer/EMA + the per-example TD+actor skeleton,
the jitted train_step / predict / q_fn factories.

An Agent is a PLAIN python object holding only static config (`cfg`, `optimizer`,
`dual_opt`); every array lives in `TrainState`, which is the only explicit jit
argument. The `make_*` methods return `eqx.filter_jit` closures that capture
`self` (static) -- bound methods are never jitted directly.

Objective-specific behaviour is injected through overridable hooks:
  - bc_loss(model, tokens_p, action, keys)          BC term (flow / regression / MeanFlow)
  - sample_action(model, tokens_p, z_key, eps_key)  one-step actor sample (actor -Q AND TD next-action)
  - predict_action(model, tokens_p, z_key, eps_key, steps)  eval-time action
  - extra_critic_term(model, tp, tp_next, tc, q_data, key) -> dict   (CQL adds the conservative term)
  - aggregate_extra(per, alpha_prime) -> (penalty, gap, metrics)     (CQL builds the penalty + diagnostics)
  - dual_step(log_alpha_prime, dual_opt_state, gap) -> (log_alpha_prime, dual_opt_state)  (CQL dual ascent)
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import Array, PRNGKeyArray

from flow_rl.models.flow_rl_model import FlowRLModel


class TrainState(eqx.Module):
    model: FlowRLModel
    target: FlowRLModel
    opt_state: optax.OptState
    log_alpha_prime: Array          # CQL Lagrange multiplier: alpha' = exp(log_alpha') >= 0
    dual_opt_state: optax.OptState   # optimizer state for the dual ascent on log_alpha'


def count_params(model: eqx.Module) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))


def make_model(cfg, key: PRNGKeyArray, use_divergence: bool | None = None) -> FlowRLModel:
    if use_divergence is None:
        use_divergence = getattr(cfg.agent, "use_divergence", False)
    return FlowRLModel(
        variant = cfg.variant,
        state_dim = cfg.state_dim,
        encoder_num = cfg.encoder_num,
        hidden_size = cfg.hidden_size,
        img_size = cfg.img_size,
        rgb_encoder_model = cfg.rgb_encoder_model,
        pretrained = cfg.pretrained_resnet,
        horizon = cfg.horizon,
        act_dim = cfg.act_dim,
        policy_depth = cfg.policy_depth,
        critic_depth = cfg.critic_depth,
        num_critics = cfg.num_critics,
        heads = cfg.heads,
        ff_mult = cfg.ff_mult,
        use_image = cfg.obs_type == "image",
        critic_sigmoid = cfg.critic_sigmoid,
        use_divergence = use_divergence,
        key = key,
    )


def load_model_weights(cfg, model: FlowRLModel, path: str, key: PRNGKeyArray) -> FlowRLModel:
    """Deserialize a checkpoint into `model`. If `model` has a divergence head but the checkpoint predates it
    (e.g. warm-starting from a div-less bc_flow ckpt), load into a div-less template and graft every
    non-div component over, keeping `model`'s fresh div head params."""
    try:
        return eqx.tree_deserialise_leaves(path, model)            # matching structure (incl. div-full ckpts)
    except Exception:
        pass
    if getattr(cfg.agent, "use_divergence", False):
        tmpl = make_model(cfg, key, use_divergence = False)
        loaded = eqx.tree_deserialise_leaves(path, tmpl)
        return eqx.tree_at(
            lambda m: (m.policy_encoder, m.critic_encoder, m.critics,
                       m.policy.time_embed, m.policy.mf_time_embed, m.policy.action_in_proj,
                       m.policy.action_query, m.policy.action_pos_embed, m.policy.type_embed,
                       m.policy.blocks, m.policy.final_ln, m.policy.head, m.policy.log_std_head,
                       m.policy.r_type_embed),
            model,
            replace = (loaded.policy_encoder, loaded.critic_encoder, loaded.critics,
                       loaded.policy.time_embed, loaded.policy.mf_time_embed, loaded.policy.action_in_proj,
                       loaded.policy.action_query, loaded.policy.action_pos_embed, loaded.policy.type_embed,
                       loaded.policy.blocks, loaded.policy.final_ln, loaded.policy.head, loaded.policy.log_std_head,
                       loaded.policy.r_type_embed),
            is_leaf = lambda x: x is None,
        )
    raise ValueError(f"checkpoint '{path}' does not match the model structure")


def make_lr_schedule(cfg) -> optax.Schedule:
    """Configurable LR schedule. Default: linear warmup over warmup_steps -> constant."""
    lr, warmup = cfg.learning_rate, cfg.warmup_steps
    if cfg.lr_schedule == "constant":
        return optax.constant_schedule(lr)
    if cfg.lr_schedule == "warmup_constant":
        # linear_schedule clamps at end_value after transition_steps -> warmup then constant
        return optax.linear_schedule(0.0, lr, warmup)
    if cfg.lr_schedule == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value = 0.0, peak_value = lr, warmup_steps = warmup,
            decay_steps = cfg.total_steps, end_value = 0.0,
        )
    if cfg.lr_schedule == "sqrt_decay":
        # JAX-F2D2 reference: constant until lr_decay_start + lr_decay_steps, then lr / sqrt(elapsed / lr_decay_steps)
        return lambda step: lr / jnp.sqrt(jnp.maximum((step - cfg.lr_decay_start) / cfg.lr_decay_steps, 1.0))
    raise ValueError(f"unknown lr_schedule '{cfg.lr_schedule}'")


def make_optimizer(cfg) -> optax.GradientTransformation:
    if cfg.optimizer == "muon":
        # Newton-Schulz-orthogonalized updates on the 2-D weight matrices, Adam on the 1-D leaves
        # (biases). The staged step counter (tree_get "count") is optimizer-agnostic: muon's count
        # nodes all tick +1/step from 0, same as adamw (verified scratch/test_muon_count.py).
        inner = optax.contrib.muon(learning_rate = make_lr_schedule(cfg),
                                   adam_learning_rate = cfg.muon_adam_lr,
                                   weight_decay = cfg.weight_decay)
    else:
        inner = optax.adamw(learning_rate = make_lr_schedule(cfg), weight_decay = cfg.weight_decay)
    return optax.chain(optax.clip_by_global_norm(cfg.grad_clip), inner)


def make_dual_optimizer(cfg) -> optax.GradientTransformation:
    """Plain Adam on log_alpha' for the CQL dual ascent (separate from the model's clipped optimizer)."""
    return optax.adam(learning_rate = cfg.agent.cql_dual_lr)


def _detach_critics(model: FlowRLModel) -> list:
    return jax.tree_util.tree_map(
        lambda x: jax.lax.stop_gradient(x) if eqx.is_inexact_array(x) else x, model.critics
    )


def _ema_update(target: FlowRLModel, model: FlowRLModel, tau: float) -> FlowRLModel:
    m_arr = eqx.filter(model, eqx.is_inexact_array)
    t_arr = eqx.filter(target, eqx.is_inexact_array)
    new_t_arr = optax.incremental_update(m_arr, t_arr, tau)
    return eqx.combine(new_t_arr, target)


class BaseAgent:
    """Shared model/optimizer/EMA + the TD+actor skeleton. Subclasses inject the objective via hooks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.optimizer = make_optimizer(cfg)
        self.dual_opt = make_dual_optimizer(cfg)

    # ---------- construction ----------
    def init_train_state(self, key: PRNGKeyArray) -> tuple[TrainState, optax.GradientTransformation]:
        model = make_model(self.cfg, key)
        if self.cfg.init_ckpt:                                # warm start (paper staged recipe / distillation)
            model = load_model_weights(self.cfg, model, self.cfg.init_ckpt, key)
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        log_alpha_prime = jnp.array(0.0)                       # alpha' = exp(0) = 1 initially
        dual_opt_state = self.dual_opt.init(log_alpha_prime)
        state = TrainState(model = model, target = model, opt_state = opt_state,
                           log_alpha_prime = log_alpha_prime, dual_opt_state = dual_opt_state)
        return state, self.optimizer

    # ---------- overridable hooks (defaults = no extra critic term, no dual) ----------
    def bc_loss(self, model, tokens_p, action, keys) -> dict:
        raise NotImplementedError

    def sample_action(self, model, tokens_p, z_key, eps_key) -> Array:
        raise NotImplementedError

    def predict_action(self, model, tokens_p, z_key, eps_key, steps) -> Array:
        raise NotImplementedError

    def aux_actor_term(self, model, target, tokens_p, tokens_p_tgt, action, keys) -> dict:
        # extra per-example actor term keyed to a free RNG slot (divergence-head agents add "l_div");
        # target/tokens_p_tgt = the EMA teacher for the F2D2 anchors (tokens_p_tgt is None unless use_divergence)
        return {}

    def extra_critic_term(self, model, tokens_p, tokens_p_next, tokens_c, q_data, key) -> dict:
        return {}

    def aggregate_extra(self, per, alpha_prime) -> tuple:
        return 0.0, 0.0, {}

    def dual_step(self, log_alpha_prime, dual_opt_state, gap):
        return log_alpha_prime, dual_opt_state

    # ---------- shared per-example loss skeleton ----------
    def _per_example_loss(self, model, target, batch, key) -> dict:
        cfg = self.cfg
        h = cfg.horizon
        keys = jax.random.split(key, 12)   # [0]p-tok [1..2]bc [3]c-tok [4,9]actor [5]next-p [6,10]td [7]tgt-c [8]extra

        state = batch["obs_state"]
        img = batch["obs_img"] if cfg.obs_type == "image" else None
        action = batch["action"]

        tokens_p = model.policy_tokens(img, state, keys[0])

        bc = self.bc_loss(model, tokens_p, action, keys)
        if "l2" in bc:                                          # regression BC -> early return
            return bc
        l_fm = bc["l_fm"]
        # teacher (EMA target) policy tokens for the F2D2 divergence anchors; skipped unless the agent needs them
        tokens_p_tgt = (target.policy_tokens(img, state, keys[0])
                        if getattr(cfg.agent, "use_divergence", False) else None)
        aux = self.aux_actor_term(model, target, tokens_p, tokens_p_tgt, action, keys)   # {} default; {"l_div": ...} for OFQL-CQL
        if not cfg.agent.use_q:                                # BC-only (flow / MeanFlow)
            return {"l_fm": l_fm, "q_pi": 0.0, "l_td": 0.0, "q_data": 0.0, **aux}

        tokens_c = model.critic_tokens(img, state, keys[3])

        # --- actor Q-max: one-step sample, MEAN over detached critics ---
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
        l_td = jnp.sum((q_data - y) ** 2)                      # per-critic TD, summed over the ensemble

        out = {"l_fm": l_fm, "q_pi": q_pi, "l_td": l_td, "q_data": jnp.mean(q_data), **aux}
        out.update(self.extra_critic_term(model, tokens_p, on_tokens_p_next, tokens_c, q_data, keys[8]))
        return out

    # ---------- shared loss aggregation ----------
    def make_loss(self):
        cfg = self.cfg
        regression = cfg.policy_type == "regression"

        def loss_fn(model, target, batch, key, alpha_prime):
            keys = jax.random.split(key, batch["action"].shape[0])
            per = jax.vmap(lambda b, k: self._per_example_loss(model, target, b, k))(batch, keys)
            if regression:
                l2 = jnp.mean(per["l2"])
                return l2, ({"loss/total": l2, "actor/regression_loss": l2}, 0.0)
            l_fm = jnp.mean(per["l_fm"])
            l_div = jnp.mean(per["l_div"]) if "l_div" in per else 0.0   # F2D2 divergence head (OFQL-CQL)
            div_term = cfg.agent.div_coef * l_div if "l_div" in per else 0.0
            if not cfg.agent.use_q:
                total_bc = l_fm + div_term
                return total_bc, ({"loss/total": total_bc, "actor/BC_flow_loss": l_fm,
                                   "actor/div_loss": l_div}, 0.0)

            q_pi, q_data = per["q_pi"], per["q_data"]
            if cfg.agent.normalize_q_loss:
                denom = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_pi))) + 1e-6
                q_term = jnp.mean(q_pi) / denom
            else:
                q_term = jnp.mean(q_pi)
            l_td = jnp.mean(per["l_td"])

            penalty, gap, extra_metrics = self.aggregate_extra(per, alpha_prime)
            critic_loss = l_td + penalty
            actor_loss = cfg.agent.alpha * l_fm - q_term + div_term   # div_term trains the F2D2 divergence head
            total = actor_loss + critic_loss
            metrics = {
                "loss/total": total,
                "actor/loss": actor_loss,
                "actor/BC_flow_loss": l_fm,
                "actor/div_loss": l_div,
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

    # ---------- jitted factories ----------
    def make_train_step(self):
        cfg = self.cfg
        optimizer = self.optimizer
        loss_fn = self.make_loss()
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux = True)

        @eqx.filter_jit
        def train_step(state: TrainState, batch, key):
            alpha_prime = jnp.clip(jnp.exp(state.log_alpha_prime), 0.0, 1e6)
            (_, (metrics, gap)), grads = grad_fn(state.model, state.target, batch, key, alpha_prime)
            params = eqx.filter(state.model, eqx.is_inexact_array)
            gnorm = optax.global_norm(grads)                 # pre-clip gradient norm
            updates, opt_state = optimizer.update(grads, state.opt_state, params)
            model = eqx.apply_updates(state.model, updates)
            target = _ema_update(state.target, model, cfg.agent.tau)

            log_alpha_prime, dual_opt_state = self.dual_step(state.log_alpha_prime, state.dual_opt_state, gap)

            pnorm = optax.global_norm(eqx.filter(model, eqx.is_inexact_array))
            metrics = {**metrics, "params/grad_norm": gnorm, "params/param_norm": pnorm}
            return TrainState(model = model, target = target, opt_state = opt_state,
                              log_alpha_prime = log_alpha_prime, dual_opt_state = dual_opt_state), metrics

        return train_step

    def make_q_fn(self):
        """Jitted Q(s, a_norm) = mean over critics, for eval-time visualization."""
        use_image = self.cfg.obs_type == "image"

        @eqx.filter_jit
        def q_fn(model, img, state, a_norm, key):
            keys = jax.random.split(key, state.shape[0])

            def one(state_i, a_i, k, img_i):
                tokens_c = model.critic_tokens(img_i, state_i, k)
                return jnp.mean(model.q(tokens_c, a_i))      # mean over the critic ensemble

            if use_image:
                return jax.vmap(lambda im, s, a, k: one(s, a, k, im))(img, state, a_norm, keys)
            return jax.vmap(lambda s, a, k: one(s, a, k, None))(state, a_norm, keys)

        return q_fn

    def make_predict(self, steps: int = 1):
        """Returns a jitted batch action predictor (one predictor per integration-step value)."""
        cfg = self.cfg
        use_image = cfg.obs_type == "image"

        @eqx.filter_jit
        def predict(model: FlowRLModel, img, state, key):
            keys = jax.random.split(key, state.shape[0])

            def one(state_i, k, img_i):
                k_enc, k_z, k_eps = jax.random.split(k, 3)
                tokens_p = model.policy_tokens(img_i, state_i, k_enc)
                return self.predict_action(model, tokens_p, k_z, k_eps, steps)

            if use_image:
                return jax.vmap(lambda im, s, k: one(s, k, im))(img, state, keys)
            return jax.vmap(lambda s, k: one(s, k, None))(state, keys)

        return predict
