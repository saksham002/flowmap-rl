"""
CQL: FQL + a conservative importance-sampled log-sum-exp penalty on the
critic (aviralkumar2907/CQL min_q_version 3, with_lagrange dual ascent).

penalty = alpha' * (gap - budget), gap = logsumexp_j (Q(s,a_j) - log q(a_j)) -
Q(s,a_data) over a_j = cql_n_actions uniform-random + cql_n_actions current-policy
+ cql_n_actions next-policy chunks. alpha' = exp(log_alpha') >= 0 is a learned
multiplier updated by dual ascent toward the fixed cql_budget.

CQLMixin injects the penalty/dual into any base agent. The per-agent proposal + log q(a) recipe is the
overridable `_policy_logp_block`:
  - FQLAgent (default) -> flow one-step sample with EXACT change-of-variables log q (_flow_sample_logp).
  - ShortcutCQLAgent   -> one-step endpoint a = e + u(e,0,1) with the F2D2 divergence-head log q.
Proposal actions and their log q are always stop-gradiented (the divergence head is trained only by its own
F2D2 loss, never by the CQL penalty) -- the decoupling that makes CQL + MeanFlow clean.
"""

import math
import jax
import jax.numpy as jnp

from flow_rl.flow import sample_action as flow_sample
from flow_rl.agents.fql import FQLAgent


def _flow_sample_logp(vf, z, steps):
    """One flow sample a = g(z), z ~ N(0, I), with the EXACT change-of-variables log-density
    log q(a) = log N(z;0,I) - log|det da/dz|.

    The Jacobian da/dz of the `steps`-step Euler sampler is built column-by-column with a lax.scan of
    jax.jvp (one tangent live at a time -> peak memory ~ one velocity forward; a parallel jacfwd OOMs at
    N x batch), then slogdet. Exact for the one-step straight-flow map (steps=1, a = z + v(z,0)); for
    steps>1 it is the exact density of the discrete sampler. Returns (a, log q); caller stop-gradients both.
    """
    h, act_dim = z.shape
    d = h * act_dim
    g = lambda zf: flow_sample(vf, zf.reshape(h, act_dim), steps = steps).reshape(-1)
    z_flat = z.reshape(-1)
    a_flat = g(z_flat)
    _, jac = jax.lax.scan(lambda _, e: (None, jax.jvp(g, (z_flat,), (e,))[1]), None, jnp.eye(d))  # rows = da/dz cols
    _, logabsdet = jnp.linalg.slogdet(jac)                          # det invariant to transpose
    log_base = -0.5 * jnp.sum(z_flat ** 2) - 0.5 * d * math.log(2.0 * math.pi)   # log N(z; 0, I)
    return a_flat.reshape(h, act_dim), log_base - logabsdet


def _cql_term(agent, model, tokens_p, tokens_p_next, tokens_c, key, cfg):
    """Importance-sampled CQL log-sum-exp term for one transition (aviralkumar2907/CQL min_q_version 3).

    Returns ood_i = log-mean-exp_j (Q_i(s, a_j) - log q(a_j)) for each critic i (per-critic vector). The
    conservative gap_i = ood_i - Q_i(s, a_data) and the penalty are formed in aggregate_extra. a_j =
    cql_n_actions uniform-random + cql_n_actions current-policy pi(s) + cql_n_actions next-policy pi(s')
    chunks (3N total). Uniform log density = -d*log2; the two policy proposals use agent._policy_logp_block.
    Samples and log q are stop-gradiented (in _policy_logp_block).
    """
    h, act_dim, N = cfg.horizon, cfg.act_dim, cfg.agent.cql_n_actions
    d = h * act_dim
    k_rand, k_cur, k_next = jax.random.split(key, 3)

    a_rand = jax.random.uniform(k_rand, (N, h, act_dim), minval = -1.0, maxval = 1.0)
    logp_rand = jnp.full((N,), -d * math.log(2.0))                   # uniform density on [-1,1]^d

    a_cur, lp_cur, base_cur, logdet_cur = agent._policy_logp_block(model, tokens_p, k_cur)        # pi(s)
    a_next, lp_next, base_next, logdet_next = agent._policy_logp_block(model, tokens_p_next, k_next)  # pi(s')

    q_rand = jax.vmap(lambda a: model.q(tokens_c, a))(a_rand)        # (N, num_critics) raw Q per proposal type
    q_cur = jax.vmap(lambda a: model.q(tokens_c, a))(a_cur)
    q_next = jax.vmap(lambda a: model.q(tokens_c, a))(a_next)
    # proposals INCLUDED in the penalty: uniform + policy(s) + policy(s'); cql_no_uniform -> policy proposals only
    if cfg.agent.cql_no_uniform:
        q_used = jnp.concatenate([q_cur, q_next], axis = 0)          # (2N, num_critics)
        used_logp = jnp.concatenate([lp_cur, lp_next])
    else:
        q_used = jnp.concatenate([q_rand, q_cur, q_next], axis = 0)  # (3N, num_critics)
        used_logp = jnp.concatenate([logp_rand, lp_cur, lp_next])
    M = q_used.shape[0]
    # importance-sampled: subtract log q (CQL(H)); else plain log-mean-exp of Q over the OOD actions
    q_corr = q_used.T - used_logp[None, :] if cfg.agent.cql_importance_sampling else q_used.T  # (num_critics, M)
    ood = (jax.scipy.special.logsumexp(q_corr / cfg.agent.cql_temp, axis = -1) - math.log(M)) * cfg.agent.cql_temp  # (num_critics,)
    # diagnostics: split log q = log N(z) [logq_base ~ -45, bounded] - log|det da/dz| [logdet -> -inf if flow collapses]
    diag = {"logq_rand": logp_rand[0], "logq_cur": jnp.mean(lp_cur), "logq_next": jnp.mean(lp_next),
            "q_ood": jnp.mean(q_used), "logq_base": jnp.mean(base_cur),
            "logdet_cur": jnp.mean(logdet_cur), "logdet_next": jnp.mean(logdet_next),
            "q_ood_rand": jnp.mean(q_rand), "q_ood_cur": jnp.mean(q_cur),
            "q_ood_next": jnp.mean(q_next)}    # raw Q split by proposal type (uniform/current/next)
    return ood, diag                                              # per-critic log-sum-exp + IS-logp diagnostics


class CQLMixin:
    """CQL conservative penalty + dual ascent, mixed into a base agent. Subclasses may override
    `_policy_logp_block` to supply the proposal + log q recipe (flow slogdet, or the F2D2 divergence head)."""

    def _policy_logp_block(self, model, toks, key):
        """N detached policy proposals + EXACT one-step flow change-of-variables log q (default / FQL-CQL).
        Returns (a, logp, log_base, logdet), all stop-gradiented."""
        cfg = self.cfg
        h, act_dim, N = cfg.horizon, cfg.act_dim, cfg.agent.cql_n_actions
        d = h * act_dim
        steps = cfg.agent.inner_flow_steps
        k_z, k_e = jax.random.split(key)                            # k_e unused: keeps z byte-identical to prior CQL
        z = jax.random.normal(k_z, (N, h, act_dim))
        vf = lambda x, tt: model.velocity(toks, x, tt)
        a, logp = jax.vmap(lambda zz: _flow_sample_logp(vf, zz, steps))(z)
        log_base = -0.5 * jnp.sum(z ** 2, axis = (1, 2)) - 0.5 * d * math.log(2.0 * math.pi)   # log N(z;0,I), per sample
        logdet = log_base - logp                                   # logp = log_base - log|det da/dz| -> recover log|det|
        a = jax.lax.stop_gradient(jnp.clip(a, -1.0, 1.0))          # clip for Q eval (matches inference)
        return a, jax.lax.stop_gradient(logp), jax.lax.stop_gradient(log_base), jax.lax.stop_gradient(logdet)

    def extra_critic_term(self, model, tokens_p, tokens_p_next, tokens_c, q_data, key) -> dict:
        ood, diag = _cql_term(self, model, tokens_p, tokens_p_next, tokens_c, key, self.cfg)   # log-sum-exp + IS-logp diag
        out = {"cql": ood - q_data, "cql_ood": ood}                  # (num_critics,) per-critic conservative gap
        out.update({f"cql_{k}": v for k, v in diag.items()})
        return out

    def aggregate_extra(self, per, alpha_prime) -> tuple:
        cfg = self.cfg
        # CQL Lagrange penalty applied PER CRITIC and summed over the ensemble, matching the reference.
        gap_per_critic = jnp.mean(per["cql"], axis = 0)             # (num_critics,) batch-mean gap per critic
        gap = jnp.mean(gap_per_critic)                              # mean gap drives the dual ascent
        budget = float(cfg.agent.cql_budget)                       # fixed target gap (target_action_gap)
        cql_penalty = alpha_prime * jnp.sum(gap_per_critic - budget)
        m = jnp.mean
        metrics = {
            "cql/logsumexp": m(per["cql_ood"]),      # importance-sampled log-mean-exp of Q over the OOD actions
            "cql/Q_data": m(per["q_data"]),          # Q of dataset actions; gap = cql/logsumexp - cql/Q_data
            "cql/gap": gap,                          # cql/logsumexp - cql/Q_data (the conservative gap)
            "cql/penalty": cql_penalty,              # alpha' * sum_i(gap_i - budget): the term added to critic/loss
            "cql/alpha_prime": alpha_prime,          # learned Lagrange multiplier (dual ascent)
            "cql/budget": budget,                    # fixed target gap the dual ascent drives toward
            "cql/logq_rand": m(per["cql_logq_rand"]),   # IS log-prob of the uniform proposal = -d*log2 (constant)
            "cql/logq_cur": m(per["cql_logq_cur"]),     # mean EXACT IS log-prob of current-policy samples
            "cql/logq_next": m(per["cql_logq_next"]),
            "cql/q_ood": m(per["cql_q_ood"]),           # mean RAW Q over the OOD samples (no -log q correction)
            "cql/q_ood_rand": m(per["cql_q_ood_rand"]),
            "cql/q_ood_cur": m(per["cql_q_ood_cur"]),
            "cql/q_ood_next": m(per["cql_q_ood_next"]),
            "cql/logq_base": m(per["cql_logq_base"]),   # mean log N(z;0,I) of policy samples (~ -45, bounded)
            "cql/logdet_cur": m(per["cql_logdet_cur"]), # mean log|det da/dz| current policy (-> -inf as the flow collapses)
            "cql/logdet_next": m(per["cql_logdet_next"]),
        }
        return cql_penalty, gap, metrics

    def dual_step(self, log_alpha_prime, dual_opt_state, gap):
        # dual ascent: minimize alpha'_loss = -alpha'*(gap - budget) -> drives gap -> budget.
        cfg = self.cfg
        ap = lambda log_ap: jnp.clip(jnp.exp(log_ap), 0.0, 1e6)
        g = jax.grad(lambda log_ap: -ap(log_ap) * (jax.lax.stop_gradient(gap) - cfg.agent.cql_budget))(log_alpha_prime)
        d_updates, dual_opt_state = self.dual_opt.update(g, dual_opt_state)
        return log_alpha_prime + d_updates, dual_opt_state


class CQLAgent(CQLMixin, FQLAgent):
    """FQL + CQL conservative penalty (default flow-slogdet log q). Byte-identical to the prior CQLAgent."""
    pass
