# flow_rl — one-step flow policies for offline RL on Push-T

Offline-RL testbed (JAX + Equinox) for **one-step flow / flow-map policies** on Push-T. A transformer
policy maps a noise vector to an action chunk in a single network evaluation; the same backbone optionally
carries a critic (for Q-learning / CQL) and an F2D2 **divergence head** that yields the policy's own
log-density `log q(a)` in one forward pass (needed by the importance-sampled CQL penalty).

Two families of one-step map are implemented and compared:
- **MeanFlow** — the average-velocity identity, one JVP (`agents/meanflow.py`, `agents/meanflow_cql.py`).
- **Shortcut-Distill (F2D2)** — tangent distillation + midpoint semigroup self-consistency
  (`agents/shortcut.py`, `agents/shortcut_cql.py`).

Task: Push-T, 18-D keypoint state (`lerobot/pusht_keypoints`) or image obs (`lerobot/pusht_image`).
Action chunk length `H = 16`, `act_dim = 2`, so the flow acts on `d = H*act_dim = 32` dims.

---

## Commands

Training and eval are driven by a named `Config` plus optional `field=value` overrides that route to the
top-level `Config` or the nested `cfg.agent` automatically.

```bash
# train a registered config (optionally override any field)
python flow_rl/train.py <config> [field=value ...]
python flow_rl/train.py fql alpha=100 total_steps=200000
python flow_rl/train.py meanflow_bc_p1 adaptive_weight=false exp_name=meanflow_bc_p1_noaw

# evaluate a checkpoint (closed-loop gym-pusht rollouts)
python flow_rl/eval.py <config> <ckpt.eqx> [field=value ...]

# CPU smoke test: build + one train step + predict for every agent on synthetic data (no dataset)
python smoke_test.py
```

`train.py` logs to wandb (project `flow_rl`), runs `num_eval_rollouts` closed-loop gym-pusht rollouts every
`eval_interval` steps at each `eval_flow_steps` integration-step count, and checkpoints every `save_interval`
to `ckpt_dir/<exp_name>/model_<step>.eqx`. The venv is `/data/user_data/saksham3/uv/flow_rl/bin/python`.

Convention (flow.py, all one-step-map agents): `e ~ N(0, I)` is noise, `a` the data action chunk,
`x_t = (1-t)*e + t*a` (so `t=0` is noise, `t=1` is data), ground-truth conditional velocity `v = a - e`,
and the one-step endpoint is `a_hat = clip(e + u(e, 0, 1), -1, 1)`.

---

## Repository layout

```
flow_rl/
  train.py              training loop (wandb, periodic eval + checkpoints)
  eval.py               load a checkpoint, run closed-loop gym-pusht rollouts
  flow.py               interpolate() + Euler sample_action() for the flow map
  configs/__init__.py   Config + per-agent AgentConfig dataclasses + the named-config registry
  agents/               one agent class per method (see below)
  models/               TransformerPolicy (+ divergence head), critics, ResNet obs encoder
  data/                 lerobot Push-T dataset loader + action/state normalization
  envs/pusht_eval.py    closed-loop evaluation in gym-pusht
smoke_test.py           CPU verification harness (all agents, synthetic data)
```

---

## Model backbone (`models/`)

- **`TransformerPolicy`** — the flow policy. `velocity(obs, x_t, t)` is the instantaneous flow-matching
  velocity (Fourier time token); `velocity_avg(obs, x_t, t, s)` is the MeanFlow/shortcut **average** field
  from current time `t` to target `s`. `regress(obs)` is a plain deterministic head for L2-regression BC.
- **Divergence head** — gated by `use_divergence` (a static field; the head's params are `None` when off,
  so old checkpoints still load). `divergence(obs, x_t, t, s) -> scalar` runs the flow tokens plus a CLS
  token through the shared blocks. It predicts `-div(u)/rescale`, giving a **one-forward log-density**
  `log q(a) = log N(e; 0, I) + divergence_rescale * D(e, 0, 1)`.
- **Critics** — `num_critics` (default 2) clipped double-Q heads, sigmoid-bounded (sparse reward → return in
  (0, 1]). The observation encoder (ResNet-18 for images; MLP for keypoint state) can be shared or separate.

Shared skeleton (`agents/base.py`): all arrays live in `TrainState(model, target, opt_state,
log_alpha_prime, dual_opt_state)`; `target` is the EMA of `model` (rate `tau`), used for the critic TD
backup. The critic TD target for an `H`-step action chunk is

```
reward   = gamma**steps_to_end   if steps_to_end < H else 0      # sparse, terminal reward 1
y        = sg( reward + gamma**H * (1 - done) * min_i Q_target_i(s', a') )
L_TD     = ( Q(s, a_data) - y )**2
```

---

## Agents and their optimization objectives

`sg(.)` = stop-gradient. Losses are per-example unless noted; the runner averages over the batch. Q-max
terms detach the critic so the actor only moves the policy. For the flow-map agents, `t <= s` are the
current/target times, `t == s` a fraction of the time (the "diagonal"), and `x_r` the midpoint state.

### `FQLAgent` — `agents/fql.py` (configs: `bc_l2`, `bc_flow`, `fql`)
One-step flow + DDPG-style Q-max, regularized by flow-matching BC. Also covers BC-only (`use_q=False`) and
L2 regression (`policy_type="regression"`).

```
L_FM  = || v_pred(x_t, t) - (a - e) ||^2          # flow-matching BC; t ~ U[0, max_t], or t=0 (straight_flow)
                                                   # regression variant: L2 = || regress(obs) - a ||^2
a_hat = one-step (or inner_flow_steps-step) Euler flow sample
total = alpha * L_FM  -  Q(s, a_hat)  +  L_TD      # actor (alpha*BC - Q) + critic TD; use_q=False drops -Q + L_TD
```

### `CQLAgent` — `agents/cql.py` (config: `cql`)
FQL plus a conservative importance-sampled CQL penalty on the critic (aviralkumar2907/CQL,
min_q_version 3, `with_lagrange` dual ascent).

```
gap      = logsumexp_j ( Q(s, a_j) - log q(a_j) )  -  Q(s, a_data)
           # a_j = cql_n_actions each of: uniform-random + current-policy pi(s) + next-policy pi(s')
           # log q: uniform = -d*log2; policy proposals = EXACT one-step flow change-of-variables (slogdet)
penalty  = alpha' * sum_i ( gap_i - cql_budget )   # per critic; alpha' = exp(log_alpha') >= 0, learned
critic   = L_TD + penalty
dual     = min over alpha' of  -alpha' * (gap - cql_budget)   # drives gap -> cql_budget
```

### `MeanFlowAgent` — `agents/meanflow.py` (configs: `meanflow`, `meanflow_bc_p1`)
Staged MeanFlow-F2D2 BC in one run (`phase_steps`). Phase 1 learns the average-velocity map; phase 2 adds
the divergence head. `du/dt`, `dD/dt` are total derivatives via one `jax.jvp` with tangents `(v, 1)`.

```
Phase 1 (u map):   L_mf     = adaptive( || u(x_t,t,s) - sg( (s-t)*du/dt + v ) ||^2 )
Phase 2 (+ D head): L_mf  +  L_div = ( D(x_t,t,s) - sg( (s-t)*dD/dt - div(u(x_t,t,t))/rescale ) )^2
total  = L_mf + L_div          # div anchor div(u) via Hutchinson (k = div_hutch_samples), sg'd
```

`adaptive(err) = err / sg((err + norm_eps)**norm_p)` when `adaptive_weight` (MeanFlow L2 weighting; applied
to `L_mf` only). `log q(a) = log N(e) + divergence_rescale * D(e,0,1)`. Variants: `frozen_teacher` (phase-2
tangent + div anchor from a frozen end-of-phase-1 snapshot — deterministic and data-anchored) and
`self_tangent` (from a tracking EMA — diverges; kept only as an ablation).

### `MeanFlowCQLAgent` — `agents/meanflow_cql.py` (config: `meanflow_cql`)
MeanFlow policy + double-Q critic + CQL, warm-started from a MeanFlow BC checkpoint (`init_ckpt`). The CQL
log q comes from the divergence head. `alpha` weights only the `t == s` (diagonal) part of `L_mf`.

```
actor  = alpha * L_mf(t==s)  +  L_mf(t<s)  +  L_div  -  E[ mean_i Q_i(s, a_hat) ]
critic = L_TD + penalty        # penalty as in CQL, with log q(a_j) = log N(e) + rescale * D(e,0,1)
total  = actor + critic  (+ dual ascent on alpha')
```

The divergence head is trained ONLY by `L_div`; CQL proposals and their log q are stop-gradiented, so the
penalty never trains the map or `D`. `divergence_rescale` MUST match the BC run that produced `init_ckpt`.

### `ShortcutAgent` — `agents/shortcut.py` (configs: `shortcut_staged`, `shortcut_bc_direct`)
Staged Shortcut-Distill-F2D2 BC. One `(t, s)` draw per example, `t == s` a `diag_fraction`. Tangent losses
on the diagonal, midpoint **semigroup** (two half-jumps, `r = (t+s)/2`, `x_r = sg(x_t + (r-t)*u(x_t,t,r))`)
off it.

```
L_u  (t==s):  || u(x_t,t,t) - sg( TARGET ) ||^2
              TARGET = v_phi(x_t,t) if distill_f2d2 else the ground-truth v = a - e
     (t< s):  || u(x_t,t,s) - 1/2 * sg( u(x_t,t,r) + u(x_r,r,s) ) ||^2
L_D  (t==s):  ( D(x_t,t,t) + sg( div(FIELD)(x_t,t) ) / rescale )^2
              FIELD = v_phi if distill_f2d2 else the student's own diagonal u(.,t,t)
     (t< s):  ( D(x_t,t,s) - 1/2 * sg( D(x_t,t,r) + D(x_r,r,s) ) )^2
total = L_u + L_D
```

- `distill_f2d2=True` (3 phases): (1) `v_phi` flow matching; (2) tangent distill vs a **frozen snapshot**
  `v_phi` + semigroup; (3) + D head. `TrainState.target` is snapshotted at phase boundaries (not EMA).
- `distill_f2d2=False` (2 phases, default): (1) learn the u map directly (diagonal = ground-truth-v FM +
  semigroup off-diagonal); (2) + D head anchored on the student's own diagonal. No teacher anywhere.

### `ShortcutCQLAgent` — `agents/shortcut_cql.py` (config: `shortcut_cql`)
Shortcut policy + double-Q critic + CQL, warm-started from a shortcut BC checkpoint (`init_ckpt`; plus
`teacher_ckpt` when `distill_f2d2=True`). `L_VMSC` is the diagonal tangent term, `L_u-sc` the off-diagonal
semigroup term.

```
actor  = alpha * L_VMSC  +  L_u-sc  +  L_div  -  E[ mean_i Q_i(s, a_hat) ]
critic = L_TD + penalty        # log q(a_j) = log N(e) + rescale * D(e,0,1)
total  = actor + critic  (+ dual ascent on alpha')
```

`alpha` weights only the diagonal BC term; the off-diagonal and divergence terms have weight 1. With
`distill_f2d2=False` the VM-SC tangent target is the ground-truth `v = a - e` (no teacher).

---

## Registered configs (`configs/__init__.py`)

| Config | Agent | What |
|---|---|---|
| `bc_l2` / `bc_flow` | FQL (BC-only) | L2-regression / flow-matching behavior cloning |
| `fql` | FQL | one-step flow + Q-max |
| `fql_shared*` | FQL | shared-backbone / stop-gradient ablations |
| `cql` | CQL | FQL + CQL conservative critic (state obs) |
| `meanflow` | MeanFlow | 2-phase MeanFlow-F2D2 BC |
| `meanflow_bc_p1` | MeanFlow | phase-1-only MeanFlow BC (500-eval every 25k) |
| `meanflow_cql` | MeanFlowCQL | MeanFlow policy + CQL (needs `init_ckpt`) |
| `shortcut_staged` | Shortcut | 3-phase Shortcut-Distill-F2D2 BC (`distill_f2d2=True`) |
| `shortcut_bc_direct` | Shortcut | 2-phase teacherless shortcut BC |
| `shortcut_cql` | ShortcutCQL | shortcut policy + CQL (needs `init_ckpt`) |
| `smoke` | FQL | tiny CPU smoke config |
