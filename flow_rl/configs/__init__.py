"""Config dataclasses (shared knobs + per-method AgentConfig) + registry of named instances.

Run:  python flow_rl/train.py <config_name> [field=value ...]

`Config` holds task / data / model-architecture / optimizer / eval knobs that every
method shares. Method-specific objective knobs live in a nested `AgentConfig`
(FQLConfig / CQLConfig / MeanFlowConfig / ...); `cfg.agent.name` selects the agent class.
Flat `field=value` overrides route to `Config` or `cfg.agent` automatically.
"""

from dataclasses import dataclass, replace, fields, field
from typing import Any


# ----------------------------- agent (method) configs -----------------------------
@dataclass
class AgentConfig:
    # shared across all value-based agents
    name: str = "fql"                  # selects the agent class (see agents/__init__)
    use_q: bool = True                 # False -> BC only (flow-matching, or L2 if policy_type=regression)
    alpha: float = 1.0                 # actor BC/Q balance: actor_loss = alpha * L_BC - Q
    gamma: float = 0.99
    tau: float = 0.005                 # EMA target rate
    normalize_q_loss: bool = False     # FQL default off; unnecessary for sparse reward
    cql_dual_lr: float = 3e-4          # Adam lr for the CQL dual ascent on log_alpha' (dual state exists for all agents)


@dataclass
class FQLConfig(AgentConfig):
    name: str = "fql"
    inner_flow_steps: int = 1          # Euler steps for the training Q-max sample (1 = one-step)
    straight_flow: bool = False        # FM loss at fixed t=0 (one-step path v(z,0)=a-z) vs t~U[0,max_t]
    max_t: float = 1.0


@dataclass
class CQLConfig(FQLConfig):
    # CQL conservative penalty on the critic (aviralkumar2907/CQL with_lagrange). penalty = alpha' *
    # (gap - cql_budget), gap = logsumexp_j (Q(s,a_j) - log q(a_j)) - Q(s,a_data) over a_j = cql_n_actions
    # uniform-random + cql_n_actions current-policy + cql_n_actions next-policy chunks (importance-sampled).
    name: str = "cql"
    use_cql: bool = True
    cql_n_actions: int = 8             # actions per proposal (uniform / current / next policy) for the logsumexp
    cql_temp: float = 1.0
    cql_importance_sampling: bool = True   # subtract log q(a_j) (CQL(H)); False -> plain log-mean-exp of Q
    cql_budget: float = 5.0            # target gap (target_action_gap) the dual ascent drives toward
    cql_no_uniform: bool = False       # drop the uniform-random proposals from the penalty (policy proposals only)


@dataclass
class ShortcutConfig(AgentConfig):
    # F2D2 paper staged pipeline, ALL PHASES IN ONE RUN (agents/shortcut.py; toy-validated, logq corr 0.997):
    # phase 1 v_phi FM (no ckpt loading) -> phase 2 shortcut distill (teacher = end-of-phase-1 snapshot) ->
    # phase 3 + divergence head (teacher = end-of-phase-2 snapshot). total_steps must equal sum(phase_steps).
    name: str = "shortcut"
    use_q: bool = False                # BC-only
    # distill_f2d2=True: the 3-phase teacher recipe (v_phi FM -> tangent distill vs frozen snapshot ->
    # + D head, D tangent anchored on the frozen teacher). False (default): NO teacher in any phase --
    # 2 phases: (1) u map directly (diag FM vs ground-truth v = a - e + semigroup offdiag),
    # (2) + D head (tangent anchor = the STUDENT's own diagonal). phase_steps length must match (3 vs 2).
    distill_f2d2: bool = False
    phase_steps: tuple = (100_000, 100_000)            # distill_f2d2=True: (v_phi FM, distill, + div head)
    diag_fraction: float = 0.5         # fraction of exactly-diagonal (s == t) samples (reference 0.5)
    # --- phase 3: F2D2 divergence head ---
    use_divergence: bool = True        # div head params exist from the start; trained only in phase 3
    div_coef: float = 1.0
    div_hutch_samples: int = 1
    divergence_rescale: float = 200.0  # D ~ -div(u)/rescale; 200 for the d=32 action chunk (checker d=2 used 10)


@dataclass
class ShortcutCQLConfig(CQLConfig):
    # Shortcut-model policy + CQL + F2D2 divergence-head log q (agents/shortcut_cql.py). SINGLE stage;
    # requires the staged BC run's final checkpoint: pass init_ckpt=<ckpt> at launch (+ teacher_ckpt=<ckpt>
    # when distill_f2d2=True).
    name: str = "shortcut_cql"
    alpha: float = 10.0                # weight of the tangent VM-SC distill term (the "BC weight")
    # distill_f2d2=True: VM-SC tangent target = frozen v_phi from teacher_ckpt (current recipe).
    # False (default): NO teacher anywhere -- VM-SC target = ground-truth v = a - e; teacher_ckpt unused.
    # Either way init_ckpt (a u + D BC checkpoint) is REQUIRED as the warm start.
    distill_f2d2: bool = False
    teacher_ckpt: str = ""             # distill_f2d2=True only: frozen v_phi = the staged BC checkpoint
    diag_fraction: float = 0.5         # P(t == s): tangent losses on the diagonal, semigroup off it (reference)
    use_divergence: bool = True
    div_hutch_samples: int = 1
    divergence_rescale: float = 200.0  # MUST match the BC run's value (log q = log N + rescale*D)


@dataclass
class MeanFlowConfig(AgentConfig):
    # MeanFlow-F2D2 BC (paper Eq 3.11-3.13), BOTH phases in one run (agents/meanflow.py):
    # phase 1 L_mf from scratch -> phase 2 + L_div-mf (div anchor = sg(student diagonal); no teacher/EMA).
    name: str = "meanflow"
    use_q: bool = False                # BC-only
    phase_steps: tuple = (100_000, 100_000)   # (L_mf, + divergence head)
    flow_ratio: float = 0.5            # fraction of exactly-diagonal (t == s) samples (MeanFlow convention)
    time_dist: str = "logitnormal"     # logitnormal | uniform
    t_mean: float = 0.4                # flow.py convention (data at t=1): mass toward the data end
    t_std: float = 1.0
    adaptive_weight: bool = True       # MeanFlow adaptive L2 on L_mf ONLY (L_div-mf is raw)
    norm_p: float = 1.0
    norm_eps: float = 0.001
    use_divergence: bool = True        # div head params exist from the start; trained only in phase 2
    div_coef: float = 1.0
    div_hutch_samples: int = 1
    # divergence_rescale: D predicts -div/rescale (raw squared loss); log q = log N(e) + rescale * D(e,0,1).
    divergence_rescale: float = 1.0
    # self_tangent: in phase 2 replace every stochastic v = a - e (JVP tangents + additive u target term)
    # with sg(u_theta(x_t,t,t)), the student's own diagonal (deterministic; diagonal L_mf becomes 0).
    self_tangent: bool = False
    # frozen_teacher: phase-2 tangent/anchor source = FROZEN end-of-phase-1 snapshot (deterministic AND
    # data-anchored -- the shortcut recipe's choice): target is snapshotted at step p1 instead of EMA-updated,
    # v -> u_phi(x_t,t,t) and the Hutchinson div anchor -> div(u_phi) in phase 2. Mutually exclusive with
    # self_tangent (whose EMA target runs away -- mf13).
    frozen_teacher: bool = False


@dataclass
class MeanFlowCQLConfig(CQLConfig):
    # MeanFlow-F2D2 policy + CQL + divergence-head log q (agents/meanflow_cql.py, mirror of shortcut_cql).
    # SINGLE stage; requires init_ckpt = the meanflow BC run's final checkpoint. NO teacher ckpt and NO
    # actor-side EMA (paper Eq 3.11-3.12: targets sg'd, div anchor = the student's own diagonal). alpha weights
    # ONLY the t==s diagonal L_mf (off-diagonal + L_div weight 1); divergence_rescale MUST match the BC run
    # (D ~ -div/rescale; log q = log N(e) + rescale*D). adaptive L2 (norm_eps, norm_p) on L_mf only.
    name: str = "meanflow_cql"
    alpha: float = 10.0                # weight of L_mf on the t==s diagonal (off-diagonal weight 1)
    flow_ratio: float = 0.5
    time_dist: str = "logitnormal"
    t_mean: float = 0.4                # flow.py convention (data at t=1): mass toward the data end
    t_std: float = 1.0
    adaptive_weight: bool = True
    norm_p: float = 1.0
    norm_eps: float = 0.001
    use_divergence: bool = True
    div_hutch_samples: int = 1
    divergence_rescale: float = 1.0    # MUST match the meanflow BC run (D ~ -div/rescale; log q = log N + rescale*D)


# ----------------------------- shared config -----------------------------
@dataclass
class Config:
    # --- identity / logging ---
    algo: str = "FQL"                  # wandb group
    exp_name: str = "fql_pusht_image"  # wandb run name
    seed: int = 86

    # --- task / data ---
    dataset_repo_id: str = "lerobot/pusht_image"
    cache_dir: str = "/data/user_data/saksham3/flow_rl/cache"
    obs_type: str = "image"            # image (ResNet vision + 2-D agent state) | state (18-D keypoints, no vision)
    state_dim: int = 2                 # overridden to KEYPOINT_DIM=18 when obs_type="state" (see get_config)
    act_dim: int = 2
    horizon: int = 16                  # action-chunk length H
    img_size: int = 96
    encoder_num: int = 1               # number of cameras
    norm_stats_path: str = "/data/user_data/saksham3/flow_rl/cache/norm_stats.json"
    norm_mode: str = "min_max"         # min_max | mean_std (state/action normalization)

    # --- model (plain transformers; vision tokens via modified ResNet) ---
    policy_type: str = "flow"          # flow | regression
    variant: str = "separate"          # separate | shared_no_stopgrad | shared_stopgrad_critic | shared_stopgrad_policy
    hidden_size: int = 512
    policy_depth: int = 5              # ~30M total incl. 13.6M ResNet
    critic_depth: int = 5
    num_critics: int = 2               # clipped double-Q (min over target critics in the TD backup)
    critic_sigmoid: bool = True        # sigmoid output on Q (bounds Q in (0,1); sparse reward => return in (0,1])
    heads: int = 8
    ff_mult: int = 4
    rgb_encoder_model: str = "resnet-18"
    pretrained_resnet: bool = True

    # --- objective (method-specific knobs) ---
    agent: AgentConfig = field(default_factory = FQLConfig)
    init_ckpt: str = ""                # warm-start model weights from this checkpoint (staged recipes)

    # --- optimization ---
    learning_rate: float = 3e-4        # shared actor+critic (FQL agents/fql.py default)
    lr_schedule: str = "warmup_constant"   # warmup_constant | constant | warmup_cosine | sqrt_decay
    warmup_steps: int = 1000           # linear LR warmup steps
    lr_decay_steps: int = 50_000       # sqrt_decay: constant this long past lr_decay_start, then 1/sqrt (F2D2 ref)
    lr_decay_start: int = 0            # sqrt_decay: step offset where the decay clock starts (e.g. phase-3 start)
    grad_clip: float = 1.0             # global-norm gradient clip (logged pre-clip)
    weight_decay: float = 0.0
    optimizer: str = "adamw"           # adamw | muon (muon: Newton-Schulz on 2-D weights, Adam on the rest)
    muon_adam_lr: float = 3e-4         # muon only: LR for the Adam-fallback params (biases / 1-D leaves)
    batch_size: int = 256
    total_steps: int = 500_000

    # --- eval / logging / ckpt ---
    eval_interval: int = 10_000
    num_eval_rollouts: int = 30
    eval_max_steps: int = 300
    eval_flow_steps: tuple = (1, 10)   # integration-step counts to evaluate (flow only; 30 rollouts each)
    sample_interval: int = 1000        # log sampling/ reconstruction losses every this many steps
    log_interval: int = 100
    save_interval: int = 50_000
    ckpt_dir: str = "/data/group_data/rl/saksham3/flow_rl/checkpoints"
    wandb_project: str = "flow_rl"
    wandb_mode: str = "online"         # online | offline | disabled


# alpha sweep for experiment 3 (5 geometric values from the FQL grid, Park et al. 2025).
ALPHA_SWEEP = (0.1, 0.3, 1.0, 3.0, 10.0)

# state-based obs: 18-D keypoints = observation.state (2, agent) + observation.environment_state (16).
KEYPOINT_DIM = 18


CONFIGS: dict[str, Config] = {
    # --- first experiment: three policies to compare ---
    # 1) BC with plain L2 regression
    "bc_l2": Config(algo = "regression", exp_name = "bc_l2_pusht", policy_type = "regression",
                    agent = FQLConfig(use_q = False), eval_flow_steps = (1,)),
    # 2) BC with flow-matching (evaluated at 1 and 10 integration steps)
    "bc_flow": Config(algo = "flow_bc", exp_name = "bc_flow_pusht", policy_type = "flow",
                      agent = FQLConfig(use_q = False), eval_flow_steps = (1, 10)),
    # 3) our FQL variant (one-step flow + Q-max)
    "fql": Config(algo = "flow_rl", exp_name = "fql_pusht", policy_type = "flow",
                  variant = "separate", agent = FQLConfig(use_q = True), eval_flow_steps = (1, 10)),

    # --- experiment 2: shared-backbone / stop-gradient variants ---
    "fql_shared": Config(algo = "FQL_shared", exp_name = "fql_shared_no_sg", variant = "shared_no_stopgrad",
                         agent = FQLConfig()),
    "fql_shared_sg_critic": Config(algo = "FQL_shared", exp_name = "fql_shared_sg_critic",
                                   variant = "shared_stopgrad_critic", agent = FQLConfig()),
    "fql_shared_sg_policy": Config(algo = "FQL_shared", exp_name = "fql_shared_sg_policy",
                                   variant = "shared_stopgrad_policy", agent = FQLConfig()),

    # --- CQL conservative critic (state-based; one-step straight-flow slogdet log q) ---
    "cql": Config(algo = "flow_rl_cql", exp_name = "cql_pusht", obs_type = "state", policy_type = "flow",
                  variant = "separate", eval_flow_steps = (1,),
                  agent = CQLConfig(straight_flow = True, alpha = 10.0)),

    # --- Shortcut-Distill-F2D2 staged pipeline, ALL 3 PHASES IN ONE RUN (100k each; no ckpt loading).
    # v_phi trains on the velocity_avg diagonal (sinusoidal time embeddings); its cache = model_100000.eqx.
    # lr: constant 1e-4 through phases 1-2, reference sqrt decay inside phase 3 (lr_decay_start = p1+p2).
    # eval k=1 -> one-step map endpoint (garbage during phase 1); k=10 -> Euler on the diagonal (v_phi). ---
    "shortcut_staged": Config(algo = "shortcut", exp_name = "shortcut_staged", obs_type = "state",
                              policy_type = "flow", variant = "separate", eval_flow_steps = (1, 10),
                              total_steps = 300_000, save_interval = 10_000,
                              learning_rate = 1e-4, lr_schedule = "sqrt_decay",
                              lr_decay_steps = 50_000, lr_decay_start = 200_000,
                              agent = ShortcutConfig(distill_f2d2 = True,
                                                     phase_steps = (100_000, 100_000, 100_000))),

    # --- Teacherless 2-phase shortcut BC (distill_f2d2=False): u map directly (diag FM + semigroup),
    # then + D head (student-anchored). 500 one-step rollouts + ckpt every 25k. ---
    "shortcut_bc_direct": Config(algo = "shortcut", exp_name = "shortcut_bc_direct", obs_type = "state",
                                 policy_type = "flow", variant = "separate", eval_flow_steps = (1,),
                                 total_steps = 400_000, save_interval = 25_000,
                                 eval_interval = 25_000, num_eval_rollouts = 500,
                                 learning_rate = 1e-4, lr_schedule = "sqrt_decay",
                                 lr_decay_steps = 50_000, lr_decay_start = 200_000,
                                 agent = ShortcutConfig(phase_steps = (200_000, 200_000))),

    # --- Shortcut-CQL: shortcut policy + CQL + divergence-head log q; single stage, needs the staged BC
    # ckpt: launch with init_ckpt=<ckpt> teacher_ckpt=<ckpt> (e.g. shortcut_staged/model_300000.eqx) ---
    "shortcut_cql": Config(algo = "shortcut_cql", exp_name = "shortcut_cql", obs_type = "state",
                           policy_type = "flow", variant = "separate", eval_flow_steps = (1,),
                           total_steps = 200_000, save_interval = 10_000,
                           agent = ShortcutCQLConfig()),

    # --- MeanFlow-CQL: launch with init_ckpt=<meanflow BC final ckpt> ---
    "meanflow_cql": Config(algo = "meanflow_cql", exp_name = "meanflow_cql", obs_type = "state",
                           policy_type = "flow", variant = "separate", eval_flow_steps = (1,),
                           total_steps = 200_000, save_interval = 10_000, agent = MeanFlowCQLConfig()),

    # --- MeanFlow-F2D2 staged BC, BOTH phases in one run (100k each; no ckpt loading) ---
    "meanflow": Config(algo = "meanflow", exp_name = "meanflow_bc", obs_type = "state",
                              policy_type = "flow", variant = "separate", eval_flow_steps = (1, 10),
                              total_steps = 200_000, save_interval = 10_000,
                              learning_rate = 1e-4, lr_schedule = "sqrt_decay",
                              lr_decay_steps = 50_000, lr_decay_start = 100_000,
                              agent = MeanFlowConfig()),

    # --- MeanFlow BC, PHASE 1 ONLY (L_mf u map, no divergence head) for 200k steps: the meanflow-map
    # counterpart of shortcut_bc_direct's phase 1. lr constant 1e-4 (decay never kicks in), 500-eval + ckpt
    # every 25k, one-step eval. Override adaptive_weight=false/true per run. ---
    "meanflow_bc_p1": Config(algo = "meanflow", exp_name = "meanflow_bc_p1", obs_type = "state",
                             policy_type = "flow", variant = "separate", eval_flow_steps = (1,),
                             total_steps = 200_000, save_interval = 25_000,
                             eval_interval = 25_000, num_eval_rollouts = 500,
                             learning_rate = 1e-4, lr_schedule = "sqrt_decay",
                             lr_decay_steps = 50_000, lr_decay_start = 200_000,
                             agent = MeanFlowConfig(phase_steps = (200_000, 0))),

    # tiny CPU smoke config
    "smoke": Config(
        algo = "smoke", exp_name = "smoke", hidden_size = 64, policy_depth = 2, critic_depth = 2,
        heads = 2, horizon = 2, batch_size = 4, total_steps = 3, num_eval_rollouts = 0,
        pretrained_resnet = False, wandb_mode = "disabled", eval_flow_steps = (1, 4),
        agent = FQLConfig(),
    ),
}


def _coerce(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, (tuple, list)):
        return tuple(int(x) for x in value.split(","))
    if current is None:
        try:
            return int(value)
        except ValueError:
            return value
    return type(current)(value)


def get_config(name: str, overrides: list[str] | None = None) -> Config:
    if name not in CONFIGS:
        raise KeyError(f"unknown config '{name}'. available: {list(CONFIGS)}")
    cfg = CONFIGS[name]
    if overrides:
        cfg_fields = {f.name for f in fields(Config)}
        agent_fields = {f.name for f in fields(type(cfg.agent))}
        top_kw, agent_kw = {}, {}
        for ov in overrides:
            k, v = ov.split("=", 1)
            if k in cfg_fields:
                top_kw[k] = _coerce(v, getattr(cfg, k))
            elif k in agent_fields:
                agent_kw[k] = _coerce(v, getattr(cfg.agent, k))
            else:
                raise KeyError(f"unknown config field '{k}' (Config or {type(cfg.agent).__name__})")
        if agent_kw:
            cfg = replace(cfg, agent = replace(cfg.agent, **agent_kw))
        if top_kw:
            cfg = replace(cfg, **top_kw)
    # state-based configs use the 18-D keypoints state (no vision encoder)
    if cfg.obs_type == "state" and cfg.state_dim != KEYPOINT_DIM:
        cfg = replace(cfg, state_dim = KEYPOINT_DIM)
    return cfg
