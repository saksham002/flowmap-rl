"""
Offline training entrypoint.

    python flow_rl/train.py <config> [field=value ...]

<config> is a registered Config in flow_rl/configs. Logs to W&B
(project=flow_rl, group=<algo>, name=<exp_name>); evaluates every
eval_interval steps with num_eval_rollouts gym-pusht rollouts.
"""

import os
import sys
import time
from dataclasses import asdict

import numpy as np
import jax
# TF32 matmuls on Ampere/Ada (L40S): ~2-4x faster than full fp32 for negligible accuracy loss at our
# scale (Q in [0,1], CQL gap ~40). Big win for the CQL runs (many OOD-chunk Q-evals per step).
jax.config.update("jax_default_matmul_precision", "high")
import jax.numpy as jnp
import equinox as eqx

from flow_rl.configs import get_config
from flow_rl.data.pusht_dataset import PushTChunkDataset
from flow_rl.agents import make_agent, make_lr_schedule, count_params


def save_checkpoint(model, cfg, step: int) -> None:
    out_dir = os.path.join(cfg.ckpt_dir, cfg.exp_name)
    os.makedirs(out_dir, exist_ok = True)
    # tree_serialise_leaves writes the array leaves (replicated arrays gather to host fine); do NOT
    # device_put the whole module -- that would turn the Python static fields into arrays.
    eqx.tree_serialise_leaves(os.path.join(out_dir, f"model_{step}.eqx"), model)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python flow_rl/train.py <config> [field=value ...]")
    cfg = get_config(sys.argv[1], sys.argv[2:])
    print("config:", cfg)

    # fail fast if a GPU failed to initialize: SLURM may expose N GPUs (CUDA_VISIBLE_DEVICES) yet JAX
    # enumerate fewer (bad/occupied device), which would silently run single-GPU (slow + OOM-prone).
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_expected = len([x for x in cvd.split(",") if x != ""]) if cvd else jax.device_count()
    if jax.device_count() < n_expected:
        raise SystemExit(f"JAX sees {jax.device_count()} GPU(s) but CUDA_VISIBLE_DEVICES={cvd} expects "
                         f"{n_expected}; a GPU failed to init -> aborting (resubmit / exclude this node)")

    key = jax.random.PRNGKey(cfg.seed)
    key, mk = jax.random.split(key)
    agent = make_agent(cfg)
    state, optimizer = agent.init_train_state(mk)

    enc_p = count_params(state.model.policy_encoder)
    enc_c = enc_p if state.model.critic_encoder is None else count_params(state.model.critic_encoder)
    p_policy = enc_p + count_params(state.model.policy)
    p_critic = enc_c + count_params(state.model.critics)
    print(f"params: policy={p_policy/1e6:.2f}M  critic={p_critic/1e6:.2f}M  total={count_params(state.model)/1e6:.2f}M")

    dataset = PushTChunkDataset(cfg)
    train_step = agent.make_train_step()
    # one predictor per integration-step count (flow); regression / OFQL ignore steps
    predictors = {n: agent.make_predict(steps = n) for n in cfg.eval_flow_steps}
    q_fn = agent.make_q_fn() if cfg.agent.use_q else None     # critic Q(s,a) for the eval-video subplot
    lr_sched = make_lr_schedule(cfg)
    rng = np.random.default_rng(cfg.seed)

    import wandb
    group = (cfg.algo + ("-state" if cfg.obs_type == "state" else "")).replace("_", "-")   # hyphenated groups
    cfg_d = asdict(cfg)
    agent_d = cfg_d.pop("agent")
    wandb_config = {**cfg_d, **agent_d}              # flatten the agent sub-config to top-level keys
    wandb.init(
        project = cfg.wandb_project, group = group, name = cfg.exp_name,
        config = wandb_config, mode = cfg.wandb_mode,
    )

    # data-parallel across all visible GPUs: replicate params, shard each batch along axis 0.
    # 1 GPU -> trivial single shard (no-op); N GPUs -> ~N x throughput + N x activation-memory headroom
    # (per-device batch = batch_size / N), which is what lets the N=8 CQL runs fit and run faster.
    ndev = jax.device_count()
    assert cfg.batch_size % ndev == 0, f"batch_size {cfg.batch_size} must divide device count {ndev}"
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    state = eqx.filter_shard(state, repl)
    print(f"devices: {ndev} -> data-parallel, per-device batch {cfg.batch_size // ndev}", flush = True)

    # step 0: log metrics of the INITIAL (untrained) state. One train_step gives the forward metrics
    # (computed on the input state, before the update); we drop the returned state so no update is applied.
    b0 = eqx.filter_shard({k: jnp.asarray(v) for k, v in dataset.sample(cfg.batch_size, rng).items()}, data_shard)
    key, k0 = jax.random.split(key)
    _, m0 = train_step(state, b0, k0)
    log0 = {k: float(v) for k, v in m0.items()}
    log0["perf/steps_per_sec"], log0["train/lr"] = 0.0, float(lr_sched(0))
    wandb.log(log0, step = 0)
    print("[0] " + " ".join(f"{k}={v:.4g}" for k, v in sorted(log0.items())), flush = True)

    t0 = time.time()
    for step in range(1, cfg.total_steps + 1):
        batch = {k: jnp.asarray(v) for k, v in dataset.sample(cfg.batch_size, rng).items()}
        batch = eqx.filter_shard(batch, data_shard)            # split batch across GPUs (data parallel)
        key, sk = jax.random.split(key)
        state, metrics = train_step(state, batch, sk)

        if step % cfg.log_interval == 0:
            sps = step / (time.time() - t0)
            log = {k: float(v) for k, v in metrics.items()}     # keys already carry actor/ critic/ loss/ train/
            log["perf/steps_per_sec"] = sps
            log["train/lr"] = float(lr_sched(step))
            # batch/: stats of state & action exactly as passed to the networks (normalized)
            for nm, arr in (("state", batch["obs_state"]), ("action", batch["action"])):
                log[f"batch/{nm}_mean"] = float(jnp.mean(arr))
                log[f"batch/{nm}_std"] = float(jnp.std(arr))
                log[f"batch/{nm}_min"] = float(jnp.min(arr))
                log[f"batch/{nm}_max"] = float(jnp.max(arr))
            wandb.log(log, step = step)
            print(f"[{step}] " + " ".join(f"{k}={v:.4g}" for k, v in sorted(log.items())), flush = True)

        if step % cfg.sample_interval == 0:
            # sampling/: reconstruction (MSE + L1) of sampled action chunks vs the batch's
            # dataset actions, for each integration-step count in eval_flow_steps
            recon = {}
            img = batch["obs_img"] if cfg.obs_type == "image" else None   # state config carries no image
            for nsteps, pf in predictors.items():
                key, rk = jax.random.split(key)
                diff = pf(state.model, img, batch["obs_state"], rk) - batch["action"]
                recon[f"sampling/k{nsteps}/mse_loss"] = float(jnp.mean(diff ** 2))
                recon[f"sampling/k{nsteps}/l1_loss"] = float(jnp.mean(jnp.abs(diff)))
            wandb.log(recon, step = step)

        if cfg.num_eval_rollouts > 0 and step % cfg.eval_interval == 0:
            from flow_rl.envs.pusht_eval import evaluate
            # eval/: closed-loop gym-pusht rollouts at each integration-step count. Env-interleaved, so
            # gather the replicated model to one device and run eval single-device. Partition first so
            # only ARRAY leaves are device_put -- device_put-ing the whole module turns the Python static
            # fields (shared / stopgrad_*) into traced arrays, which then break `if self.stopgrad_*` under jit.
            _arr, _static = eqx.partition(state.model, eqx.is_array)
            eval_model = eqx.combine(jax.device_put(_arr, jax.devices()[0]), _static)
            for nsteps, pf in predictors.items():
                key, ek = jax.random.split(key)
                video_prefix = os.path.join(cfg.ckpt_dir, cfg.exp_name, f"rollout_{step}_k{nsteps}")
                em = evaluate(eval_model, pf, dataset, cfg, ek, cfg.num_eval_rollouts, video_prefix, q_fn = q_fn)
                tagged = {k.replace("eval/", f"eval/k{nsteps}/"): v for k, v in em.items()}
                wandb.log(tagged, step = step)
                print(f"[step {step}] k{nsteps}: {em}")

        if step % cfg.save_interval == 0:
            save_checkpoint(state.model, cfg, step)

    save_checkpoint(state.model, cfg, cfg.total_steps)
    wandb.finish()


if __name__ == "__main__":
    main()
