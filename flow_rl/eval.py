"""
Standalone evaluation from a checkpoint.

    python flow_rl/eval.py <config> <ckpt.eqx> [field=value ...]
"""

import sys
import os
import jax
import equinox as eqx

from flow_rl.configs import get_config
from flow_rl.data.pusht_dataset import PushTChunkDataset
from flow_rl.agents import make_agent


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: python flow_rl/eval.py <config> <ckpt.eqx> [field=value ...]")
    name, ckpt = sys.argv[1], sys.argv[2]
    cfg = get_config(name, sys.argv[3:])

    agent = make_agent(cfg)
    state, _ = agent.init_train_state(jax.random.PRNGKey(cfg.seed))
    model = eqx.tree_deserialise_leaves(ckpt, state.model)

    dataset = PushTChunkDataset(cfg)
    from flow_rl.envs.pusht_eval import evaluate
    n = max(cfg.num_eval_rollouts, 30)
    q_fn = agent.make_q_fn() if cfg.agent.use_q else None
    for nsteps in cfg.eval_flow_steps:
        predict_fn = agent.make_predict(steps = nsteps)
        video_prefix = os.path.join(cfg.ckpt_dir, cfg.exp_name, f"eval_rollout_k{nsteps}")
        metrics = evaluate(model, predict_fn, dataset, cfg, jax.random.PRNGKey(cfg.seed), n, video_prefix, q_fn = q_fn)
        print(f"k{nsteps}: {metrics}")


if __name__ == "__main__":
    main()
