"""
CPU smoke test (no dataset download): validates model build, param counts,
train_step and predict for every agent / backbone variant, on synthetic data.

    python smoke_test.py
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from dataclasses import replace
from flow_rl.configs import (get_config, FQLConfig, CQLConfig, MeanFlowConfig, MeanFlowCQLConfig,
                             ShortcutConfig, ShortcutCQLConfig)
from flow_rl.models.flow_rl_model import VARIANTS
from flow_rl.agents import make_agent, count_params


def synthetic_batch(cfg, seed = 86):
    rng = np.random.default_rng(seed)
    B, h = cfg.batch_size, cfg.horizon
    img_shape = (B, cfg.encoder_num, 3, cfg.img_size, cfg.img_size)
    state_shape = (B, cfg.state_dim)
    return {
        "obs_img": jnp.asarray(rng.standard_normal(img_shape), dtype = jnp.float32),
        "obs_state": jnp.asarray(rng.standard_normal(state_shape), dtype = jnp.float32),
        "action": jnp.asarray(rng.uniform(-1, 1, (B, h, cfg.act_dim)), dtype = jnp.float32),
        "steps_to_end": jnp.asarray(rng.integers(0, 30, B)),
        "done": jnp.asarray((rng.random(B) > 0.8).astype(np.float32)),
        "next_obs_img": jnp.asarray(rng.standard_normal(img_shape), dtype = jnp.float32),
        "next_obs_state": jnp.asarray(rng.standard_normal(state_shape), dtype = jnp.float32),
    }


def run_variant(label, cfg):
    agent = make_agent(cfg)
    state, optimizer = agent.init_train_state(jax.random.PRNGKey(cfg.seed))
    train_step = agent.make_train_step()
    predict_fn = agent.make_predict(steps = 4)
    key = jax.random.PRNGKey(0)
    batch = synthetic_batch(cfg)

    for i in range(3):
        key, sk = jax.random.split(key)
        state, metrics = train_step(state, batch, sk)
    m = {k: float(v) for k, v in metrics.items()}
    assert all(np.isfinite(v) for v in m.values()), f"non-finite metrics: {m}"

    key, sk = jax.random.split(key)
    a = predict_fn(state.model, batch["obs_img"], batch["obs_state"], sk)
    assert a.shape == (cfg.batch_size, cfg.horizon, cfg.act_dim), a.shape
    assert np.all(np.isfinite(np.asarray(a)))
    print(f"  [{label:26s}] ok  metrics={ {k: round(v, 4) for k, v in m.items()} }")
    return state


def main():
    print("== smoke: agents + backbone variants (tiny config) ==")
    smoke = get_config("smoke")
    run_variant("regression BC", replace(smoke, policy_type = "regression", agent = FQLConfig(use_q = False)))
    run_variant("flow BC",       replace(smoke, policy_type = "flow", agent = FQLConfig(use_q = False)))
    for v in VARIANTS:
        run_variant(f"fql {v}", replace(smoke, variant = v, policy_type = "flow", agent = FQLConfig(use_q = True)))
    run_variant("cql",  replace(smoke, policy_type = "flow", agent = CQLConfig()))
    # staged BC agents (phase switching on tiny phase budgets) + the CQL agents warm-started from them
    mf_state = run_variant("meanflow", replace(smoke, policy_type = "flow", total_steps = 4,
                                                      agent = MeanFlowConfig(phase_steps = (2, 2))))
    sc_state = run_variant("shortcut_staged", replace(smoke, policy_type = "flow", total_steps = 6,
                                                      agent = ShortcutConfig(distill_f2d2 = True,
                                                                             phase_steps = (2, 2, 2))))
    run_variant("shortcut_direct", replace(smoke, policy_type = "flow", total_steps = 4,
                                           agent = ShortcutConfig(phase_steps = (2, 2))))
    with tempfile.TemporaryDirectory() as td:
        mf_ck, sc_ck = os.path.join(td, "mf.eqx"), os.path.join(td, "sc.eqx")
        eqx.tree_serialise_leaves(mf_ck, mf_state.model)
        eqx.tree_serialise_leaves(sc_ck, sc_state.model)
        run_variant("meanflow_cql", replace(smoke, policy_type = "flow", init_ckpt = mf_ck,
                                            agent = MeanFlowCQLConfig()))
        run_variant("shortcut_cql", replace(smoke, policy_type = "flow", init_ckpt = sc_ck,
                                            agent = ShortcutCQLConfig(distill_f2d2 = True,
                                                                      teacher_ckpt = sc_ck)))
        run_variant("shortcut_cql direct", replace(smoke, policy_type = "flow", init_ckpt = sc_ck,
                                                   agent = ShortcutCQLConfig()))

    print("== full-size param counts (pretrained_resnet=False) ==")
    cfg = replace(get_config("fql"), pretrained_resnet = False)
    state, _ = make_agent(cfg).init_train_state(jax.random.PRNGKey(cfg.seed))
    enc = count_params(state.model.policy_encoder)
    p_policy = enc + count_params(state.model.policy)
    p_critic = enc + count_params(state.model.critics)
    print(f"  encoder={enc/1e6:.2f}M  policy(total)={p_policy/1e6:.2f}M  critic(total)={p_critic/1e6:.2f}M")
    assert p_policy > 5e6 and p_critic > 5e6, "models unexpectedly tiny"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
