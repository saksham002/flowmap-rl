"""
Closed-loop Push-T evaluation: roll out the policy in gym-pusht (image obs),
executing each predicted action chunk open-loop and replanning every H steps.
Reports success rate / coverage / length; saves at most ONE success and ONE
failure rollout video per eval.

If a critic Q-function is provided (q_fn), the saved video is a side-by-side of
the Push-T frame and a Q(s, a) chart (one point per replan step) with a vertical
red bar marking the current timestep.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import jax
import jax.numpy as jnp


def _save_video(path, frames, fps = 10):
    import imageio
    imageio.mimsave(path, frames, fps = fps, codec = "libx264", quality = 8, macro_block_size = 1)


def _save_q_video(path, frames, q_steps, q_vals, max_steps, fps = 10):
    """Side-by-side: env frame | Q(s,a)-at-replan chart with a red bar at the current step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio

    ylo = min(0.0, min(q_vals) - 0.05)
    yhi = max(1.05, max(q_vals) + 0.05)
    fig, (ax_img, ax_q) = plt.subplots(1, 2, figsize = (10, 5))
    with imageio.get_writer(path, fps = fps, codec = "libx264", quality = 8, macro_block_size = 1) as w:
        for f, frame in enumerate(frames):
            ax_img.clear(); ax_img.imshow(frame); ax_img.axis("off"); ax_img.set_title("Push-T rollout")
            ax_q.clear()
            ax_q.plot(q_steps, q_vals, "-o", ms = 3, color = "tab:blue")
            ax_q.axvline(f, color = "red", lw = 2)                       # current timestep
            ax_q.set_xlim(0, max_steps); ax_q.set_ylim(ylo, yhi)
            ax_q.set_xlabel("env step"); ax_q.set_ylabel("Q(s, a)  (mean over critics)")
            ax_q.set_title("Q at replan steps")
            fig.canvas.draw()
            w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)


def evaluate(model, predict_fn, dataset, cfg, key, num_episodes: int,
             video_prefix: str | None = None, q_fn = None) -> dict:
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the env)

    use_image = cfg.obs_type == "image"
    obs_mode = "pixels_agent_pos" if use_image else "environment_state_agent_pos"
    env = gym.make("gym_pusht/PushT-v0", obs_type = obs_mode, render_mode = "rgb_array")

    successes, coverages, lengths = 0, [], []
    saved = {"success": False, "fail": False}     # at most one video of each outcome
    for ep in range(num_episodes):
        obs, info = env.reset(seed = cfg.seed + 1000 + ep)
        ep_max_cov, ep_success, steps = 0.0, False, 0
        record = video_prefix is not None and not (saved["success"] and saved["fail"])
        frames, q_steps, q_vals = [], [], []

        ep_done = False
        while steps < cfg.eval_max_steps:
            if use_image:
                img = jnp.asarray(dataset._prep_img(obs["pixels"][None]))                    # (1, K=1, 3, 96, 96)
                state = jnp.asarray(dataset.normalize_state(np.asarray(obs["agent_pos"])[None]))  # (1, 2)
            else:
                img = None                                                                  # vision encoder stripped
                kp = np.concatenate([np.asarray(obs["agent_pos"], dtype = np.float32),
                                     np.asarray(obs["environment_state"], dtype = np.float32)])[None]  # (1, 18)
                state = jnp.asarray(dataset.normalize_state(kp))
            key, sk = jax.random.split(key)
            a_norm = np.asarray(predict_fn(model, img, state, sk))[0]                        # (H, act_dim), normalized
            a_chunk = dataset.unnormalize_action(a_norm)

            # Q of the chunk about to be executed (recorded once per replan)
            if record and q_fn is not None:
                key, qk = jax.random.split(key)
                q_vals.append(float(q_fn(model, img, state, jnp.asarray(a_norm)[None], qk)[0]))
                q_steps.append(steps)

            for ai in range(cfg.horizon):
                obs, r, term, trunc, info = env.step(a_chunk[ai])
                ep_max_cov = max(ep_max_cov, float(info["coverage"]))
                ep_success = ep_success or bool(info["is_success"])
                if record:
                    frames.append(np.asarray(env.render(), dtype = np.uint8))
                steps += 1
                if term or trunc or steps >= cfg.eval_max_steps:
                    ep_done = True
                    break
            if ep_done:
                break

        successes += int(ep_success)
        coverages.append(ep_max_cov)
        lengths.append(steps)

        # save one video per outcome (success / fail)
        if record and frames:
            tag = "success" if ep_success else "fail"
            if not saved[tag]:
                os.makedirs(os.path.dirname(video_prefix), exist_ok = True)
                out = f"{video_prefix}_{tag}.mp4"
                if q_fn is not None and len(q_vals) > 0:
                    _save_q_video(out, frames, q_steps, q_vals, cfg.eval_max_steps)
                else:
                    _save_video(out, frames)
                saved[tag] = True

    env.close()

    return {
        "eval/success_rate": successes / num_episodes,
        "eval/coverage_mean": float(np.mean(coverages)),
        "eval/coverage_max": float(np.max(coverages)),
        "eval/length": float(np.mean(lengths)),
    }
