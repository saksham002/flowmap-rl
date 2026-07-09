"""
Push-T offline dataset for chunked flow-RL.

Cache build (raw per-frame; NO history stacking, NO action chunking here):
  - images   come from `lerobot/pusht_image`, after de-duplicating (episode_index,
    frame_index) pairs;
  - all other fields (state, action, episode/frame index) come from
    `lerobot/pusht_keypoints` (the image dataset's reward is known-broken),
    aligned to the image frames by (episode_index, frame_index).

Loader (chunking + RL fields built here):
  - every frame is a valid chunk start;
  - the action chunk a_{t:t+h} clamps indices to the episode terminal, i.e. it
    REPEATS the terminal action past the episode boundary (no masking);
  - sparse reward: 1 on the transition entering the episode terminal state;
  - chunked TD: done / next-state clamp at the terminal (Q in [0, 1]).
"""

import os
import numpy as np

from flow_rl.data.norm import Normalizer, build_and_save

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype = np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype = np.float32).reshape(1, 3, 1, 1)
_KEY_MULT = 100_000   # frame_index < this; (episode, frame) -> episode*MULT + frame


def _to_chw_uint8(im) -> np.ndarray:
    """Normalize a LeRobot image (CHW or HWC, float[0,1] or uint8) to CHW uint8."""
    arr = np.asarray(im)
    if arr.dtype.kind == "f":
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.shape[0] != 3 and arr.shape[-1] == 3:   # HWC -> CHW
        arr = arr.transpose(2, 0, 1)
    return arr


def _dedup_first(keys: np.ndarray):
    """Indices of first occurrence of each unique key, in original order."""
    _, idx = np.unique(keys, return_index = True)
    return np.sort(idx)


def _build_cache(cache_path: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("[pusht] building cache: keypoints fields + pusht_image images ...")
    # --- canonical fields from keypoints (dedup (ep, frame)) ---
    kp = LeRobotDataset("lerobot/pusht_keypoints").hf_dataset
    kp_ep = np.asarray(kp["episode_index"], dtype = np.int64).reshape(-1)
    kp_fr = np.asarray(kp["frame_index"], dtype = np.int64).reshape(-1)
    states = np.asarray(kp["observation.state"], dtype = np.float32)             # (N, 2) agent pos
    env_state = np.asarray(kp["observation.environment_state"], dtype = np.float32)  # (N, 16) T-block keypoints
    actions = np.asarray(kp["action"], dtype = np.float32)
    kp_key = kp_ep * _KEY_MULT + kp_fr
    keep = _dedup_first(kp_key)
    kp_ep, kp_fr, states, env_state, actions, kp_key = (
        a[keep] for a in (kp_ep, kp_fr, states, env_state, actions, kp_key))
    keypoints = np.concatenate([states, env_state], axis = 1)                    # (N, 18) state-based obs

    # --- images from pusht_image (dedup (ep, frame), align to keypoints) ---
    im = LeRobotDataset("lerobot/pusht_image").hf_dataset
    im_ep = np.asarray(im["episode_index"], dtype = np.int64).reshape(-1)
    im_fr = np.asarray(im["frame_index"], dtype = np.int64).reshape(-1)
    im_key = im_ep * _KEY_MULT + im_fr
    im_keep = _dedup_first(im_key)
    im_sub = im.select([int(i) for i in im_keep])
    sub_keys = im_key[im_keep]
    key_to_pos = {int(k): p for p, k in enumerate(sub_keys)}
    missing = [int(k) for k in kp_key if int(k) not in key_to_pos]
    assert not missing, f"{len(missing)} keypoints frames have no matching image"
    sub_images = im_sub["observation.image"]
    images = np.stack([_to_chw_uint8(sub_images[key_to_pos[int(k)]]) for k in kp_key])

    os.makedirs(os.path.dirname(cache_path), exist_ok = True)
    np.savez(cache_path, images = images, states = states, keypoints = keypoints, actions = actions,
             episode_index = kp_ep, frame_index = kp_fr)
    print(f"[pusht] cached {len(states)} frames "
          f"(image dedup {len(im_keep)}/{len(im_ep)}, keypoints dedup {len(keep)}) -> {cache_path}")


class PushTChunkDataset:
    def __init__(self, cfg):
        self.h = cfg.horizon
        cache_path = os.path.join(cfg.cache_dir, "pusht_merged.npz")
        if not os.path.exists(cache_path):
            _build_cache(cache_path)
        data = np.load(cache_path)
        self.use_image = cfg.obs_type == "image"
        self.images = data["images"]                          # (N, 3, 96, 96) uint8
        self.actions = data["actions"].astype(np.float32)     # (N, 2) raw
        # obs state: 2-D agent pos (image config) or 18-D keypoints (state config)
        if cfg.obs_type == "state":
            assert "keypoints" in data.files, "cache lacks 'keypoints'; rebuild pusht_merged.npz"
            self.state_raw = data["keypoints"].astype(np.float32)   # (N, 18)
        else:
            self.state_raw = data["states"].astype(np.float32)      # (N, 2)
        ep_idx = data["episode_index"]                        # (N,)
        N = len(self.state_raw)

        # modular normalization: per-obs_type stats json (2-D vs 18-D state), computed once
        base, ext = os.path.splitext(cfg.norm_stats_path)
        norm_path = f"{base}_{cfg.obs_type}{ext}"
        if not os.path.exists(norm_path):
            build_and_save(norm_path, self.state_raw, self.actions)
        self.normalizer = Normalizer.from_json(norm_path, cfg.norm_mode)

        # per-frame terminal index (last frame of the frame's episode); every frame is a start
        self.ep_end = np.empty(N, dtype = np.int64)
        for ep in np.unique(ep_idx):
            idx = np.where(ep_idx == ep)[0]
            self.ep_end[idx] = idx[-1]
        self.offsets = np.arange(self.h, dtype = np.int64)
        self.N = N
        print(f"[pusht] {N} frames, {len(np.unique(ep_idx))} episodes "
              f"(all frames are chunk starts, H={self.h}); norm={cfg.norm_mode}; "
              f"reward=gamma^steps_to_end (terminal reward 1)")

    # eval-time helpers (delegate to the normalizer transform)
    def normalize_state(self, s: np.ndarray) -> np.ndarray:
        return self.normalizer.normalize(s, "state")

    def unnormalize_action(self, a: np.ndarray) -> np.ndarray:
        return self.normalizer.unnormalize(a, "action")

    def _prep_img(self, img_uint8: np.ndarray) -> np.ndarray:
        # (B, 3, 96, 96) CHW or (B, 96, 96, 3) HWC uint8 -> (B, K=1, 3, 96, 96) ImageNet-normalized
        x = img_uint8.astype(np.float32) / 255.0
        if x.shape[-1] == 3:                                  # HWC (gym-pusht obs) -> CHW
            x = x.transpose(0, 3, 1, 2)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x[:, None]                                     # (B, 1, 3, 96, 96)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict:
        t = rng.integers(0, self.N, size = batch_size)        # every frame is a valid start
        ee = self.ep_end[t]                                   # (B,) terminal state index
        steps_to_end = (ee - t).astype(np.int64)              # ep_len - 1 - t_local
        chunk = np.minimum(t[:, None] + self.offsets, ee[:, None])   # clamp -> repeat terminal action
        done = (t + self.h > ee).astype(np.float32)           # t_local + H >= ep_len  <=>  t + H > ep_end
        nxt = np.minimum(t + self.h, ee)                      # next state, clamped

        out = {
            "obs_state": self.normalizer.normalize(self.state_raw[t], "state"),    # (B, state_dim)
            "action": self.normalizer.normalize(self.actions[chunk], "action"),    # (B, H, act_dim)
            "steps_to_end": steps_to_end,                                          # (B,)
            "done": done,                                                          # (B,)
            "next_obs_state": self.normalizer.normalize(self.state_raw[nxt], "state"),
        }
        if self.use_image:                                                         # state config carries no images
            out["obs_img"] = self._prep_img(self.images[t])                        # (B, 1, 3, 96, 96)
            out["next_obs_img"] = self._prep_img(self.images[nxt])
        return out
