"""Client for the frozen DM0.5 base policy.

Two modes:
  - HTTP mode: talks to the OpenDM inference service (POST /v1/infer, JSON —
    the officially recommended API; legacy /process_frame is being phased out).
    Action-only; feature extraction must happen separately (prepare_latents).
  - Local mode: loads the checkpoint inside the OpenDM container and registers
    a forward hook on the VLM's second-to-last layer to capture patch tokens.

/v1/infer request (official spec, docs/en/dm05_inference.md):
    {
      "observation": {
        "prompt": str,                  # task instruction
        "state":  [floats],             # must match checkpoint norm stats
        "images": {"1": b64, "2": b64, "3": b64},   # 1-based slots, one-to-one
                                            # with --inference-config.image-prompts
                                            # (Head / Left wrist / Right wrist)
        "history_images": [b64, ...],   # oldest -> newest, only when the
                                            # service runs --data-config.is-history
        "robot_type": str,              # norm profile selector
        "control_mode": "joint",        # text conditioning
        "speed": "0.1"
      },
      "sampling": {"num_steps": 10, "seed": int}   # seed -> deterministic
    }                                               # sampling (candidates!)
Response:
    {"actions": [[floats] * chunk_len], "metadata": {"latency_ms": ...}}

Only the base policy's *outputs* are used; no base weights are trained.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np


@dataclass
class DM05Config:
    server_url: str = "http://127.0.0.1:7891"
    api_path: str = "/v1/infer"            # official JSON API (legacy: /process_frame)
    action_dim: int = 14
    chunk_len: int = 50
    # env-side camera keys; mapped to 1-based request slots "1","2","3" BY ORDER.
    # Order must match the service's --inference-config.image-prompts
    # (Head / Left wrist / Right wrist) — never reorder.
    image_keys: tuple = ("images_1", "images_2", "images_3")
    # DM05-MEM-Robodojo-Sim is memory-enabled: up to 20 head-camera history
    # frames at 1 FPS, left-padded at episode start (official protocol;
    # requires the service to run with --data-config.is-history).
    history_frames: int = 20
    history_fps: int = 1
    # robot / conditioning profile — TODO(server): confirm the exact robot_type
    # string for the ARX X5 embodiment from the RoboDojo guide
    # (docs/en/dm05_robodojo.md) or the checkpoint's norm_stats.json.
    robot_type: str | None = None
    control_mode: str | None = None       # e.g. "joint" (MEM default)
    speed: str | None = None              # e.g. "0.1" (MEM default)
    # must match the service's fixed diffusion steps (--inference-config.
    # diffusion-steps, default 10)
    num_diffusion_steps: int = 10
    # image transport
    image_format: str = "JPEG"
    image_quality: int = 95


class BasePolicyClient:
    """HTTP client for the OpenDM DM0.5 inference service (/v1/infer)."""

    def __init__(self, cfg: DM05Config) -> None:
        self.cfg = cfg
        self._history: list[np.ndarray] = []   # head-camera frames, oldest first
        self.last_latency_ms: float | None = None  # from metadata (scaling curve)

    # ------------------------------------------------------------ episode
    def reset_history(self) -> None:
        """Call at every episode start (official protocol left-pads warm-up)."""
        self._history = []

    # ------------------------------------------------------------ sampling
    def sample(self, obs: dict, seed: int | None = None) -> np.ndarray:
        """Single action chunk [chunk_len, action_dim] for the observation."""
        self._update_history(obs)
        return self._request(obs, seed)

    def sample_candidates(self, obs: dict, n: int) -> list[np.ndarray]:
        """C candidate chunks via different flow-matching noise seeds.

        sampling.seed gives deterministic sampling per the official API, so
        candidate diversity is exactly flow-matching noise diversity. Validate
        it on day 1 (W1 smoke test): same obs, n seeds -> measure pairwise
        action distance; fall back to perturbation-based candidates if flat.
        """
        self._update_history(obs)   # once per decision, not per candidate
        seeds = np.random.SeedSequence(obs.get("episode_seed", 0)).spawn(n)
        return [self._request(obs, int(s.generate_state(1)[0])) for s in seeds]

    # ------------------------------------------------------------ internal
    def _update_history(self, obs: dict) -> None:
        head_key = self.cfg.image_keys[0]
        self._history.append(obs["images"][head_key])
        self._history = self._history[-self.cfg.history_frames:]
        # TODO(server): official adapter samples history at 1 FPS; if decision
        # points are faster than 1 Hz in wall/sim time, downsample here.

    def _encode_image(self, arr: np.ndarray) -> str:
        """numpy RGB frame -> base64-encoded JPEG string."""
        from PIL import Image  # deferred: server-side dependency

        buf = io.BytesIO()
        Image.fromarray(arr).save(
            buf, format=self.cfg.image_format, quality=self.cfg.image_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _request(self, obs: dict, seed: int | None) -> np.ndarray:
        import requests  # deferred: server-side dependency

        observation: dict = {
            "prompt": obs["instruction"],
            "state": [float(x) for x in np.asarray(obs["state"]).ravel()],
            "images": {str(i + 1): self._encode_image(obs["images"][key])
                       for i, key in enumerate(self.cfg.image_keys)},
        }
        # history: oldest -> newest, only when the service accepts it
        if self._history:
            observation["history_images"] = [
                self._encode_image(f) for f in self._history]
        if self.cfg.robot_type:
            observation["robot_type"] = self.cfg.robot_type
        if self.cfg.control_mode:
            observation["control_mode"] = self.cfg.control_mode
        if self.cfg.speed:
            observation["speed"] = self.cfg.speed

        payload: dict = {"observation": observation}
        # num_steps must match the service's fixed diffusion steps; if it is
        # unset we omit it and let the service use its own default (safer than
        # guessing --inference-config.diffusion-steps wrong).
        sampling: dict = {}
        if self.cfg.num_diffusion_steps:
            sampling["num_steps"] = int(self.cfg.num_diffusion_steps)
        if seed is not None:
            sampling["seed"] = int(seed)
        if sampling:
            payload["sampling"] = sampling

        resp = requests.post(
            f"{self.cfg.server_url}{self.cfg.api_path}",
            json=payload, timeout=10,
        )
        resp.raise_for_status()
        body = resp.json()
        self.last_latency_ms = (body.get("metadata") or {}).get("latency_ms")
        actions = np.asarray(body["actions"], dtype=np.float32)
        if actions.shape != (self.cfg.chunk_len, self.cfg.action_dim):
            raise ValueError(
                f"unexpected action chunk shape {actions.shape}, expected "
                f"{(self.cfg.chunk_len, self.cfg.action_dim)}")
        return actions


class FeatureHook:
    """Capture VLM second-to-last-layer patch tokens in local mode.

    Usage (inside the OpenDM container):
        hook = FeatureHook(model.vlm, layer_index=-2)
        ... run inference ...
        z = hook.last_patch_tokens()   # [P, 2560]
    """

    def __init__(self, vlm_module, layer_index: int = -2) -> None:
        self._captured: list = []
        layers = vlm_module.model.layers   # Gemma3-style; adjust for other VLMs
        layers[layer_index].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured.append(hidden.detach().cpu())

    def reset(self) -> None:
        """Drop captured tensors — call before every batched forward."""
        self._captured.clear()

    def last_batch_patch_tokens(self) -> np.ndarray:
        """Full [B, seq, d] hidden states of the most recent forward pass.

        Use this for batched offline feature extraction (prepare_latents);
        `last_patch_tokens` only returns batch element 0.
        """
        if not self._captured:
            raise RuntimeError("no forward pass captured yet")
        return self._captured[-1].float().numpy()

    def last_patch_tokens(self) -> np.ndarray:
        if not self._captured:
            raise RuntimeError("no forward pass captured yet")
        h = self._captured[-1]                      # [1, seq, d]
        return h[0].float().numpy()
