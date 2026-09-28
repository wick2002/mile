"""Client for the frozen DM0.5 base policy.

Two modes:
  - HTTP mode: talks to the OpenDM inference service (/process_frame).
    Action-only; feature extraction must happen separately (prepare_latents).
  - Local mode: loads the checkpoint inside the OpenDM container and registers
    a forward hook on the VLM's second-to-last layer to capture patch tokens.

Only the base policy's *outputs* are used; no base weights are trained.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


@dataclass
class DM05Config:
    server_url: str = "http://127.0.0.1:7891"
    action_dim: int = 14
    chunk_len: int = 50
    image_keys: tuple = ("images_1", "images_2", "images_3")  # head / left / right wrist
    # DM05-MEM-Robodojo-Sim is memory-enabled: up to 20 head-camera history
    # frames at 1 FPS, left-padded at episode start (official protocol).
    history_frames: int = 20
    history_fps: int = 1


class BasePolicyClient:
    """HTTP client for the OpenDM DM0.5 inference service."""

    def __init__(self, cfg: DM05Config) -> None:
        self.cfg = cfg
        self._history: list[np.ndarray] = []   # head-camera frames, oldest first

    def reset_history(self) -> None:
        """Call at every episode start (official protocol left-pads warm-up)."""
        self._history = []

    def sample(self, obs: dict, seed: int | None = None) -> np.ndarray:
        """Single action chunk for the current observation."""
        self._update_history(obs)
        return self._request(obs, seed)

    def sample_candidates(self, obs: dict, n: int) -> list[np.ndarray]:
        """C candidate chunks via different flow-matching noise seeds."""
        self._update_history(obs)   # once per decision, not per candidate
        seeds = np.random.SeedSequence(obs.get("episode_seed", 0)).spawn(n)
        return [self._request(obs, int(s.generate_state(1)[0])) for s in seeds]

    def _update_history(self, obs: dict) -> None:
        head_key = self.cfg.image_keys[0]
        self._history.append(obs["images"][head_key])
        self._history = self._history[-self.cfg.history_frames:]

    def _request(self, obs: dict, seed: int | None) -> np.ndarray:
        import requests  # deferred: server-side dependency

        fields = {
            "text": obs["instruction"],
            "states": obs["state"].astype(np.float32).tobytes(),
        }
        if seed is not None:
            fields["seed"] = str(seed)
        files = {}
        for i, key in enumerate(self.cfg.image_keys):
            buf = io.BytesIO()
            np.save(buf, obs["images"][key])
            files[f"image_{i}"] = (f"{key}.npy", buf.getvalue())

        # ---- memory: head-camera history buffer (1 FPS, left-padded) ----
        # TODO(server): align history field names / sampling with the official
        # RoboDojo adapter (XPolicyLab PR #101). The service expects up to
        # history_frames head frames at history_fps; at episode start the
        # buffer is short and the server left-pads.
        for j, frame in enumerate(self._history):
            buf = io.BytesIO()
            np.save(buf, frame)
            files[f"history_{j}"] = (f"history_{j}.npy", buf.getvalue())

        resp = requests.post(
            f"{self.cfg.server_url}/process_frame",
            data=fields, files=files, timeout=10,
        )
        resp.raise_for_status()
        chunk = np.frombuffer(resp.content, dtype=np.float32)
        return chunk.reshape(self.cfg.chunk_len, self.cfg.action_dim)


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

    def last_patch_tokens(self) -> np.ndarray:
        if not self._captured:
            raise RuntimeError("no forward pass captured yet")
        h = self._captured[-1]                      # [1, seq, d]
        return h[0].float().numpy()
