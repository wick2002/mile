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
    image_keys: tuple = ("images_1", "images_2", "images_3")


class BasePolicyClient:
    """HTTP client for the OpenDM DM0.5 inference service."""

    def __init__(self, cfg: DM05Config) -> None:
        self.cfg = cfg

    def sample(self, obs: dict, seed: int | None = None) -> np.ndarray:
        """Single action chunk for the current observation."""
        return self._request(obs, seed)

    def sample_candidates(self, obs: dict, n: int) -> list[np.ndarray]:
        """C candidate chunks via different flow-matching noise seeds."""
        seeds = np.random.SeedSequence(obs.get("episode_seed", 0)).spawn(n)
        return [self._request(obs, int(s.generate_state(1)[0])) for s in seeds]

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
