"""Build the base policy client from config: HTTP service or in-process.

`base.mode`:
  "http"  -> mile.base.dm05_client.BasePolicyClient  (OpenDM service, /v1/infer)
  "local" -> mile.base.dm05_local.LocalPolicyClient  (no HTTP, in-process)

Both expose the same surface so the controller and evaluation loop do not care:
    sample(obs, seed) -> [chunk_len, action_dim]
    sample_candidates(obs, n) -> list[[chunk_len, action_dim]]
    reset_history()
    last_latency_ms
"""
from __future__ import annotations


def build_base_client(cfg: dict):
    b = cfg["base"]
    mode = str(b.get("mode", "http")).lower()

    if mode == "local":
        from mile.base.dm05_local import build_local_policy

        return build_local_policy(cfg)

    if mode == "http":
        from mile.base.dm05_client import BasePolicyClient, DM05Config

        return BasePolicyClient(DM05Config(
            server_url=b["server_url"],
            api_path=b.get("api_path", "/v1/infer"),
            action_dim=b["action_dim"],
            chunk_len=b["chunk_len"],
            image_keys=tuple(b["image_keys"]),
            history_frames=b.get("history_frames", 20),
            history_fps=b.get("history_fps", 1),
            robot_type=b.get("robot_type"),
            control_mode=b.get("control_mode"),
            speed=b.get("speed"),
            num_diffusion_steps=b.get("num_diffusion_steps", 10),
            image_format=b.get("image_format", "JPEG"),
            image_quality=b.get("image_quality", 95),
        ))

    raise ValueError(f"base.mode must be 'http' or 'local', got {mode!r}")
