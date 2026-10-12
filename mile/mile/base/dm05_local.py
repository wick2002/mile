"""In-process (no HTTP) DM0.5 base policy for MILE.

WHY THIS EXISTS
---------------
The official OpenDM path is an HTTP service (`/v1/infer`). Running MILE against
it costs a base64/JPEG round-trip per decision and, more importantly, makes the
VLM's intermediate features unreachable — the WM needs the same patch tokens at
eval time that `prepare_latents.py` writes at training time.

`opendm/exp/dm05_exp.py` shows that the HTTP handler is a thin shell:

    _infer()            = request.get_json() -> _apply_v1_sampling -> _prepare_input
                          -> _predict -> jsonify
    _prepare_input(body)  takes a PLAIN DICT (no Flask)
    _predict(data)        takes the dict from _prepare_input, returns np.ndarray

So we can skip Flask entirely and call the official preprocessing + model
directly. That keeps us byte-identical to the official pipeline (tokenization,
pixel transform, normalization, history packing, de-normalization) instead of
reimplementing it and drifting.

We call `_prepare_model_input` rather than `_prepare_input` so we can pass PIL
images straight through — no base64, no JPEG round-trip, and the eval-time
latents match the training-time latents better.

USAGE (inside the OpenDM container / on the GPU server)
-------------------------------------------------------
    from mile.base.dm05_local import LocalPolicyClient, LocalDM05Config
    client = LocalDM05Config.from_cfg(cfg)   # cfg = configs/default.yaml
    client = LocalPolicyClient(client)
    chunk = client.sample(obs, seed=0)                 # [50, 14]
    cands = client.sample_candidates(obs, n=4)         # 4 x [50, 14]
    feats = client.last_features                       # [P, D] if enabled

CAVEATS
-------
* Uses private OpenDM methods (`_prepare_model_input`, `_predict`). They are
  pinned to the commit recorded in docs/DEPLOY_LOCAL_zh.md; re-verify after
  upgrading OpenDM.
* Do NOT set the env var DM05_INFERENCE_SEED: `_predict` applies it on every
  call, which would force identical noise for every candidate and kill the
  candidate-diversity mechanism MILE depends on.
* `sampling.num_steps` must equal the exp's `diffusion_steps`; we omit it and
  let the exp decide.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LocalDM05Config:
    """Everything needed to build an in-process DM0.5 policy."""

    # Path to the OpenDM *repository root* (needed to import the exp entry).
    opendm_root: str = "."
    # Exp entry file, as you would pass to --exp. For RoboDojo-Sim use the
    # official RoboDojo entry (mem SFT, 20 history frames, Aloha profile).
    exp_entry: str = "playground/dm05_mem_sft_robodojo_cover_blocks.py"
    checkpoint: str = "./checkpoints/DM05-MEM-Robodojo-Sim"

    chunk_len: int = 50
    action_dim: int = 14
    image_keys: tuple = ("images_1", "images_2", "images_3")
    history_frames: int = 20
    history_fps: int = 1

    # Model-side robot profile. RoboDojo is a 14-DoF BIMANUAL ARX X5, whose
    # state description is joint x6 + gripper + joint x6 + gripper — i.e.
    # opendm.constants.robot.RobotType.ALOHA ("Aloha"), NOT RobotType.ARX5
    # ("ARX5"), which is a 7-DoF SINGLE arm. This is what the official
    # RoboDojo deploy.yml uses.
    robot_type: str | None = "Aloha"
    control_mode: str | None = "joint"
    speed: str | None = None          # None -> exp default ("0.5" upstream)

    # Capture VLM patch tokens for the world model (see `last_features`).
    enable_features: bool = False
    feature_layer: int = -2
    num_patches: int = 256
    vlm_dim: int = 2560

    device: str = "cuda"
    # Extra launcher flags forwarded verbatim to tyro (e.g. precision policy).
    extra_args: list[str] = field(default_factory=list)

    # ---- local-mode knobs mirrored from configs/default.yaml ----
    dataset_name: str | None = None
    action_mode: str | None = "absolute"
    add_state: bool | None = None
    is_history: bool = True
    max_history_images: int = 20      # OpenDM default is 5 -> must be 20 here
    model_max_length: int = 1536      # deploy.yml value; OpenDM default is 1024
    backend: str = "default"          # "fast" (TensorRT) would break the hook
    precision_policy: str | None = "bf16_mixed"
    output_action_dim: int | None = None

    @staticmethod
    def from_cfg(cfg: dict) -> "LocalDM05Config":
        b = cfg["base"]
        return LocalDM05Config(
            opendm_root=b.get("opendm_root", "."),
            exp_entry=b.get("exp_entry", LocalDM05Config.exp_entry),
            checkpoint=b.get("checkpoint", LocalDM05Config.checkpoint),
            chunk_len=b["chunk_len"],
            action_dim=b["action_dim"],
            image_keys=tuple(b["image_keys"]),
            history_frames=b.get("history_frames", 20),
            history_fps=b.get("history_fps", 1),
            robot_type=b.get("robot_type"),
            control_mode=b.get("control_mode"),
            speed=b.get("speed"),
            enable_features=bool(b.get("enable_features", False)),
            feature_layer=int(b.get("feature_layer", -2)),
            num_patches=int(b["num_patches"]),
            vlm_dim=int(b["vlm_dim"]),
            dataset_name=b.get("dataset_name"),
            action_mode=b.get("action_mode"),
            add_state=b.get("add_state"),
            is_history=bool(b.get("is_history", True)),
            max_history_images=int(b.get("max_history_images", 20)),
            model_max_length=int(b.get("model_max_length", 1536)),
            backend=str(b.get("backend", "default")),
            precision_policy=b.get("precision_policy"),
            output_action_dim=b.get("output_action_dim"),
        )


def build_local_policy(cfg: dict) -> "LocalPolicyClient":
    """Convenience entry point used by the scripts: cfg -> in-process client."""
    return LocalPolicyClient(LocalDM05Config.from_cfg(cfg))


class LocalPolicyClient:
    """In-process DM0.5 policy. Same surface as `BasePolicyClient`."""

    def __init__(self, cfg: LocalDM05Config) -> None:
        self.cfg = cfg
        self._history: list[np.ndarray] = []
        self.last_latency_ms: float | None = None
        self.last_features: np.ndarray | None = None

        self.exp, self.ic = self._build_exp()
        self._hook = None
        if cfg.enable_features:
            self._hook = self._register_hook()

    # ------------------------------------------------------------- build
    def _build_exp(self):
        """Import the exp entry by path and initialise the inference runtime."""
        import importlib.util
        import sys

        import tyro                                   # server-side dependency

        root = Path(self.cfg.opendm_root).expanduser().resolve()
        entry = Path(self.cfg.exp_entry)
        entry = entry if entry.is_absolute() else root / entry
        if not entry.exists():
            raise FileNotFoundError(f"exp entry not found: {entry}")

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        spec = importlib.util.spec_from_file_location("mile_dm05_exp_entry", entry)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        exp_cls = getattr(mod, "DM05Exp")

        argv = [
            "--model-config.model-name-or-path", str(self.cfg.checkpoint),
            "--model-config.chunk-size", str(self.cfg.chunk_len),
            "--inference-config.output-action-dim", str(self.cfg.action_dim),
            "--inference-config.image-prompts", "Head", "Left wrist", "Right wrist",
            "--inference-config.backend", str(self.cfg.backend),
            "--inference-config.max-history-images",
            str(self.cfg.max_history_images),
            "--model-config.model-max-length", str(self.cfg.model_max_length),
        ]
        # MEM checkpoint: history frames are mandatory, and the flag name is
        # --data-config.is-history (a store_true-style option on the data config).
        if self.cfg.is_history:
            argv.append("--data-config.is-history")
        if self.cfg.dataset_name:
            argv += ["--data-config.dataset-name", str(self.cfg.dataset_name)]
        if self.cfg.action_mode:
            argv += ["--data-config.action-mode", str(self.cfg.action_mode)]
        if self.cfg.add_state is not None:
            argv += ["--data-config.add-state", str(bool(self.cfg.add_state))]
        if self.cfg.precision_policy:
            argv += ["--model-config.precision-policy", str(self.cfg.precision_policy)]
        if self.cfg.extra_args:
            argv += list(self.cfg.extra_args)

        exp = tyro.cli(exp_cls, args=argv)
        # NB: exp.inference() would start Flask; we only want the runtime.
        exp._initialize_inference_runtime()
        return exp, exp.inference_config

    def _register_hook(self):
        """Hook the VLM's second-to-last layer to capture patch tokens."""
        from mile.base.dm05_client import FeatureHook

        backbone = self._resolve_vlm_backbone()
        return FeatureHook(backbone, layer_index=self.cfg.feature_layer)

    def _resolve_vlm_backbone(self):
        """Find the module exposing `.model.layers` (what FeatureHook taps).

        Probed against the DM05 layout: dm05.model.vlm -> .model.layers
        (see opendm/exp/dm05_exp.py: `dm05_model.model.vlm.model`).
        """
        model = self.ic.model
        candidates = []
        try:
            from opendm.exp.dm05_exp import unwrap_dm05_model
            model = unwrap_dm05_model(model)
        except Exception:                              # noqa: BLE001
            pass
        for path in ("model.vlm", "vlm", "language_model", "model"):
            obj = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "model") and hasattr(obj.model, "layers"):
                return obj
            candidates.append(path)
        raise RuntimeError(
            "could not locate the VLM backbone exposing `.model.layers`; tried "
            f"{candidates}. Set base.enable_features=false and use the offline "
            "prepare_latents.py path, or extend _resolve_vlm_backbone().")

    def current_patch_tokens(self) -> np.ndarray:
        """[P, D] patch tokens captured during the most recent forward.

        Zero extra cost: they come from the forward we already ran for
        sampling. Requires ``base.enable_features: true``.
        """
        if self.last_features is None:
            raise RuntimeError(
                "no patch tokens captured — set base.enable_features: true "
                "(local mode needs them: the controller reads obs['patch_tokens'] "
                "at every verification point)")
        return self.last_features

    def encode(self, obs: dict) -> np.ndarray:
        """[P, D] patch tokens for one observation.

        WARNING: this costs a FULL model forward (VLM + action expert), so do
        not call it per control step — throttle with ``eval.record_stride``
        (default 5). Prefer `current_patch_tokens()`, which is free.
        """
        self._predict(obs, seed=None)
        return self.current_patch_tokens()

    # ---------------------------------------------------------- episode
    def reset_history(self) -> None:
        self._history = []

    def _update_history(self, obs: dict) -> None:
        head_key = self.cfg.image_keys[0]
        self._history.append(obs["images"][head_key])
        self._history = self._history[-self.cfg.history_frames:]

    # --------------------------------------------------------- sampling
    def sample(self, obs: dict, seed: int | None = None) -> np.ndarray:
        self._update_history(obs)
        return self._predict(obs, seed)

    def sample_candidates(self, obs: dict, n: int) -> list[np.ndarray]:
        """C chunks via different flow-matching noise seeds.

        `_apply_v1_sampling({"seed": s})` sets np.random + torch + CUDA seeds,
        which is exactly how the official service makes sampling deterministic.
        Verify diversity on day 1 (W1 smoke test): same obs, n seeds -> measure
        pairwise action distance. If flat, fall back to perturbation candidates.
        """
        self._update_history(obs)          # once per decision, not per candidate
        seeds = np.random.SeedSequence(obs.get("episode_seed", 0)).spawn(n)
        return [self._predict(obs, int(s.generate_state(1)[0])) for s in seeds]

    # ---------------------------------------------------------- internal
    @staticmethod
    def _to_pil(arr: np.ndarray):
        from PIL import Image                          # server-side dependency

        return Image.fromarray(np.asarray(arr)).convert("RGB")

    def _build_data(self, obs: dict) -> dict:
        pil_images = [self._to_pil(obs["images"][k]) for k in self.cfg.image_keys]
        pil_history = [self._to_pil(f) for f in self._history]
        state = np.asarray(obs["state"], dtype=np.float32).ravel()
        return self.ic._prepare_model_input(
            text=obs.get("instruction", ""),
            images=pil_images,
            states=state.tolist(),
            robot_type=self.cfg.robot_type,
            speed=self.cfg.speed,
            control_mode=self.cfg.control_mode,
            history_images=pil_history or None,
        )

    def _predict(self, obs: dict, seed: int | None) -> np.ndarray:
        data = self._build_data(obs)
        if seed is not None:
            self.ic._apply_v1_sampling({"seed": int(seed)})
        if self._hook is not None:
            self._hook.reset()

        t0 = time.perf_counter()
        actions = self.ic._predict(data)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0

        if self._hook is not None:
            hidden = self._hook.last_patch_tokens()     # [seq, D]
            if hidden.shape[0] < self.cfg.num_patches:
                raise RuntimeError(
                    f"captured {hidden.shape[0]} positions < num_patches="
                    f"{self.cfg.num_patches}")
            self.last_features = hidden[: self.cfg.num_patches]

        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.cfg.chunk_len, self.cfg.action_dim):
            raise ValueError(
                f"unexpected action chunk shape {actions.shape}, expected "
                f"{(self.cfg.chunk_len, self.cfg.action_dim)}")
        return actions
