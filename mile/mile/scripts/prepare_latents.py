"""Prepare world-model training data: raw demos -> per-trajectory .npz (stage S2 prerequisite).

WHAT THIS PRODUCES
------------------
One flat directory of ``.npz`` files (``--out-dir``), one per demonstration,
each with exactly the keys ``mile/wm/data.py`` expects:

    latents    : [T, P, vlm_dim]   VLM patch features, one per control step
    actions    : [T, action_dim]
    milestones : [K] int           frame indices of milestone boundaries
    text_emb   : [K+1, text_tokens, text_dim]   stage-descriptor embeddings

``SlidingWindowDataset`` globs ``*.npz`` NON-recursively, so files are written
flat as ``<task_id>_<idx:06d>.npz`` — do not put them in per-task subfolders.

WHY IT MUST RUN INSIDE THE OPENDM CONTAINER
-------------------------------------------
``POST /v1/infer`` returns actions only; patch features are not exposed over
HTTP. Latents therefore have to be captured in-process with
``mile.base.dm05_client.FeatureHook`` on the VLM's second-to-last layer
(``configs/default.yaml: base.feature_layer = -2``), taking the first
``base.num_patches`` (256) sequence positions — see W1 smoke-test item 4 in
``docs/SETUP_zh.md``.

DISK COST — READ BEFORE RUNNING THE FULL 3,500 DEMOS
----------------------------------------------------
Latents dominate: T x P x vlm_dim x dtype. With the defaults
(T~300, P=256, D=2560, float16) that is ~375 MiB per demo, i.e. **~1.3 TiB for
all 3,500**. Do NOT stride frames to shrink this: the WM imagination window must
stay exactly ``exec_horizon`` (25) *control steps*, and striding would silently
redefine 25 stored steps as 25x real ones. Use ``--estimate-only`` first, then
either (a) pilot on a subset via ``--limit-per-task``, or (b) provision the
disk. Run ``--estimate-only`` to see the projection for your own data.

USAGE
-----
    # 1. cost projection (no GPU needed)
    python -m mile.scripts.prepare_latents --demo-dir /data/demos --estimate-only

    # 2. real run, inside the OpenDM container
    python -m mile.scripts.prepare_latents \
        --demo-dir /data/demos --out-dir data/robodojo_demos_latents \
        --model-path ./checkpoints/DM05-MEM-Robodojo-Sim --batch-size 32

    # 3. smoke test on one container / 2 demos per task
    python -m mile.scripts.prepare_latents --demo-dir /data/demos \
        --out-dir data/smoke --model-path ... --tasks task_a --limit-per-task 2

If the OpenDM internals do not match the attribute paths probed by
``LocalVLMFeatureBackend``, pass your own with ``--adapter pkg.mod:ClassName``
(subclass ``FeatureBackend``) instead of editing this file.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------- config
class PrepConfig:
    """Resolved settings shared by the extractor and the writer."""

    def __init__(self, cfg: dict, args) -> None:
        b = cfg["base"]
        w = cfg["world_model"]
        self.num_patches: int = int(b["num_patches"])
        self.vlm_dim: int = int(b["vlm_dim"])
        self.action_dim: int = int(b["action_dim"])
        self.text_tokens: int = int(w["text_tokens"])
        self.text_dim: int = int(b["vlm_dim"])
        self.image_keys: tuple = tuple(b["image_keys"])
        self.feature_layer: int = int(b.get("feature_layer", -2))
        self.template: str = args.stage_text_template or w.get(
            "stage_text_template", "{task}: stage {i}")
        # WM window == exec_horizon == 25 control steps; never subsample frames.
        self.chunk_len: int = int(b.get("exec_horizon", b["chunk_len"]))


# ------------------------------------------------------------------- backends
class FeatureBackend:
    """Interface for turning frames / stage texts into arrays.

    Subclass and pass via ``--adapter pkg.mod:ClassName`` if the OpenDM
    container layout differs from what ``LocalVLMFeatureBackend`` probes.
    """

    def encode_frames(self, frames: list[list[np.ndarray]]) -> np.ndarray:
        """frames: [B][C] RGB arrays -> [B, P, D] patch tokens."""
        raise NotImplementedError

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """[N] stage strings -> [N, text_tokens, text_dim]."""
        raise NotImplementedError

    def close(self) -> None:
        pass


def segment_pool(hidden: np.ndarray, n_tokens: int) -> np.ndarray:
    """[S, D] (or [B, S, D]) -> [n_tokens, D] (or [B, n_tokens, D]).

    Contiguous-segment mean pooling: a training-free way to turn a variable-
    length text encoding into a fixed number of stage tokens. Deterministic, so
    offline (this script) and online (server) agree as long as both use it.
    """
    S = hidden.shape[-2]
    if n_tokens <= 1:
        return hidden.mean(axis=-2, keepdims=True)
    edges = np.linspace(0, S, n_tokens + 1).astype(int)
    return np.stack(
        [hidden[..., edges[i]:max(edges[i + 1], edges[i] + 1), :].mean(axis=-2)
         for i in range(n_tokens)],
        axis=-2,
    )


class LocalVLMFeatureBackend(FeatureBackend):
    """In-process VLM feature extraction via ``FeatureHook``.

    Probes a few known attribute paths for the language-model backbone that
    owns ``.model.layers`` (which is what ``FeatureHook`` taps).
    """

    CANDIDATES = ("vlm", "language_model", "model")

    def __init__(self, model_path: str, cfg: PrepConfig, device: str = "cuda",
                 dtype: str = "bf16", frame_prompt: str = "") -> None:
        import torch                                    # server-side dependency
        from transformers import AutoModel, AutoProcessor  # server-side

        self.torch = torch
        self.cfg = cfg
        self.device = device
        self.frame_prompt = frame_prompt

        self.model = AutoModel.from_pretrained(model_path, torch_dtype=(
            torch.bfloat16 if dtype == "bf16" else torch.float32))
        self.model.to(device).eval()

        self.backbone = self._resolve_backbone(self.model)
        self.processor = AutoProcessor.from_pretrained(model_path)

        from mile.base.dm05_client import FeatureHook
        self.hook = FeatureHook(self.backbone, layer_index=cfg.feature_layer)

    def _resolve_backbone(self, model):
        for name in self.CANDIDATES:
            cand = getattr(model, name, None)
            if cand is not None and hasattr(cand, "model") \
                    and hasattr(cand.model, "layers"):
                return cand
        raise RuntimeError(
            "could not locate the language-model backbone (a module exposing "
            f".model.layers) on the loaded model; tried {list(self.CANDIDATES)}. "
            "Pass --adapter pkg.mod:ClassName with your own FeatureBackend, or "
            "extend LocalVLMFeatureBackend.CANDIDATES."
        )

    def encode_frames(self, frames: list[list[np.ndarray]]) -> np.ndarray:
        """frames: [B][C] RGB arrays (camera order = cfg.image_keys)."""
        torch = self.torch
        from PIL import Image

        pil_batch = [[Image.fromarray(c) for c in per_sample] for per_sample in frames]
        prompt = self.frame_prompt or ""
        try:
            inputs = self.processor(images=pil_batch, text=[prompt] * len(frames),
                                    return_tensors="pt", padding=True)
        except Exception as exc:                       # noqa: BLE001
            raise RuntimeError(
                "the processor rejected a batch of multi-image samples; adapt "
                "LocalVLMFeatureBackend.encode_frames or supply --adapter. "
                f"original error: {exc!r}") from None
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.hook.reset()
        with torch.no_grad():
            self.backbone(**inputs)
        hidden = self.hook.last_batch_patch_tokens()    # [B, seq, D]

        B = hidden.shape[0]
        P, D = self.cfg.num_patches, self.cfg.vlm_dim
        if hidden.shape[1] < P:
            raise RuntimeError(
                f"captured sequence has {hidden.shape[1]} positions but "
                f"num_patches={P}; check base.num_patches / feature_layer")
        if hidden.shape[2] != D:
            raise RuntimeError(
                f"captured hidden dim {hidden.shape[2]} != base.vlm_dim {D}")
        return np.ascontiguousarray(hidden[:B, :P, :])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        enc = self.processor(text=texts, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        self.hook.reset()
        with torch.no_grad():
            self.backbone(**enc)
        hidden = self.hook.last_batch_patch_tokens()    # [N, seq, D]
        return np.ascontiguousarray(
            segment_pool(hidden, self.cfg.text_tokens)).astype(np.float32)


def load_backend(args, cfg: PrepConfig) -> FeatureBackend:
    if args.adapter:
        mod_name, cls_name = args.adapter.split(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls(args.model_path, cfg, device=args.device, dtype=args.dtype,
                   frame_prompt=args.frame_prompt)
    return LocalVLMFeatureBackend(args.model_path, cfg, device=args.device,
                                  dtype=args.dtype, frame_prompt=args.frame_prompt)


# --------------------------------------------------------------- demo loading
def load_demo(path: Path, cfg: PrepConfig) -> dict:
    """Read one raw demonstration. Adapt here if RoboDojo ships another format.

    Accepted keys inside the .npz (first match wins):
      frames : stacked [T, C, H, W, 3]
      or the per-camera arrays named by ``base.image_keys``, each [T, H, W, 3]
      actions     : [T, action_dim]                    (required)
      official_substeps : [K] int                      (v1 milestone source)
    """
    with np.load(path, allow_pickle=True) as z:
        if "frames" in z.files:
            frames = z["frames"]                       # [T, C, H, W, 3]
            per_step = [list(frames[t]) for t in range(frames.shape[0])]
        else:
            cams = [z[k] for k in cfg.image_keys if k in z.files]
            if not cams:
                raise KeyError(
                    f"{path}: need 'frames' or {list(cfg.image_keys)}; got {z.files}")
            per_step = [[c[t] for c in cams] for t in range(cams[0].shape[0])]

        if "actions" not in z.files:
            raise KeyError(f"{path}: missing 'actions'")
        actions = np.asarray(z["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != cfg.action_dim:
            raise ValueError(
                f"{path}: actions {actions.shape} != [T, {cfg.action_dim}]")

        substeps = (np.asarray(z["official_substeps"], dtype=np.int64).tolist()
                    if "official_substeps" in z.files else None)
    return {"frames": per_step, "actions": actions, "official_substeps": substeps}


def resolve_milestones(demo: dict, T: int, args) -> np.ndarray:
    """Milestone frame indices. v1 = official RoboDojo sub-step boundaries."""
    ms = demo["official_substeps"]
    if ms:
        idx = sorted(int(m) for m in ms if 0 <= int(m) < T)
        if idx:
            return np.asarray(idx, dtype=np.int64)
    if args.k_fallback is None:
        raise ValueError(
            "demo has no usable 'official_substeps' and --k-fallback was not "
            "given; refusing to invent milestone positions")
    k = int(args.k_fallback)
    return np.linspace(0, T - 1, k + 2).astype(np.int64)[1:-1]


def stage_texts(task_id: str, k: int, template: str) -> list[str]:
    """K milestones -> K+1 stage descriptors (stage i leads to milestone i)."""
    return [template.format(task=task_id, i=i + 1) for i in range(k + 1)]


# ---------------------------------------------------------------------- write
def process_demo(task_id: str, idx: int, path: Path, backend: FeatureBackend,
                 cfg: PrepConfig, args, out_dir: Path,
                 text_cache: dict) -> dict:
    demo = load_demo(path, cfg)
    T = len(demo["frames"])
    if T <= cfg.chunk_len:
        raise ValueError(
            f"{path}: T={T} <= chunk_len={cfg.chunk_len}; too short for a window")

    latents = np.empty((T, cfg.num_patches, cfg.vlm_dim),
                       dtype=np.float16 if args.dtype == "fp16" else np.float32)
    bs = max(1, args.batch_size)
    for start in range(0, T, bs):
        batch = demo["frames"][start:start + bs]
        latents[start:start + len(batch)] = backend.encode_frames(batch)

    milestones = resolve_milestones(demo, T, args)
    texts = stage_texts(task_id, len(milestones), cfg.template)
    missing = [t for t in texts if t not in text_cache]
    if missing:
        emb = backend.encode_texts(missing)
        for t, e in zip(missing, emb):
            text_cache[t] = e
    text_emb = np.stack([text_cache[t] for t in texts])   # [K+1, Tt, Dt]

    out = out_dir / f"{task_id}_{idx:06d}.npz"
    np.savez_compressed(
        out,
        latents=latents,
        actions=demo["actions"],
        milestones=milestones,
        text_emb=text_emb,
        task_id=np.array(task_id),
    )
    return {"file": out.name, "task": task_id, "T": T, "K": len(milestones),
            "bytes": out.stat().st_size}


def estimate(demo_dir: Path, cfg: PrepConfig, args) -> None:
    """Project disk cost without loading the VLM."""
    tasks = pick_tasks(demo_dir, args.tasks)
    total_T = 0
    n = 0
    for task in tasks:
        for f in sorted(task.glob("*.npz"))[:args.limit_per_task or None]:
            with np.load(f) as z:
                if "frames" in z.files:
                    T = int(z["frames"].shape[0])
                else:
                    T = int(z[cfg.image_keys[0]].shape[0])
            total_T += T
            n += 1
    per_frame = cfg.num_patches * cfg.vlm_dim * (2 if args.dtype == "fp16" else 4)
    total = total_T * per_frame
    print(f"[estimate] demos={n} frames={total_T} "
          f"per_frame={per_frame / 1024:.0f} KiB ({args.dtype})")
    print(f"[estimate] projected latents size ~= {total / 2**30:.1f} GiB "
          f"(plus actions/text, compression will reduce the actual file size)")
    if args.dtype != "fp16":
        print("[estimate] hint: --dtype fp16 halves this")


def pick_tasks(demo_dir: Path, task_filter: list[str] | None) -> list[Path]:
    dirs = sorted(d for d in demo_dir.iterdir() if d.is_dir())
    if task_filter:
        wanted = set(task_filter)
        dirs = [d for d in dirs if d.name in wanted]
        missing = wanted - {d.name for d in dirs}
        if missing:
            raise FileNotFoundError(f"tasks not found under {demo_dir}: {missing}")
    return dirs


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--demo-dir", required=True,
                    help="root with one subdir per task, each holding demo .npz")
    ap.add_argument("--out-dir", default="data/robodojo_demos_latents")
    ap.add_argument("--model-path", default=None,
                    help="DM0.5 checkpoint (required unless --estimate-only)")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of task ids")
    ap.add_argument("--limit-per-task", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                    help="storage dtype for latents (default fp16)")
    ap.add_argument("--bf16", action="store_true", help="run the VLM in bf16")
    ap.add_argument("--frame-prompt", default="",
                    help="prompt used when encoding frames; keep constant so "
                         "latents stay purely visual (stage text is separate)")
    ap.add_argument("--stage-text-template", default=None,
                    help="override world_model.stage_text_template")
    ap.add_argument("--k-fallback", type=int, default=None,
                    help="uniform milestone count when official_substeps absent")
    ap.add_argument("--adapter", default=None,
                    help="pkg.mod:ClassName implementing FeatureBackend")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print projected disk cost and exit (no GPU needed)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pcfg = PrepConfig(cfg, args)
    demo_dir = Path(args.demo_dir)

    if args.estimate_only:
        estimate(demo_dir, pcfg, args)
        return

    if not args.model_path:
        ap.error("--model-path is required unless --estimate-only")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out_dir.glob("*.npz")}
    if existing and not args.overwrite:
        print(f"[prep] {len(existing)} .npz already in {out_dir}; "
              f"pass --overwrite to regenerate")
        return

    backend = load_backend(args, pcfg)
    text_cache: dict[str, np.ndarray] = {}
    manifest: list[dict] = []
    failures: list[str] = []

    try:
        for task in pick_tasks(demo_dir, args.tasks):
            files = sorted(task.glob("*.npz"))
            if args.limit_per_task:
                files = files[:args.limit_per_task]
            for i, f in enumerate(files):
                try:
                    rec = process_demo(task.name, i, f, backend, pcfg, args,
                                       out_dir, text_cache)
                except Exception as exc:                # noqa: BLE001
                    failures.append(f"{f}: {type(exc).__name__}: {exc}")
                    print(f"[prep] SKIP {f.name}: {exc}", file=sys.stderr)
                    continue
                manifest.append(rec)
            print(f"[prep] {task.name}: {len(files)} demos")
    finally:
        backend.close()

    (out_dir / "manifest.json").write_text(
        json.dumps({"config": {"num_patches": pcfg.num_patches,
                               "vlm_dim": pcfg.vlm_dim,
                               "text_tokens": pcfg.text_tokens,
                               "chunk_len": pcfg.chunk_len,
                               "stage_text_template": pcfg.template,
                               "dtype": args.dtype},
                    "records": manifest,
                    "failures": failures}, indent=2))
    total = sum(r["bytes"] for r in manifest) / 2**30
    print(f"[prep] wrote {len(manifest)} trajectories ({total:.2f} GiB) -> {out_dir}")
    if failures:
        print(f"[prep] {len(failures)} failures (see manifest.json)", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
