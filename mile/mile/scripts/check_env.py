"""Pre-flight check for the GPU server. Run this BEFORE anything else.

    conda activate opendm
    python -m mile.scripts.check_env --config configs/server.yaml

It fails fast (exit 1) on anything that will bite you later, so you do not
discover a missing ``norm_stats.json`` after a 3-hour latent extraction run.

Checks, in order:
  1. interpreter / core packages (torch+CUDA, numpy<2, yaml, PIL, tyro)
  2. opendm + mile importable
  3. paths declared in the config really exist
     (checkpoint dir + its norm_stats.json, opendm_root, exp_entry)
  4. GPU: name / free memory / bf16 support
  5. free disk on the output dirs (the latent dump is ~1.3 TiB at full scale)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

OK, WARN, FAIL = "  ok ", " warn", "FAIL "


def _say(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}")


def _disk_free(path: Path) -> float | None:
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    hard_fail = False

    # ---- 1. interpreter + core packages -----------------------------------
    print("== 1. interpreter / packages ==")
    _say(OK if sys.version_info >= (3, 10) else FAIL,
         f"python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        hard_fail = True

    import importlib

    for mod, extra in (("torch", "cuda"), ("numpy", "<2"), ("yaml", None),
                       ("PIL", None), ("tyro", None)):
        try:
            m = importlib.import_module(mod)
        except Exception as exc:  # noqa: BLE001
            _say(FAIL, f"{mod} not importable: {exc}")
            hard_fail = True
            continue
        ver = getattr(m, "__version__", "?")
        note = ""
        if mod == "numpy" and int(ver.split(".")[0]) >= 2:
            note = " -- numpy>=2 breaks the Isaac Sim ecosystem"
            hard_fail = True
            _say(FAIL, f"numpy {ver}{note}")
            continue
        if mod == "PIL":
            import PIL
            ver = PIL.__version__
        _say(OK, f"{mod} {ver}")

    try:
        import torch
        if not torch.cuda.is_available():
            _say(FAIL, "torch.cuda.is_available() == False "
                       "(check driver / CUDA runtime)")
            hard_fail = True
        else:
            _say(OK, f"cuda {torch.version.cuda} | "
                     f"device0={torch.cuda.get_device_name(0)} | "
                     f"bf16={torch.cuda.is_bf16_supported()}")
            free, total = torch.cuda.mem_get_info()
            _say(OK, f"gpu mem free={free / 1024**3:.1f} GiB "
                     f"/ total={total / 1024**3:.1f} GiB")
            if free / 1024 ** 3 < 30:
                _say(WARN, "less than 30 GiB free -- DM0.5 (4B) + WM needs more, "
                           "close anything else holding the card")
    except Exception as exc:  # noqa: BLE001
        _say(FAIL, f"torch probe failed: {exc}")
        hard_fail = True

    # ---- 2. project packages ----------------------------------------------
    print("== 2. project packages ==")
    for mod in ("opendm", "mile"):
        try:
            m = importlib.import_module(mod)
            _say(OK, f"{mod} -> {Path(getattr(m, '__file__', '.')).parent}")
        except Exception as exc:  # noqa: BLE001
            _say(FAIL, f"{mod} not importable: {exc} "
                       f"({'pip install -e . in the opendm repo' if mod == 'opendm' else 'pip install -e . --no-deps in the mile repo'})")
            hard_fail = True

    # ---- 3. config paths ---------------------------------------------------
    print("== 3. config paths ==")
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        _say(FAIL, f"config not found: {cfg_path}")
        return 1
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    b = cfg.get("base", {})

    if b.get("mode") != "local":
        _say(WARN, f"base.mode={b.get('mode')!r} -- the HTTP path cannot give "
                   "you VLM patch tokens; you almost certainly want 'local'")

    ckpt = Path(str(b.get("checkpoint", "")).replace("~", str(Path.home())))
    if not ckpt.exists():
        _say(FAIL, f"base.checkpoint missing: {ckpt}")
        hard_fail = True
    else:
        _say(OK, f"checkpoint {ckpt}")
        if not (ckpt / "norm_stats.json").exists():
            _say(WARN, "no norm_stats.json next to the checkpoint -- OpenDM "
                       "falls back to ./norm_stats/ and action scaling may drift")
        else:
            _say(OK, "norm_stats.json present")

    root = Path(str(b.get("opendm_root", ".")).replace("~", str(Path.home())))
    if not root.exists():
        _say(FAIL, f"base.opendm_root missing: {root}")
        hard_fail = True
    else:
        _say(OK, f"opendm_root {root}")
        entry = Path(b.get("exp_entry", ""))
        entry = entry if entry.is_absolute() else root / entry
        if entry.exists():
            _say(OK, f"exp_entry {entry.name}")
        else:
            _say(FAIL, f"exp_entry not found: {entry} -- list "
                       f"{root / 'playground'} and pick the real MEM/RoboDojo entry")
            hard_fail = True

    # ---- 4. disk ------------------------------------------------------------
    print("== 4. disk ==")
    seen: set[str] = set()
    for p in (cfg.get("train", {}).get("data_dir"),
              cfg.get("train", {}).get("out_dir"),
              "."):
        if not p:
            continue
        d = Path(str(p)).resolve()
        key = str(d)
        if key in seen:
            continue
        seen.add(key)
        free = _disk_free(d if d.exists() else Path.cwd())
        if free is None:
            _say(WARN, f"cannot stat {d}")
        else:
            tag = OK if free > 200 else WARN
            _say(tag, f"{d}: {free:.0f} GiB free")
            if free < 1500:
                _say(WARN, "full-scale latents (3,500 demos, fp16, T~300) need "
                           "~1.3 TiB -- run prepare_latents.py --estimate-only "
                           "and start with a subset")

    print()
    if hard_fail:
        print("RESULT: FAILED -- fix the FAIL lines above before continuing.")
        return 1
    print("RESULT: OK -- proceed to "
          "`python -m mile.scripts.probe_dm05_local`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
