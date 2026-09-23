"""Sliding-window dataset for world-model training.

Expects preprocessed per-trajectory .npz files produced by
mile/scripts/prepare_latents.py, each containing:
    latents    : [T, P, vlm_dim]  VLM patch features per frame
    actions    : [T, action_dim]  executed actions
    milestones : [K]              milestone frame indices
    text_emb   : [K+1, text_tokens, text_dim] stage-descriptor embeddings
                 (text_emb[k] = description *of stage leading to milestone k*)
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    def __init__(self, data_dir: str, chunk_len: int = 50, stride: int = 10) -> None:
        self.data_dir = Path(data_dir)
        self.chunk_len = chunk_len
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz trajectories in {self.data_dir}")

        # index all valid windows: (file_idx, start_t)
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            with np.load(f) as z:
                T = z["actions"].shape[0]
            for t in range(0, T - chunk_len, stride):
                self.index.append((fi, t))

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def progress_label(t: int, milestones: np.ndarray, T: int) -> float:
        """Piecewise-linear progress anchored at milestones (design doc §4).

        milestone i (1-based) marks progress i / (K+1); task end = 1.0.
        """
        K = len(milestones)
        anchors = np.concatenate([[0], milestones, [T - 1]]).astype(np.float64)
        values = np.linspace(0.0, 1.0, K + 2)
        return float(np.interp(t, anchors, values))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, t = self.index[idx]
        with np.load(self.files[fi]) as z:
            latents = z["latents"]          # [T, P, D]
            actions = z["actions"]          # [T, A]
            milestones = z["milestones"]    # [K]
            text_emb = z["text_emb"]        # [K+1, Tt, Dt]

            T = actions.shape[0]
            end = t + self.chunk_len

            x = latents[t]                          # [P, D]
            a = actions[t:end]                      # [chunk, A]
            y = latents[min(end, T - 1)]            # [P, D]

            # stage descriptor = next milestone after current position
            k_next = int(np.searchsorted(milestones, t, side="right"))
            te = text_emb[min(k_next, len(text_emb) - 1)]   # [Tt, Dt]

            v = self.progress_label(end, milestones, T)

        return {
            "patch_tokens": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "action_chunk": torch.from_numpy(np.ascontiguousarray(a)).float(),
            "text_emb": torch.from_numpy(np.ascontiguousarray(te)).float(),
            "target_latents": torch.from_numpy(np.ascontiguousarray(y)).float(),
            "target_value": torch.tensor(v, dtype=torch.float32),
        }


def make_dataloader(data_dir: str, chunk_len: int, batch_size: int,
                    num_workers: int = 8, shuffle: bool = True):
    from torch.utils.data import DataLoader

    ds = SlidingWindowDataset(data_dir, chunk_len=chunk_len)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
