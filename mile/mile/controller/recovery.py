"""Recovery buffer + hindsight distillation (paper §4.6, self-improvement).

Every rollback that is later followed by successfully reaching the next
milestone yields a hindsight-labeled recovery segment — a free demonstration
of "how to recover from the boundary of failure". Every R deployment
episodes the buffer is distilled back into the system (value head always,
optional LoRA adapter on the base's action expert).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RecoverySegment:
    task_id: str
    anchor_rank: int              # milestone rank retreated to
    latents: np.ndarray           # [T_seg, P, D] realized latents of the recovery
    actions: np.ndarray           # [T_seg, action_dim]
    reached_rank: int             # milestone rank eventually verified


@dataclass
class RecoveryBuffer:
    segments: list[RecoverySegment] = field(default_factory=list)

    def add(self, seg: RecoverySegment) -> None:
        self.segments.append(seg)

    def __len__(self) -> int:
        return len(self.segments)

    def value_distill_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch for value-head distillation.

        States near the *start* of a recovery segment are exactly the
        failure-boundary states where the base policy's progress estimate
        is least reliable; their labels are the normalized position within
        the successful recovery.
        """
        idx = np.random.choice(len(self.segments), size=min(batch_size, len(self.segments)),
                               replace=False)
        zs, vs = [], []
        for i in idx:
            seg = self.segments[i]
            T = len(seg.latents)
            t = np.random.randint(0, T)
            zs.append(seg.latents[t])
            vs.append(t / max(T - 1, 1))   # 0 = at anchor, 1 = milestone reached
        return {
            "latents": torch.from_numpy(np.stack(zs)).float(),
            "targets": torch.tensor(vs, dtype=torch.float32),
        }


def distill_value_head(world_model, buffer: RecoveryBuffer, steps: int = 500,
                       batch_size: int = 64, lr: float = 1e-4,
                       device: str = "cuda") -> dict[str, float]:
    """Fine-tune ONLY the value head on recovery data (no base, no WM trunk)."""
    if len(buffer) == 0:
        return {"skipped": 1.0}

    for p in world_model.parameters():
        p.requires_grad_(False)
    for p in world_model.value_head.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(world_model.value_head.parameters(), lr=lr)
    last = 0.0
    world_model.train()
    for _ in range(steps):
        batch = buffer.value_distill_batch(batch_size)
        z = batch["latents"].to(device)
        y = batch["targets"].to(device)
        v = world_model.value_path(z)          # differentiable value-only forward
        loss = F.huber_loss(v, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()

    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(True)
    return {"final_loss": last, "segments": float(len(buffer))}
