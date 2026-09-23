"""Three cue channels for automatic milestone extraction (design doc §2).

Each function returns a per-frame score in [0, 1] for a trajectory.
"""
from __future__ import annotations

import numpy as np


def kinematic_cue(gripper: np.ndarray, ee_vel: np.ndarray,
                  grip_thresh: float = 0.3, vel_ratio: float = 0.05) -> np.ndarray:
    """Gripper open/close transitions + end-effector velocity turning points.

    Args:
        gripper: [T] or [T, 2] normalized gripper aperture (0..1), one per arm.
        ee_vel:  [T, 3] or [T, 2, 3] end-effector velocity.
    Returns:
        cue: [T] score in [0, 1].
    """
    T = gripper.shape[0]
    g = gripper.reshape(T, -1)
    v = ee_vel.reshape(T, -1, 3) if ee_vel.ndim == 3 else ee_vel.reshape(T, 1, -1)

    cue = np.zeros(T, dtype=np.float64)

    # gripper transitions: |dg| crosses 30% of range -> score 1
    dg = np.abs(np.diff(g, axis=0, prepend=g[:1]))
    cue += (dg.max(axis=1) > grip_thresh).astype(np.float64)

    # velocity minima / direction reversal per arm
    for arm in range(v.shape[1]):
        va = v[:, arm, :]                                   # [T, 3]
        speed = np.linalg.norm(va, axis=1)
        peak = speed.max() + 1e-8
        slow = speed < vel_ratio * peak
        # sign reversal of the dominant velocity component
        pc = va[:, np.argmax(va.var(axis=0))]
        sign_flip = np.zeros(T, dtype=bool)
        sign_flip[1:] = np.sign(pc[1:]) != np.sign(pc[:-1])
        cue += 0.5 * (slow | sign_flip)

    return np.clip(cue, 0.0, 1.0)


def contact_cue(contact: np.ndarray) -> np.ndarray:
    """Contact onset/offset events from (privileged or probed) contact flags.

    Args:
        contact: [T, F] per-finger contact flags or probabilities.
    Returns:
        cue: [T] score in [0, 1].
    """
    c = (contact > 0.5).astype(np.int32)
    any_contact = c.max(axis=1)
    diff = np.diff(any_contact, prepend=any_contact[:1])
    cue = np.where(diff > 0, 1.0, np.where(diff < 0, 0.5, 0.0))
    return cue.astype(np.float64)


def visual_cue(latents: np.ndarray, attn_mask: np.ndarray | None = None) -> np.ndarray:
    """Local maxima of frame-to-frame latent distance (attention-weighted).

    Args:
        latents: [T, P, D] per-frame patch features.
        attn_mask: optional [T, P] instruction-relevance weights per patch.
    Returns:
        cue: [T] score in [0, 1].
    """
    T = latents.shape[0]
    if attn_mask is not None:
        w = attn_mask / (attn_mask.sum(axis=1, keepdims=True) + 1e-8)
        pooled = np.einsum("tpd,tp->td", latents, w)
    else:
        pooled = latents.mean(axis=1)                        # [T, D]

    d = np.zeros(T, dtype=np.float64)
    d[1:] = np.linalg.norm(np.diff(pooled, axis=0), axis=1)

    # local maxima
    is_peak = np.zeros(T, dtype=bool)
    is_peak[1:-1] = (d[1:-1] > d[:-2]) & (d[1:-1] >= d[2:])
    if d.max() > 0:
        d = d / d.max()
    return np.where(is_peak, d, 0.0)


def consensus(kin: np.ndarray, con: np.ndarray, vis: np.ndarray,
              alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Weighted vote of the three cue channels."""
    return alpha * kin + beta * con + gamma * vis
