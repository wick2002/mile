"""Split-conformal calibration of the acceptance threshold (paper §4.4).

Given calibration pairs (J_i, d_i) — verification score and realized
deviation of executed chunks collected from BASE-POLICY rollouts —
choose the smallest threshold tau such that, among accepted points,
the empirical (1-alpha) quantile of deviations is within radius r.

Guarantee (marginal coverage, finite-sample):
    P(accepted AND deviation <= r) >= 1 - alpha - 1/(n+1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalResult:
    tau_hat: float
    alpha: float
    radius: float
    n_calib: int
    guarantee: float          # 1 - alpha - 1/(n+1)
    rejection_rate: float     # fraction of calibration points below tau_hat


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) empirical quantile."""
    n = len(values)
    if n == 0:
        return np.inf
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    k = min(max(k, 1), n)
    return float(np.sort(values)[k - 1])


def calibrate_tau(J: np.ndarray, d: np.ndarray, radius: float,
                  alpha: float = 0.05,
                  tau_grid: np.ndarray | None = None) -> ConformalResult:
    """Choose the smallest acceptance threshold satisfying the coverage constraint.

    Args:
        J: [n] verification scores of executed candidates (calibration).
        d: [n] realized deviations to the target milestone.
        radius: verification radius r.
        alpha: target miscoverage level (e.g. 0.05).
        tau_grid: candidate thresholds; default = sorted unique J values.
    """
    J = np.asarray(J, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if tau_grid is None:
        tau_grid = np.unique(J)

    best_tau = None
    for tau in np.sort(tau_grid):
        accepted = d[J >= tau]
        if len(accepted) == 0:
            continue
        if conformal_quantile(accepted, alpha) <= radius:
            best_tau = float(tau)
            break

    if best_tau is None:
        # no threshold satisfies coverage at this radius: reject everything
        best_tau = float(np.max(tau_grid)) + 1.0

    n = len(J)
    return ConformalResult(
        tau_hat=best_tau,
        alpha=alpha,
        radius=radius,
        n_calib=n,
        guarantee=1.0 - alpha - 1.0 / (n + 1),
        rejection_rate=float((J < best_tau).mean()),
    )


def empirical_coverage(J: np.ndarray, d: np.ndarray, tau_hat: float,
                       radius: float) -> float:
    """Coverage check on HELD-OUT data: fraction of accepted points within r."""
    accepted = d[J >= tau_hat]
    if len(accepted) == 0:
        return 1.0
    return float((accepted <= radius).mean())
