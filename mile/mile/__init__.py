"""MILE: Milestone-grounded closed-Loop imagination and Execution.

A model-based wrapper that turns a frozen chunked VLA policy (DM0.5 / pi0)
into a closed-loop long-horizon agent:
  - milestone extraction from demonstrations (mile.extractor)
  - latent world model + progress value head (mile.wm)
  - verify-then-act controller with safe-anchor rollback (mile.controller)
"""

__version__ = "0.1.0"
