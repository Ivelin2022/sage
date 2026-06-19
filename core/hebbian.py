"""
core/hebbian.py - gradient-free Hebbian write/update (Stage 1).

Ported from cube_core_v3_torch.py (Force 6 / Langevin). The ONLY change vs the
cube is that this operates on the substrate's payload tensor regardless of where
the grid points sit - the rule is coordinate-agnostic.

THESIS GUARD: pure tensor ops. No loss.backward, no optimizer, no autograd.
  - hebbian_write: strengthen activated points toward their targets (+ optional
    tangent-space Langevin noise, dimension-scaled exactly as in the cube).
  - decay: pull inactive points gently back / weaken them (controlled forgetting).
  - direct_store: alpha=1 limit used by Stage 1 (write the embedding as-is).
"""

import math
import torch
import torch.nn.functional as F


def hebbian_write(payloads, active_idx, targets, scores,
                  alpha=0.01, langevin_T=0.0):
    """Move activated payloads toward their targets; renormalize to the sphere.

    payloads:   (N, D) unit-norm, MODIFIED IN PLACE and returned
    active_idx: (M,) long - which points received signal
    targets:    (M, D) - desired direction for each active point
    scores:     (M,) - activation weight per point (e.g. softmax mass)
    """
    targets = F.normalize(targets, p=2, dim=1)
    payloads[active_idx] = (payloads[active_idx]
                            + alpha * scores.unsqueeze(1)
                            * (targets - payloads[active_idx]))

    # Force 6: tangent-space Langevin noise, scaled by sqrt(2T/d) so the
    # rotation angle is dimension-independent (same perturbation at 64d & 768d).
    if langevin_T > 0:
        d = payloads.shape[1]
        noise = torch.randn_like(payloads[active_idx]) * math.sqrt(2 * langevin_T / d)
        radial = (noise * payloads[active_idx]).sum(1, keepdim=True) * payloads[active_idx]
        payloads[active_idx] = payloads[active_idx] + (noise - radial)

    # normalize ONLY the touched rows, in place, so the caller's tensor stays
    # consistent with the return value (others are already unit-norm).
    payloads[active_idx] = F.normalize(payloads[active_idx], p=2, dim=1)
    return payloads


def decay(payloads, active_idx, rate=0.0):
    """Controlled forgetting: shrink non-active points toward zero-strength.
    For unit-norm payloads decay is a no-op on direction; kept as the hook for
    the Stage 7 radius channel where magnitude carries memory strength."""
    if rate <= 0:
        return payloads
    mask = torch.ones(payloads.shape[0], dtype=torch.bool, device=payloads.device)
    mask[active_idx] = False
    payloads[mask] = payloads[mask] * (1.0 - rate)
    return payloads


def direct_store(target):
    """Stage-1 write: store the embedding as-is (alpha=1 Hebbian limit)."""
    return F.normalize(target, p=2, dim=-1)
