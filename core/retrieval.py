"""
core/retrieval.py - cosine routing over a substrate (Stage 1).

Identical on cube or sphere: retrieval is a single parallel GPU pass of cosine
similarity over the stored 768-D payloads. No Python loops over points, no
autograd. This is why the substrate swap is a near no-op for retrieval - the
metric never changes.
"""

import torch
import torch.nn.functional as F


def cosine_topk(query, payloads, k=10):
    """Top-k cosine routing.

    query:    (D,) or (B, D)   - need not be unit-norm (normalized here)
    payloads: (N, D) unit-norm tensor stored at the grid points
    returns:  (scores (B,k), indices (B,k))  - squeezed to (k,) if query was 1-D
    """
    single = query.dim() == 1
    if single:
        query = query.unsqueeze(0)
    query = F.normalize(query, p=2, dim=1)
    sims = query @ payloads.T                      # (B, N) - one matmul
    k = min(k, payloads.shape[0])
    scores, idx = torch.topk(sims, k, dim=1)
    if single:
        return scores.squeeze(0), idx.squeeze(0)
    return scores, idx
