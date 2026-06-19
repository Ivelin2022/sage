"""
core/isotropy.py - All-but-the-Top isotropy preprocessing (Stage 2).

Mu, Bhat & Viswanath, "All-but-the-Top: Simple and Effective Postprocessing for
Word Representations" (ICLR 2018). Gradient-free:
  1. subtract the common mean vector,
  2. project out the top-D principal directions,
  3. renormalize to the unit sphere.

Conditional on Stage 0: HIGH anisotropy here -> REQUIRED. Brief caveat: isotropy
helps raw embeddings but can HURT fine-tuned encoders - so Stage 2 is an
experiment; if it degrades retrieval we drop it.

THESIS GUARD: torch tensor ops + SVD only. No autograd / optimizer / backward.
"""

import torch
import torch.nn.functional as F


def all_but_the_top(embeddings, D):
    """Remove common-mean + top-D principal directions, renormalize to S^(d-1).
    D <= 0 returns the raw embeddings unit-normalized (the untreated baseline)."""
    X = embeddings.float()
    if D <= 0:
        return F.normalize(X, p=2, dim=1)
    Xc = X - X.mean(dim=0)
    # top-D principal directions (rows of Vh, desc by singular value)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    U = Vh[:D]                                   # (D, dim)
    proj = Xc - (Xc @ U.T) @ U                   # project out the top-D subspace
    # A row whose energy lived entirely in the removed subspace collapses to ~0;
    # normalizing it would yield a noise/NaN direction. Fall back to its
    # mean-removed direction so it stays a valid unit vector.
    collapsed = proj.norm(dim=1) < 1e-8
    if collapsed.any():
        proj = proj.clone()
        proj[collapsed] = Xc[collapsed]
    return F.normalize(proj, p=2, dim=1)


def anisotropy_stats(embeddings, n_pairs=10000, seed=0):
    """Stage-0-style isotropy diagnostic on an embedding matrix.
    Returns mean cosine, mean|cos|, mean-vector norm, top-1 PC variance share."""
    Mn = F.normalize(embeddings.float().cpu(), p=2, dim=1)
    n = Mn.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    a = torch.randint(0, n, (n_pairs,), generator=g)
    b = torch.randint(0, n, (n_pairs,), generator=g)
    keep = a != b
    a, b = a[keep], b[keep]
    cos = (Mn[a] * Mn[b]).sum(1)
    Xc = Mn - Mn.mean(0)
    var = torch.linalg.svdvals(Xc) ** 2
    return {
        "mean_cosine": float(cos.mean()),
        "mean_abs_cosine": float(cos.abs().mean()),
        "mean_vector_norm": float(Mn.mean(0).norm()),
        "top1_pc_variance": float(var[0] / var.sum()),
    }
