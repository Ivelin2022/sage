"""
core/sphere_substrate.py - Fibonacci-sphere storage substrate (Stage 1).

Replaces the Cartesian cube grid with a golden-angle lattice on S^2. Key design
decisions (kept faithful to the brief):

  - The GRID is low-dim (3-D, visualizable). The 768-D embedding is the PAYLOAD
    stored at each point. Grid dim and payload dim are DECOUPLED.
  - Grid points are FIXED at deterministic golden-angle positions. Positions are
    never optimized (no gradient anywhere).
  - place() needs to map a 768-D unit embedding onto a 3-D lattice point "by
    cosine". Since dims differ, we use a deterministic gradient-free PCA
    projection 768 -> 3 (fit once on the corpus), normalize to S^2, and snap to
    the nearest lattice point by angle. Collisions (many words -> one point) are
    kept in a per-point bucket.
  - RETRIEVAL uses the full 768-D payloads (exact cosine), so it is provably
    identical to the cube - the substrate swap cannot regress retrieval. The 3-D
    positions exist for the downstream graph (Stage 3) and partition (Stage 6).

THESIS GUARD: numpy/torch tensor ops + PCA-via-SVD only. No autograd/optimizer.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from .retrieval import cosine_topk


def fibonacci_sphere(n_points, device='cpu'):
    """Deterministic golden-angle lattice on the unit 2-sphere -> (n_points, 3)."""
    i = torch.arange(n_points, dtype=torch.float64, device=device)
    z = 1.0 - 2.0 * (i + 0.5) / n_points              # even spacing in z, (-1,1)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    golden = math.pi * (1.0 + 5.0 ** 0.5)             # golden angle
    theta = golden * i
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y, z], dim=1).float()      # (N, 3) unit-norm


class SphereSubstrate:
    def __init__(self, n_points=4096, payload_dim=768, seed=42, device=None):
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        torch.manual_seed(seed)
        self.n_points = n_points
        self.payload_dim = payload_dim
        self.positions = fibonacci_sphere(n_points, self.device).to(self.device)

        # populated by place()
        self.payloads = None        # (n_items, D) stored embeddings (unit-norm)
        self.item_words = None      # list[str] aligned with payloads
        self.item_point = None      # (n_items,) long: which grid point holds item
        self.buckets = None         # list[list[int]] grid point -> item indices
        self._pca_mean = None       # (D,)
        self._pca_comps = None      # (3, D)
        print("SphereSubstrate: %d points on S^2 | payload_dim=%d | device=%s"
              % (n_points, payload_dim, self.device))

    # ---- gradient-free PCA projection 768 -> 3 -------------------------------

    def fit_projection(self, embeddings):
        """Fit top-3 principal directions (SVD) on the corpus. Gradient-free.
        float32 is enough for the top-3 directions and avoids the slow/fragile
        float64 linalg path on CUDA."""
        X = embeddings.float()
        self._pca_mean = X.mean(dim=0)
        Xc = X - self._pca_mean
        # economy SVD; rows of Vh are principal directions (desc by sing. value)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        self._pca_comps = Vh[:3].clone()              # (3, D) float32
        return self

    def project_raw(self, embeddings):
        """768-D -> raw 3-D PCA coordinates (no normalization). (M, 3) float32."""
        return (embeddings.float() - self._pca_mean) @ self._pca_comps.T

    def project_to_sphere(self, embeddings):
        """768-D -> 3-D PCA coords -> unit vectors on S^2. (M, 3) float32."""
        return F.normalize(self.project_raw(embeddings), p=2, dim=1)

    # ---- placement -----------------------------------------------------------

    def place(self, embeddings, words):
        """Assign each unit embedding to its nearest lattice point and store the
        full 768-D payload in that point's bucket. Collisions -> per-point list."""
        embeddings = F.normalize(embeddings.to(self.device).float(), p=2, dim=1)
        if self._pca_comps is None:
            self.fit_projection(embeddings)
        proj = self.project_to_sphere(embeddings)     # (M, 3) on S^2

        # nearest lattice point by angle == max dot product (both unit-norm)
        sims = proj @ self.positions.T                # (M, N)
        nearest = torch.argmax(sims, dim=1)           # (M,)

        self.payloads = embeddings
        self.item_words = list(words)
        self.item_point = nearest
        self.buckets = [[] for _ in range(self.n_points)]
        for item_idx, pt in enumerate(nearest.tolist()):
            self.buckets[pt].append(item_idx)
        return self

    # ---- retrieval -----------------------------------------------------------

    def query_exact(self, query_vec, k=10):
        """Exact retrieval: cosine over ALL stored 768-D payloads (substrate-
        independent). Returns (scores, item_indices)."""
        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.tensor(query_vec, dtype=torch.float32)
        return cosine_topk(query_vec.to(self.device), self.payloads, k=k)

    def query_positional(self, query_vec, k=10, n_cells=16):
        """Positional retrieval: route the query to its nearest lattice point,
        gather candidates from that point + its n_cells nearest grid neighbours,
        then rank within by exact 768-D cosine. Measures how usable the coarse
        3-D geometry is for routing (it loses info vs exact, by design)."""
        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.tensor(query_vec, dtype=torch.float32)
        q = F.normalize(query_vec.to(self.device).float(), p=2, dim=0)
        qproj = self.project_to_sphere(q.unsqueeze(0))            # (1, 3)
        cell_sims = (qproj @ self.positions.T).squeeze(0)        # (N,)
        n_cells = min(n_cells, self.n_points)
        _, cells = torch.topk(cell_sims, n_cells)
        cand = [it for c in cells.tolist() for it in self.buckets[c]]
        if not cand:
            return (torch.empty(0, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device))
        cand_t = torch.tensor(cand, dtype=torch.long, device=self.device)
        sims = (q @ self.payloads[cand_t].T)
        k = min(k, len(cand))
        sc, loc = torch.topk(sims, k)
        return sc, cand_t[loc]

    # ---- diagnostics ---------------------------------------------------------

    def collision_stats(self):
        sizes = torch.tensor([len(b) for b in self.buckets])
        used = (sizes > 0).sum().item()
        collided = (sizes > 1).sum().item()
        return {
            "n_points": self.n_points,
            "n_items": int(self.payloads.shape[0]),
            "points_used": int(used),
            "points_collided": int(collided),
            "max_bucket": int(sizes.max().item()),
            "mean_nonempty_bucket": float(sizes[sizes > 0].float().mean().item()),
            "pct_items_in_collision": float(
                100.0 * (sizes[sizes > 1].sum().item()) / self.payloads.shape[0]),
        }
