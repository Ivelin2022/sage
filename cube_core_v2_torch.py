"""
SpatialCube V2 — PyTorch GPU Version
======================================
Author: Ivelin Likov

V1 base (cube_core_torch.py) + four V2 learning forces, all GPU-vectorised:

  Force 1 — Momentum
    Velocity buffer accumulates across steps — prevents oscillation.

  Force 2 — Contrastive repulsion
    Random non-target points pushed away from target direction.
    Prevents representational collapse.

  Force 3 — Direction training
    Teaches the direction vector (target - query), not just destination.
    Enables king - man + woman = queen analogy arithmetic.
    THIS is the critical force for the GloVe delta encoding test.

  Force 4 — Adaptive temperature
    Softmax sharpens from 0.3 → 0.05 over first 500 steps.

All forces fully vectorised — no Python loops over points.
"""

import torch
import torch.nn.functional as F
import json
import os


class SpatialCubeV2Torch:
    def __init__(self, cube_size=16, embed_dim=64, seed=42, device=None):
        torch.manual_seed(seed)

        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.cube_size = cube_size
        self.embed_dim = embed_dim
        self.n_points  = cube_size ** 3

        print(f"SpatialCubeV2Torch: {cube_size}^3 = {self.n_points:,} | "
              f"dim={embed_dim} | device={self.device}")

        self._init_positions()

        self.embeddings = F.normalize(
            torch.randn(self.n_points, embed_dim, device=self.device), p=2, dim=1)

        # Momentum buffer — same shape as embeddings, starts at zero
        self._velocity = torch.zeros(self.n_points, embed_dim, device=self.device)

        self.activations  = torch.zeros(self.n_points, device=self.device)
        self.labels       = {}
        self.loss_history = []
        self.step_count   = 0
        print("Ready.\n")

    def _init_positions(self):
        coords = torch.linspace(-1, 1, self.cube_size, device=self.device)
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        self.positions = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    # ── Temperature schedule ──────────────────────────────────

    def _temperature(self, base=0.1):
        """Force 4: warm early, sharp later."""
        warmup = 500
        factor = 1.0 + 2.0 * max(0.0, 1.0 - self.step_count / warmup)
        return base * factor

    # ── Query ─────────────────────────────────────────────────

    def query(self, query_vec, top_k=10, temperature=None, update_activations=True):
        if temperature is None:
            temperature = self._temperature()

        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.tensor(query_vec, dtype=torch.float32, device=self.device)
        query_vec = F.normalize(query_vec.to(self.device), p=2, dim=0)

        sims   = self.embeddings @ query_vec
        scores = F.softmax(sims / temperature, dim=0)
        top_scores, top_indices = torch.topk(scores, top_k)

        if update_activations:
            self.activations[top_indices] += top_scores

        return {
            'indices':    top_indices,
            'scores':     top_scores,
            'positions':  self.positions[top_indices],
            'embeddings': self.embeddings[top_indices],
            'labels':     [self.labels.get(i.item(), f'point_{i.item()}')
                           for i in top_indices],
        }

    def label_point(self, query_vec, label):
        r   = self.query(query_vec, top_k=1, update_activations=False)
        idx = r['indices'][0].item()
        self.labels[idx] = label
        return idx

    # ── V2 learn_batch ────────────────────────────────────────

    def learn_batch(self, batch_pairs, alpha=0.01, beta=0.005,
                    momentum=0.9, neg_weight=0.3, teach_directions=True):
        """
        V2 learning — all four forces, GPU vectorised.

        batch_pairs: list of (query_vec, target_vec) — numpy or torch
        """
        B = len(batch_pairs)
        temperature = self._temperature()

        # Convert batch to tensors on device
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device, dtype=torch.float32)
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        Q = F.normalize(torch.stack([to_tensor(q) for q, _ in batch_pairs]), p=2, dim=1)
        T = F.normalize(torch.stack([to_tensor(t) for _, t in batch_pairs]), p=2, dim=1)

        # Direction vectors (target - query) — Force 3
        D = T - Q
        D_norms = D.norm(dim=1, keepdim=True)
        D_valid = (D_norms > 1e-8).squeeze(1)
        D = torch.where(D_norms > 1e-8, D / (D_norms + 1e-8), D)

        # ── Positive pass ─────────────────────────────────────
        # Similarities: (B, N)
        all_sims   = Q @ self.embeddings.T
        all_scores = F.softmax(all_sims / temperature, dim=1)

        # Loss
        top_k = 20
        top_scores, top_indices = torch.topk(all_scores, top_k, dim=1)
        batch_emb   = self.embeddings[top_indices]                          # (B, K, D)
        responses   = torch.einsum('bk,bkd->bd', top_scores, batch_emb)    # (B, D)
        responses_n = F.normalize(responses, p=2, dim=1)
        loss        = (1.0 - (responses_n * T).sum(dim=1)).mean()

        # Positive gradient accumulator — vectorised
        # grad[i] = sum_b scores[b,i] * (T[b] - E[i])
        # = (all_scores.T @ T) - (all_scores.sum(0).unsqueeze(1) * E)
        pos_grad   = all_scores.T @ T                                       # (N, D)
        weight_sum = all_scores.sum(0).unsqueeze(1)                         # (N, 1)
        pos_grad   = pos_grad - weight_sum * self.embeddings                # subtract E[i] * w[i]

        # Force 3: direction training — same query activations, different target
        if teach_directions:
            dir_sims   = Q @ self.embeddings.T                              # (B, N) — same Q
            dir_scores = F.softmax(dir_sims / temperature, dim=1) * 0.3    # reduced weight
            # Grad toward direction vectors
            dir_grad = dir_scores.T @ D                                     # (N, D)
            dir_w    = dir_scores.sum(0).unsqueeze(1)
            dir_grad = dir_grad - dir_w * self.embeddings
            pos_grad   = pos_grad + dir_grad
            weight_sum = weight_sum + dir_scores.sum(0).unsqueeze(1)

        # Normalise gradient
        active = weight_sum.squeeze() > 1e-6
        avg_grad = torch.zeros_like(self.embeddings)
        avg_grad[active] = pos_grad[active] / (weight_sum[active] + 1e-8)

        # Force 1: momentum
        self._velocity = momentum * self._velocity + (1.0 - momentum) * avg_grad
        self.embeddings = self.embeddings + alpha * self._velocity

        # Force 2: contrastive repulsion — vectorised
        if neg_weight > 0:
            n_neg   = min(30, self.n_points)
            neg_idx = torch.randperm(self.n_points, device=self.device)[:n_neg]
            # For each negative point: average push away from all targets
            T_mean  = T.mean(0, keepdim=True)                               # (1, D)
            push    = self.embeddings[neg_idx] - T_mean                     # (n_neg, D)
            push_n  = push.norm(dim=1, keepdim=True)
            push    = torch.where(push_n > 1e-8, push / (push_n + 1e-8), push)
            self.embeddings[neg_idx] = self.embeddings[neg_idx] + alpha * neg_weight * push

        # Base spatial cohesion (every 5 steps, limited)
        if beta > 0 and self.step_count % 5 == 0:
            active_idx = torch.where(active)[0][:10]
            for idx in active_idx:
                dists   = torch.norm(self.positions - self.positions[idx], dim=1)
                nearby  = torch.where((dists < 0.35) & (dists > 0))[0]
                if len(nearby) > 0:
                    sims_n = (self.embeddings[idx] @ self.embeddings[nearby].T)
                    cohese = nearby[sims_n > 0.3]
                    if len(cohese) > 0:
                        s = sims_n[sims_n > 0.3]
                        self.embeddings[idx] += (
                            beta * (s.unsqueeze(1) *
                            (self.embeddings[cohese] - self.embeddings[idx])).sum(0)
                        )

        # Renormalise
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        self.loss_history.append(loss.item())
        self.step_count += 1
        return loss.item()

    def continuous_learn(self, query_vec, target_vec, alpha=0.01):
        """No-momentum single-step update for inference time."""
        q = F.normalize(
            torch.tensor(query_vec, dtype=torch.float32, device=self.device), p=2, dim=0)
        t = F.normalize(
            torch.tensor(target_vec, dtype=torch.float32, device=self.device), p=2, dim=0)
        sims   = self.embeddings @ q
        scores = F.softmax(sims / 0.1, dim=0)
        _, top = torch.topk(scores, 10)
        for idx in top:
            self.embeddings[idx] += alpha * scores[idx] * (t - self.embeddings[idx])
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

    # ── Save / Load ───────────────────────────────────────────

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.embeddings.cpu(), os.path.join(path, 'embeddings.pt'))
        torch.save(self.activations.cpu(), os.path.join(path, 'activations.pt'))
        meta = {
            'cube_size':   self.cube_size,
            'embed_dim':   self.embed_dim,
            'step_count':  self.step_count,
            'labels':      {str(k): v for k, v in self.labels.items()},
            'loss_history': self.loss_history[-1000:],
        }
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        cube = cls(cube_size=meta['cube_size'], embed_dim=meta['embed_dim'])
        cube.embeddings  = torch.load(
            os.path.join(path, 'embeddings.pt')).to(cube.device)
        cube.activations = torch.load(
            os.path.join(path, 'activations.pt')).to(cube.device)
        cube.step_count  = meta['step_count']
        cube.labels      = {int(k): v for k, v in meta['labels'].items()}
        cube.loss_history = meta['loss_history']
        print(f"Loaded from {path} (step {cube.step_count})")
        return cube
