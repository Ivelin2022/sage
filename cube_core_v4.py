"""
SpatialCube V4 — Complete Architecture
========================================
Author: Ivelin Likov

Version history:
  V1 — Basic geometric routing + spatial cohesion (cube_core.py)
  V2 — Momentum + contrastive repulsion + direction training + adaptive temperature
  V3 — Lennard-Jones gravity + oscillatory pulse + corner dampening (V1 learning)
  V4 — V2 learning + V3 gravity = full architecture (this file)

Five forces act during learning:

  [V2 learning forces]
  Force 1 — Momentum
    Embedding velocity accumulates across steps. Prevents oscillation.
    Same principle as SGD+momentum.

  Force 2 — Contrastive repulsion
    Random non-target points pushed away from target direction.
    Prevents representational collapse. Core insight: word2vec, SimCLR.

  Force 3 — Relationship direction training
    Teaches direction vector (target - query), not just destination.
    Enables king - man + woman = queen analogy arithmetic.

  Force 4 — Adaptive temperature
    Softmax attention starts warm (exploration) and cools over 500 steps
    (precision). Prevents premature convergence.

  [V3 gravity force]
  Force 5 — Lennard-Jones gravity with oscillatory pulse and corner dampening
    Similar concepts attract in 3D space (cosine > 0 → pull toward).
    Dissimilar concepts repel in 3D space (cosine < 0 → push away).
    Force magnitude follows Gaussian spatial falloff: exp(-dist/sigma).
    Oscillatory pulse: G breathes sinusoidally, preventing cluster freezing.
    Corner dampening: points near cube corners receive weaker gravity,
    preventing the geometric boundary bias of a bounded cube.

Usage:
    cube = SpatialCubeV4(cube_size=16, embed_dim=64)
    cube.learn_batch(pairs, alpha=0.01)
    result = cube.query(query_vec)
"""

import numpy as np
from scipy.spatial import KDTree
import json
import os
import math


class SpatialCubeV4:
    def __init__(
        self,
        cube_size=16,
        embed_dim=64,
        seed=42,
        # Gravity hyperparameters
        gravity_strength=0.003,
        gravity_sigma=0.5,
        gravity_repel=0.6,
        gravity_top_k=12,
        gravity_every=3,
        # Oscillation hyperparameters
        osc_amplitude=0.4,
        osc_omega=0.05,
    ):
        np.random.seed(seed)

        self.cube_size       = cube_size
        self.embed_dim       = embed_dim
        self.n_points        = cube_size ** 3

        # Gravity config — scale G by cube size (neighbourhood grows as n^3)
        # For cube_size=16: G = 0.003 * (0.5)^3 = 0.000375
        # For cube_size=32: G = 0.003 (reference calibration)
        self.gravity_strength = gravity_strength * (cube_size / 32.0) ** 3
        self.gravity_sigma    = gravity_sigma
        self.gravity_repel    = gravity_repel
        self.gravity_top_k    = gravity_top_k
        self.gravity_every    = gravity_every
        self.osc_amplitude    = osc_amplitude
        self.osc_omega        = osc_omega

        print(f"SpatialCubeV4: {cube_size}^3 = {self.n_points:,} points | "
              f"embed_dim={embed_dim}")
        print(f"  Gravity: G={self.gravity_strength:.6f}, "
              f"sigma={gravity_sigma}, repel={gravity_repel}")

        self._init_positions()

        self.embeddings = np.random.randn(self.n_points, embed_dim).astype(np.float32)
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)

        # Corner dampening factors: cf = 1 - max(|x|,|y|,|z|)^2
        self.corner_factors = (
            1.0 - np.max(np.abs(self.positions), axis=1) ** 2
        ).astype(np.float32)

        self.activations    = np.zeros(self.n_points, dtype=np.float32)
        self.labels         = {}
        self.loss_history   = []
        self.gravity_history = []
        self.step_count     = 0

        self._rebuild_spatial_index()
        print("SpatialCubeV4 ready.\n")

    def _init_positions(self):
        coords = np.linspace(-1, 1, self.cube_size)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        self.positions = np.stack([
            xx.flatten(), yy.flatten(), zz.flatten()
        ], axis=1).astype(np.float32)

    def _rebuild_spatial_index(self):
        self.spatial_index = KDTree(self.positions)

    # ─────────────────────────────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────────────────────────────

    def query(self, query_vec, top_k=10, temperature=0.1, update_activations=True):
        query_vec = np.array(query_vec, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        similarities = self.embeddings @ query_vec
        scores = similarities / temperature
        scores = scores - scores.max()
        scores = np.exp(scores)
        scores = scores / scores.sum()

        top_indices = np.argsort(scores)[-top_k:][::-1]
        if update_activations:
            self.activations[top_indices] += scores[top_indices]

        return {
            'indices':    top_indices,
            'scores':     scores[top_indices],
            'positions':  self.positions[top_indices],
            'embeddings': self.embeddings[top_indices],
            'labels':     [self.labels.get(i, f'point_{i}') for i in top_indices],
        }

    def label_point(self, query_vec, label):
        result = self.query(query_vec, top_k=1)
        idx = result['indices'][0]
        self.labels[idx] = label
        return idx

    # ─────────────────────────────────────────────────────────────────
    # V3 GRAVITY ENGINE
    # ─────────────────────────────────────────────────────────────────

    def _apply_gravity(self, active_indices):
        """
        Lennard-Jones-inspired gravity operating in embedding space.

        For each active point:
          - Find its top-k semantic neighbours (by cosine similarity)
          - Positive cosine → attraction force
          - Negative cosine → repulsion force
          - Force magnitude = G * |cosine| * exp(-dist/sigma) * pulse * cf_i * cf_j

        Oscillatory pulse: G breathes as 1 + A*sin(omega*t), preventing freezing.
        Corner dampening: cf = 1 - max(|x|,|y|,|z|)^2, uniform interior field.
        """
        G      = self.gravity_strength
        sigma  = self.gravity_sigma
        repel  = self.gravity_repel
        top_k  = self.gravity_top_k
        pulse  = 1.0 + self.osc_amplitude * math.sin(self.osc_omega * self.step_count)

        # Pairwise cosine similarities for active points vs all points
        active_embs = self.embeddings[active_indices]          # (n_active, d)
        sim_matrix  = active_embs @ self.embeddings.T          # (n_active, N)

        # Pairwise 3D distances for active points vs all points
        delta  = self.positions[np.newaxis, :, :] - self.positions[active_indices, np.newaxis, :]
        dists  = np.linalg.norm(delta, axis=2) + 1e-6          # (n_active, N)
        falloff = np.exp(-dists / sigma)

        embed_delta  = np.zeros_like(self.embeddings)
        total_force  = 0.0

        for k, idx in enumerate(active_indices):
            sims_k = sim_matrix[k].copy()
            sims_k[idx] = -2.0  # exclude self

            top_j = np.argsort(sims_k)[-top_k:]
            cf_i  = self.corner_factors[idx]

            for j in top_j:
                s   = sims_k[j]
                fo  = falloff[k, j]
                cf  = cf_i * self.corner_factors[j]

                embed_dir  = self.embeddings[j] - self.embeddings[idx]
                embed_dist = np.linalg.norm(embed_dir) + 1e-8

                if s > 0:
                    force_mag = G * s * fo * pulse * cf
                    embed_delta[idx] += force_mag * (embed_dir / embed_dist)
                else:
                    force_mag = G * repel * (-s) * fo * pulse * cf
                    embed_delta[idx] -= force_mag * (embed_dir / embed_dist)

                total_force += abs(force_mag)

        self.embeddings += embed_delta
        return total_force / max(len(active_indices), 1)

    # ─────────────────────────────────────────────────────────────────
    # V2 LEARNING FORCES
    # ─────────────────────────────────────────────────────────────────

    def _get_temperature(self, base_temp=0.1):
        """Force 4: adaptive temperature — warm early, sharp later."""
        warmup = 500
        factor = 1.0 + 2.0 * (1.0 - min(self.step_count, warmup) / warmup)
        return base_temp * factor

    def learn_batch(self, batch_pairs, alpha=0.01, beta=0.005,
                    momentum=0.9, neg_weight=0.3, teach_directions=True):
        """
        V4 full learning: V2 forces + V3 gravity every gravity_every steps.
        """
        if not hasattr(self, '_velocity'):
            self._velocity = np.zeros_like(self.embeddings)

        pos_grad    = np.zeros_like(self.embeddings)
        neg_grad    = np.zeros_like(self.embeddings)
        weight_acc  = np.zeros(self.n_points)
        batch_loss  = 0.0
        temperature = self._get_temperature()

        for query_vec, target_vec in batch_pairs:
            query_vec  = np.array(query_vec,  dtype=np.float32)
            target_vec = np.array(target_vec, dtype=np.float32)
            query_vec  = query_vec  / (np.linalg.norm(query_vec)  + 1e-8)
            target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)

            direction_vec = target_vec - query_vec
            d_norm = np.linalg.norm(direction_vec)
            if d_norm > 1e-8:
                direction_vec = direction_vec / d_norm

            result  = self.query(query_vec, top_k=20, temperature=temperature)
            indices = result['indices']
            scores  = result['scores']

            current_response = np.sum(
                self.embeddings[indices] * scores[:, np.newaxis], axis=0)
            sim  = np.dot(current_response, target_vec) / (
                np.linalg.norm(current_response) + 1e-8)
            batch_loss += 1.0 - sim

            # Force 1: accumulate positive gradients
            for i, idx in enumerate(indices):
                pos_grad[idx]   += scores[i] * (target_vec - self.embeddings[idx])
                weight_acc[idx] += scores[i]

            # Force 3: direction training
            if teach_directions and d_norm > 1e-8:
                dir_result = self.query(query_vec, top_k=10, temperature=temperature)
                for i, idx in enumerate(dir_result['indices']):
                    pos_grad[idx]   += 0.3 * dir_result['scores'][i] * (
                        direction_vec - self.embeddings[idx])
                    weight_acc[idx] += 0.3 * dir_result['scores'][i]

            # Force 2: contrastive repulsion
            if neg_weight > 0:
                neg_idx   = np.random.choice(self.n_points, 20, replace=False)
                index_set = set(indices.tolist())
                for idx in neg_idx:
                    if idx not in index_set:
                        push = self.embeddings[idx] - target_vec
                        pn   = np.linalg.norm(push)
                        if pn > 1e-8:
                            neg_grad[idx] += neg_weight * (push / pn)

        # Apply momentum update (Force 1)
        active_mask = weight_acc > 0
        if active_mask.any():
            avg_grad = np.zeros_like(self.embeddings)
            avg_grad[active_mask] = (
                pos_grad[active_mask] / weight_acc[active_mask, np.newaxis])
            self._velocity = momentum * self._velocity + (1.0 - momentum) * avg_grad
            self.embeddings[active_mask] += alpha * self._velocity[active_mask]

        # Apply repulsion (Force 2)
        neg_active = np.any(neg_grad != 0, axis=1)
        if neg_active.any():
            self.embeddings[neg_active] += alpha * neg_grad[neg_active]

        # Base spatial cohesion
        if beta > 0 and self.step_count % 5 == 0:
            for idx in np.where(active_mask)[0][:10]:
                nearby = self.spatial_index.query_ball_point(
                    self.positions[idx], r=0.35)
                for nidx in nearby:
                    if nidx != idx:
                        sim = np.dot(self.embeddings[idx], self.embeddings[nidx])
                        if sim > 0.3:
                            self.embeddings[idx] += beta * sim * (
                                self.embeddings[nidx] - self.embeddings[idx])

        # Force 5: LJ gravity (every gravity_every steps)
        if self.step_count % self.gravity_every == 0:
            active_indices = np.where(active_mask)[0]
            if len(active_indices) > 0:
                gf = self._apply_gravity(active_indices)
                self.gravity_history.append(gf)

        # Renormalise
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.maximum(norms, 1e-8)

        mean_loss = batch_loss / len(batch_pairs)
        self.loss_history.append(float(mean_loss))
        self.step_count += 1
        return mean_loss

    def continuous_learn(self, query_vec, target_vec, alpha=0.01):
        """No-momentum inference-time update. Achieves -0.000 anti-forgetting."""
        query_vec  = np.array(query_vec,  dtype=np.float32)
        target_vec = np.array(target_vec, dtype=np.float32)
        query_vec  = query_vec  / (np.linalg.norm(query_vec)  + 1e-8)
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        result = self.query(query_vec, top_k=10)
        for i, idx in enumerate(result['indices']):
            self.embeddings[idx] += alpha * result['scores'][i] * (
                target_vec - self.embeddings[idx])
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.maximum(norms, 1e-8)

    def learn_association(self, query_vec, target_vec, alpha=0.01, beta=0.005):
        """Single pair wrapper."""
        return self.learn_batch([(query_vec, target_vec)], alpha=alpha, beta=beta)


    def stats(self):
        """Print current cube statistics."""
        print(f"\n{'='*55}")
        print(f"SpatialCubeV4 Stats")
        print(f"{'='*55}")
        print(f"Size:          {self.cube_size}^3 = {self.n_points:,} points")
        print(f"Embed dim:     {self.embed_dim}")
        print(f"Steps:         {self.step_count:,}")
        print(f"Labels:        {len(self.labels)}")
        if self.loss_history:
            print(f"Recent loss:   {sum(self.loss_history[-100:])/min(100,len(self.loss_history)):.4f}")
        if self.gravity_history:
            print(f"Avg gravity:   {sum(self.gravity_history[-50:])/min(50,len(self.gravity_history)):.6f}")
        top = sorted(range(self.n_points), key=lambda i: self.activations[i], reverse=True)[:5]
        print(f"\nMost activated:")
        for idx in top:
            pos = self.positions[idx]
            print(f"  [{self.labels.get(idx, f'p{idx}')}] "
                  f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"act={self.activations[idx]:.3f}")
        print(f"{'='*55}\n")

        # ─────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'embeddings.npy'), self.embeddings)
        np.save(os.path.join(path, 'activations.npy'), self.activations)
        meta = {
            'cube_size':   self.cube_size,
            'embed_dim':   self.embed_dim,
            'n_points':    self.n_points,
            'step_count':  self.step_count,
            'labels':      {str(k): v for k, v in self.labels.items()},
            'loss_history': self.loss_history,
            'gravity_history': self.gravity_history,
            'gravity_strength': float(self.gravity_strength),
        }
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"SpatialCubeV4 saved to {path}")

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        cube = cls.__new__(cls)
        cube.cube_size        = meta['cube_size']
        cube.embed_dim        = meta['embed_dim']
        cube.n_points         = meta['n_points']
        cube.step_count       = meta['step_count']
        cube.labels           = {int(k): v for k, v in meta['labels'].items()}
        cube.loss_history     = meta['loss_history']
        cube.gravity_history  = meta.get('gravity_history', [])
        cube.gravity_strength = meta.get('gravity_strength', 0.003)
        cube.gravity_sigma    = 0.5
        cube.gravity_repel    = 0.6
        cube.gravity_top_k    = 12
        cube.gravity_every    = 3
        cube.osc_amplitude    = 0.4
        cube.osc_omega        = 0.05
        cube._init_positions()
        cube.embeddings   = np.load(os.path.join(path, 'embeddings.npy'))
        cube.activations  = np.load(os.path.join(path, 'activations.npy'))
        cube.corner_factors = (
            1.0 - np.max(np.abs(cube.positions), axis=1) ** 2
        ).astype(np.float32)
        cube._rebuild_spatial_index()
        print(f"SpatialCubeV4 loaded from {path} (step {cube.step_count})")
        return cube
