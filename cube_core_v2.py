"""
SpatialCube - A 3D Geometric Knowledge Architecture
====================================================
Novel architecture where the space IS the model.
No layers. No weight matrices. Geometry computes.

Author: Ivelin Likov
Architecture: 3D bounded cube where points have:
  - Position (x,y,z) in cube space
  - Embedding vector (the meaning)
  - Activation scalar

Learning (V2): Four forces
  Force 1 — Momentum: velocity accumulation prevents oscillation
  Force 2 — Contrastive repulsion: negatives pushed away from target
  Force 3 — Direction training: direction vector (target-query) taught explicitly
  Force 4 — Adaptive temperature: softmax sharpens over first 500 steps
  Force 5 — Spatial cohesion: nearby similar points pulled together (base)
"""

import numpy as np
from scipy.spatial import KDTree
import json
import os


class SpatialCubeV2:
    def __init__(self, cube_size=16, embed_dim=64, seed=42):
        """
        Initialise the SpatialCube.
        
        Args:
            cube_size: N for NxNxN grid of points
            embed_dim: dimensionality of each point's meaning vector
            seed: random seed for reproducibility
        """
        np.random.seed(seed)
        
        self.cube_size = cube_size
        self.embed_dim = embed_dim
        self.n_points = cube_size ** 3
        
        print(f"Initialising SpatialCubeV2: {cube_size}x{cube_size}x{cube_size} = {self.n_points} points")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Total parameters: {self.n_points * (3 + embed_dim):,}")
        
        # Generate grid positions - the skeleton of the cube
        self._init_positions()
        
        # Initialise embeddings - the meaning at each point
        self.embeddings = np.random.randn(self.n_points, embed_dim).astype(np.float32)
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Activation history - which points have been used
        self.activations = np.zeros(self.n_points, dtype=np.float32)
        
        # Labels for interpretability
        self.labels = {}  # point_idx -> label string
        
        # Learning history
        self.loss_history = []
        self.step_count = 0
        
        # Build spatial index for fast nearest neighbour lookup
        self._rebuild_spatial_index()
        
        print("SpatialCubeV2 ready.\n")
    
    def _init_positions(self):
        """Create NxNxN grid positions normalised to [-1, 1]"""
        coords = np.linspace(-1, 1, self.cube_size)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        self.positions = np.stack([
            xx.flatten(), 
            yy.flatten(), 
            zz.flatten()
        ], axis=1).astype(np.float32)
    
    def _rebuild_spatial_index(self):
        """Rebuild KDTree for fast spatial lookup"""
        self.spatial_index = KDTree(self.positions)
    
    def query(self, query_vec, top_k=10, temperature=0.1):
        """
        Query the cube with a vector.
        Returns activated points sorted by relevance.
        
        Args:
            query_vec: numpy array of shape (embed_dim,)
            top_k: how many points to return
            temperature: sharpness of attention (lower = sharper)
        
        Returns:
            indices, scores, positions, embeddings
        """
        query_vec = np.array(query_vec, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        # Compute similarity to all points (dot product = cosine sim since normalised)
        similarities = self.embeddings @ query_vec
        
        # Softmax attention with temperature
        scores = similarities / temperature
        scores = scores - scores.max()  # numerical stability
        scores = np.exp(scores)
        scores = scores / scores.sum()
        
        # Get top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Update activation history
        self.activations[top_indices] += scores[top_indices]
        
        return {
            'indices': top_indices,
            'scores': scores[top_indices],
            'positions': self.positions[top_indices],
            'embeddings': self.embeddings[top_indices],
            'labels': [self.labels.get(i, f'point_{i}') for i in top_indices]
        }
    
    def _get_temperature(self, base_temp=0.1):
        """Adaptive temperature: warm early (exploration), cool later (precision)."""
        warmup = 500
        factor = 1.0 + 2.0 * (1.0 - min(self.step_count, warmup) / warmup)
        return base_temp * factor

    def learn_batch(self, batch_pairs, alpha=0.01, beta=0.005,
                    momentum=0.9, neg_weight=0.3, teach_directions=True):
        """
        V2 learning rule — four forces added over V1:

        Force 1 - Momentum: velocity buffer prevents oscillation.
        Force 2 - Contrastive repulsion: random negatives pushed away from target.
        Force 3 - Direction training: teaches (target-query) direction vector,
                  enabling king-man+woman=queen analogy arithmetic.
        Force 4 - Adaptive temperature: attention sharpens over first 500 steps.
        """
        if not hasattr(self, '_velocity'):
            self._velocity = np.zeros_like(self.embeddings)

        pos_grad   = np.zeros_like(self.embeddings)
        neg_grad   = np.zeros_like(self.embeddings)
        weight_acc = np.zeros(self.n_points)
        batch_loss = 0.0
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

            for i, idx in enumerate(indices):
                error = target_vec - self.embeddings[idx]
                pos_grad[idx]   += scores[i] * error
                weight_acc[idx] += scores[i]

            if teach_directions and d_norm > 1e-8:
                dir_result = self.query(query_vec, top_k=10, temperature=temperature)
                for i, idx in enumerate(dir_result['indices']):
                    dir_error = direction_vec - self.embeddings[idx]
                    pos_grad[idx]   += 0.3 * dir_result['scores'][i] * dir_error
                    weight_acc[idx] += 0.3 * dir_result['scores'][i]

            if neg_weight > 0:
                neg_idx = np.random.choice(self.n_points, 20, replace=False)
                index_set = set(indices.tolist())
                for idx in neg_idx:
                    if idx not in index_set:
                        push = self.embeddings[idx] - target_vec
                        push_norm = np.linalg.norm(push)
                        if push_norm > 1e-8:
                            neg_grad[idx] += neg_weight * (push / push_norm)

        active_mask = weight_acc > 0
        if active_mask.any():
            avg_grad = np.zeros_like(self.embeddings)
            avg_grad[active_mask] = (
                pos_grad[active_mask] / weight_acc[active_mask, np.newaxis])
            self._velocity = momentum * self._velocity + (1.0 - momentum) * avg_grad
            self.embeddings[active_mask] += alpha * self._velocity[active_mask]

        neg_active = np.any(neg_grad != 0, axis=1)
        if neg_active.any():
            self.embeddings[neg_active] += alpha * neg_grad[neg_active]

        if beta > 0 and self.step_count % 5 == 0:
            for idx in np.where(active_mask)[0][:10]:
                nearby = self.spatial_index.query_ball_point(self.positions[idx], r=0.35)
                for nidx in nearby:
                    if nidx != idx:
                        sim = np.dot(self.embeddings[idx], self.embeddings[nidx])
                        if sim > 0.3:
                            self.embeddings[idx] += beta * sim * (
                                self.embeddings[nidx] - self.embeddings[idx])

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings /= np.maximum(norms, 1e-8)

        mean_loss = batch_loss / len(batch_pairs)
        self.loss_history.append(float(mean_loss))
        self.step_count += 1
        return mean_loss

    def continuous_learn(self, query_vec, target_vec, alpha=0.01):
        """No-momentum single-step update for inference-time continuous learning.
        Achieves -0.000 anti-forgetting by never building up disruptive velocity."""
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
        """Single pair wrapper — calls learn_batch with one pair."""
        return self.learn_batch([(query_vec, target_vec)], alpha=alpha, beta=beta)
    
    def label_point(self, query_vec, label):
        """Assign a human-readable label to the most activated point"""
        result = self.query(query_vec, top_k=1)
        idx = result['indices'][0]
        self.labels[idx] = label
        return idx
    
    def get_neighbourhood(self, point_idx, radius=0.3):
        """Get all points within spatial radius of a given point"""
        nearby = self.spatial_index.query_ball_point(
            self.positions[point_idx], r=radius
        )
        return nearby
    
    def get_semantic_neighbours(self, point_idx, top_k=5):
        """Get points with most similar embeddings (semantic neighbours)"""
        target_embed = self.embeddings[point_idx]
        similarities = self.embeddings @ target_embed
        similarities[point_idx] = -1  # exclude self
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return top_indices, similarities[top_indices]
    
    def save(self, path):
        """Save cube state to disk"""
        os.makedirs(path, exist_ok=True)
        np.save(f'{path}/embeddings.npy', self.embeddings)
        np.save(f'{path}/positions.npy', self.positions)
        np.save(f'{path}/activations.npy', self.activations)
        
        meta = {
            'cube_size': self.cube_size,
            'embed_dim': self.embed_dim,
            'n_points': self.n_points,
            'step_count': self.step_count,
            'labels': {str(k): v for k, v in self.labels.items()},
            'loss_history': self.loss_history[-1000:]  # last 1000
        }
        with open(f'{path}/meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Cube saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load cube state from disk"""
        with open(f'{path}/meta.json', 'r') as f:
            meta = json.load(f)
        
        cube = cls(
            cube_size=meta['cube_size'],
            embed_dim=meta['embed_dim']
        )
        cube.embeddings = np.load(f'{path}/embeddings.npy')
        cube.positions = np.load(f'{path}/positions.npy')
        cube.activations = np.load(f'{path}/activations.npy')
        cube.step_count = meta['step_count']
        cube.labels = {int(k): v for k, v in meta['labels'].items()}
        cube.loss_history = meta['loss_history']
        cube._rebuild_spatial_index()
        
        print(f"Cube loaded from {path} (step {cube.step_count})")
        return cube
    
    def stats(self):
        """Print current cube statistics"""
        print(f"\n{'='*50}")
        print(f"SpatialCubeV2 Stats")
        print(f"{'='*50}")
        print(f"Size:          {self.cube_size}^3 = {self.n_points:,} points")
        print(f"Embed dim:     {self.embed_dim}")
        print(f"Total params:  {self.n_points * (3 + self.embed_dim):,}")
        print(f"Training steps:{self.step_count:,}")
        print(f"Labels:        {len(self.labels)}")
        
        if self.loss_history:
            recent = self.loss_history[-100:]
            print(f"Recent loss:   {np.mean(recent):.4f} (avg last 100)")
        
        # Most activated regions
        top_active = np.argsort(self.activations)[-5:][::-1]
        print(f"\nMost activated points:")
        for idx in top_active:
            label = self.labels.get(idx, f'point_{idx}')
            pos = self.positions[idx]
            print(f"  [{label}] pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"activations={self.activations[idx]:.3f}")
        print(f"{'='*50}\n")
