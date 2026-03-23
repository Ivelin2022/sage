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

Learning: Two forces (V1)
  Force 1 — Semantic attraction: embeddings pulled toward targets via Hebbian update
  Force 2 — Spatial cohesion: nearby similar points pulled together
"""

import numpy as np
from scipy.spatial import KDTree
import json
import os


class SpatialCube:
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
        
        print(f"Initialising SpatialCube: {cube_size}x{cube_size}x{cube_size} = {self.n_points} points")
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
        
        print("SpatialCube ready.\n")
    
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
    
    def learn_batch(self, batch_pairs, alpha=0.01, beta=0.005):
        """
        FIXED learning rule using batch averaging.
        
        Problem with previous rule:
          - Each pair updated immediately, overwriting previous updates
          - Conflicting targets caused oscillation
          - Normalisation fought the learning signal
        
        Fix:
          - Accumulate gradient signals across ALL pairs in batch
          - Apply ONE averaged update at the end
          - Stable convergence point exists now
        
        Args:
            batch_pairs: list of (query_vec, target_vec) tuples
            alpha: semantic attraction learning rate
            beta: spatial cohesion rate
        
        Returns:
            mean loss across batch
        """
        # Accumulate gradients here â€” shape (n_points, embed_dim)
        grad_accumulator = np.zeros_like(self.embeddings)
        weight_accumulator = np.zeros(self.n_points)
        
        batch_loss = 0.0
        
        for query_vec, target_vec in batch_pairs:
            query_vec = np.array(query_vec, dtype=np.float32)
            target_vec = np.array(target_vec, dtype=np.float32)
            
            query_vec  = query_vec  / (np.linalg.norm(query_vec)  + 1e-8)
            target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)
            
            # Query â€” get activated points and scores
            result = self.query(query_vec, top_k=20)
            indices = result['indices']
            scores  = result['scores']
            
            # Current response = weighted sum of activated embeddings
            current_response = np.sum(
                self.embeddings[indices] * scores[:, np.newaxis], axis=0
            )
            
            # Loss = cosine distance (1 - similarity) â€” more stable than L2
            similarity = np.dot(current_response, target_vec) / (
                np.linalg.norm(current_response) + 1e-8
            )
            loss = 1.0 - similarity
            batch_loss += loss
            
            # ACCUMULATE gradient: each activated point pulled toward target
            # weighted by how much it activated
            for i, idx in enumerate(indices):
                error = target_vec - self.embeddings[idx]
                grad_accumulator[idx] += scores[i] * error
                weight_accumulator[idx] += scores[i]
        
        # APPLY averaged update â€” this is the key fix
        active_mask = weight_accumulator > 0
        if active_mask.any():
            # Average the accumulated gradients
            avg_grad = np.zeros_like(self.embeddings)
            avg_grad[active_mask] = (
                grad_accumulator[active_mask] / 
                weight_accumulator[active_mask, np.newaxis]
            )
            
            # Apply update
            self.embeddings[active_mask] += alpha * avg_grad[active_mask]
            
            # Force 2: Spatial cohesion
            # Spatially close points with similar embeddings reinforce each other
            if beta > 0 and self.step_count % 5 == 0:
                active_indices = np.where(active_mask)[0]
                for idx in active_indices[:10]:  # sample for efficiency
                    nearby = self.spatial_index.query_ball_point(
                        self.positions[idx], r=0.35
                    )
                    for nidx in nearby:
                        if nidx != idx:
                            sim = np.dot(self.embeddings[idx], self.embeddings[nidx])
                            if sim > 0.3:  # only cohese if already somewhat similar
                                cohesion = beta * sim * (
                                    self.embeddings[nidx] - self.embeddings[idx]
                                )
                                self.embeddings[idx] += cohesion
            
            # Renormalise ONCE after all updates â€” not per-pair
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self.embeddings /= norms
        
        mean_loss = batch_loss / len(batch_pairs)
        self.loss_history.append(float(mean_loss))
        self.step_count += 1
        
        return mean_loss

    def learn_association(self, query_vec, target_vec, alpha=0.01, beta=0.005):
        """Single pair wrapper â€” calls learn_batch with one pair"""
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
        print(f"SpatialCube Stats")
        print(f"{'='*50}")
        print(f"Size:          {self.cube_size}Â³ = {self.n_points:,} points")
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
