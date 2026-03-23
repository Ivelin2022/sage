"""
SpatialCube PyTorch GPU Version
================================
CUDA-accelerated 3D Geometric Knowledge Architecture.
Replaces numpy with torch tensors for GPU computation.

Author: Ivelin Likov
"""

import torch
import torch.nn.functional as F
import json
import os


class SpatialCubeTorch:
    def __init__(self, cube_size=16, embed_dim=64, seed=42, device=None):
        """
        Initialise the SpatialCube with PyTorch tensors.

        Args:
            cube_size: N for NxNxN grid of points
            embed_dim: dimensionality of each point's meaning vector
            seed: random seed for reproducibility
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        torch.manual_seed(seed)

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.cube_size = cube_size
        self.embed_dim = embed_dim
        self.n_points = cube_size ** 3

        print(f"Initialising SpatialCubeTorch: {cube_size}x{cube_size}x{cube_size} = {self.n_points:,} points")
        print(f"Embedding dimension: {embed_dim}")
        print(f"Total parameters: {self.n_points * (3 + embed_dim):,}")
        print(f"Device: {self.device}")

        # Generate grid positions - the skeleton of the cube
        self._init_positions()

        # Initialise embeddings - the meaning at each point
        self.embeddings = torch.randn(self.n_points, embed_dim, device=self.device, dtype=torch.float32)
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        # Activation history - which points have been used
        self.activations = torch.zeros(self.n_points, device=self.device, dtype=torch.float32)

        # Labels for interpretability
        self.labels = {}  # point_idx -> label string

        # Learning history
        self.loss_history = []
        self.step_count = 0

        print("SpatialCubeTorch ready.\n")

    def _init_positions(self):
        """Create NxNxN grid positions normalised to [-1, 1]"""
        coords = torch.linspace(-1, 1, self.cube_size, device=self.device)
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')
        self.positions = torch.stack([
            xx.flatten(),
            yy.flatten(),
            zz.flatten()
        ], dim=1).float()

    def query(self, query_vec, top_k=10, temperature=0.1, update_activations=True):
        """
        Query the cube with a vector (GPU-accelerated).
        Returns activated points sorted by relevance.

        Args:
            query_vec: tensor of shape (embed_dim,)
            top_k: how many points to return
            temperature: sharpness of attention (lower = sharper)
            update_activations: whether to update activation history

        Returns:
            dict with indices, scores, positions, embeddings, labels
        """
        if not isinstance(query_vec, torch.Tensor):
            query_vec = torch.tensor(query_vec, device=self.device, dtype=torch.float32)
        else:
            query_vec = query_vec.to(self.device)

        query_vec = F.normalize(query_vec.unsqueeze(0), p=2, dim=1).squeeze(0)

        # Compute similarity to all points (matrix-vector multiply on GPU)
        similarities = self.embeddings @ query_vec

        # Softmax attention with temperature
        scores = F.softmax(similarities / temperature, dim=0)

        # Get top-k
        top_scores, top_indices = torch.topk(scores, top_k)

        # Update activation history (optional)
        if update_activations:
            self.activations[top_indices] += top_scores

        return {
            'indices': top_indices,
            'scores': top_scores,
            'positions': self.positions[top_indices],
            'embeddings': self.embeddings[top_indices],
            'labels': [self.labels.get(i.item(), f'point_{i.item()}') for i in top_indices]
        }

    def learn_batch(self, batch_pairs, alpha=0.01, beta=0.005, record_loss=True):
        """
        GPU-accelerated batch learning.

        Args:
            batch_pairs: list of (query_vec, target_vec) tuples
            alpha: semantic attraction learning rate
            beta: spatial cohesion rate

        Returns:
            mean loss across batch
        """
        batch_size = len(batch_pairs)

        # Convert batch to tensors
        queries = torch.stack([
            torch.tensor(q, device=self.device, dtype=torch.float32) if not isinstance(q, torch.Tensor)
            else q.to(self.device)
            for q, _ in batch_pairs
        ])
        targets = torch.stack([
            torch.tensor(t, device=self.device, dtype=torch.float32) if not isinstance(t, torch.Tensor)
            else t.to(self.device)
            for _, t in batch_pairs
        ])

        # Normalize
        queries = F.normalize(queries, p=2, dim=1)
        targets = F.normalize(targets, p=2, dim=1)

        # Compute all similarities at once: (batch_size, n_points)
        all_similarities = queries @ self.embeddings.T

        # Softmax attention for each query
        all_scores = F.softmax(all_similarities / 0.1, dim=1)

        # Get top-k indices for each query
        top_k = 20
        top_scores, top_indices = torch.topk(all_scores, top_k, dim=1)

        # Compute current responses: weighted sum of embeddings
        # Shape: (batch_size, top_k, embed_dim)
        batch_embeddings = self.embeddings[top_indices]
        # Weighted sum: (batch_size, embed_dim)
        current_responses = torch.einsum('bk,bkd->bd', top_scores, batch_embeddings)

        # Loss = 1 - cosine similarity
        current_responses_norm = F.normalize(current_responses, p=2, dim=1)
        similarities = (current_responses_norm * targets).sum(dim=1)
        losses = 1.0 - similarities
        batch_loss = losses.mean()

        # Accumulate gradients
        grad_accumulator = torch.zeros_like(self.embeddings)
        weight_accumulator = torch.zeros(self.n_points, device=self.device)

        # For each sample in batch
        for b in range(batch_size):
            indices = top_indices[b]
            scores_b = top_scores[b]
            target_b = targets[b]

            # Error for each activated point
            errors = target_b.unsqueeze(0) - self.embeddings[indices]  # (top_k, embed_dim)
            weighted_errors = scores_b.unsqueeze(1) * errors  # (top_k, embed_dim)

            # Accumulate
            grad_accumulator.index_add_(0, indices, weighted_errors)
            weight_accumulator.index_add_(0, indices, scores_b)

        # Apply averaged update
        active_mask = weight_accumulator > 0
        if active_mask.any():
            avg_grad = torch.zeros_like(self.embeddings)
            avg_grad[active_mask] = grad_accumulator[active_mask] / weight_accumulator[active_mask].unsqueeze(1)

            # Apply update
            self.embeddings[active_mask] += alpha * avg_grad[active_mask]

            # Spatial cohesion (every 5 steps, sample for efficiency)
            if beta > 0 and self.step_count % 5 == 0:
                active_indices = torch.where(active_mask)[0][:10]
                for idx in active_indices:
                    # Find nearby points using positions
                    distances = torch.norm(self.positions - self.positions[idx], dim=1)
                    nearby_mask = (distances < 0.35) & (distances > 0)
                    nearby_indices = torch.where(nearby_mask)[0]

                    if len(nearby_indices) > 0:
                        sims = (self.embeddings[idx] @ self.embeddings[nearby_indices].T)
                        cohesion_mask = sims > 0.3
                        if cohesion_mask.any():
                            cohesion_indices = nearby_indices[cohesion_mask]
                            cohesion_sims = sims[cohesion_mask]
                            cohesion = beta * cohesion_sims.unsqueeze(1) * (
                                self.embeddings[cohesion_indices] - self.embeddings[idx]
                            )
                            self.embeddings[idx] += cohesion.sum(dim=0)

            # Renormalize
            self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        mean_loss = batch_loss.item()
        if record_loss:
            self.loss_history.append(mean_loss)
            self.step_count += 1

        return mean_loss

    def learn_association(self, query_vec, target_vec, alpha=0.01, beta=0.005):
        """Single pair wrapper"""
        return self.learn_batch([(query_vec, target_vec)], alpha=alpha, beta=beta)

    def query_batch(self, queries, top_k=10, temperature=0.1):
        """
        Batched query - THE KEY TO GPU PERFORMANCE.

        Instead of launching N separate GPU kernels (slow due to overhead),
        this processes all queries in ONE parallel operation.

        Args:
            queries: tensor of shape (batch_size, embed_dim)
            top_k: how many points to return per query
            temperature: sharpness of attention

        Returns:
            dict with batched results
        """
        if not isinstance(queries, torch.Tensor):
            queries = torch.tensor(queries, device=self.device, dtype=torch.float32)
        else:
            queries = queries.to(self.device)

        batch_size = queries.shape[0]

        # Normalize all queries at once
        queries = F.normalize(queries, p=2, dim=1)

        # ONE matrix multiplication for ALL queries: (batch, embed) @ (embed, n_points) -> (batch, n_points)
        # This is where GPU parallelism shines - thousands of CUDA cores working simultaneously
        all_similarities = queries @ self.embeddings.T

        # Softmax attention for all queries at once
        all_scores = F.softmax(all_similarities / temperature, dim=1)

        # Get top-k for all queries in one operation
        top_scores, top_indices = torch.topk(all_scores, top_k, dim=1)

        # Gather embeddings for all top-k results
        # Shape: (batch_size, top_k, embed_dim)
        top_embeddings = self.embeddings[top_indices]

        # Gather positions
        top_positions = self.positions[top_indices]

        return {
            'indices': top_indices,           # (batch_size, top_k)
            'scores': top_scores,             # (batch_size, top_k)
            'positions': top_positions,       # (batch_size, top_k, 3)
            'embeddings': top_embeddings,     # (batch_size, top_k, embed_dim)
            'batch_size': batch_size
        }

    def label_point(self, query_vec, label):
        """Assign a human-readable label to the most activated point"""
        result = self.query(query_vec, top_k=1)
        idx = result['indices'][0].item()
        self.labels[idx] = label
        return idx

    def get_neighbourhood(self, point_idx, radius=0.3):
        """Get all points within spatial radius of a given point"""
        distances = torch.norm(self.positions - self.positions[point_idx], dim=1)
        nearby = torch.where(distances < radius)[0]
        return nearby.cpu().numpy()

    def get_semantic_neighbours(self, point_idx, top_k=5):
        """Get points with most similar embeddings"""
        target_embed = self.embeddings[point_idx]
        similarities = self.embeddings @ target_embed
        similarities[point_idx] = -1  # exclude self
        top_sims, top_indices = torch.topk(similarities, top_k)
        return top_indices.cpu().numpy(), top_sims.cpu().numpy()

    def save(self, path):
        """Save cube state to disk"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.embeddings.cpu(), f'{path}/embeddings.pt')
        torch.save(self.positions.cpu(), f'{path}/positions.pt')
        torch.save(self.activations.cpu(), f'{path}/activations.pt')

        meta = {
            'cube_size': self.cube_size,
            'embed_dim': self.embed_dim,
            'n_points': self.n_points,
            'step_count': self.step_count,
            'labels': {str(k): v for k, v in self.labels.items()},
            'loss_history': self.loss_history[-1000:],
            'device': str(self.device)
        }
        with open(f'{path}/meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Cube saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        """Load cube state from disk"""
        with open(f'{path}/meta.json', 'r') as f:
            meta = json.load(f)

        cube = cls(
            cube_size=meta['cube_size'],
            embed_dim=meta['embed_dim'],
            device=device
        )
        cube.embeddings = torch.load(f'{path}/embeddings.pt').to(cube.device)
        cube.positions = torch.load(f'{path}/positions.pt').to(cube.device)
        cube.activations = torch.load(f'{path}/activations.pt').to(cube.device)
        cube.step_count = meta['step_count']
        cube.labels = {int(k): v for k, v in meta['labels'].items()}
        cube.loss_history = meta['loss_history']

        print(f"Cube loaded from {path} (step {cube.step_count})")
        return cube

    def stats(self):
        """Print current cube statistics"""
        print(f"\n{'='*50}")
        print(f"SpatialCubeTorch Stats")
        print(f"{'='*50}")
        print(f"Size:          {self.cube_size}^3 = {self.n_points:,} points")
        print(f"Embed dim:     {self.embed_dim}")
        print(f"Total params:  {self.n_points * (3 + self.embed_dim):,}")
        print(f"Device:        {self.device}")
        print(f"Training steps:{self.step_count:,}")
        print(f"Labels:        {len(self.labels)}")

        if self.loss_history:
            recent = self.loss_history[-100:]
            print(f"Recent loss:   {sum(recent)/len(recent):.4f} (avg last 100)")

        # Most activated regions
        top_active = torch.argsort(self.activations)[-5:].flip(0)
        print(f"\nMost activated points:")
        for idx in top_active:
            idx = idx.item()
            label = self.labels.get(idx, f'point_{idx}')
            pos = self.positions[idx].cpu().numpy()
            print(f"  [{label}] pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"activations={self.activations[idx].item():.3f}")
        print(f"{'='*50}\n")

    def to_numpy(self):
        """Convert tensors to numpy for visualization"""
        return {
            'embeddings': self.embeddings.cpu().numpy(),
            'positions': self.positions.cpu().numpy(),
            'activations': self.activations.cpu().numpy()
        }
