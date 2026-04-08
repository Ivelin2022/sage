"""
Test 8B — Shared Grid Coordinates (O(1) Cross-Cube Association)
================================================================
Author: Ivelin Likov

Tests whether storing related concepts at the SAME grid position
across two specialist cubes creates gradient-free O(1) association.

Inspired by the Tolman-Eichenbaum Machine (Whittington et al., Cell 2020)
which separates "where" (structural code = grid position) from
"what" (content = embedding vector).

The experiment:
  - Cube A (perception): stores situation embeddings
  - Cube B (action): stores action embeddings
  - Condition 1 (ALIGNED): related pairs stored at SAME grid position
  - Condition 2 (RANDOM): related pairs stored at DIFFERENT positions
  - Condition 3 (DIRECT): baseline, query Cube B directly with situation

Hypothesis: aligned > random >> direct
  If true: shared grid coordinates give O(1) cross-cube association for free.

Results: test_8b_shared_coords_results.json + test_8b_shared_coords.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# ── Import cube ──────────────────────────────────────────────────────────────
try:
    from cube_core_v2_torch import SpatialCubeV2Torch as Cube
    print("Using SpatialCubeV2Torch")
except ImportError:
    class Cube:
        def __init__(self, cube_size=32, embed_dim=64, seed=42, device=None):
            torch.manual_seed(seed)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.n_points = cube_size ** 3
            self.embed_dim = embed_dim
            self.embeddings = F.normalize(
                torch.randn(self.n_points, embed_dim, device=self.device), p=2, dim=1)
            print(f"Fallback: {cube_size}^3 | {self.device}")

        def learn_batch(self, pairs, alpha=0.05, **kwargs):
            Q = F.normalize(torch.stack([p[0] for p in pairs]).to(self.device), p=2, dim=1)
            T = F.normalize(torch.stack([p[1] for p in pairs]).to(self.device), p=2, dim=1)
            sims = Q @ self.embeddings.T
            scores = F.softmax(sims / 0.1, dim=1)
            grad = scores.T @ T
            wt   = scores.sum(0).unsqueeze(1)
            mask = wt.squeeze() > 1e-6
            self.embeddings[mask] += alpha * (grad[mask] / wt[mask] - self.embeddings[mask])
            self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
            return 0.0

# ── Config ───────────────────────────────────────────────────────────────────
CUBE_SIZE   = 32
EMBED_DIM   = 64
N_CONCEPTS  = 200       # concepts per domain
EPOCHS      = 30        # more epochs for proper convergence
BATCH_SIZE  = 32
ALPHA       = 0.05
TEMPERATURE = 0.05
N_RUNS      = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print(f"Cube: {CUBE_SIZE}³ | Embed: {EMBED_DIM}d | Concepts: {N_CONCEPTS}")


# ── Force a concept to a specific grid position ───────────────────────────────

def force_store(cube, concept_vec, target_grid_pos, alpha=0.2, n_steps=50):
    """
    Force-store a concept at a specific grid position by
    repeatedly updating that point toward the target.
    This bypasses normal nearest-neighbour assignment.
    """
    concept_vec = F.normalize(concept_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
    for _ in range(n_steps):
        error = concept_vec - cube.embeddings[target_grid_pos]
        cube.embeddings[target_grid_pos] += alpha * error
        cube.embeddings[target_grid_pos] = F.normalize(
            cube.embeddings[target_grid_pos].unsqueeze(0), p=2, dim=1).squeeze(0)


def natural_store(cube, concept_vec, alpha=0.05, n_steps=50):
    """
    Store a concept using natural nearest-neighbour assignment.
    Returns (final_sims, grid_position) where grid_position is the
    top-1 index AFTER all training steps complete.
    """
    concept_vec = F.normalize(concept_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
    for _ in range(n_steps):
        sims = cube.embeddings @ concept_vec
        best_idx = sims.argmax().item()
        error = concept_vec - cube.embeddings[best_idx]
        cube.embeddings[best_idx] += alpha * error
        cube.embeddings[best_idx] = F.normalize(
            cube.embeddings[best_idx].unsqueeze(0), p=2, dim=1).squeeze(0)
    # Recompute final sims AFTER all training to get the correct final position
    final_sims = cube.embeddings @ concept_vec
    final_pos  = final_sims.argmax().item()
    return final_sims, final_pos


def retrieve(cube, query, temperature=TEMPERATURE):
    """Return (response_vector, top1_grid_idx, similarities)."""
    q = F.normalize(query.unsqueeze(0), p=2, dim=1).squeeze(0)
    sims = cube.embeddings @ q
    scores = F.softmax(sims / temperature, dim=0)
    response = (scores.unsqueeze(1) * cube.embeddings).sum(0)
    top1_idx = sims.argmax().item()
    return F.normalize(response.unsqueeze(0), p=2, dim=1).squeeze(0), top1_idx, sims


def shared_coord_retrieve(cube_a, cube_b, query, temperature=TEMPERATURE):
    """
    O(1) cross-cube association via shared grid coordinate:
    1. Find top-1 grid position in Cube A
    2. Directly look up the SAME position in Cube B
    """
    _, top1_a, _ = retrieve(cube_a, query, temperature)
    # Directly index Cube B at the same grid position
    associated = cube_b.embeddings[top1_a]
    return associated, top1_a


# ── Run one experiment condition ──────────────────────────────────────────────

def run_condition(seed, condition):
    """
    condition: 'aligned' | 'random' | 'direct'

    aligned: related pairs forced to same grid positions
    random:  related pairs stored at random positions (unrelated)
    direct:  baseline — query Cube B directly with Cube A's query
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate concept pairs
    situations = F.normalize(
        torch.randn(N_CONCEPTS, EMBED_DIM, device=device), p=2, dim=1)

    # Actions related to situations via fixed rotation
    torch.manual_seed(seed + 999)
    R = torch.linalg.qr(torch.randn(EMBED_DIM, EMBED_DIM, device=device))[0]
    actions = F.normalize(situations @ R.T, p=2, dim=1)

    cube_a = Cube(cube_size=CUBE_SIZE, embed_dim=EMBED_DIM, seed=seed, device=str(device))
    cube_b = Cube(cube_size=CUBE_SIZE, embed_dim=EMBED_DIM, seed=seed+1, device=str(device))

    # Assign grid positions
    n_train = int(N_CONCEPTS * 0.8)
    train_situ = situations[:n_train]
    train_acts = actions[:n_train]

    if condition == 'aligned':
        # Find where Cube A naturally stores each concept
        # then force Cube B to store the corresponding action at the SAME position
        grid_positions = []
        for i in range(n_train):
            _, gpos = natural_store(cube_a, train_situ[i], alpha=ALPHA, n_steps=EPOCHS)
            grid_positions.append(gpos)
            force_store(cube_b, train_acts[i], gpos, alpha=ALPHA, n_steps=EPOCHS)

    elif condition == 'random':
        # Store situations in Cube A naturally
        # Store actions in Cube B naturally (different positions, no alignment)
        for i in range(n_train):
            natural_store(cube_a, train_situ[i], alpha=ALPHA, n_steps=EPOCHS)
            natural_store(cube_b, train_acts[i], alpha=ALPHA, n_steps=EPOCHS)

    elif condition == 'direct':
        # Baseline: store actions in Cube B naturally, query directly with situation
        for i in range(n_train):
            natural_store(cube_b, train_acts[i], alpha=ALPHA, n_steps=EPOCHS)

    # Evaluate on test set
    test_situ = situations[n_train:]
    test_acts  = actions[n_train:]
    n_test = len(test_situ)

    correct = 0
    cosines = []

    for i in range(n_test):
        q = test_situ[i]
        target = test_acts[i]

        if condition == 'aligned':
            output, _ = shared_coord_retrieve(cube_a, cube_b, q)
        elif condition == 'random':
            output, _ = shared_coord_retrieve(cube_a, cube_b, q)
        elif condition == 'direct':
            output, _, _ = retrieve(cube_b, q)

        cos = F.cosine_similarity(output.unsqueeze(0), target.unsqueeze(0)).item()
        cosines.append(cos)
        all_sims = test_acts @ output
        if all_sims.argmax().item() == i:
            correct += 1

    return correct / n_test, float(np.mean(cosines))


# ── Main ─────────────────────────────────────────────────────────────────────
conditions = ['aligned', 'random', 'direct']
all_results = {c: {'acc': [], 'cosine': []} for c in conditions}

print("\nRunning shared grid coordinate test...")
print("="*55)

for run in range(N_RUNS):
    print(f"\nRun {run+1}/{N_RUNS}:")
    t0 = time.perf_counter()
    for cond in conditions:
        acc, cos = run_condition(seed=42 + run * 100, condition=cond)
        all_results[cond]['acc'].append(acc)
        all_results[cond]['cosine'].append(cos)
        print(f"  {cond:<8}: Top-1={acc:.3f}  Cosine={cos:.4f}")
    print(f"  ({time.perf_counter()-t0:.1f}s)")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
for cond in conditions:
    acc_mean = float(np.mean(all_results[cond]['acc']))
    acc_std  = float(np.std(all_results[cond]['acc']))
    cos_mean = float(np.mean(all_results[cond]['cosine']))
    print(f"  {cond:<8}: Top-1={acc_mean:.3f}±{acc_std:.3f}  Cosine={cos_mean:.4f}")

baseline = float(np.mean(all_results['direct']['acc']))
aligned  = float(np.mean(all_results['aligned']['acc']))
random   = float(np.mean(all_results['random']['acc']))

print(f"\nAligned vs Direct: {aligned - baseline:+.3f}")
print(f"Random  vs Direct: {random  - baseline:+.3f}")

if aligned > random and aligned > baseline + 0.05:
    verdict = "✓ SHARED COORDS WORK — aligned >> random >> direct"
elif aligned > baseline + 0.01:
    verdict = "~ MARGINAL — aligned slightly better than baseline"
else:
    verdict = "✗ NO BENEFIT — spatial alignment doesn't help"
print(f"\nVerdict: {verdict}")

# ── Save + Plot ───────────────────────────────────────────────────────────────
out_data = {
    'metadata': {
        'experiment': 'Test 8B — Shared Grid Coordinates',
        'run_date': datetime.now().isoformat(),
        'cube_size': CUBE_SIZE, 'embed_dim': EMBED_DIM,
        'n_concepts': N_CONCEPTS, 'epochs': EPOCHS,
        'n_runs': N_RUNS, 'device': str(device),
    },
    'summary': {
        cond: {
            'acc_mean': float(np.mean(all_results[cond]['acc'])),
            'acc_std':  float(np.std(all_results[cond]['acc'])),
            'cos_mean': float(np.mean(all_results[cond]['cosine'])),
        }
        for cond in conditions
    },
    'verdict': verdict,
    'aligned_vs_direct': aligned - baseline,
    'random_vs_direct':  random  - baseline,
}

out_json = 'test_8b_shared_coords_results.json'
with open(out_json, 'w') as f:
    json.dump(out_data, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Test 8B — Shared Grid Coordinates\nO(1) Cross-Cube Association via TEM-style Spatial Alignment',
             fontsize=12, fontweight='bold')

colors = {'aligned': '#2E8B57', 'random': '#FF8C00', 'direct': '#CC3333'}
labels = {'aligned': 'Aligned\n(same grid pos)', 'random': 'Random\n(diff grid pos)',
          'direct': 'Direct\n(baseline)'}

accs  = [float(np.mean(all_results[c]['acc']))    for c in conditions]
stds  = [float(np.std(all_results[c]['acc']))     for c in conditions]
coses = [float(np.mean(all_results[c]['cosine'])) for c in conditions]

ax = axes[0]
ax.bar([labels[c] for c in conditions], accs, yerr=stds,
       color=[colors[c] for c in conditions], alpha=0.85, capsize=5)
ax.set_ylabel('Top-1 Accuracy')
ax.set_title('Retrieval Accuracy by Condition')
ax.set_ylim(0, 1.05)

ax = axes[1]
ax.bar([labels[c] for c in conditions], coses,
       color=[colors[c] for c in conditions], alpha=0.85)
ax.set_ylabel('Mean Cosine Similarity (response vs target)')
ax.set_title('Response Quality by Condition')

plt.tight_layout()
out_png = 'test_8b_shared_coords.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResults: {out_json}")
print(f"Plot:    {out_png}")
print("Done.")
