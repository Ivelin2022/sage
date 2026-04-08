"""
Test 8A — Residual Query Chaining (Sequential Cross-Cube Composition)
=====================================================================
Author: Ivelin Likov

Tests whether SAGE can chain two specialist cubes geometrically:
  Cube A (perception/context) → residual chain → Cube B (action/response)

The experiment:
  - Cube A stores: {situation: situation_embedding}
  - Cube B stores: {situation_context: action_embedding}
  - Query: a situation
  - Baseline: query Cube B directly with situation
  - Chained: query Cube A first, mix output with query, then query Cube B

Expected: chaining beats direct query when domains are sequentially dependent.

Results saved to: test_8a_chaining_results.json + test_8a_chaining.png
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

# ── Import cube ─────────────────────────────────────────────────────────────
try:
    from cube_core_v2_torch import SpatialCubeV2Torch as Cube
    print("Using SpatialCubeV2Torch")
except ImportError:
    # fallback: minimal cube
    class Cube:
        def __init__(self, cube_size=32, embed_dim=64, seed=42, device=None):
            torch.manual_seed(seed)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.n_points = cube_size ** 3
            self.embed_dim = embed_dim
            self.embeddings = F.normalize(
                torch.randn(self.n_points, embed_dim, device=self.device), p=2, dim=1)
            print(f"Fallback cube: {cube_size}^3={self.n_points:,} | {self.device}")

        def learn_batch(self, pairs, alpha=0.05, **kwargs):
            Q = F.normalize(torch.stack([p[0] for p in pairs]).to(self.device), p=2, dim=1)
            T = F.normalize(torch.stack([p[1] for p in pairs]).to(self.device), p=2, dim=1)
            sims = Q @ self.embeddings.T
            scores = F.softmax(sims / 0.1, dim=1)
            grad = scores.T @ (T - (scores.unsqueeze(2) * self.embeddings.unsqueeze(0)).sum(1))
            self.embeddings += alpha * grad
            self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
            return (1 - (scores.unsqueeze(2) * self.embeddings.unsqueeze(0)).sum(1) *
                    T.unsqueeze(1)).mean().item()

# ── Config ──────────────────────────────────────────────────────────────────
CUBE_SIZE   = 32
EMBED_DIM   = 64
N_CONCEPTS  = 100       # concepts per domain
EPOCHS      = 20
BATCH_SIZE  = 32
ALPHA       = 0.05
TEMPERATURE = 0.05
N_RUNS      = 5         # repeat for stability
LAMBDAS     = [0.0, 0.3, 0.5, 0.7, 1.0]   # mixing coefficients to sweep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
print(f"Cube: {CUBE_SIZE}³ | Embed: {EMBED_DIM}d | Concepts: {N_CONCEPTS} | Epochs: {EPOCHS}")


# ── Core functions ────────────────────────────────────────────────────────────

def retrieve(cube, query, temperature=TEMPERATURE):
    """Softmax-weighted retrieval. Returns (response_vector, top1_idx, top1_score)."""
    q = F.normalize(query.unsqueeze(0), p=2, dim=1).squeeze(0)
    sims = cube.embeddings @ q
    scores = F.softmax(sims / temperature, dim=0)
    response = (scores.unsqueeze(1) * cube.embeddings).sum(0)
    top1_idx = sims.argmax().item()
    top1_score = sims.max().item()
    return F.normalize(response.unsqueeze(0), p=2, dim=1).squeeze(0), top1_idx, top1_score


def chained_retrieve(cube_a, cube_b, query, lam=0.5, temperature=TEMPERATURE):
    """
    Residual query chaining:
      1. Retrieve from Cube A
      2. Mix output with original query: q_b = normalize(q + λ * output_a)
      3. Retrieve from Cube B using q_b
    """
    # Step 1: Cube A retrieval
    output_a, _, _ = retrieve(cube_a, query, temperature)

    # Step 2: Cosine-gated mixing
    cos_sim = F.cosine_similarity(query.unsqueeze(0), output_a.unsqueeze(0)).item()
    effective_lam = lam * max(0.0, cos_sim)  # scale by relevance
    q_b = F.normalize(query + effective_lam * output_a, p=2, dim=0)

    # Step 3: Cube B retrieval
    output_b, top1_idx, top1_score = retrieve(cube_b, q_b, temperature)
    return output_b, top1_idx, top1_score, cos_sim


def train_cube(cube, concept_vecs, epochs=EPOCHS, batch_size=BATCH_SIZE, alpha=ALPHA):
    """Train a cube to store concept vectors (identity: query=target)."""
    n = len(concept_vecs)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch = concept_vecs[idx]
            pairs = [(batch[j], batch[j]) for j in range(len(idx))]
            cube.learn_batch(pairs, alpha=alpha, beta=0.003,
                             teach_directions=False, momentum=0.9,
                             neg_weight=0.1)


def evaluate(cube_a, cube_b, query_vecs, target_vecs, lam):
    """
    Evaluate cross-cube retrieval.
    query_vecs: queries to present
    target_vecs: what we expect Cube B to return (ground truth)

    Returns: top1_accuracy, mean_cosine
    """
    correct = 0
    cosines = []

    for i in range(len(query_vecs)):
        q = query_vecs[i]

        if lam == 0.0:
            # Baseline: direct query to Cube B
            output, _, _ = retrieve(cube_b, q)
        else:
            # Chained: A → B
            output, _, _, _ = chained_retrieve(cube_a, cube_b, q, lam=lam)

        target = target_vecs[i]
        cos = F.cosine_similarity(output.unsqueeze(0), target.unsqueeze(0)).item()
        cosines.append(cos)

        # Top-1: is output closest to the correct target?
        all_sims = target_vecs @ output
        if all_sims.argmax().item() == i:
            correct += 1

    return correct / len(query_vecs), float(np.mean(cosines))


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(seed, lam_sweep):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate concept pairs: (situation, action)
    # Situation → Cube A context
    # Action    → Cube B response
    # They share a directional relationship: action ≈ rotate(situation)
    situations = F.normalize(torch.randn(N_CONCEPTS, EMBED_DIM, device=device), p=2, dim=1)

    # Actions are NOT random — they are the situations rotated by a fixed
    # orthogonal matrix. This creates a clean sequential dependency:
    # knowing the situation (from Cube A) helps find the right action (in Cube B).
    torch.manual_seed(seed + 999)
    R = torch.linalg.qr(torch.randn(EMBED_DIM, EMBED_DIM, device=device))[0]  # random rotation
    actions = F.normalize(situations @ R.T, p=2, dim=1)

    # Context vectors — what Cube A actually stores:
    # Cube A stores situations, but also *implicitly encodes* the action direction
    # via direction training. A pure situation embedding + action embedding
    # together give the chain its power.
    context_a = situations           # Cube A: situation space
    context_b = actions              # Cube B: action space

    # Train both cubes
    cube_a = Cube(cube_size=CUBE_SIZE, embed_dim=EMBED_DIM, seed=seed, device=str(device))
    cube_b = Cube(cube_size=CUBE_SIZE, embed_dim=EMBED_DIM, seed=seed+1, device=str(device))

    train_cube(cube_a, context_a)
    train_cube(cube_b, context_b)

    # Evaluation: use the first 80% as train, last 20% as test
    n_test = max(1, N_CONCEPTS // 5)
    test_q = situations[-n_test:]   # test queries (situations)
    test_t = actions[-n_test:]      # correct answers (actions)

    results = {}
    for lam in lam_sweep:
        acc, cos = evaluate(cube_a, cube_b, test_q, test_t, lam)
        results[lam] = {'acc': acc, 'cosine': cos}

    return results


print("\nRunning cross-cube composition test...")
print("="*55)

all_results = {lam: {'acc': [], 'cosine': []} for lam in LAMBDAS}

for run in range(N_RUNS):
    t0 = time.perf_counter()
    run_results = run_experiment(seed=42 + run * 100, lam_sweep=LAMBDAS)
    t1 = time.perf_counter()

    print(f"\nRun {run+1}/{N_RUNS} ({t1-t0:.1f}s):")
    for lam in LAMBDAS:
        r = run_results[lam]
        all_results[lam]['acc'].append(r['acc'])
        all_results[lam]['cosine'].append(r['cosine'])
        tag = "(baseline)" if lam == 0.0 else f"(λ={lam})"
        print(f"  {tag:<12} Top-1={r['acc']:.3f}  Cosine={r['cosine']:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
baseline_acc = float(np.mean(all_results[0.0]['acc']))
best_lam, best_acc = 0.0, baseline_acc

for lam in LAMBDAS:
    acc_mean = float(np.mean(all_results[lam]['acc']))
    acc_std  = float(np.std(all_results[lam]['acc']))
    cos_mean = float(np.mean(all_results[lam]['cosine']))
    delta    = acc_mean - baseline_acc
    sign     = "↑" if delta > 0 else ("↓" if delta < 0 else "—")
    tag      = "(baseline)" if lam == 0.0 else f"(λ={lam})"
    print(f"  {tag:<12} Top-1={acc_mean:.3f}±{acc_std:.3f}  "
          f"Cosine={cos_mean:.4f}  Δ={delta:+.3f} {sign}")
    if acc_mean > best_acc:
        best_acc, best_lam = acc_mean, lam

improvement = best_acc - baseline_acc
print(f"\nBest λ={best_lam}: +{improvement:.3f} Top-1 accuracy vs baseline")
verdict = "✓ CHAINING HELPS" if improvement > 0.05 else \
          ("~ MARGINAL" if improvement > 0.01 else "✗ NO BENEFIT")
print(f"Verdict: {verdict}")

# ── Save results ──────────────────────────────────────────────────────────────
results_data = {
    'metadata': {
        'experiment': 'Test 8A — Residual Query Chaining',
        'run_date': datetime.now().isoformat(),
        'cube_size': CUBE_SIZE, 'embed_dim': EMBED_DIM,
        'n_concepts': N_CONCEPTS, 'epochs': EPOCHS,
        'n_runs': N_RUNS, 'lambdas': LAMBDAS,
        'device': str(device),
    },
    'summary': {
        lam: {
            'acc_mean': float(np.mean(all_results[lam]['acc'])),
            'acc_std':  float(np.std(all_results[lam]['acc'])),
            'cos_mean': float(np.mean(all_results[lam]['cosine'])),
        }
        for lam in LAMBDAS
    },
    'verdict': verdict,
    'best_lambda': best_lam,
    'improvement': improvement,
}

out_json = 'test_8a_chaining_results.json'
with open(out_json, 'w') as f:
    json.dump(results_data, f, indent=2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Test 8A — Residual Query Chaining\nCross-Cube Sequential Composition',
             fontsize=13, fontweight='bold')

accs  = [float(np.mean(all_results[l]['acc']))  for l in LAMBDAS]
stds  = [float(np.std(all_results[l]['acc']))   for l in LAMBDAS]
coses = [float(np.mean(all_results[l]['cosine'])) for l in LAMBDAS]

ax = axes[0]
colors = ['#CC3333' if l == 0.0 else '#3366CC' for l in LAMBDAS]
bars = ax.bar([str(l) for l in LAMBDAS], accs, yerr=stds,
              color=colors, alpha=0.85, capsize=5)
ax.axhline(baseline_acc, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_acc:.3f})')
ax.set_xlabel('Lambda (mixing coefficient)')
ax.set_ylabel('Top-1 Accuracy')
ax.set_title('Retrieval Accuracy by Mixing Coefficient')
ax.legend()
ax.set_ylim(0, 1.05)

ax = axes[1]
ax.plot([str(l) for l in LAMBDAS], coses, 'o-', color='#3366CC', linewidth=2, markersize=8)
ax.axhline(coses[0], color='red', linestyle='--', alpha=0.5, label=f'Baseline ({coses[0]:.4f})')
ax.set_xlabel('Lambda (mixing coefficient)')
ax.set_ylabel('Mean Cosine Similarity (response vs target)')
ax.set_title('Response Quality by Mixing Coefficient')
ax.legend()

plt.tight_layout()
out_png = 'test_8a_chaining.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResults: {out_json}")
print(f"Plot:    {out_png}")
print(f"Done.")
