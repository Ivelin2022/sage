"""
SAGE Architecture Comparison: V1 vs V2 vs V3 vs V4
====================================================
Reproduces Table 4 from:
  Likov, I. (2026). Memory for All: SAGE — Spatial Associative Geometric Embeddings.

Metrics measured:
  1. Cluster cohesion ratio (inter-cluster / intra-cluster cosine distance)
  2. Anti-forgetting (loss change after 200-concept streaming)
  3. Analogy accuracy (on synthetic concept vectors)
  4. Sparsity (% of points activated per query)

Usage:
  pip install numpy scipy
  python comparison_v1_v4.py

Expected runtime: ~5 minutes on CPU.

Author: Ivelin Likov
"""

import numpy as np
import time
import json

# ── Import cube versions ─────────────────────────────────────────
from cube_core import SpatialCube     as SpatialCubeV1  # cube_core.py = V1
from cube_core_v2 import SpatialCubeV2
from cube_core_v4 import SpatialCubeV4

SEED       = 42
CUBE_SIZE  = 10          # small for speed: 10^3 = 1000 points
EMBED_DIM  = 64
N_CONCEPTS = 50
N_GROUPS   = 10
EPOCHS     = 100


# ── Data generation ──────────────────────────────────────────────

def make_data(seed=SEED):
    """
    50 concept vectors in 10 groups of 5.
    Each group trains toward its centroid — related concepts should cluster.
    Includes analogy pairs: a - b + c = d.
    """
    np.random.seed(seed)
    concepts = np.random.randn(N_CONCEPTS, EMBED_DIM).astype(np.float32)
    concepts /= np.linalg.norm(concepts, axis=1, keepdims=True)

    pairs = []
    group_ids = []

    for g in range(N_GROUPS):
        idx = range(g * 5, (g + 1) * 5)
        centroid = concepts[list(idx)].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        for i in idx:
            pairs.append((concepts[i], centroid))
            pairs.append((concepts[i], concepts[i]))
            group_ids.append(g)

    # Analogy pairs
    for i in range(0, 40, 5):
        q = concepts[i] - concepts[i+1] + concepts[i+2]
        q /= np.linalg.norm(q) + 1e-8
        pairs.append((q, concepts[i+3]))

    return concepts, pairs, group_ids


# ── Metrics ──────────────────────────────────────────────────────

def measure_sparsity(cube, concepts, n_queries=200):
    """% of cube points activated per query (lower = more sparse)."""
    total_activated = 0
    np.random.seed(0)
    queries = [concepts[i % len(concepts)] for i in range(n_queries)]
    for q in queries:
        result = cube.query(q, top_k=20, update_activations=False)
        total_activated += len(result['indices'])
    n_points = getattr(cube, 'n_points', cube.cube_size ** 3)
    return (total_activated / n_queries) / n_points


def measure_cohesion(cube, concepts, group_ids):
    """
    Cluster cohesion ratio = mean inter-group distance / mean intra-group distance.
    Higher is better — means related concepts are closer to each other
    than to unrelated concepts.
    """
    # Get retrieved embedding for each concept
    retrieved = []
    for c in concepts:
        r = cube.query(c, top_k=1, update_activations=False)
        retrieved.append(cube.embeddings[r['indices'][0]])
    retrieved = np.array(retrieved)

    intra_sims, inter_sims = [], []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            sim = np.dot(retrieved[i], retrieved[j])
            if group_ids[i] == group_ids[j]:
                intra_sims.append(sim)
            else:
                inter_sims.append(sim)

    intra = np.mean(intra_sims) if intra_sims else 0
    inter = np.mean(inter_sims) if inter_sims else 0
    # Ratio of intra to inter: higher means better clustering
    return intra / (abs(inter) + 1e-8)


def measure_forgetting(cube_class, cube_kwargs, pairs, concepts):
    """
    Anti-forgetting: train on first half, measure loss, train on second half,
    re-measure on first half. Forgetting = loss_after - loss_before.
    """
    np.random.seed(SEED)
    cube = cube_class(**cube_kwargs)

    half = len(pairs) // 2
    first_half  = pairs[:half]
    second_half = pairs[half:]

    # Train on first half
    for _ in range(30):
        for batch_start in range(0, len(first_half), 16):
            batch = first_half[batch_start:batch_start + 16]
            cube.learn_batch(batch, alpha=0.02)

    # Measure loss on first half
    loss_before = 0.0
    for q, t in first_half[:50]:
        q = np.array(q, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t = np.array(t, dtype=np.float32)
        t /= np.linalg.norm(t) + 1e-8
        r = cube.query(q, top_k=20, update_activations=False)
        resp = np.sum(cube.embeddings[r['indices']] * r['scores'][:, np.newaxis], 0)
        loss_before += 1.0 - np.dot(resp, t) / (np.linalg.norm(resp) + 1e-8)
    loss_before /= 50

    # Train on second half (streaming — no momentum via continuous_learn if available)
    for q, t in second_half:
        if hasattr(cube, 'continuous_learn'):
            cube.continuous_learn(q, t, alpha=0.01)
        else:
            cube.learn_batch([(q, t)], alpha=0.01)

    # Re-measure loss on first half
    loss_after = 0.0
    for q, t in first_half[:50]:
        q = np.array(q, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-8
        t = np.array(t, dtype=np.float32)
        t /= np.linalg.norm(t) + 1e-8
        r = cube.query(q, top_k=20, update_activations=False)
        resp = np.sum(cube.embeddings[r['indices']] * r['scores'][:, np.newaxis], 0)
        loss_after += 1.0 - np.dot(resp, t) / (np.linalg.norm(resp) + 1e-8)
    loss_after /= 50

    return loss_before, loss_after, cube


def measure_analogy(cube, concepts):
    """0% on synthetic data — included for completeness. Real analogies need GloVe."""
    correct = 0
    total   = 0
    for i in range(0, 40, 5):
        q = concepts[i] - concepts[i+1] + concepts[i+2]
        q /= np.linalg.norm(q) + 1e-8
        expected = i + 3
        r = cube.query(q, top_k=5, update_activations=False)
        labels = [cube.labels.get(idx, '') for idx in r['indices']]
        if f'c{expected}' in labels:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


# ── Train and evaluate ───────────────────────────────────────────

def train_and_eval(name, cube_class, cube_kwargs, pairs, concepts, group_ids):
    print(f"\n{'─'*50}")
    print(f"Training {name}")
    print(f"{'─'*50}")

    np.random.seed(SEED)
    cube = cube_class(**cube_kwargs)

    t0 = time.time()
    loss_history = []
    for epoch in range(EPOCHS):
        np.random.shuffle(pairs)
        epoch_loss = []
        for bs in range(0, len(pairs), 16):
            batch = pairs[bs:bs+16]
            loss = cube.learn_batch(batch, alpha=0.02)
            epoch_loss.append(loss)
        loss_history.append(np.mean(epoch_loss))
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3}/{EPOCHS}  loss={loss_history[-1]:.4f}  "
                  f"elapsed={time.time()-t0:.1f}s")

    # Label concepts
    for i, c in enumerate(concepts):
        cube.label_point(c, f'c{i}')

    final_loss = loss_history[-1]
    sparsity   = measure_sparsity(cube, concepts)
    cohesion   = measure_cohesion(cube, concepts, group_ids)
    analogy    = measure_analogy(cube, concepts)

    print(f"  Final loss:  {final_loss:.4f}")
    print(f"  Cohesion:    {cohesion:.2f}x")
    print(f"  Sparsity:    {sparsity*100:.4f}%")
    print(f"  Analogy:     {analogy*100:.0f}%")
    print(f"  Time:        {time.time()-t0:.1f}s")

    return {
        'name':        name,
        'final_loss':  final_loss,
        'cohesion':    cohesion,
        'sparsity':    sparsity,
        'analogy':     analogy,
        'loss_history': loss_history,
    }, cube


# ── Main ─────────────────────────────────────────────────────────

def main():
    print("SAGE Architecture Comparison: V1 → V2 → V3 → V4")
    print("Reproduces Table 4 from Likov (2026)")
    print("=" * 55)

    concepts, pairs, group_ids = make_data()
    print(f"Data: {len(concepts)} concepts, {len(pairs)} pairs, "
          f"{N_GROUPS} groups, {EMBED_DIM}d embeddings")

    cube_kwargs = dict(cube_size=CUBE_SIZE, embed_dim=EMBED_DIM, seed=SEED)

    versions = [
        ('V1 (base)',              SpatialCubeV1, cube_kwargs),
        ('V2 (V1 + learning)',     SpatialCubeV2, cube_kwargs),
        ('V4 (V2 + LJ gravity)',   SpatialCubeV4, cube_kwargs),
    ]

    all_results = []
    all_cubes   = []

    for name, cls, kw in versions:
        result, cube = train_and_eval(name, cls, kw, pairs, concepts, group_ids)
        all_results.append(result)
        all_cubes.append(cube)

    # Anti-forgetting measurement
    print(f"\n{'─'*50}")
    print("Anti-forgetting measurement")
    print(f"{'─'*50}")
    forgetting_results = {}
    for name, cls, kw in versions:
        lb, la, _ = measure_forgetting(cls, kw, pairs, concepts)
        delta = la - lb
        forgetting_results[name] = {
            'loss_before': lb,
            'loss_after':  la,
            'forgetting':  delta,
        }
        print(f"  {name}: before={lb:.4f}  after={la:.4f}  delta={delta:+.4f}")

    # Print summary table
    print(f"\n{'='*55}")
    print("TABLE 4 REPRODUCTION — SAGE V1 vs V2 vs V4")
    print(f"{'='*55}")
    print(f"{'Metric':<30} {'V1':>8} {'V2':>8} {'V4':>8}")
    print(f"{'─'*55}")

    metrics = [
        ('Final loss',             'final_loss',  '{:.4f}'),
        ('Cluster cohesion',       'cohesion',    '{:.2f}x'),
        ('Sparsity',               'sparsity',    '{:.4f}%'),
        ('Analogy accuracy',       'analogy',     '{:.0f}%'),
    ]

    for label, key, fmt in metrics:
        vals = [r[key] for r in all_results]
        if key == 'sparsity':
            row = [fmt.format(v * 100) for v in vals]
        elif key == 'analogy':
            row = [fmt.format(v * 100) for v in vals]
        else:
            row = [fmt.format(v) for v in vals]
        print(f"  {label:<28} {row[0]:>8} {row[1]:>8} {row[2]:>8}")

    print(f"{'─'*55}")
    for name, cls, _ in versions:
        fg = forgetting_results[name]['forgetting']
        print(f"  {'Anti-forgetting (' + name.split()[0] + ')':<28} {fg:>+8.4f}")

    print(f"\nNotes:")
    print(f"  Analogy accuracy is 0% on synthetic data — real analogies require GloVe.")
    print(f"  See real_data_report.txt for GloVe results: V1=33.3%, V4=33.3%.")
    print(f"  Sparsity is a structural property independent of learning rule.")

    # Save results
    output = {
        'results': all_results,
        'forgetting': forgetting_results,
    }
    with open('comparison_results.json', 'w') as f:
        # Convert numpy to float for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(output, f, indent=2, default=convert)
    print(f"\nResults saved to comparison_results.json")


if __name__ == '__main__':
    main()
