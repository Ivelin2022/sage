"""
core/knn_graph.py - kNN-relation graph construction (Stage 3).

Builds the graph that traversal (Stage 4) will run on. NO traversal here -
structure only. Per the Stage 1 finding (3-D positions hold only ~8% variance),
the graph is built in FULL 768-D cosine; the 3-D sphere positions are kept only
for visualization / the Stage 6 partition.

THESIS GUARD: exact cosine kNN (one matmul + topk) + integer co-occurrence
counting. No gradient descent, no autograd, no learned edge embeddings.
"""

import torch


def build_knn(embeddings, k):
    """Exact cosine k-nearest-neighbour over the rows of `embeddings`.

    embeddings: (N, D) unit-norm tensor (the 768-D payloads).
    returns: idx (N, k) neighbour indices, sim (N, k) cosine similarities.
    Single parallel pass; self excluded.
    """
    sims = embeddings @ embeddings.T
    sims.fill_diagonal_(-2.0)                       # exclude self
    sim, idx = torch.topk(sims, k, dim=1)
    return idx, sim


def label_edges(knn_idx, triples):
    """Hebbian co-occurrence edge labelling (gradient-free).

    knn_idx: (N, k) neighbour indices from build_knn.
    triples: iterable of (head_idx, relation_str, tail_idx) observed relations.

    For every observed (h, r, t) whose endpoints are DIRECTLY connected in the
    kNN graph, increment a per-relation counter on that (undirected) edge. Edges
    can carry multiple relation labels with counts. Endpoints not directly
    connected are left for multi-hop traversal (Stage 4) - they are reported as
    'uncovered' so the cold-start gap is visible.

    returns: (labels, n_covered, n_total)
      labels: dict[(min,max)] -> {relation: count}
    """
    nbr = [set(row.tolist()) for row in knn_idx]
    labels = {}
    n_covered = 0
    n_total = 0
    for h, r, t in triples:
        n_total += 1
        if t in nbr[h] or h in nbr[t]:             # edge exists (union kNN)
            n_covered += 1
            key = (min(h, t), max(h, t))
            d = labels.setdefault(key, {})
            d[r] = d.get(r, 0) + 1
    return labels, n_covered, n_total
