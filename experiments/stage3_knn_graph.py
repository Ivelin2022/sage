"""
SAGE-Sphere - Stage 3 - kNN-relation graph construction
=========================================================
Build the graph traversal will run on (NO traversal yet - structure only).
Per the Stage 1 flag, the kNN graph is built in FULL 768-D cosine (3-D positions
hold only ~8% variance). Built on the Stage 2 ISOTROPIC embeddings, with RAW as a
comparison so Stage 4 knows which to prefer.

For k in {10, 20, 50} and embeddings in {isotropic, raw}, measure:
  - CONNECTIVITY: #components, giant-component fraction, degree distribution.
  - BENCHMARK COVERAGE: for known relation pairs (Google analogy: capitals,
    family, past-tense, nationality, ...), what fraction are connected by a SHORT
    path. 1-hop = partner is a nearest neighbour (the strong signal for traversal);
    <=2 / <=3 hop = reachable by a short walk.
  - EDGE LABELLING: Hebbian co-occurrence counts on directly-connected pairs.

GATE: >=70% of benchmark relation pairs connected within a small number of hops
(<=3). Report 1-hop and <=2-hop too, as those are what traversal actually needs.
STOP if pairs are mostly disconnected even at high k.

Loads data/embeddings_isotropic.npz (+ _cache.npz for raw) and the analogy
benchmark. No Ollama. Run:  python experiments/stage3_knn_graph.py
"""

import os
import sys
import json

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWARCH = os.path.abspath(os.path.join(ROOT, '..', '..'))
sys.path.insert(0, ROOT)
from core.knn_graph import build_knn, label_edges  # noqa: E402

ISO_NPZ  = os.path.join(ROOT, 'data', 'embeddings_isotropic.npz')
RAW_NPZ  = os.path.join(ROOT, 'data', 'embeddings_cache.npz')
BENCH    = os.path.join(NEWARCH, 'sage_revision', 'experiments',
                        'exp2_full_analogy', 'questions-words.txt')
KNN_OUT  = os.path.join(ROOT, 'data', 'stage3_knn.npz')
RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

K_GRID   = [10, 20, 50]
HOP_GATE = 2           # "small number of hops": <=2 is the discriminating short
                       # path (<=3 is ~100% by small-world, trivial)
COVER_TARGET = 70.0    # percent
MIN_PAIRS = 10         # ignore categories with too few in-vocab pairs
SEED = 42


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    words = [str(w) for w in z['words']]
    embs = F.normalize(torch.tensor(np.asarray(z['embs'], dtype=np.float32)),
                       p=2, dim=1)
    return words, embs


def load_benchmark_pairs(path, word2idx):
    """category -> list of (a_idx, b_idx) relation pairs fully in vocab."""
    cats = {}
    cur = None
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(':'):
                cur = ln[1:].strip()
                cats.setdefault(cur, set())
                continue
            w = ln.lower().split()
            if len(w) != 4:
                continue
            for a, b in ((w[0], w[1]), (w[2], w[3])):
                if a in word2idx and b in word2idx and a != b:
                    cats[cur].add((word2idx[a], word2idx[b]))
    return {c: sorted(p) for c, p in cats.items() if len(p) >= MIN_PAIRS}


def build_csr(idx_np, n):
    """Symmetric (union) unweighted adjacency from kNN indices."""
    k = idx_np.shape[1]
    rows = np.repeat(np.arange(n), k)
    cols = idx_np.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def evaluate(emb, k, cats, device):
    n = emb.shape[0]
    idx, sim = build_knn(emb.to(device), k)
    idx_np = idx.cpu().numpy()
    A = build_csr(idx_np, n)

    # connectivity
    ncomp, lbl = connected_components(A, directed=False)
    sizes = np.bincount(lbl)
    giant = int(sizes.max())
    deg = np.asarray((A > 0).sum(1)).ravel()

    # hop distances from all source nodes appearing in benchmark pairs
    srcs = sorted({a for pairs in cats.values() for a, _ in pairs})
    src_pos = {s: i for i, s in enumerate(srcs)}
    # unweighted=True -> hop counts (BFS-equivalent); scipy has no 'BFS' method,
    # 'D' (Dijkstra) with unit weights gives the same hop distances.
    dist = shortest_path(A, method='D', unweighted=True, directed=False,
                         indices=srcs)                     # (len(srcs), n)
    nbr = [set(r.tolist()) for r in idx]

    per_cat = {}
    tot = {"n": 0, "h1": 0, "h2": 0, "h3": 0}
    for c, pairs in cats.items():
        c1 = c2 = c3 = 0
        for a, b in pairs:
            if b in nbr[a] or a in nbr[b]:
                c1 += 1
            d = dist[src_pos[a], b]
            if d <= 2:
                c2 += 1
            if d <= 3:
                c3 += 1
        m = len(pairs)
        per_cat[c] = {"n": m,
                      "hop1_pct": 100.0 * c1 / m,
                      "hop2_pct": 100.0 * c2 / m,
                      "hop3_pct": 100.0 * c3 / m}
        tot["n"] += m; tot["h1"] += c1; tot["h2"] += c2; tot["h3"] += c3

    overall = {
        "hop1_pct": 100.0 * tot["h1"] / tot["n"],
        "hop2_pct": 100.0 * tot["h2"] / tot["n"],
        "hop3_pct": 100.0 * tot["h3"] / tot["n"],
        "n_pairs": tot["n"],
    }
    stats = {
        "k": k, "n_components": int(ncomp),
        "giant_frac": 100.0 * giant / n,
        "isolated_nodes": int((deg == 0).sum()),
        "mean_degree": float(deg.mean()), "max_degree": int(deg.max()),
    }
    return idx_np, stats, overall, per_cat


def main():
    torch.manual_seed(SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % device)

    if not os.path.exists(ISO_NPZ):
        print("ERROR: %s missing - run Stage 2 first." % ISO_NPZ)
        sys.exit(1)
    words, iso = load_npz(ISO_NPZ)
    word2idx = {w: i for i, w in enumerate(words)}
    raw_words, raw = load_npz(RAW_NPZ)
    # raw cache and isotropic share the same word order (built from same source)
    assert raw_words == words, "raw/isotropic vocab order mismatch"

    cats = load_benchmark_pairs(BENCH, word2idx)
    if not cats:
        print("ERROR: no benchmark categories with >=%d in-vocab pairs. "
              "Check %s" % (MIN_PAIRS, BENCH))
        sys.exit(1)
    print("Benchmark relation categories in vocab (>=%d pairs):" % MIN_PAIRS)
    for c, p in cats.items():
        print("  %-28s %4d pairs" % (c, len(p)))
    print()

    runs = []   # (emb_name, k, idx_np, stats, overall, per_cat)
    for emb_name, emb in (("isotropic", iso), ("raw", raw)):
        for k in K_GRID:
            idx_np, stats, overall, per_cat = evaluate(emb, k, cats, device)
            runs.append((emb_name, k, idx_np, stats, overall, per_cat))
            print("[%-9s k=%2d] giant=%.1f%% comps=%d meanDeg=%.1f | "
                  "1hop=%.1f%% <=2hop=%.1f%% <=3hop=%.1f%%"
                  % (emb_name, k, stats["giant_frac"], stats["n_components"],
                     stats["mean_degree"], overall["hop1_pct"],
                     overall["hop2_pct"], overall["hop3_pct"]))

    # ---- choose best config: smallest k (parsimony) that passes the gate,
    #      isotropic preferred; tie-break on 1-hop coverage (what traversal needs)
    def passes(r):
        return r[4]["hop%d_pct" % HOP_GATE] >= COVER_TARGET
    iso_runs = [r for r in runs if r[0] == "isotropic"]
    passing = [r for r in iso_runs if passes(r)]
    if passing:
        best = min(passing, key=lambda r: (r[1], -r[4]["hop1_pct"]))
        gate_pass = True
    else:
        best = max(iso_runs, key=lambda r: r[4]["hop%d_pct" % HOP_GATE])
        gate_pass = False
    b_name, b_k, b_idx, b_stats, b_over, b_percat = best

    print("\n" + "=" * 66)
    print("STAGE 3 GATE: >=%.0f%% of benchmark pairs within <=%d hops"
          % (COVER_TARGET, HOP_GATE))
    print("  chosen: %s k=%d -> 1hop=%.1f%% <=2hop=%.1f%% <=3hop=%.1f%% "
          "(gate <=%dhop >= %.0f%%) -> %s"
          % (b_name, b_k, b_over["hop1_pct"], b_over["hop2_pct"],
             b_over["hop3_pct"], HOP_GATE, COVER_TARGET,
             "PASS" if gate_pass else "FAIL"))
    print("=" * 66)

    # ---- edge labelling (Hebbian co-occurrence) on the chosen graph ----
    triples = [(a, c, b) for c, pairs in cats.items() for a, b in pairs]
    b_idx_t = torch.tensor(b_idx)
    labels, n_cov, n_tot = label_edges(b_idx_t, triples)
    print("\nEdge labelling: %d/%d benchmark triples are DIRECT edges (%.1f%%); "
          "%d labelled edges, %d relation types."
          % (n_cov, n_tot, 100.0 * n_cov / n_tot, len(labels),
             len({r for d in labels.values() for r in d})))

    # ---- save chosen kNN for Stage 4 ----
    np.savez(KNN_OUT, knn_idx=b_idx, words=np.array(words, dtype=object),
             k=b_k, emb=b_name)
    print("Saved chosen kNN graph -> %s" % KNN_OUT)

    # ---- JSON ----
    out = {
        "stage": 3, "n_words": len(words), "k_grid": K_GRID,
        "hop_gate": HOP_GATE, "cover_target": COVER_TARGET,
        "chosen": {"emb": b_name, "k": b_k},
        "gate_pass": bool(gate_pass),
        "edge_labelling": {"direct_edge_triples": n_cov, "total_triples": n_tot,
                           "labelled_edges": len(labels)},
        "runs": [{"emb": e, "k": k, "stats": s, "overall": o, "per_category": pc}
                 for (e, k, _, s, o, pc) in runs],
    }
    res_path = os.path.join(RES_DIR, 'stage3_knn_graph.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("Saved: %s" % res_path)

    # ---- plots (dark theme) ----
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0a0a0a')
    for ax in axs:
        ax.set_facecolor('#111111')

    # (1) coverage vs k, isotropic vs raw (<=2-hop, the meaningful short-path)
    for emb_name, color in (("isotropic", '#4fc3f7'), ("raw", '#ff7043')):
        ks = [r[1] for r in runs if r[0] == emb_name]
        h1 = [r[4]["hop1_pct"] for r in runs if r[0] == emb_name]
        h2 = [r[4]["hop2_pct"] for r in runs if r[0] == emb_name]
        axs[0].plot(ks, h1, 'o-', color=color, label='%s 1-hop' % emb_name)
        axs[0].plot(ks, h2, 'o--', color=color, alpha=0.6, label='%s <=2-hop' % emb_name)
    axs[0].axhline(COVER_TARGET, color='#888888', ls=':', lw=1, label='target')
    axs[0].set_title('benchmark pair coverage vs k', color='white')
    axs[0].set_xlabel('k (neighbours)', color='white')
    axs[0].set_ylabel('% pairs connected', color='white')
    axs[0].legend(facecolor='#111111', edgecolor='#333333', fontsize=7)

    # (2) per-category 1-hop at chosen config
    cs = list(b_percat.keys())
    v1 = [b_percat[c]["hop1_pct"] for c in cs]
    yp = np.arange(len(cs))
    axs[1].barh(yp, v1, color='#26c281')
    axs[1].set_yticks(yp)
    axs[1].set_yticklabels([c[:22] for c in cs], fontsize=7, color='white')
    axs[1].invert_yaxis()
    axs[1].set_title('1-hop coverage by relation (%s k=%d)' % (b_name, b_k),
                     color='white')
    axs[1].set_xlabel('% pairs with partner in kNN', color='white')

    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage3_knn_graph.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    # ---- findings ----
    cat_lines = "".join(
        "    - %-28s %4d pairs | 1hop %.0f%% <=2hop %.0f%%\n"
        % (c, b_percat[c]["n"], b_percat[c]["hop1_pct"], b_percat[c]["hop2_pct"])
        for c in b_percat)
    entry = (
        "\n## Stage 3 - kNN-relation graph\n\n"
        "- Built in FULL 768-D cosine (per Stage 1 flag) on ISOTROPIC embeddings; "
        "raw compared. %d words, k in %s.\n"
        "- **GATE: %s** (>=%.0f%% within <=%d hops) - chosen %s k=%d: "
        "1-hop = %.1f%%, <=2-hop = %.1f%%, <=3-hop = %.1f%%.\n"
        "- Connectivity: giant component %.1f%% of nodes, %d components, mean "
        "degree %.1f.\n"
        "- Edge labelling (Hebbian co-occurrence): %d/%d benchmark triples are "
        "DIRECT edges (%.1f%%), %d labelled edges.\n"
        "- Per-relation 1-hop coverage (chosen config):\n%s"
        "- Saved chosen kNN to data/stage3_knn.npz for Stage 4. NOTE: <=3-hop "
        "coverage is inflated by small-world connectivity; 1-hop (partner is a "
        "nearest neighbour) is the signal that matters for typed traversal.\n"
        % (len(words), str(K_GRID),
           "PASS" if gate_pass else "FAIL", COVER_TARGET, HOP_GATE, b_name, b_k,
           b_over["hop1_pct"], b_over["hop2_pct"], b_over["hop3_pct"],
           b_stats["giant_frac"], b_stats["n_components"], b_stats["mean_degree"],
           n_cov, n_tot, 100.0 * n_cov / n_tot, len(labels), cat_lines)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 3 entry to findings.md")
    print("\nNext: Stage 4 - Dijkstra traversal (the GO/NO-GO for the whole direction).")


if __name__ == '__main__':
    main()
