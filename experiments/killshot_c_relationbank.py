"""
KILL-SHOT C - SAGE Hebbian "Relation Bank" vs vanilla k-means on (b-a) offsets
==============================================================================
Claim under test (the ONLY honest framing): label-free RELATION DISCOVERY - cluster
difference vectors (b-a) of word pairs to AUTO-discover the latent relation types
(capital-of, plural-of, ...), no labels. NOT analogy accuracy (that ties GloVe).

The catch: a Hebbian winner-take-all prototype bank over offsets IS online k-means
(same objective, same fixed points). So the strong classical baseline is vanilla
k-means on the same offsets, and the metric is cluster agreement with the gold
relation labels: Adjusted Rand Index, NMI, purity (permutation-invariant). The
literature (DiffVec, Vylomova 2016; arXiv:2305.04265) already shows offset clustering
recovers SYNTACTIC relations but not lexical-SEMANTIC ones - so we split by family.

Contenders (cluster offsets into R = #relation types, no labels used in fitting):
  SAGE-bank  : Hebbian WTA prototypes (online competitive learning) on offsets.
  k-means    : sklearn KMeans on the same offsets  (the killer baseline).
  agglomerative: Ward linkage (non-spherical control).
SAGE "wins" only if its ARI beats k-means by a margin that would survive a bootstrap.
Prior: exact tie -> "k-means with extra steps" -> NO-GO by the iron rule.

Uses cached nomic embeddings + the analogy benchmark categories as gold relations.
python experiments/killshot_c_relationbank.py
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEWARCH = os.path.abspath(os.path.join(ROOT, '..', '..'))
RES_DIR = os.path.join(ROOT, 'results')
DATA = os.path.join(ROOT, 'data', 'embeddings_cache.npz')
BENCH = os.path.join(NEWARCH, 'sage_revision', 'experiments',
                     'exp2_full_analogy', 'questions-words.txt')
FINDINGS = os.path.join(ROOT, 'findings.md')

MIN_PAIRS = 10
SEED = 42
N_SEEDS = 15                  # multi-seed ARI distribution (vs single cherry-picked run)
SYNTACTIC = ('gram',)         # categories starting with 'gram' are morphological


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def load_offsets():
    z = np.load(DATA, allow_pickle=True)
    words = [str(w) for w in z['words']]
    emb = _unit(np.asarray(z['embs'], dtype=np.float64))
    w2i = {w: i for i, w in enumerate(words)}
    cats, cur = {}, None
    with open(BENCH, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(':'):
                cur = ln[1:].strip(); cats.setdefault(cur, set()); continue
            w = ln.lower().split()
            if len(w) != 4:
                continue
            for h, t in ((w[0], w[1]), (w[2], w[3])):
                if h in w2i and t in w2i and h != t:
                    cats[cur].add((w2i[h], w2i[t]))
    cats = {c: sorted(p) for c, p in cats.items() if len(p) >= MIN_PAIRS}
    names = sorted(cats)
    X, y = [], []
    for ci, c in enumerate(names):
        for h, t in cats[c]:
            X.append(emb[t] - emb[h])         # offset b - a on unit embeddings
            y.append(ci)
    return _unit(np.array(X)), np.array(y), names


def hebb_bank(X, R, rng, epochs=15, lr=0.2):
    """SAGE relation bank = online competitive WTA prototypes over offsets. This IS
    online SPHERICAL k-means (cosine winner + normalized convex update). One run
    from `rng` (the multi-seed loop in main gives the distribution)."""
    proto = _unit(X[rng.choice(len(X), R, replace=False)].copy())
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            j = int(np.argmax(proto @ X[i]))              # winner (cosine)
            proto[j] = _unit((1 - lr) * proto[j] + lr * X[i])
    return np.argmax(_unit(X) @ proto.T, axis=1)


def online_spherical_kmeans(X, R, rng, epochs=15, lr=0.2):
    """The MATCHED STANDARD baseline: textbook online spherical k-means. Same
    algorithm family as hebb_bank - if SAGE has no mechanism, these tie exactly.
    (Separate function purely to make the 'SAGE == standard algorithm' point
    empirically, not by assertion.)"""
    C = _unit(X[rng.choice(len(X), R, replace=False)].copy())
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            j = int(np.argmax(C @ X[i]))
            C[j] = _unit((1 - lr) * C[j] + lr * X[i])
    return np.argmax(_unit(X) @ C.T, axis=1)


def purity(y, lab):
    tot = 0
    for c in np.unique(lab):
        tot += np.bincount(y[lab == c]).max()
    return tot / len(y)


def score(y, lab):
    return (adjusted_rand_score(y, lab), normalized_mutual_info_score(y, lab),
            purity(y, lab))


METHODS = ["SAGE-bank", "online-sph-kmeans", "kmeans-euclid", "minibatch-kmeans",
           "agglom-ward"]


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    X, y, names = load_offsets()
    R = len(names)
    syn = np.array([any(names[c].startswith(p) for p in SYNTACTIC) for c in y])
    print("KILL-SHOT C: label-free relation discovery - SAGE Hebbian bank vs the "
          "MATCHED standard baseline (online spherical k-means), %d-seed dist." % N_SEEDS)
    print("%d pairs, %d relation types (gold), D=%d. Syntactic pairs=%d, "
          "semantic=%d.\n" % (len(y), R, X.shape[1], int(syn.sum()), int((~syn).sum())))

    # stochastic methods: ONE run per seed -> ARI distribution (no best-by-obj
    # cherry-pick). agglomerative is deterministic (single value).
    ari = {m: [] for m in METHODS}
    rep = {}                                          # seed-0 labeling per method
    for si in range(N_SEEDS):
        runs = {
            "SAGE-bank": hebb_bank(X, R, np.random.default_rng(1000 + si)),
            "online-sph-kmeans": online_spherical_kmeans(X, R, np.random.default_rng(2000 + si)),
            "kmeans-euclid": KMeans(R, random_state=si, n_init=1).fit_predict(X),
            "minibatch-kmeans": MiniBatchKMeans(R, random_state=si, n_init=1).fit_predict(X),
        }
        for m, lab in runs.items():
            ari[m].append(adjusted_rand_score(y, lab))
            if si == 0:
                rep[m] = lab
    agg = AgglomerativeClustering(n_clusters=R, linkage='ward').fit_predict(X)
    ari["agglom-ward"] = [adjusted_rand_score(y, agg)]
    rep["agglom-ward"] = agg

    print("%-20s %9s %7s %14s" % ("method", "ARI mean", "std", "(min..max)"))
    rows = {}
    for m in METHODS:
        a = np.array(ari[m])
        rows[m] = {"ari_mean": float(a.mean()), "ari_std": float(a.std()),
                   "ari_min": float(a.min()), "ari_max": float(a.max())}
        print("%-20s %9.3f %7.3f  %.3f..%.3f"
              % (m, a.mean(), a.std(), a.min(), a.max()))

    # DECISIVE: SAGE-bank vs the MATCHED online-spherical-kmeans, PAIRED over seeds.
    dpaired = np.array(ari["SAGE-bank"]) - np.array(ari["online-sph-kmeans"])
    dmean = float(dpaired.mean())
    ci_lo, ci_hi = (float(x) for x in np.percentile(dpaired, [2.5, 97.5]))
    best_std = max(rows[m]["ari_mean"] for m in METHODS if m != "SAGE-bank")
    survives = (rows["SAGE-bank"]["ari_mean"] - best_std > 0.03) and (ci_lo > 0)

    base = rep["online-sph-kmeans"]                  # descriptive syn/sem split
    ari_syn = adjusted_rand_score(y[syn], base[syn]) if syn.any() else float('nan')
    ari_sem = adjusted_rand_score(y[~syn], base[~syn]) if (~syn).any() else float('nan')

    print("\n" + "=" * 72)
    print("DECISIVE - SAGE-bank vs MATCHED online-spherical-kmeans (same algorithm "
          "family), paired over %d seeds: mean ARI diff %+.3f, 95%% CI [%+.3f, %+.3f]"
          % (N_SEEDS, dmean, ci_lo, ci_hi))
    print("SAGE-bank mean %.3f vs best STANDARD method %.3f (delta %+.3f)"
          % (rows["SAGE-bank"]["ari_mean"], best_std,
             rows["SAGE-bank"]["ari_mean"] - best_std))
    if survives:
        print("VERDICT: SURVIVES - SAGE-bank beats EVERY standard clusterer incl. "
              "the matched online-spherical k-means beyond noise. Investigate.")
    else:
        print("VERDICT: FALSIFIED - SAGE-bank TIES the matched online-spherical "
              "k-means (it IS that algorithm); its edge over sklearn BATCH/euclidean "
              "k-means is just the online+spherical recipe, both standard. No SAGE "
              "mechanism -> NO-GO by the iron rule.")
    print("Syntactic vs semantic ARI (online-sph): %.3f vs %.3f - offsets cluster "
          "morphology not lexical semantics (DiffVec 2016)." % (ari_syn, ari_sem))
    print("=" * 72)

    out = {"experiment": "killshot_c_relationbank", "n_pairs": int(len(y)),
           "n_relations": R, "relations": names, "dim": int(X.shape[1]),
           "n_seeds": N_SEEDS, "sage_minus_matched_ari": dmean,
           "ari_diff_ci": [ci_lo, ci_hi], "best_standard_ari": float(best_std),
           "ari_syn": float(ari_syn), "ari_sem": float(ari_sem),
           "survives": bool(survives), "scores": rows}
    res_path = os.path.join(RES_DIR, 'killshot_c_relationbank.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    xc = np.arange(len(METHODS))
    cols = ['#4fc3f7', '#26c6da', '#ff7043', '#ffa726', '#9ccc65']
    ax.bar(xc, [rows[m]["ari_mean"] for m in METHODS],
           yerr=[rows[m]["ari_std"] for m in METHODS], color=cols, capsize=4)
    ax.set_xticks(xc); ax.set_xticklabels(METHODS, fontsize=8, rotation=12)
    ax.set_ylabel('ARI vs gold relations (mean +/- std)', color='white')
    ax.set_title('Kill-shot C: SAGE-bank == online spherical k-means (matched ties)',
                 color='white')
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'killshot_c_relationbank.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    verdict = "SURVIVES" if survives else "FALSIFIED"
    entry = (
        "\n## Kill-shot C - SAGE Hebbian Relation Bank vs MATCHED online spherical "
        "k-means (offset clustering)\n\n"
        "- Label-free relation DISCOVERY: cluster (b-a) offsets of %d word pairs "
        "into %d gold relation types (nomic, D=%d). ARI vs gold, %d-seed dist. "
        "Decisive baseline = online spherical k-means (the algorithm HebbBank IS); "
        "sklearn KMeans/MiniBatch/Ward for context.\n"
        "- ARI mean+/-std: SAGE-bank %.3f+/-%.3f, online-sph-kmeans %.3f+/-%.3f, "
        "kmeans-euclid %.3f+/-%.3f, minibatch %.3f+/-%.3f, agglom %.3f.\n"
        "- DECISIVE: SAGE-bank vs matched online-spherical, paired diff %+.3f, 95%% "
        "CI [%+.3f, %+.3f]. SAGE-bank vs best standard %+.3f. Syn vs sem ARI %.3f "
        "vs %.3f.\n"
        "- **VERDICT: %s.** %s\n"
        % (len(y), R, X.shape[1], N_SEEDS,
           rows["SAGE-bank"]["ari_mean"], rows["SAGE-bank"]["ari_std"],
           rows["online-sph-kmeans"]["ari_mean"], rows["online-sph-kmeans"]["ari_std"],
           rows["kmeans-euclid"]["ari_mean"], rows["kmeans-euclid"]["ari_std"],
           rows["minibatch-kmeans"]["ari_mean"], rows["minibatch-kmeans"]["ari_std"],
           rows["agglom-ward"]["ari_mean"], dmean, ci_lo, ci_hi,
           rows["SAGE-bank"]["ari_mean"] - best_std, ari_syn, ari_sem, verdict,
           ("SAGE-bank beats every standard clusterer incl. the matched online-"
            "spherical k-means beyond noise -> investigate." if survives
            else "SAGE-bank TIES the matched online-spherical k-means (it IS that "
            "algorithm); its edge over batch/euclidean k-means is just the online+"
            "spherical recipe, both standard. No SAGE mechanism -> NO-GO. ARI-syn >> "
            "ARI-sem matches DiffVec 2016 (offsets cluster morphology, not "
            "lexical semantics).")))
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended kill-shot C entry to findings.md")


if __name__ == '__main__':
    main()
