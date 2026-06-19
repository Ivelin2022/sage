"""
SAGE-Sphere - Stage 2 - Isotropy preprocessing (REQUIRED; Stage 0 = HIGH)
==========================================================================
All-but-the-Top (Mu & Viswanath 2018), gradient-free. Sweep D in {0,1,2,3,5,7,10}
(D=0 = untreated baseline) and for each measure:

  - ANISOTROPY (Stage-0 diagnostic): mean|cos|, mean-vector norm, top-1 PC.
    GATE part 1: processed mean|cos| meaningfully BELOW raw.
  - STRUCTURE NOT SCRAMBLED (sanity, NOT a quality score): mean Jaccard overlap
    of each word's exact top-10 neighbours, raw vs processed. Isotropy is SUPPOSED
    to shift some neighbourhoods, so this is only a catastrophe check - we require
    it not destroy local structure (Jaccard above a floor), not that it stay high.
    Whether isotropy actually HELPS is decided at STAGE 4 (run the analogy task on
    raw vs isotropic), per the brief's caveat that isotropy can hurt some encoders.
  - DOWNSTREAM SIGNAL: continuous PCA-3D vs 768-D cosine Spearman - does removing
    the common-mode RAISE how much the 3-D projection tracks semantics?
    (Stage 1 raw baseline was r=+0.269.)

Pick the D with the largest anisotropy reduction that does NOT hurt retrieval,
save those processed embeddings for downstream stages, and re-confirm the Stage 0
verdict dropped. If no D qualifies -> flag for human review (brief STOP).

Loads data/embeddings_cache.npz. No Ollama. Run:
  python experiments/stage2_isotropy.py
"""

import os
import sys
import json

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.isotropy import all_but_the_top, anisotropy_stats  # noqa: E402

DATA_NPZ = os.path.join(ROOT, 'data', 'embeddings_cache.npz')
ISO_NPZ  = os.path.join(ROOT, 'data', 'embeddings_isotropic.npz')
RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

D_GRID        = [0, 1, 2, 3, 5, 7, 10]   # 0 = untreated baseline
N_PAIRS       = 20000
JACCARD_FLOOR = 0.50                      # min neighbour preservation to "not hurt"
ANISO_DROP    = 0.10                      # require >=10% relative mean|cos| drop
NEAR_OPT_TOL  = 0.15                      # "near-minimal" anisotropy band for parsimony
SEED          = 42


def load_embeddings(device):
    if not os.path.exists(DATA_NPZ):
        print("ERROR: %s missing - run Stage 0 first." % DATA_NPZ)
        sys.exit(1)
    z = np.load(DATA_NPZ, allow_pickle=True)
    words = list(z['words'])
    embs = F.normalize(torch.tensor(np.asarray(z['embs'], dtype=np.float32)),
                       p=2, dim=1).to(device)
    print("Loaded %d words x %d dims." % (embs.shape[0], embs.shape[1]))
    return words, embs


def topk_neighbor_sets(embs, k=10):
    """Each row's exact top-k neighbour indices (self excluded), as sets."""
    sims = embs @ embs.T
    sims.fill_diagonal_(-2.0)
    _, idx = torch.topk(sims, k, dim=1)
    return [set(r.tolist()) for r in idx]


def mean_jaccard(sets_a, sets_b):
    tot = 0.0
    for a, b in zip(sets_a, sets_b):
        u = len(a | b)
        tot += (len(a & b) / u) if u else 1.0
    return tot / len(sets_a)


def continuous_3d_corr(embs, device, n_pairs=N_PAIRS, seed=SEED):
    """Spearman( true 768-D cosine , cosine of continuous PCA-3D direction )."""
    X = embs.float()
    Xc = X - X.mean(0)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    dir3 = F.normalize(Xc @ Vh[:3].T, p=2, dim=1)
    Mn = F.normalize(X, p=2, dim=1)
    g = np.random.default_rng(seed)
    n = X.shape[0]
    ii = g.integers(0, n, n_pairs); jj = g.integers(0, n, n_pairs)
    keep = ii != jj
    it = torch.tensor(ii[keep], device=device); jt = torch.tensor(jj[keep], device=device)
    cos768 = (Mn[it] * Mn[jt]).sum(1).cpu().numpy()
    cos3 = (dir3[it] * dir3[jt]).sum(1).cpu().numpy()
    r = spearmanr(cos768, cos3).correlation
    return float(r) if r == r else None


def main():
    torch.manual_seed(SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % device)

    words, raw = load_embeddings(device)
    raw_neighbors = topk_neighbor_sets(raw, k=10)

    rows = []
    processed = {}
    print("%-4s %-11s %-11s %-9s %-9s %-9s" %
          ("D", "mean|cos|", "meanVecNrm", "top1PC%", "nbrJacc", "cont3D_r"))
    for D in D_GRID:
        proc = all_but_the_top(raw, D)
        processed[D] = proc
        st = anisotropy_stats(proc, n_pairs=N_PAIRS, seed=SEED)
        # retrieval sanity: self-retrieval stays 100% (still distinct vectors)
        sims = proc @ proc.T
        self_acc = 100.0 * (sims.argmax(1) ==
                            torch.arange(proc.shape[0], device=device)).float().mean().item()
        jacc = mean_jaccard(raw_neighbors, topk_neighbor_sets(proc, k=10))
        r3 = continuous_3d_corr(proc, device)
        rows.append({
            "D": D, "mean_abs_cosine": st["mean_abs_cosine"],
            "mean_vector_norm": st["mean_vector_norm"],
            "top1_pc_variance": st["top1_pc_variance"],
            "self_retrieval_pct": self_acc,
            "neighbor_jaccard_vs_raw": jacc,
            "continuous_3d_spearman": r3,
        })
        print("%-4d %-11.4f %-11.4f %-9.2f %-9.3f %-9s" %
              (D, st["mean_abs_cosine"], st["mean_vector_norm"],
               100 * st["top1_pc_variance"], jacc,
               ("%+.3f" % r3) if r3 is not None else "nan"))

    raw_aniso = rows[0]["mean_abs_cosine"]   # D=0 baseline

    # ---- pick best D: largest anisotropy reduction that does NOT hurt retrieval ----
    eligible = [r for r in rows[1:]
                if r["neighbor_jaccard_vs_raw"] >= JACCARD_FLOOR
                and r["mean_abs_cosine"] <= raw_aniso * (1 - ANISO_DROP)]
    if eligible:
        # Anisotropy reduction saturates fast (typically the 1st PC does ~all the
        # work). Minimizing mean|cos| outright rewards OVER-removal, which scrambles
        # neighbours and lowers the downstream 3-D signal for negligible extra
        # de-anisotropy. So: pick the SMALLEST D (parsimony / Occam) whose mean|cos|
        # is within NEAR_OPT_TOL of the best achievable.
        m_min = min(r["mean_abs_cosine"] for r in eligible)
        near_opt = [r for r in eligible
                    if r["mean_abs_cosine"] <= m_min * (1 + NEAR_OPT_TOL)]
        best = min(near_opt, key=lambda r: r["D"])
        gate_pass = True
    else:
        # nothing both reduces anisotropy enough AND preserves neighbours
        best = min(rows[1:], key=lambda r: r["mean_abs_cosine"])
        gate_pass = False
    best_D = best["D"]
    drop_pct = (100 * (1 - best["mean_abs_cosine"] / raw_aniso)
                if raw_aniso > 0 else 0.0)

    print("\n" + "=" * 64)
    print("STAGE 2 GATE: mean|cos| drops >=%d%% AND neighbour Jaccard >= %.2f"
          % (int(ANISO_DROP * 100), JACCARD_FLOOR))
    print("  raw (D=0) mean|cos| = %.4f" % raw_aniso)
    print("  best D = %d -> mean|cos| = %.4f (%.1f%% drop), Jaccard = %.3f, "
          "cont3D r = %s"
          % (best_D, best["mean_abs_cosine"], drop_pct,
             best["neighbor_jaccard_vs_raw"],
             ("%+.3f" % best["continuous_3d_spearman"])
             if best["continuous_3d_spearman"] is not None else "nan"))
    print("  -> %s" % ("PASS" if gate_pass else
                       "FAIL - no D both de-anisotropizes and preserves retrieval"))
    print("=" * 64)

    # ---- save processed embeddings at best D for downstream stages ----
    if gate_pass:
        pb = processed[best_D].cpu().float().numpy()
        np.savez(ISO_NPZ, words=np.array(words, dtype=object),
                 embs=pb.astype(np.float32), D=best_D)
        print("\nSaved isotropic embeddings (D=%d) -> %s" % (best_D, ISO_NPZ))
    else:
        print("\nNOT saving processed embeddings - gate failed, keep raw; flag "
              "for human review per brief STOP condition.")

    # ---- save JSON ----
    out = {
        "stage": 2, "method": "all-but-the-top", "n_words": raw.shape[0],
        "raw_mean_abs_cosine": raw_aniso, "best_D": best_D,
        "gate_pass": bool(gate_pass),
        "jaccard_floor": JACCARD_FLOOR, "aniso_drop_required": ANISO_DROP,
        "sweep": rows,
    }
    res_path = os.path.join(RES_DIR, 'stage2_isotropy.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("Saved: %s" % res_path)

    # ---- plots (dark theme) ----
    Ds = [r["D"] for r in rows]
    mac = [r["mean_abs_cosine"] for r in rows]
    jac = [r["neighbor_jaccard_vs_raw"] for r in rows]
    r3s = [r["continuous_3d_spearman"] if r["continuous_3d_spearman"] is not None
           else 0.0 for r in rows]

    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), facecolor='#0a0a0a')
    for ax in axs:
        ax.set_facecolor('#111111')
        ax.set_xlabel('D (top PCs removed)', color='white')

    axs[0].plot(Ds, mac, 'o-', color='#4fc3f7')
    axs[0].axhline(raw_aniso, color='#888888', ls='--', lw=1, label='raw')
    axs[0].axhline(raw_aniso * (1 - ANISO_DROP), color='#ff5252', ls=':', lw=1,
                   label='%d%% drop target' % int(ANISO_DROP * 100))
    axs[0].axvline(best_D, color='#26c281', lw=1.2, label='chosen D=%d' % best_D)
    axs[0].set_title('anisotropy (mean|cos|) vs D', color='white')
    axs[0].legend(facecolor='#111111', edgecolor='#333333', fontsize=8)

    axs[1].plot(Ds, jac, 'o-', color='#ffb74d')
    axs[1].axhline(JACCARD_FLOOR, color='#ff5252', ls=':', lw=1, label='floor')
    axs[1].set_title('neighbour preservation (Jaccard vs raw)', color='white')
    axs[1].set_ylim(0, 1.0)
    axs[1].legend(facecolor='#111111', edgecolor='#333333', fontsize=8)

    axs[2].plot(Ds, r3s, 'o-', color='#26c281')
    axs[2].set_title('continuous 3-D vs 768-D cosine (Spearman)', color='white')

    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage2_isotropy.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    # ---- findings entry ----
    bj = best["neighbor_jaccard_vs_raw"]
    br = best["continuous_3d_spearman"]
    entry = (
        "\n## Stage 2 - Isotropy preprocessing (All-but-the-Top)\n\n"
        "- Swept D in %s on %d words (D=0 = untreated baseline).\n"
        "- Raw mean|cos| = %.4f. **Best D = %d -> mean|cos| = %.4f (%.1f%% drop), "
        "neighbour Jaccard vs raw = %.3f, continuous-3D Spearman = %s.**\n"
        "- **GATE: %s** (require >=%d%% mean|cos| drop AND Jaccard >= %.2f).\n"
        "- Re-run Stage 0 diagnostic on processed embeddings: mean|cos| %.4f -> "
        "%.4f. %s\n"
        "- Downstream: continuous-3D correlation moved %s -> %s vs the Stage 1 raw "
        "baseline (+0.269); %s.\n"
        "- %s\n"
        % (str(D_GRID), raw.shape[0], raw_aniso, best_D, best["mean_abs_cosine"],
           drop_pct, bj,
           ("%+.3f" % br) if br is not None else "nan",
           "PASS" if gate_pass else "FAIL",
           int(ANISO_DROP * 100), JACCARD_FLOOR,
           raw_aniso, best["mean_abs_cosine"],
           "Anisotropy reduced as expected." if best["mean_abs_cosine"] < raw_aniso
           else "No reduction - unexpected.",
           "+0.269", ("%+.3f" % br) if br is not None else "nan",
           "isotropy helps the 3-D projection" if (br is not None and br > 0.269)
           else "little/no change to the 3-D projection",
           ("Saved processed embeddings to data/embeddings_isotropic.npz (D=%d); "
            "downstream stages use these." % best_D) if gate_pass else
           "Gate FAILED - kept raw embeddings; flagged for human review "
           "(brief: isotropy can hurt fine-tuned encoders).")
    )
    entry += ("- NOTE: neighbour-Jaccard-vs-raw is a SCRAMBLE check (isotropy is "
              "meant to shift some neighbourhoods), NOT a quality score. Whether "
              "isotropy HELPS composition is decided at Stage 4 (analogy on raw vs "
              "isotropic); both embedding sets are kept (raw cache + isotropic).\n")
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 2 entry to findings.md")
    print("\nNext: Stage 3 - kNN-relation graph (build in FULL 768-D per Stage 1 flag).")


if __name__ == '__main__':
    main()
