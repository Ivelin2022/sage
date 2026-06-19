"""
SAGE-Sphere - Stage 4b - Hybrid follow-up to the NO-GO
=======================================================
Stage 4 found: greedy re-grounding HURTS (Bg-global=38% vs arith=62%), but the
kNN-restricted variant beat arithmetic (B=70.6%). So B's apparent win is the
candidate RESTRICTION, not the traversal mechanism. This stage isolates that
honestly and asks: is there a hybrid that ROBUSTLY beats one-shot arithmetic on
multi-hop, attributable to a real mechanism?

Multi-hop methods (capital -> country -> continent), same leakage-safe split as
Stage 4 (dir1/dir2 from TRAIN countries, test = held-out countries):
  A    : one-shot arithmetic, GLOBAL nearest (the Stage 4 baseline, ~62%).
  A_r2 : one-shot arithmetic, restricted to cap's 2-HOP neighbourhood. Isolates
         "does restriction help arithmetic WITHOUT re-grounding?"
  B    : greedy re-grounding, kNN pool (the Stage 4 deployed traversal, ~71%).
  Beam : re-grounding with a BEAM (soft commitment), kNN pool. Isolates "is the
         greedy single-hop commitment the problem?"

Read: if A_r2 >= A + MARGIN, the benefit is restriction and is achievable with
PLAIN arithmetic (no traversal needed). If Beam >> B, greedy commitment was the
flaw. If nothing clearly/attributably beats A, multi-hop composition is not
improvable here and the NO-GO stands strengthened.

Reuses Stage 4's loaders. No Ollama.  python experiments/stage4b_hybrid.py
"""

import os
import sys
import json
import random

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)
from core.traversal import (relation_prototype, neighbors,                      # noqa: E402
                            multi_hop_traverse, multi_hop_arithmetic, beam_traverse)
import stage4_traversal as s4                                                   # noqa: E402

RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

K_CAND = 50
BEAM   = 5
MARGIN = 5.0
SEED   = 42


def two_hop_pool(start, emb, k):
    """Union of start's kNN and the kNN of each of those neighbours (start excluded)."""
    nb = neighbors(start, emb, k)
    pool = set(nb.tolist())
    for nidx in nb.tolist():
        pool.update(neighbors(nidx, emb, k).tolist())
    pool.discard(int(start))
    return torch.tensor(sorted(pool), device=emb.device)


def eval_hybrid(emb, cap_of):
    # SAME split as Stage 4 eval_multihop for direct comparability
    rng = random.Random(SEED)
    countries = list(cap_of.keys())
    rng.shuffle(countries)
    half = max(1, len(countries) // 2)
    train_c, test_c = countries[:half], countries[half:]
    if not test_c:
        return None
    dir1 = relation_prototype([(cap_of[c][0], cap_of[c][1]) for c in train_c], emb)
    dir2 = relation_prototype([(cap_of[c][1], cap_of[c][2]) for c in train_c], emb)
    if dir1 is None or dir2 is None:
        return None
    A = Ar2 = B = Bm = 0
    n = 0
    for c in test_c:
        cap, ctry, cont = cap_of[c]
        n += 1
        # A: one-shot global arithmetic
        if multi_hop_arithmetic(cap, [dir1, dir2], emb) == cont:
            A += 1
        # A_r2: one-shot arithmetic restricted to the 2-hop neighbourhood
        pool = two_hop_pool(cap, emb, K_CAND)
        if pool.numel() > 0:
            v = F.normalize(emb[cap] + dir1 + dir2, p=2, dim=0)
            if int(pool[torch.argmax(emb[pool] @ v)]) == cont:
                Ar2 += 1
        # B: greedy re-grounding, kNN
        if multi_hop_traverse(cap, [dir1, dir2], emb, K_CAND) == cont:
            B += 1
        # Beam: re-grounding with a beam, kNN
        if beam_traverse(cap, [dir1, dir2], emb, K_CAND, BEAM) == cont:
            Bm += 1
    return {"n": n, "A_pct": 100.0 * A / n, "Ar2_pct": 100.0 * Ar2 / n,
            "B_pct": 100.0 * B / n, "Beam_pct": 100.0 * Bm / n}


def main():
    torch.manual_seed(SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % device)

    words, iso = s4.load_npz(s4.ISO_NPZ)
    _, raw = s4.load_npz(s4.RAW_NPZ)
    word2idx = {w: i for i, w in enumerate(words)}
    iso = iso.to(device); raw = raw.to(device)
    cap_of = s4.build_multihop(s4.BENCH, word2idx)
    print("Multi-hop countries: %d\n" % len(cap_of))

    results = {}
    for name, emb in (("isotropic", iso), ("raw", raw)):
        r = eval_hybrid(emb, cap_of)
        results[name] = r
        if r:
            print("[%-9s] A(arith-global)=%.1f%%  A_r2(arith-2hop)=%.1f%%  "
                  "B(greedy-kNN)=%.1f%%  Beam(beam-kNN)=%.1f%%  (n=%d)"
                  % (name, r["A_pct"], r["Ar2_pct"], r["B_pct"], r["Beam_pct"],
                     r["n"]))
    print()

    iso_r = results["isotropic"]
    if iso_r is None:
        print("Isotropic multi-hop produced no data - cannot run hybrid verdict.")
        return
    A = iso_r["A_pct"]; Ar2 = iso_r["Ar2_pct"]; B = iso_r["B_pct"]; Bm = iso_r["Beam_pct"]
    methods = {"A_r2 (restricted arithmetic)": Ar2,
               "B (greedy traversal)": B, "Beam (beam traversal)": Bm}
    best_name = max(methods, key=methods.get)
    best = methods[best_name]
    restriction_helps = Ar2 >= A + MARGIN                # clean: no re-grounding
    if best < A + MARGIN:
        outcome = ("HYBRID NULL: nothing beats one-shot arithmetic by >=%.0f - "
                   "multi-hop composition not improvable here; NO-GO stands" % MARGIN)
    elif "A_r2" in best_name:
        outcome = ("HYBRID PROMISING: restricted PLAIN arithmetic (A_r2) is best, "
                   "beats A by %+.1f - gain is candidate restriction, no traversal "
                   "needed" % (best - A))
    else:
        outcome = ("HYBRID MIXED: %s is best (%+.1f vs A); restricted arithmetic "
                   "alone %s (A_r2 %+.1f) - gain entangled with re-grounding"
                   % (best_name, best - A,
                      "also helps" if restriction_helps else "does NOT help",
                      Ar2 - A))

    print("=" * 70)
    print("STAGE 4b HYBRID (isotropic): %s" % outcome)
    print("  A_r2 - A (restriction on arithmetic) = %+.1f" % (Ar2 - A))
    print("  Beam - B (soft vs greedy commitment) = %+.1f" % (Bm - B))
    print("  best hybrid (%.1f) - A (%.1f) = %+.1f" % (best, A, best - A))
    print("=" * 70)

    out = {"stage": "4b", "outcome": outcome, "k_cand": K_CAND, "beam": BEAM,
           "results": results,
           "deltas_isotropic": {"Ar2_minus_A": Ar2 - A, "Beam_minus_B": Bm - B,
                                "best_minus_A": best - A}}
    res_path = os.path.join(RES_DIR, 'stage4b_hybrid.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # plot
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    labels = ['A\narith-glob', 'A_r2\narith-2hop', 'B\ngreedy-kNN', 'Beam\nbeam-kNN']
    xc = np.arange(len(labels))
    for off, (name, col) in zip((-0.2, 0.2),
                                (("isotropic", '#4fc3f7'), ("raw", '#ff7043'))):
        r = results[name]
        vals = ([r["A_pct"], r["Ar2_pct"], r["B_pct"], r["Beam_pct"]]
                if r else [0, 0, 0, 0])
        ax.bar(xc + off, vals, width=0.38, color=col, label=name)
    ax.axhline(A, color='#888888', ls='--', lw=1, label='arith baseline')
    ax.set_xticks(xc); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('multi-hop chain accuracy %', color='white')
    ax.set_title('Stage 4b hybrid: can anything beat one-shot arithmetic?',
                 color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage4b_hybrid.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    raw_r = results["raw"]
    entry = (
        "\n## Stage 4b - Hybrid follow-up (multi-hop)\n\n"
        "- Same leakage-safe split as Stage 4. Methods isolate restriction vs "
        "greedy commitment. K_CAND=%d, beam=%d.\n"
        "- Multi-hop (isotropic): A arith-global=%.1f%%, A_r2 arith-2hop=%.1f%%, "
        "B greedy-kNN=%.1f%%, Beam beam-kNN=%.1f%%.\n"
        "- Multi-hop (raw): A=%.1f%%, A_r2=%.1f%%, B=%.1f%%, Beam=%.1f%%.\n"
        "- A_r2 - A = %+.1f (restriction on PLAIN arithmetic); Beam - B = %+.1f "
        "(soft vs greedy).\n"
        "- **%s**\n"
        % (K_CAND, BEAM, A, Ar2, B, Bm,
           (raw_r["A_pct"] if raw_r else float('nan')),
           (raw_r["Ar2_pct"] if raw_r else float('nan')),
           (raw_r["B_pct"] if raw_r else float('nan')),
           (raw_r["Beam_pct"] if raw_r else float('nan')),
           Ar2 - A, Bm - B, outcome)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 4b entry to findings.md")


if __name__ == '__main__':
    main()
