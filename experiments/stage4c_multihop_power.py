"""
SAGE-Sphere - Stage 4c - Powered multi-hop test (valid significance)
=====================================================================
Stage 4/4b left the multi-hop verdict UNDERPOWERED (B beat arithmetic by ~6-9pp
on only n=34). This stage resolves it with a STATISTICALLY VALID design.

Why a single held-out split, not k-fold CV: under k-fold, every test item in a
fold shares that fold's prototype and folds overlap in training, so the ~N test
predictions are NOT independent - which violates the independence assumption of
both McNemar and the bootstrap and can manufacture false significance. Instead we
fit ONE prototype on a disjoint TRAIN set of countries and test on the held-out
REST; conditional on that single fixed prototype the paired test items are
independent, so the paired test is valid.

  TRAIN_FRAC of countries -> build dir1 (capital->country), dir2 (country->cont).
  The remaining countries are the held-out test set (each tested once).

Methods (multi-hop capital -> country -> continent), paired per test country:
  A = one-shot arithmetic (sum prototype offsets, global nearest)  [baseline]
  B = greedy graph re-grounding, kNN pool                          [traversal]
(4b showed B is the best traversal variant; beam/restriction-alone did not help.)

Significance (two complementary, distinct estimands):
  - McNemar's exact test on the DISCORDANT pairs (paired A-vs-B test).
  - Bootstrap 95% CI on the marginal ACCURACY DIFFERENCE (B_acc - A_acc).
GO only if B>A AND McNemar p<0.05 AND bootstrap CI excludes 0. Else TIE/NO-GO.
A 5-fold CV accuracy is also reported, DESCRIPTIVELY only (not for inference).

Run:  python experiments/stage4c_multihop_power.py
"""

import os
import sys
import json
import random

import numpy as np
from scipy.stats import binomtest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)
from core.traversal import (relation_prototype, multi_hop_traverse,             # noqa: E402
                            multi_hop_arithmetic)
import stage4_traversal as s4                                                   # noqa: E402

RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

CAPITAL_CATS = {"capital-common-countries", "capital-world"}
K_CAND     = 50
TRAIN_FRAC = 0.30      # countries used to fit ONE prototype; rest are held-out test
N_FOLDS    = 5         # descriptive CV accuracy only
N_BOOT     = 10000
SEED       = 42

# Country -> continent over the benchmark capital-pair countries in vocab (the
# continent word must also be in vocab: asia/europe/africa/america/australia).
# Transcontinental cases use the conventional assignment; 'australia' doubles as
# the Oceania target.
CONTINENT = {
    'afghanistan': 'asia', 'armenia': 'asia', 'azerbaijan': 'asia',
    'bahrain': 'asia', 'bangladesh': 'asia', 'bhutan': 'asia', 'china': 'asia',
    'georgia': 'asia', 'indonesia': 'asia', 'iran': 'asia', 'iraq': 'asia',
    'japan': 'asia', 'jordan': 'asia', 'kazakhstan': 'asia', 'kyrgyzstan': 'asia',
    'laos': 'asia', 'lebanon': 'asia', 'nepal': 'asia', 'oman': 'asia',
    'pakistan': 'asia', 'philippines': 'asia', 'qatar': 'asia', 'syria': 'asia',
    'taiwan': 'asia', 'tajikistan': 'asia', 'thailand': 'asia', 'turkey': 'asia',
    'turkmenistan': 'asia', 'uzbekistan': 'asia', 'vietnam': 'asia',
    'albania': 'europe', 'austria': 'europe', 'belarus': 'europe',
    'belgium': 'europe', 'bulgaria': 'europe', 'croatia': 'europe',
    'cyprus': 'europe', 'denmark': 'europe', 'england': 'europe',
    'estonia': 'europe', 'finland': 'europe', 'france': 'europe',
    'germany': 'europe', 'greece': 'europe', 'hungary': 'europe',
    'ireland': 'europe', 'italy': 'europe', 'latvia': 'europe',
    'liechtenstein': 'europe', 'lithuania': 'europe', 'macedonia': 'europe',
    'malta': 'europe', 'moldova': 'europe', 'montenegro': 'europe',
    'norway': 'europe', 'poland': 'europe', 'portugal': 'europe',
    'romania': 'europe', 'russia': 'europe', 'serbia': 'europe',
    'slovakia': 'europe', 'slovenia': 'europe', 'spain': 'europe',
    'sweden': 'europe', 'switzerland': 'europe', 'ukraine': 'europe',
    'algeria': 'africa', 'angola': 'africa', 'botswana': 'africa',
    'burundi': 'africa', 'egypt': 'africa', 'eritrea': 'africa',
    'gabon': 'africa', 'gambia': 'africa', 'ghana': 'africa', 'guinea': 'africa',
    'kenya': 'africa', 'liberia': 'africa', 'libya': 'africa',
    'madagascar': 'africa', 'malawi': 'africa', 'mali': 'africa',
    'mauritania': 'africa', 'morocco': 'africa', 'mozambique': 'africa',
    'namibia': 'africa', 'niger': 'africa', 'nigeria': 'africa', 'rwanda': 'africa',
    'senegal': 'africa', 'somalia': 'africa', 'sudan': 'africa',
    'tunisia': 'africa', 'uganda': 'africa', 'zambia': 'africa',
    'zimbabwe': 'africa',
    'bahamas': 'america', 'belize': 'america', 'canada': 'america',
    'chile': 'america', 'cuba': 'america', 'dominica': 'america',
    'ecuador': 'america', 'greenland': 'america', 'guyana': 'america',
    'honduras': 'america', 'jamaica': 'america', 'nicaragua': 'america',
    'peru': 'america', 'suriname': 'america', 'uruguay': 'america',
    'venezuela': 'america',
    'australia': 'australia', 'fiji': 'australia', 'samoa': 'australia',
    'tuvalu': 'australia',
}


def build_triples(word2idx):
    cap_of = {}
    cur = None
    with open(s4.BENCH, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith(':'):
                cur = ln[1:].strip(); continue
            if cur not in CAPITAL_CATS or not ln:
                continue
            w = ln.lower().split()
            if len(w) != 4:
                continue
            for cap, ctry in ((w[0], w[1]), (w[2], w[3])):
                if ctry in CONTINENT and cap in word2idx and ctry in word2idx:
                    cont = CONTINENT[ctry]
                    if cont in word2idx:
                        cap_of.setdefault(ctry, (word2idx[cap], word2idx[ctry],
                                                 word2idx[cont]))
    return cap_of


def predict(emb, cap_of, train, test):
    """Fit ONE prototype on `train` countries, predict A and B for each `test`
    country. Returns paired 0/1 correctness arrays (a, b) or None if degenerate."""
    dir1 = relation_prototype([(cap_of[c][0], cap_of[c][1]) for c in train], emb)
    dir2 = relation_prototype([(cap_of[c][1], cap_of[c][2]) for c in train], emb)
    if dir1 is None or dir2 is None or not test:
        return None
    a, b = [], []
    for c in test:
        cap, ctry, cont = cap_of[c]
        a.append(int(multi_hop_arithmetic(cap, [dir1, dir2], emb) == cont))
        b.append(int(multi_hop_traverse(cap, [dir1, dir2], emb, K_CAND) == cont))
    return np.array(a), np.array(b)


def holdout(emb, cap_of):
    """Single fixed disjoint split -> valid paired test set."""
    countries = list(cap_of.keys())
    random.Random(SEED).shuffle(countries)
    k = max(10, int(round(TRAIN_FRAC * len(countries))))
    return predict(emb, cap_of, countries[:k], countries[k:])


def cv_accuracy(emb, cap_of):
    """5-fold CV accuracy - DESCRIPTIVE point estimate only (NOT for inference;
    fold predictions are not independent)."""
    countries = list(cap_of.keys())
    random.Random(SEED).shuffle(countries)
    folds = [countries[i::N_FOLDS] for i in range(N_FOLDS)]
    a, b = [], []
    for f in range(N_FOLDS):
        test = set(folds[f])
        out = predict(emb, cap_of, [c for c in countries if c not in test], folds[f])
        if out is not None:
            a.extend(out[0].tolist()); b.extend(out[1].tolist())
    return np.array(a), np.array(b)


def significance(a, b):
    """McNemar exact (discordant pairs) + vectorized paired bootstrap CI on the
    marginal accuracy difference."""
    n = len(a)
    A_acc = 100.0 * a.mean(); B_acc = 100.0 * b.mean()
    b_only = int(((a == 1) & (b == 0)).sum())     # A right, B wrong
    c_only = int(((a == 0) & (b == 1)).sum())     # A wrong, B right
    n_disc = b_only + c_only
    p = binomtest(c_only, n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, (N_BOOT, n))         # vectorized resample
    diffs = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = 100.0 * np.percentile(diffs, [2.5, 97.5])
    return {"n": n, "A_acc": A_acc, "B_acc": B_acc, "diff": B_acc - A_acc,
            "discordant_A_right_B_wrong": b_only, "discordant_A_wrong_B_right": c_only,
            "mcnemar_p": float(p), "boot_ci_lo": float(lo), "boot_ci_hi": float(hi)}


def main():
    torch.manual_seed(SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % device)

    words, iso = s4.load_npz(s4.ISO_NPZ)
    _, raw = s4.load_npz(s4.RAW_NPZ)
    word2idx = {w: i for i, w in enumerate(words)}
    iso = iso.to(device); raw = raw.to(device)
    cap_of = build_triples(word2idx)
    n_train = max(10, int(round(TRAIN_FRAC * len(cap_of))))
    print("Multi-hop chains in vocab: %d  (train=%d, held-out test=%d)"
          % (len(cap_of), n_train, len(cap_of) - n_train))
    print("Inference on the single held-out split; %d bootstrap resamples.\n"
          % N_BOOT)

    results = {}
    for name, emb in (("isotropic", iso), ("raw", raw)):
        res = holdout(emb, cap_of)
        if res is None:
            print("ERROR: held-out split produced no test data for %s." % name)
            sys.exit(1)
        a, b = res
        st = significance(a, b)
        acv, bcv = cv_accuracy(emb, cap_of)
        st["cv_A_acc"] = 100.0 * acv.mean(); st["cv_B_acc"] = 100.0 * bcv.mean()
        st["cv_n"] = len(acv)
        results[name] = st
        print("[%-9s] held-out n=%d  A(arith)=%.1f%%  B(traversal)=%.1f%%  "
              "diff=%+.1f  | discordant A>B=%d B>A=%d  McNemar p=%.4f  "
              "boot95%%CI[%+.1f, %+.1f]"
              % (name, st["n"], st["A_acc"], st["B_acc"], st["diff"],
                 st["discordant_A_right_B_wrong"], st["discordant_A_wrong_B_right"],
                 st["mcnemar_p"], st["boot_ci_lo"], st["boot_ci_hi"]))
        print("            descriptive 5-fold CV (n=%d): A=%.1f%%  B=%.1f%%"
              % (st["cv_n"], st["cv_A_acc"], st["cv_B_acc"]))
    print()

    s = results["isotropic"]
    sig = s["mcnemar_p"] < 0.05 and s["boot_ci_lo"] > 0
    if s["B_acc"] > s["A_acc"] and sig:
        verdict = "GO (traversal beats arithmetic, significant)"
    elif s["A_acc"] > s["B_acc"] and s["mcnemar_p"] < 0.05:
        verdict = "NO-GO (arithmetic beats traversal, significant)"
    else:
        verdict = "TIE / NOT SIGNIFICANT (cannot reject A == B)"

    print("=" * 72)
    print("STAGE 4c VERDICT (isotropic, held-out n=%d): %s" % (s["n"], verdict))
    print("  McNemar (discordant-pair test) p = %.4f  [%d discordant pairs]"
          % (s["mcnemar_p"], s["discordant_A_right_B_wrong"]
             + s["discordant_A_wrong_B_right"]))
    print("  bootstrap 95%% CI on accuracy diff (B-A) = [%+.1f, %+.1f] pp"
          % (s["boot_ci_lo"], s["boot_ci_hi"]))
    print("=" * 72)

    out = {"stage": "4c", "verdict": verdict, "k_cand": K_CAND,
           "train_frac": TRAIN_FRAC, "n_boot": N_BOOT, "results": results,
           "note": "inference on single held-out split (valid independence); "
                   "5-fold CV reported descriptively only"}
    res_path = os.path.join(RES_DIR, 'stage4c_multihop_power.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # plot: held-out A vs B per emb, p/CI annotated
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    xc = np.arange(2)
    for off, (name, col) in zip((-0.2, 0.2),
                                (("isotropic", '#4fc3f7'), ("raw", '#ff7043'))):
        st = results[name]
        ax.bar(xc + off, [st["A_acc"], st["B_acc"]], width=0.38, color=col,
               label='%s (p=%.3f, CI[%+.1f,%+.1f])'
               % (name, st["mcnemar_p"], st["boot_ci_lo"], st["boot_ci_hi"]))
    ax.set_xticks(xc); ax.set_xticklabels(['A arithmetic', 'B traversal'])
    ax.set_ylabel('multi-hop chain accuracy %', color='white')
    ax.set_title('Stage 4c held-out (n=%d): %s'
                 % (s["n"], verdict.split('(')[0].strip()), color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage4c_multihop_power.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    r = results["raw"]
    entry = (
        "\n## Stage 4c - Powered multi-hop (valid significance)\n\n"
        "- %d chains; ONE prototype on %d train countries, held-out test on the "
        "rest (valid independence, unlike k-fold). %d bootstrap resamples.\n"
        "- **VERDICT (isotropic, held-out n=%d): %s.**\n"
        "- Isotropic held-out: A(arith)=%.1f%%, B(traversal)=%.1f%%, B-A=%+.1f pp; "
        "McNemar p=%.4f (discordant A>B=%d, B>A=%d); bootstrap 95%% CI [%+.1f, "
        "%+.1f] pp. Descriptive 5-fold CV: A=%.1f%%, B=%.1f%%.\n"
        "- Raw held-out: A=%.1f%%, B=%.1f%%, B-A=%+.1f pp; McNemar p=%.4f; "
        "CI [%+.1f, %+.1f].\n"
        "- McNemar tests discordant pairs; the bootstrap CI is on the marginal "
        "accuracy difference - distinct estimands, GO requires BOTH. Single-hop "
        "analogy remains a solid null (Stage 4: B~=Aknn~=85%%).\n"
        % (len(cap_of), n_train, N_BOOT, s["n"], verdict,
           s["A_acc"], s["B_acc"], s["diff"], s["mcnemar_p"],
           s["discordant_A_right_B_wrong"], s["discordant_A_wrong_B_right"],
           s["boot_ci_lo"], s["boot_ci_hi"], s["cv_A_acc"], s["cv_B_acc"],
           r["A_acc"], r["B_acc"], r["diff"], r["mcnemar_p"],
           r["boot_ci_lo"], r["boot_ci_hi"])
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 4c entry to findings.md")


if __name__ == '__main__':
    main()
