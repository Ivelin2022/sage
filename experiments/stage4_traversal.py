"""
SAGE-Sphere - Stage 4 - Dijkstra traversal as composition primitive
====================================================================
  ***  THE GO / NO-GO STAGE FOR THE WHOLE SAGE-SPHERE DIRECTION  ***

Head-to-head, same embeddings, same questions, LEAKAGE-SAFE:

  Method A  (baseline)  : 3CosAdd arithmetic  b - a + c, nearest over ALL words.
                          The single-pair parallelogram that FAILED in 8A/9C.
  Method A' (control)   : c + dir_R, nearest over ALL words. dir_R = mean relation
                          direction from TRAINING pairs (isolates averaging).
  Method B  (new)       : c + dir_R, restricted to c's kNN graph neighbours
                          (= traversal: follow the typed edge along the graph).

Multi-hop (capital -> country -> continent):
  Method A : sum prototype offsets, ONE global nearest (drifts, no re-grounding).
  Method B : traverse hop-by-hop, RE-GROUNDING on a real node each hop.

LEAKAGE GUARDS:
  - Relation pairs split TRAIN/TEST per category; dir_R built from TRAIN only.
  - Analogy questions formed from TEST pairs; answer d never in dir_R's training.
  - Multi-hop: COUNTRIES split train/test; prototypes from train countries only,
    test chains use held-out countries' capitals. {a,b,c}/start excluded from
    candidates.

GATE (decides the project):
  GO    : B clearly beats A on multi-hop AND matches/beats A on analogy Hits@1.
  TIE   : B ~= A -> composition not really fixed; investigate.
  NO-GO : B loses to A -> stop, write the honest negative (Stage 1 still stands).

Run on raw AND isotropic to settle the isotropy question. No tuning to win - a
NO-GO is a valid, valuable result. Loads the Stage 2/3 artifacts. No Ollama.
  python experiments/stage4_traversal.py
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
NEWARCH = os.path.abspath(os.path.join(ROOT, '..', '..'))
sys.path.insert(0, ROOT)
from core.traversal import (relation_prototype, arithmetic_analogy,             # noqa: E402
                            prototype_analogy, neighbors, multi_hop_traverse,
                            multi_hop_arithmetic)
# NB: core.traversal.dijkstra is the general typed-path primitive (brief
# deliverable). The evaluated multi-hop here uses greedy re-grounding, which is
# the single-best-per-hop special case; Dijkstra/beam is the weighted general form.

ISO_NPZ = os.path.join(ROOT, 'data', 'embeddings_isotropic.npz')
RAW_NPZ = os.path.join(ROOT, 'data', 'embeddings_cache.npz')
BENCH   = os.path.join(NEWARCH, 'sage_revision', 'experiments',
                       'exp2_full_analogy', 'questions-words.txt')
RES_DIR = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

CAPITAL_CATS = {"capital-common-countries", "capital-world"}
MIN_PAIRS = 10
K_CAND    = 50               # graph neighbourhood size for Method B (traversal)
MAX_Q     = 200              # cap analogy questions per category
SEED      = 42

# country -> continent (ground-truth lookup data, not learned). Continent word
# must also be a vocab node to be a valid multi-hop target.
CONTINENT = {
    'china': 'asia', 'japan': 'asia', 'india': 'asia', 'thailand': 'asia',
    'iraq': 'asia', 'iran': 'asia', 'israel': 'asia', 'korea': 'asia',
    'vietnam': 'asia', 'indonesia': 'asia', 'pakistan': 'asia', 'jordan': 'asia',
    'lebanon': 'asia', 'taiwan': 'asia', 'laos': 'asia', 'nepal': 'asia',
    'bangladesh': 'asia', 'afghanistan': 'asia', 'syria': 'asia',
    'france': 'europe', 'germany': 'europe', 'italy': 'europe', 'spain': 'europe',
    'england': 'europe', 'greece': 'europe', 'portugal': 'europe',
    'poland': 'europe', 'russia': 'europe', 'ukraine': 'europe',
    'sweden': 'europe', 'norway': 'europe', 'finland': 'europe',
    'denmark': 'europe', 'austria': 'europe', 'switzerland': 'europe',
    'belgium': 'europe', 'netherlands': 'europe', 'ireland': 'europe',
    'hungary': 'europe', 'romania': 'europe', 'bulgaria': 'europe',
    'croatia': 'europe', 'serbia': 'europe', 'slovakia': 'europe',
    'egypt': 'africa', 'nigeria': 'africa', 'kenya': 'africa', 'ghana': 'africa',
    'morocco': 'africa', 'algeria': 'africa', 'tunisia': 'africa',
    'libya': 'africa', 'sudan': 'africa', 'angola': 'africa', 'zambia': 'africa',
    'zimbabwe': 'africa', 'uganda': 'africa', 'mali': 'africa',
    'senegal': 'africa', 'gabon': 'africa', 'namibia': 'africa',
    'botswana': 'africa', 'mozambique': 'africa', 'madagascar': 'africa',
    'canada': 'america', 'mexico': 'america', 'cuba': 'america',
    'brazil': 'america', 'argentina': 'america', 'chile': 'america',
    'peru': 'america', 'colombia': 'america', 'venezuela': 'america',
    'ecuador': 'america', 'uruguay': 'america', 'bolivia': 'america',
}


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    words = [str(w) for w in z['words']]
    embs = F.normalize(torch.tensor(np.asarray(z['embs'], dtype=np.float32)),
                       p=2, dim=1)
    return words, embs


def load_category_pairs(path, word2idx):
    """category -> list of (head_idx, tail_idx) directed relation pairs in vocab."""
    cats = {}
    cur = None
    with open(path, encoding='utf-8') as f:
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
                if h in word2idx and t in word2idx and h != t:
                    cats[cur].add((word2idx[h], word2idx[t]))
    return {c: sorted(p) for c, p in cats.items() if len(p) >= MIN_PAIRS}


def split_pairs(pairs, rng):
    p = list(pairs)
    rng.shuffle(p)
    half = max(1, len(p) // 2)
    return p[:half], p[half:]            # train, test


def eval_analogy(emb, cats):
    """Returns per-method {hit1, hit5, n, per_cat} for A, Aprime, B."""
    rng = random.Random(SEED)
    # A   = arithmetic, global       Aknn = arithmetic, kNN pool (restriction control)
    # Ap  = prototype, global        B    = prototype, kNN pool (traversal)
    # CLEAN same-pool tests: A vs Ap (global), Aknn vs B (kNN). The verdict gates
    # on B vs Aknn (same pool) so a win is the relation mechanism, not pool size.
    METHODS = ("A", "Aknn", "Ap", "B")
    res = {m: {"h1": 0, "h5": 0, "n": 0, "cat": {}} for m in METHODS}
    reach = reach_n = 0                                    # % questions with d in kNN(c)
    for cat, pairs in cats.items():
        train, test = split_pairs(pairs, rng)
        if len(train) < 3 or len(test) < 2:
            continue
        dir_R = relation_prototype(train, emb)
        # Questions from TEST pairs only. Require 4 DISTINCT words: if the answer
        # d coincides with a/b/c it is in the exclude set -> unanswerable, which
        # would depress every method equally and pollute the metric.
        qs = [(a, b, c, d) for (a, b) in test for (c, d) in test
              if (a, b) != (c, d) and len({a, b, c, d}) == 4]
        rng.shuffle(qs)
        qs = qs[:MAX_Q]
        if not qs:
            continue
        # cache c's kNN pool once per distinct c (c repeats across many questions)
        nb_cache = {c: neighbors(c, emb, K_CAND) for c in {q[2] for q in qs}}
        nb_sets = {c: set(nb.tolist()) for c, nb in nb_cache.items()}
        cstat = {m: [0, 0] for m in METHODS}              # [h1, h5]
        for a, b, c, d in qs:
            ex = (a, b, c)
            nb = nb_cache[c]
            reach_n += 1
            if d in nb_sets[c]:
                reach += 1
            preds = {
                "A":    arithmetic_analogy(a, b, c, emb, candidates=None,
                                           topk=5, exclude=ex).tolist(),
                "Aknn": arithmetic_analogy(a, b, c, emb, candidates=nb,
                                           topk=5, exclude=ex).tolist(),
                "Ap":   prototype_analogy(c, dir_R, emb, candidates=None,
                                          topk=5, exclude=ex).tolist(),
                "B":    prototype_analogy(c, dir_R, emb, candidates=nb,
                                          topk=5, exclude=ex).tolist(),
            }
            for m, pr in preds.items():
                if pr and pr[0] == d:
                    cstat[m][0] += 1
                if d in pr:
                    cstat[m][1] += 1
        nq = len(qs)
        for m in METHODS:
            res[m]["h1"] += cstat[m][0]; res[m]["h5"] += cstat[m][1]
            res[m]["n"] += nq
            res[m]["cat"][cat] = {"n": nq,
                                  "h1_pct": 100.0 * cstat[m][0] / nq,
                                  "h5_pct": 100.0 * cstat[m][1] / nq}
    for m in METHODS:
        n = max(1, res[m]["n"])
        res[m]["hit1_pct"] = 100.0 * res[m]["h1"] / n      # micro (question-weighted)
        res[m]["hit5_pct"] = 100.0 * res[m]["h5"] / n
        ch = [res[m]["cat"][c]["h1_pct"] for c in res[m]["cat"]]
        res[m]["hit1_macro_pct"] = (sum(ch) / len(ch)) if ch else 0.0   # category-weighted
    res["_reach_in_knn_pct"] = 100.0 * reach / max(1, reach_n)
    return res


def build_multihop(path, word2idx):
    """country -> (capital_idx, country_idx, continent_idx) where all in vocab."""
    cap_of = {}
    with open(path, encoding='utf-8') as f:
        cur = None
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


def eval_multihop(emb, cap_of):
    rng = random.Random(SEED)
    countries = list(cap_of.keys())
    rng.shuffle(countries)
    half = max(1, len(countries) // 2)
    train_c, test_c = countries[:half], countries[half:]
    cap_country = [(cap_of[c][0], cap_of[c][1]) for c in train_c]
    country_cont = [(cap_of[c][1], cap_of[c][2]) for c in train_c]
    dir1 = relation_prototype(cap_country, emb)      # capital -> country
    dir2 = relation_prototype(country_cont, emb)     # country -> continent
    if dir1 is None or dir2 is None or not test_c:
        return None
    N = emb.shape[0]
    A = Bg = B = hop1 = reach = 0
    n = 0
    for c in test_c:
        cap, ctry, cont = cap_of[c]
        n += 1
        # A  = arithmetic: sum offsets, ONE global nearest (no re-grounding).
        # Bg = re-grounding over the GLOBAL pool each hop -> isolates re-grounding
        #      from the kNN restriction (the clean same-pool control vs A).
        # B  = re-grounding restricted to the kNN graph (the deployed traversal).
        # All judged correct iff prediction == continent; symmetric start-exclusion.
        a_pred = multi_hop_arithmetic(cap, [dir1, dir2], emb)
        bg_pred = multi_hop_traverse(cap, [dir1, dir2], emb, N)
        b_pred = multi_hop_traverse(cap, [dir1, dir2], emb, K_CAND)
        nb = neighbors(cap, emb, K_CAND)
        mid = int(nb[torch.argmax(F.normalize(emb[nb] - emb[cap], p=2, dim=1) @ dir1)])
        cnb = set(neighbors(ctry, emb, K_CAND).tolist())   # B hop-2 reachability
        if a_pred == cont:
            A += 1
        if bg_pred == cont:
            Bg += 1
        if b_pred == cont:
            B += 1
        if mid == ctry:
            hop1 += 1
        if cont in cnb:
            reach += 1
    return {"n": n, "A_pct": 100.0 * A / n, "Bglobal_pct": 100.0 * Bg / n,
            "B_pct": 100.0 * B / n, "B_hop1_country_pct": 100.0 * hop1 / n,
            "continent_in_knn_pct": 100.0 * reach / n,
            "n_train_countries": len(train_c)}


def main():
    torch.manual_seed(SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % device)

    iso_words, iso = load_npz(ISO_NPZ)
    raw_words, raw = load_npz(RAW_NPZ)
    assert raw_words == iso_words
    words = iso_words
    word2idx = {w: i for i, w in enumerate(words)}
    iso = iso.to(device); raw = raw.to(device)

    cats = load_category_pairs(BENCH, word2idx)
    print("Analogy categories (>=%d pairs): %d" % (MIN_PAIRS, len(cats)))
    cap_of = build_multihop(BENCH, word2idx)
    print("Multi-hop countries (capital+country+continent in vocab): %d\n"
          % len(cap_of))

    results = {}
    for name, emb in (("isotropic", iso), ("raw", raw)):
        ana = eval_analogy(emb, cats)
        mh = eval_multihop(emb, cap_of)
        results[name] = {"analogy": ana, "multihop": mh}
        print("=== %s ===" % name)
        print("  analogy Hits@1 micro  A(arith-glob)=%.1f%%  Aknn(arith-kNN)=%.1f%%  "
              "A'(proto-glob)=%.1f%%  B(proto-kNN)=%.1f%%"
              % (ana["A"]["hit1_pct"], ana["Aknn"]["hit1_pct"],
                 ana["Ap"]["hit1_pct"], ana["B"]["hit1_pct"]))
        print("  analogy Hits@1 macro  A=%.1f%%  Aknn=%.1f%%  A'=%.1f%%  B=%.1f%%  "
              "| answer-in-kNN(c)=%.1f%%"
              % (ana["A"]["hit1_macro_pct"], ana["Aknn"]["hit1_macro_pct"],
                 ana["Ap"]["hit1_macro_pct"], ana["B"]["hit1_macro_pct"],
                 ana["_reach_in_knn_pct"]))
        print("  CLEAN same-pool tests: A vs A'(global) | Aknn vs B(kNN)")
        if mh:
            print("  multi-hop  A(arith)=%.1f%%  Bg(reground-global)=%.1f%%  "
                  "B(reground-kNN)=%.1f%%  (hop-1 country=%.1f%%, cont-in-kNN=%.1f%%,"
                  " n=%d)"
                  % (mh["A_pct"], mh["Bglobal_pct"], mh["B_pct"],
                     mh["B_hop1_country_pct"], mh["continent_in_knn_pct"], mh["n"]))
        print()

    # ---- VERDICT on the isotropic results (primary) ----
    # Gate on the CLEAN same-pool comparisons so a verdict reflects the relation
    # mechanism, NOT candidate-pool size:
    #   analogy   : B (proto, kNN pool)  vs  Aknn (arith, SAME kNN pool)
    #   multi-hop : Bg (re-ground, GLOBAL pool) vs A (arith, global) -> isolates
    #               re-grounding from the kNN restriction.
    MARGIN = 5.0
    iso_a = results["isotropic"]["analogy"]
    iso_m = results["isotropic"]["multihop"]
    A_h1 = iso_a["A"]["hit1_pct"]; B_h1 = iso_a["B"]["hit1_pct"]
    Aknn_h1 = iso_a["Aknn"]["hit1_pct"]          # clean same-pool baseline for B
    if iso_m is None:
        verdict = "INCOMPLETE-NO-MULTIHOP"
        A_mh = Bg_mh = B_mh = float('nan')
    else:
        A_mh = iso_m["A_pct"]; Bg_mh = iso_m["Bglobal_pct"]; B_mh = iso_m["B_pct"]
        analogy_ok = B_h1 >= Aknn_h1                  # mechanism not worse (same pool)
        if Bg_mh >= A_mh + MARGIN and analogy_ok:
            verdict = "GO"
        elif Bg_mh <= A_mh - MARGIN or B_h1 <= Aknn_h1 - MARGIN:
            verdict = "NO-GO"
        else:
            verdict = "TIE/INVESTIGATE"

    print("=" * 70)
    print("STAGE 4 VERDICT (isotropic): %s" % verdict)
    print("  CLEAN analogy   Hits@1: B(proto-kNN)=%.1f%% vs Aknn(arith-kNN)=%.1f%% "
          "(diff %+.1f) <- gates" % (B_h1, Aknn_h1, B_h1 - Aknn_h1))
    print("  CLEAN multi-hop       : Bg(reground-global)=%.1f%% vs A(arith)=%.1f%% "
          "(diff %+.1f) <- gates" % (Bg_mh, A_mh, Bg_mh - A_mh))
    print("  context: brief B-vs-A analogy %+.1f | deployed B(kNN) multi-hop=%.1f%%"
          % (B_h1 - A_h1, B_mh))
    print("  GO = re-grounding wins multi-hop(+%.0f) AND B>=Aknn on analogy | "
          "NO-GO = loses by %.0f | TIE = between" % (MARGIN, MARGIN))
    print("=" * 70)

    out = {"stage": 4, "verdict": verdict, "k_cand": K_CAND,
           "primary": "isotropic", "results": results,
           "gate_numbers": {
               "analogy_hit1_B": B_h1, "analogy_hit1_Aknn": Aknn_h1,
               "analogy_hit1_A": A_h1,
               "analogy_clean_B_minus_Aknn": B_h1 - Aknn_h1,
               "multihop_A": (None if iso_m is None else A_mh),
               "multihop_Bglobal": (None if iso_m is None else Bg_mh),
               "multihop_B_knn": (None if iso_m is None else B_mh),
               "multihop_clean_Bg_minus_A": (None if iso_m is None else Bg_mh - A_mh)}}
    res_path = os.path.join(RES_DIR, 'stage4_traversal.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # ---- plot (dark): analogy A/A'/B Hits@1 per emb + multi-hop A vs B ----
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0a0a0a')
    for ax in axs:
        ax.set_facecolor('#111111')

    labels = ['A\narith-glob', 'Aknn\narith-kNN', "A'\nproto-glob", 'B\nproto-kNN']
    xc = np.arange(len(labels))
    for off, (name, col) in zip((-0.2, 0.2),
                                (("isotropic", '#4fc3f7'), ("raw", '#ff7043'))):
        a = results[name]["analogy"]
        vals = [a["A"]["hit1_pct"], a["Aknn"]["hit1_pct"],
                a["Ap"]["hit1_pct"], a["B"]["hit1_pct"]]
        axs[0].bar(xc + off, vals, width=0.38, color=col, label=name)
    axs[0].set_xticks(xc); axs[0].set_xticklabels(labels, fontsize=8)
    axs[0].set_title('analogy Hits@1 (leakage-safe)', color='white')
    axs[0].set_ylabel('% correct', color='white')
    axs[0].legend(facecolor='#111111', edgecolor='#333333', fontsize=8)

    mh_labels = ['A\narith', 'Bg\nregrnd-glob', 'B\nregrnd-kNN']
    for off, (name, col) in zip((-0.2, 0.2),
                                (("isotropic", '#4fc3f7'), ("raw", '#ff7043'))):
        m = results[name]["multihop"]
        vals = [m["A_pct"], m["Bglobal_pct"], m["B_pct"]] if m else [0, 0, 0]
        axs[1].bar(np.arange(3) + off, vals, width=0.38, color=col, label=name)
    axs[1].set_xticks(np.arange(3)); axs[1].set_xticklabels(mh_labels, fontsize=8)
    axs[1].set_title('multi-hop chain accuracy', color='white')
    axs[1].set_ylabel('% correct', color='white')
    axs[1].legend(facecolor='#111111', edgecolor='#333333', fontsize=8)

    fig.suptitle('Stage 4 GO/NO-GO: %s' % verdict, color='white', fontsize=13)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage4_traversal.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    # ---- findings ----
    rawA = results["raw"]["analogy"]
    raw_mh = results["raw"]["multihop"]
    next_step = ("Proceed to Stage 5." if verdict == "GO"
                 else "Stop and analyse honestly; Stage 1 substrate still stands."
                 if verdict == "NO-GO" else
                 "Investigate before Stage 5 (relation mechanism vs pool restriction).")
    entry = (
        "\n## Stage 4 - Dijkstra traversal (GO/NO-GO)\n\n"
        "- Leakage-safe: dir_R from TRAIN pairs only; analogy questions from TEST "
        "pairs (4 distinct words); multi-hop countries split train/test. K_CAND=%d.\n"
        "- **VERDICT (isotropic): %s.** Gated on CLEAN same-pool comparisons.\n"
        "- Analogy Hits@1 micro (isotropic): A arith-glob=%.1f%%, Aknn arith-kNN=%.1f%%, "
        "A' proto-glob=%.1f%%, B proto-kNN=%.1f%%. Macro B=%.1f%%. answer-in-kNN(c)=%.1f%%.\n"
        "  - **CLEAN gate (B vs Aknn, same pool): %+.1f** (relation mechanism, not "
        "pool size). Context brief (B vs A): %+.1f.\n"
        "- Analogy Hits@1 micro (raw): A=%.1f%%, Aknn=%.1f%%, A'=%.1f%%, B=%.1f%%.\n"
        "- Multi-hop (isotropic): A arith=%.1f%%, Bg reground-global=%.1f%%, "
        "B reground-kNN=%.1f%%; hop-1 country=%.1f%%, cont-in-kNN=%.1f%%.\n"
        "  - **CLEAN gate (Bg vs A, isolates re-grounding): %+.1f.**\n"
        "- Multi-hop (raw): A=%.1f%%, Bg=%.1f%%, B=%.1f%%.\n"
        "- GATE: GO = re-grounding beats arith on multi-hop by >=%.0f AND B>=Aknn "
        "on analogy; NO-GO = loses by %.0f; else TIE. %s\n"
        % (K_CAND, verdict,
           A_h1, Aknn_h1, iso_a["Ap"]["hit1_pct"], B_h1,
           iso_a["B"]["hit1_macro_pct"], iso_a["_reach_in_knn_pct"],
           B_h1 - Aknn_h1, B_h1 - A_h1,
           rawA["A"]["hit1_pct"], rawA["Aknn"]["hit1_pct"],
           rawA["Ap"]["hit1_pct"], rawA["B"]["hit1_pct"],
           A_mh, Bg_mh, B_mh,
           (iso_m["B_hop1_country_pct"] if iso_m else float('nan')),
           (iso_m["continent_in_knn_pct"] if iso_m else float('nan')),
           Bg_mh - A_mh,
           (raw_mh["A_pct"] if raw_mh else float('nan')),
           (raw_mh["Bglobal_pct"] if raw_mh else float('nan')),
           (raw_mh["B_pct"] if raw_mh else float('nan')),
           MARGIN, MARGIN, next_step)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 4 entry to findings.md")


if __name__ == '__main__':
    main()
