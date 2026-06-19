"""
SAGE-Sphere - Binding memory: does structured multi-hop hold up?
=================================================================
A new direction after the traversal NO-GO. Test whether FHRR binding + geometric
cleanup can do multi-hop reasoning ROBUSTLY - the thing greedy traversal could not.

Controlled synthetic knowledge base (ideal case, random near-orthogonal vectors,
so the MECHANISM is seen cleanly without embedding correlation noise):
  - N entities, R relation types; each entity has one target per relation
    (a random multi-relational graph).
  - Store each entity's record: rec[e] = sum_r bind(relation_r, target_{e,r}).
  - 1-HOP: unbind rec[e] by relation_r, cleanup -> is it the right target?
  - 2-HOP: chain two unbind+cleanup steps -> is the 2-hop target right?
    (this is the multi-hop test; cleanup re-grounds each hop so noise can't compound)
  - 2-HOP-oracle: hop-2 given a CORRECT hop-1, to isolate per-hop quality.

Swept over R (memory load = facts per entity) and D (dimension) to map the regime
where multi-hop binding works. 2-hop ~ (1-hop)^2 if hops are independent, so the
game is keeping 1-hop high (low load / high D). Honest: this is the IDEAL case;
real correlated embeddings will be harder (the planned follow-up).

Pure numpy, no Ollama, fast.  python experiments/binding_multihop.py
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.binding import random_phasors, unbind, cleanup    # noqa: E402

RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

N_ENT    = 500            # entities (cleanup codebook size)
D_GRID   = [512, 1024, 2048]
R_GRID   = [1, 2, 4, 8, 16, 32]   # facts per entity (superposition load)
N_QUERY  = 3000
SEED     = 42
PRIOR_TRAVERSAL = 62.0    # Stage 4 multi-hop arithmetic baseline, for reference


def build_kb(n, d, r, rng):
    """Random multi-relational KB. Returns entities, relations, targets, records."""
    entities = random_phasors(n, d, rng)              # (N, D)
    relations = random_phasors(r, d, rng)             # (R, D)
    targets = rng.integers(0, n, size=(n, r))         # (N, R) -> target entity idx
    rec = np.zeros((n, d), dtype=complex)
    for rel in range(r):
        rec += relations[rel] * entities[targets[:, rel]]   # bundle of bindings
    return entities, relations, targets, rec


def eval_cell(n, d, r, rng):
    entities, relations, targets, rec = build_kb(n, d, r, rng)

    # 1-hop: (e, rel) -> target
    e = rng.integers(0, n, N_QUERY)
    rel1 = rng.integers(0, r, N_QUERY)
    pred1 = cleanup(unbind(rec[e], relations[rel1]), entities)
    true1 = targets[e, rel1]
    acc1 = 100.0 * np.mean(pred1 == true1)

    # 2-hop chained: (e, rel1, rel2) -> target[ target[e,rel1], rel2 ]
    e0 = rng.integers(0, n, N_QUERY)
    ra = rng.integers(0, r, N_QUERY)
    rb = rng.integers(0, r, N_QUERY)
    t1 = cleanup(unbind(rec[e0], relations[ra]), entities)        # hop 1 + cleanup
    t2 = cleanup(unbind(rec[t1], relations[rb]), entities)        # hop 2 + cleanup
    t1_true = targets[e0, ra]
    t2_true = targets[t1_true, rb]
    acc2 = 100.0 * np.mean(t2 == t2_true)

    # 2-hop with a CORRECT hop-1 (isolates hop-2 quality)
    t2_or = cleanup(unbind(rec[t1_true], relations[rb]), entities)
    acc2_oracle = 100.0 * np.mean(t2_or == t2_true)
    return acc1, acc2, acc2_oracle


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    print("Synthetic binding KB: N=%d entities, %d queries/cell\n" % (N_ENT, N_QUERY))

    rows = []
    print("%5s %4s %9s %9s %12s %12s" %
          ("D", "R", "1-hop%", "2-hop%", "2hop-oracle%", "(1hop)^2%"))
    for d in D_GRID:
        for r in R_GRID:
            rng = np.random.default_rng(SEED)     # deterministic per cell (KB+queries
            #                                       share the stream; each cell's 3000-
            #                                       query estimate is stable on its own)
            a1, a2, a2o = eval_cell(N_ENT, d, r, rng)
            expect2 = (a1 / 100.0) ** 2 * 100.0
            rows.append({"D": d, "R": r, "hop1": a1, "hop2": a2,
                         "hop2_oracle": a2o, "expected_hop2": expect2})
            print("%5d %4d %8.1f%% %8.1f%% %11.1f%% %11.1f%%"
                  % (d, r, a1, a2, a2o, expect2))

    # best operating regime: largest R that still keeps 2-hop usable (>= prior + 10)
    target2 = PRIOR_TRAVERSAL + 10.0
    usable = [x for x in rows if x["hop2"] >= target2]
    best = max(usable, key=lambda x: (x["R"], x["D"])) if usable else None

    print("\n" + "=" * 64)
    if best:
        print("REGIME: 2-hop binding stays usable (>= %.0f%%) up to R=%d facts/"
              "entity at D=%d -> 2-hop=%.1f%% (vs traversal %.0f%%)."
              % (target2, best["R"], best["D"], best["hop2"], PRIOR_TRAVERSAL))
        print("  PROMISING: cleanup-chained binding does multi-hop where greedy "
              "traversal could not - in the low-load / high-D regime.")
    else:
        print("NO USABLE REGIME at these D/R: 2-hop never clears %.0f%%. Binding "
              "capacity too low here; needs higher D or fewer facts/entity, or a "
              "resonator-network cleanup." % target2)
    print("=" * 64)

    out = {"experiment": "binding_multihop", "n_entities": N_ENT,
           "n_query": N_QUERY, "prior_traversal_pct": PRIOR_TRAVERSAL,
           "best_regime": best, "sweep": rows}
    res_path = os.path.join(RES_DIR, 'binding_multihop.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # plot: 1-hop and 2-hop accuracy vs R, one line set per D
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9.5, 5.5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    colors = ['#4fc3f7', '#26c281', '#ff7043']
    for d, col in zip(D_GRID, colors):
        xs = [x["R"] for x in rows if x["D"] == d]
        h1 = [x["hop1"] for x in rows if x["D"] == d]
        h2 = [x["hop2"] for x in rows if x["D"] == d]
        ax.plot(xs, h1, 'o-', color=col, label='D=%d 1-hop' % d)
        ax.plot(xs, h2, 's--', color=col, alpha=0.6, label='D=%d 2-hop' % d)
    ax.axhline(PRIOR_TRAVERSAL, color='#888888', ls=':', lw=1,
               label='traversal baseline (%.0f%%)' % PRIOR_TRAVERSAL)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R = facts per entity (memory load)', color='white')
    ax.set_ylabel('chain accuracy %', color='white')
    ax.set_title('Binding memory: multi-hop accuracy vs load (synthetic, ideal)',
                 color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=7, ncol=2)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'binding_multihop.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    # findings
    if best:
        verdict = ("PROMISING: cleanup-chained binding holds 2-hop up to R=%d "
                   "facts/entity at D=%d (2-hop=%.1f%% vs traversal %.0f%%)"
                   % (best["R"], best["D"], best["hop2"], PRIOR_TRAVERSAL))
    else:
        verdict = ("NULL at tested D/R: 2-hop never clears %.0f%%; capacity too "
                   "low - needs higher D or resonator cleanup" % target2)
    entry = (
        "\n## Binding memory - multi-hop (new direction after traversal NO-GO)\n\n"
        "- FHRR bind/unbind + geometric cleanup (= sphere nearest-neighbour). "
        "Synthetic KB, N=%d entities, ideal random vectors. Swept D in %s, R in %s.\n"
        "- **%s.**\n"
        "- Mechanism: cleanup re-grounds each hop on a clean entity, so noise does "
        "NOT compound - the property greedy traversal lacked. 2-hop ~ (1-hop)^2.\n"
        "- NOTE: this is the IDEAL case (near-orthogonal random vectors). Follow-up: "
        "repeat on the real isotropic embeddings (correlated -> lower capacity), and "
        "if capacity-limited, add a resonator-network cleanup.\n"
        % (N_ENT, str(D_GRID), str(R_GRID), verdict)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended binding-memory entry to findings.md")


if __name__ == '__main__':
    main()
