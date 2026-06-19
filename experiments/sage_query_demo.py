"""
SAGE-Sphere - Real query demo: answering multi-hop questions by binding
========================================================================
A working SAGE you can query. Loads REAL geography facts (capital -> country ->
continent) with the actual isotropic nomic embeddings as entities, stores each
entity's facts as a BOUND vector, and answers real multi-hop questions by
unbind-chaining + geometric cleanup (= the sphere's nearest-neighbour retrieval).

  "What continent is the country of Paris in?"
     rec[paris]  --unbind located_in-->  cleanup --> france
     rec[france] --unbind in_continent--> cleanup --> europe

Two things it demonstrates:
  1. The binding does real work: a city's record is OPAQUE (cos~0 to the answer)
     until unbound with the correct role key (cos~1) - structure you can only read
     with the right question.
  2. It holds under realistic MEMORY LOAD: pack R facts per entity (1 real +
     distractors) and the real query still answers up to the capacity knee.

Reuses the Stage 4c fact loader. No Ollama.  python experiments/sage_query_demo.py
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)
from core.binding import hrr_bind, hrr_unbind, cleanup, make_unitary   # noqa: E402
import stage4c_multihop_power as s4c                                   # noqa: E402

ISO_NPZ  = os.path.join(ROOT, 'data', 'embeddings_isotropic.npz')
RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

R_GRID   = [1, 2, 4, 8, 16, 32, 64]
SHOWCASE = ['france', 'japan', 'egypt', 'canada', 'germany', 'china',
            'italy', 'kenya', 'peru', 'thailand']
SEED     = 42


def unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def build_records(cap_of, embs, roles, distractors, r, rng):
    """Per-entity bound records with r facts each (1 real + r-1 distractors).
    roles = (located_in, in_continent). Returns rec_city, rec_country dicts."""
    located_in, in_continent = roles
    pool = sorted({i for trip in cap_of.values() for i in trip})
    rec_city, rec_country = {}, {}
    for cap, ctry, cont in cap_of.values():
        vc = hrr_bind(located_in, embs[ctry])
        vk = hrr_bind(in_continent, embs[cont])
        for j in range(r - 1):
            vc = vc + hrr_bind(distractors[j], embs[rng.choice(pool)])
            vk = vk + hrr_bind(distractors[j], embs[rng.choice(pool)])
        rec_city[cap] = vc
        rec_country[ctry] = vk
    return rec_city, rec_country


def eval_kb(cap_of, embs, roles, rec_city, rec_country):
    """1-hop (city->country) and 2-hop (city->continent) accuracy over all cities,
    cleanup against the FULL vocab (answer must beat every word)."""
    located_in, in_continent = roles
    h1 = h2 = 0
    n = len(cap_of)
    for cap, ctry, cont in cap_of.values():
        c_pred = int(cleanup(hrr_unbind(rec_city[cap], located_in), embs)[0])
        if c_pred == ctry:
            h1 += 1
        if c_pred in rec_country:
            k_pred = int(cleanup(hrr_unbind(rec_country[c_pred], in_continent), embs)[0])
            if k_pred == cont:
                h2 += 1
    return 100.0 * h1 / n, 100.0 * h2 / n


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    if not os.path.exists(ISO_NPZ):
        print("ERROR: %s missing - run Stage 2 first." % ISO_NPZ)
        sys.exit(1)
    z = np.load(ISO_NPZ, allow_pickle=True)
    words = [str(w) for w in z['words']]
    embs = unit_rows(np.asarray(z['embs'], dtype=np.float64))
    word2idx = {w: i for i, w in enumerate(words)}
    dim = embs.shape[1]

    cap_of = s4c.build_triples(word2idx)            # country -> (cap, ctry, cont) idx
    print("Loaded SAGE memory: %d geography facts (capital->country->continent), "
          "D=%d, vocab=%d.\n" % (len(cap_of), dim, len(words)))

    rng = np.random.default_rng(SEED)
    located_in = make_unitary(rng.standard_normal(dim))
    in_continent = make_unitary(rng.standard_normal(dim))
    roles = (located_in, in_continent)
    distractors = make_unitary(rng.standard_normal((max(R_GRID), dim)))

    # ---- clean KB (1 fact/entity) for the Q&A demo ----
    rec_city, rec_country = build_records(cap_of, embs, roles, distractors, 1, rng)

    print("=== SAGE answering real questions (continent of the country of X) ===")
    by_country = {words[t[1]]: t for t in cap_of.values()}
    for cw in SHOWCASE:
        if cw not in by_country:
            continue
        cap, ctry, cont = by_country[cw]
        city, country, continent = words[cap], words[ctry], words[cont]
        c_pred = words[int(cleanup(hrr_unbind(rec_city[cap], located_in), embs)[0])]
        k_pred = words[int(cleanup(hrr_unbind(rec_country[ctry], in_continent), embs)[0])]
        ok = "OK" if (c_pred == country and k_pred == continent) else "X"
        print("  %-10s -> %-12s -> %-9s   [true %-9s] %s"
              % (city, c_pred, k_pred, continent, ok))

    # ---- binding does real work: record is opaque until unbound with the key ----
    cap, ctry, cont = next(iter(cap_of.values()))
    opaque = cos(rec_city[cap], embs[ctry])
    revealed = cos(hrr_unbind(rec_city[cap], located_in), embs[ctry])
    print("\nBinding hides structure until you ask the right question:")
    print("  cos(record, answer)              = %+.3f  (opaque - looks unrelated)"
          % opaque)
    print("  cos(unbind(record, role), answer) = %+.3f  (revealed by the role key)"
          % revealed)

    # ---- real-KB accuracy under memory load ----
    print("\n=== Real-KB accuracy under memory load (1 real fact + distractors) ===")
    print("%6s %12s %12s" % ("R", "1-hop city->", "2-hop city->"))
    print("%6s %12s %12s" % ("", "country %", "continent %"))
    rows = []
    for r in R_GRID:
        rng_r = np.random.default_rng(SEED)
        rc, rk = build_records(cap_of, embs, roles, distractors, r, rng_r)
        a1, a2 = eval_kb(cap_of, embs, roles, rc, rk)
        rows.append({"R": r, "hop1": a1, "hop2": a2})
        print("%6d %11.1f%% %11.1f%%" % (r, a1, a2))

    usable = max((x["R"] for x in rows if x["hop2"] >= 70.0), default=0)
    print("\n" + "=" * 66)
    print("SAGE answers real multi-hop geography queries correctly; holds 2-hop "
          ">=70%% up to R=%d facts/entity of memory load." % usable)
    print("=" * 66)

    out = {"experiment": "sage_query_demo", "n_facts": len(cap_of), "dim": dim,
           "vocab": len(words), "opaque_cos": opaque, "revealed_cos": revealed,
           "usable_R": usable, "load_sweep": rows}
    res_path = os.path.join(RES_DIR, 'sage_query_demo.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    xs = [x["R"] for x in rows]
    ax.plot(xs, [x["hop1"] for x in rows], 'o-', color='#4fc3f7',
            label='1-hop city->country')
    ax.plot(xs, [x["hop2"] for x in rows], 's--', color='#26c281',
            label='2-hop city->continent')
    ax.axhline(70.0, color='#888888', ls=':', lw=1, label='usable bar')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R = facts per entity (memory load)', color='white')
    ax.set_ylabel('real-query accuracy %', color='white')
    ax.set_title('SAGE real geography KB: query accuracy vs memory load',
                 color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'sage_query_demo.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    entry = (
        "\n## SAGE real query demo (binding memory on real facts)\n\n"
        "- Loaded %d real geography facts (capital->country->continent); answered "
        "multi-hop queries by unbind-chaining + cleanup over the %d-word vocab. "
        "Holds 2-hop >=70%% up to R=%d facts/entity of load.\n"
        "- opacity cos(record,answer)=%+.3f, revealed cos(unbind,answer)=%+.3f.\n"
        "- **HONEST SCOPE (per /code-review): at low load this is lossless "
        "store-and-read-back; a plain dict {city:country, country:continent} gets "
        "the same and does NOT degrade, so this does NOT show binding beating a "
        "flat store. The 2-hop is two stored lookups joined by an index, NOT "
        "derived inference; opacity is a convolution identity. Binding's "
        "distinctive value (fixed-size SUPERPOSITION of many facts in one vector, "
        "algebraic composition) is NOT baselined here.** Conclusion so far: on a "
        "simple key-value task a dict wins; binding's edge must be shown on a task "
        "dicts cannot do (superposition footprint / algebra) against a baseline.\n"
        % (len(cap_of), len(words), usable, opaque, revealed)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended SAGE query demo entry to findings.md")


if __name__ == '__main__':
    main()
