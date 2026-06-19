"""
SAGE-Sphere - Binding memory on REAL embeddings (honest, stress-tested)
=======================================================================
Does FHRR/HRR binding + geometric cleanup work on REAL semantic embeddings, or
does their correlation break it? Earlier drafts were misleading in two ways the
code-review caught:
  - unitary roles WHITEN the unbinding noise, so embedding correlation barely
    shows unless we (a) stress the load and (b) compare against a matched control;
  - R=1 single-fact recovery is algebraically EXACT, so a "1-fact gap" is ~0 by
    construction and carries no information.

This version fixes both:
  - R swept up to 256 facts/entity, well past the capacity knee.
  - THREE conditions, so correlation is isolated cleanly:
      ideal    = random Gaussian vectors (textbook VSA reference)
      shuffled = real embeddings with each FEATURE column permuted across entities
                 (destroys inter-entity correlation, KEEPS per-dim marginals/norm)
      real     = real isotropic nomic embeddings (correlated)
    real-vs-SHUFFLED = the pure cost of correlation (matched marginals).
HRR (circular convolution); roles made frequency-unitary so unbind is exact and
the only degradation is superposition cross-talk + cleanup disambiguation.

Honest read: with unitary roles, binding CAPACITY is largely filler-distribution-
independent; correlation mainly costs the CLEANUP step (snapping a noisy vector to
the right entity among correlated neighbours), and that only bites under load.
  python experiments/binding_real.py
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
from core.binding import hrr_bind, hrr_unbind, cleanup, make_unitary   # noqa: E402

ISO_NPZ  = os.path.join(ROOT, 'data', 'embeddings_isotropic.npz')
RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

N_ENT    = 500
R_GRID   = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_QUERY  = 3000
USABLE   = 70.0           # 2-hop bar to call a regime "usable"
SEED     = 42


def unit_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def column_shuffle(x, rng):
    """Permute each feature column independently across entities: destroys
    inter-entity correlation, preserves each column's marginal distribution."""
    y = x.copy()
    for j in range(x.shape[1]):
        rng.shuffle(y[:, j])
    return unit_rows(y)


def eval_mode(entities, r, rng):
    """entities (N, D) real. Unitary Gaussian roles. Returns (acc1, acc2, oracle).
    2-hop excludes immediate self-return chains (t1==start) so accuracy is not
    inflated by degenerate self-loops."""
    n, dim = entities.shape
    relations = make_unitary(rng.standard_normal((r, dim)))     # frequency-unitary
    targets = rng.integers(0, n, size=(n, r))
    rec = np.zeros((n, dim))
    for rel in range(r):
        rec += hrr_bind(relations[rel], entities[targets[:, rel]])

    e = rng.integers(0, n, N_QUERY)
    rel1 = rng.integers(0, r, N_QUERY)
    pred1 = cleanup(hrr_unbind(rec[e], relations[rel1]), entities)
    acc1 = 100.0 * np.mean(pred1 == targets[e, rel1])

    e0 = rng.integers(0, n, N_QUERY)
    ra = rng.integers(0, r, N_QUERY)
    rb = rng.integers(0, r, N_QUERY)
    t1_true = targets[e0, ra]
    keep = t1_true != e0                                        # drop self-return
    e0, ra, rb, t1_true = e0[keep], ra[keep], rb[keep], t1_true[keep]
    t2_true = targets[t1_true, rb]
    t1 = cleanup(hrr_unbind(rec[e0], relations[ra]), entities)
    t2 = cleanup(hrr_unbind(rec[t1], relations[rb]), entities)
    acc2 = 100.0 * np.mean(t2 == t2_true)
    t2_or = cleanup(hrr_unbind(rec[t1_true], relations[rb]), entities)
    acc2_oracle = 100.0 * np.mean(t2_or == t2_true)
    return acc1, acc2, acc2_oracle


def row(rows, mode, r):
    """Lookup helper (no StopIteration: returns None if absent)."""
    for x in rows:
        if x["mode"] == mode and x["R"] == r:
            return x
    return None


def capacity(rows, mode):
    usable = [x for x in rows if x["mode"] == mode and x["hop2"] >= USABLE]
    return max((x["R"] for x in usable), default=0)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    if not os.path.exists(ISO_NPZ):
        print("ERROR: %s missing - run Stage 2 first." % ISO_NPZ)
        sys.exit(1)
    z = np.load(ISO_NPZ, allow_pickle=True)
    embs = unit_rows(np.asarray(z['embs'], dtype=np.float64))
    dim = embs.shape[1]
    print("Loaded %d isotropic embeddings, D=%d. N_ent=%d, %d queries/cell.\n"
          % (embs.shape[0], dim, N_ENT, N_QUERY))

    pick_rng = np.random.default_rng(SEED)
    real_ent = embs[pick_rng.choice(embs.shape[0], N_ENT, replace=False)]
    ideal_ent = unit_rows(pick_rng.standard_normal((N_ENT, dim)))
    shuffled_ent = column_shuffle(real_ent, pick_rng)

    conds = (("ideal", ideal_ent), ("shuffled", shuffled_ent), ("real", real_ent))
    rows = []
    print("%-9s %4s %9s %9s %12s" % ("mode", "R", "1-hop%", "2-hop%", "2hop-orac%"))
    for mode, ent in conds:
        for r in R_GRID:
            rng = np.random.default_rng(SEED)     # identical KB/queries across modes
            a1, a2, a2o = eval_mode(ent, r, rng)
            rows.append({"mode": mode, "R": r, "hop1": a1, "hop2": a2,
                         "hop2_oracle": a2o})
            print("%-9s %4d %8.1f%% %8.1f%% %11.1f%%" % (mode, r, a1, a2, a2o))

    cap = {m: capacity(rows, m) for m, _ in conds}
    # correlation cost: real vs SHUFFLED (matched marginals) at the heaviest load
    # where the shuffled control is still usable -> a fair stress point.
    stress_R = max([x["R"] for x in rows
                    if x["mode"] == "shuffled" and x["hop2"] >= USABLE], default=R_GRID[0])
    sr = row(rows, "shuffled", stress_R); rr = row(rows, "real", stress_R)
    corr_cost = (sr["hop2"] - rr["hop2"]) if (sr and rr) else float('nan')

    print("\n" + "=" * 70)
    print("CAPACITY (largest R with 2-hop >= %.0f%%):  real=%d  shuffled=%d  ideal=%d"
          % (USABLE, cap["real"], cap["shuffled"], cap["ideal"]))
    print("CORRELATION COST at R=%d (shuffled %.1f%% - real %.1f%%) = %.1f pp"
          % (stress_R, sr["hop2"] if sr else float('nan'),
             rr["hop2"] if rr else float('nan'), corr_cost))
    if cap["real"] >= 8:
        print("=> Binding WORKS on real embeddings (usable up to R=%d facts/entity)."
              % cap["real"])
    elif cap["real"] >= 2:
        print("=> Binding works on real embeddings only at LOW load (R<=%d)."
              % cap["real"])
    else:
        print("=> Binding does NOT hold on real embeddings beyond a single fact.")
    print("  (unitary roles whiten noise -> correlation hits CLEANUP, not capacity;"
          " real-vs-shuffled isolates it with matched marginals.)")
    print("=" * 70)

    out = {"experiment": "binding_real_v2", "n_entities": N_ENT, "dim": dim,
           "n_query": N_QUERY, "usable_bar": USABLE,
           "capacity": cap, "stress_R": stress_R, "correlation_cost_pp": corr_cost,
           "note": "real vs SHUFFLED = correlation cost with matched marginals; "
                   "unitary roles whiten noise so correlation hits cleanup not "
                   "capacity; R=1 recovery is algebraically exact.",
           "sweep": rows}
    res_path = os.path.join(RES_DIR, 'binding_real.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    cmap = {"ideal": '#4fc3f7', "shuffled": '#ffb74d', "real": '#ff5252'}
    for mode, _ in conds:
        xs = [x["R"] for x in rows if x["mode"] == mode]
        h2 = [x["hop2"] for x in rows if x["mode"] == mode]
        ax.plot(xs, h2, 'o-', color=cmap[mode], label='%s 2-hop' % mode)
    ax.axhline(USABLE, color='#888888', ls=':', lw=1, label='usable bar (%.0f%%)' % USABLE)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('R = facts per entity (memory load)', color='white')
    ax.set_ylabel('2-hop chain accuracy %', color='white')
    ax.set_title('Binding on real embeddings: capacity vs correlation '
                 '(HRR, D=%d, N=%d)' % (dim, N_ENT), color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'binding_real.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    verdict = ("WORKS up to R=%d facts/entity" % cap["real"] if cap["real"] >= 8
               else "low-load only (R<=%d)" % cap["real"] if cap["real"] >= 2
               else "fails beyond 1 fact")
    entry = (
        "\n## Binding memory on REAL embeddings (HRR, stress-tested)\n\n"
        "- HRR circular-convolution binding, frequency-unitary roles, + cleanup. "
        "N=%d, D=%d, R up to %d. Three conditions: ideal (Gaussian), shuffled "
        "(column-permuted real = correlation removed, marginals matched), real.\n"
        "- **Capacity (largest R with 2-hop >= %.0f%%): real=%d, shuffled=%d, "
        "ideal=%d -> binding %s on real embeddings.**\n"
        "- Correlation cost (real vs SHUFFLED at R=%d, matched marginals) = %.1f pp.\n"
        "- Honest framing: unitary roles whiten the unbinding noise, so correlation "
        "costs the CLEANUP step (disambiguating among correlated neighbours), NOT "
        "binding capacity; the real-vs-shuffled control isolates exactly that. R=1 "
        "recovery is algebraically exact (uninformative).\n"
        % (N_ENT, dim, R_GRID[-1], USABLE, cap["real"], cap["shuffled"],
           cap["ideal"], verdict, stress_R, corr_cost)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended binding-real entry to findings.md")


if __name__ == '__main__':
    main()
