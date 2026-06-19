"""
KILL-SHOT B - Fixed-footprint VSA superposition vs the strong classical baseline
================================================================================
Claim under test: "the one thing a dict structurally CAN'T do - hold K key->value
facts in a FIXED byte budget that doesn't grow with K." A dict's footprint grows
with K; a superposition vector is fixed size and degrades gracefully.

IRON RULE baseline: the strongest classical FIXED-FOOTPRINT KV = HashVote (d hashed
rows of value-ids, read by majority vote = Count-Min-for-argmax). Fixed bytes
independent of K, graceful, and - like VSA - stores NO explicit key table. We report
the BEST d in {1,2,4,8} per point so the baseline is maximally strong.

FAIRNESS FIXES (post /code-review):
  - Two VSA encodings so the verdict is robust to the byte-accounting choice:
      VSA-cplx  : bundle stored complex64 (8 B/dim, D=B/8)  - honest for a general
                  (non-unit) bundle vector.
      VSA-phase : bundle reduced to PHASE only (4 B/dim, D=B/4) - the CHARITABLE
                  minimal encoding the reviewer asked for (more dims, less crosstalk).
    VSA "wins" if EITHER beats HashVote - so we cannot be accused of handicapping it.
  - PAIRED RNG: identical value assignments fed to every method per trial.
  - Per-point mean +/- std over trials; SURVIVES requires the margin to clear BOTH
    +2pp AND ~2 standard errors (no single-lucky-point overclaim).
  - HashVote reports the best d per point (strongest baseline).
CAVEAT (disclosed): VSA also needs a value CODEBOOK (V x D) for cleanup; HashVote
needs none. We measure the per-fact-set store (bundle vs table) and treat codebooks
as shared vocabulary amortized across many independent stores - legitimate ONLY
under that multi-store assumption; for a single store VSA's true footprint is larger.

Pure numpy. No Ollama, no torch.  python experiments/killshot_b_superposition.py
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.binding import random_phasors, bind, unbind, bundle, cleanup   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

V_VALUES = 256               # value vocabulary (1 byte/value-id)
BUDGETS = [2048, 8192]       # fixed byte budgets B (per-fact-set storage)
K_GRID = [5, 10, 20, 40, 80, 160, 320]
D_CHOICES = [1, 2, 4, 8]     # hash-vote rows tried; best reported (strong baseline)
N_TRIAL = 20                 # repeats per (B,K) for mean +/- std
SEED = 42


def vsa_recall(K, D, vals, rng, phase_only):
    """FHRR bundle of K key->value facts in D-dim phasors; recall@1 via unbind+
    cleanup over the value codebook. phase_only stores only the bundle phase
    (4 B/dim, charitable) vs the full complex bundle (8 B/dim)."""
    keys = random_phasors(K, D, rng)
    vcode = random_phasors(V_VALUES, D, rng)
    M = bundle(bind(keys, vcode[vals]))                  # (D,) complex superposition
    if phase_only:
        M = M / (np.abs(M) + 1e-12)                      # discard magnitude
    rec = unbind(np.tile(M, (K, 1)), keys)               # (K, D) noisy values
    pred = cleanup(rec, vcode)                           # nearest value per key
    return float(np.mean(pred == vals))


def hashvote(K, d, w, vals, rng):
    """d hashed rows x w value-id slots; last-write-wins, read by majority vote
    (random tie-break). Fixed footprint d*w bytes, stores no keys."""
    table = np.full((d, w), -1, dtype=np.int64)
    A = rng.integers(1, 2**31 - 1, d); Bc = rng.integers(0, 2**31 - 1, d)
    keys = rng.integers(0, 2**31 - 1, K)
    for r in range(d):
        table[r, (A[r] * keys + Bc[r]) % w] = vals
    hit = 0
    for i in range(K):
        votes = [table[r, (A[r] * keys[i] + Bc[r]) % w] for r in range(d)]
        votes = [v for v in votes if v >= 0]
        if votes:
            cnt = np.bincount(votes)
            top = np.flatnonzero(cnt == cnt.max())       # random tie-break
            hit += int(rng.choice(top) == vals[i])
    return hit / K


def hashvote_best(K, B, vals, rng):
    """Strongest fixed-footprint hash KV: best recall over d in D_CHOICES at bytes B."""
    return max(hashvote(K, d, max(1, B // d), vals, rng) for d in D_CHOICES)


def _stat(a):
    a = np.array(a)
    return float(a.mean()), float(a.std())


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    print("KILL-SHOT B: fixed-footprint superposition vs hash-vote (strong "
          "classical) at EQUAL bytes.")
    print("V=%d, %d trials/point (paired RNG), budgets=%s B. VSA tested at BOTH "
          "8 B/dim (complex) and 4 B/dim (phase-only, charitable).\n"
          % (V_VALUES, N_TRIAL, BUDGETS))

    results = {}
    best_sig_margin = -9.9        # best SIGNIFICANT (mean & 2-se) VSA-minus-hash
    survives = False
    for B in BUDGETS:
        Dc, Dp = B // 8, B // 4
        print("=== Budget B=%d B  (VSA-cplx D=%d | VSA-phase D=%d | HashVote best "
              "of d=%s) ===" % (B, Dc, Dp, D_CHOICES))
        print("%-6s %14s %14s %14s %10s" %
              ("K", "VSA-cplx", "VSA-phase", "HashVote", "dict(B)"))
        row = {"Dc": Dc, "Dp": Dp, "K": [], "vsa_cplx": [], "vsa_cplx_sd": [],
               "vsa_phase": [], "vsa_phase_sd": [], "hash": [], "hash_sd": [],
               "dict_bytes": []}
        for K in K_GRID:
            vc, vp, hv = [], [], []
            for t in range(N_TRIAL):
                tr = np.random.default_rng(SEED * 7919 + B * 31 + t)
                vals = tr.integers(0, V_VALUES, K)       # PAIRED across methods
                vc.append(vsa_recall(K, Dc, vals, tr, phase_only=False))
                vp.append(vsa_recall(K, Dp, vals, tr, phase_only=True))
                hv.append(hashvote_best(K, B, vals, tr))
            (mc, sc), (mp, sp), (mh, sh) = _stat(vc), _stat(vp), _stat(hv)
            dbytes = K * 5                                # dict: ~4 B key + 1 B value
            row["K"].append(K)
            row["vsa_cplx"].append(mc); row["vsa_cplx_sd"].append(sc)
            row["vsa_phase"].append(mp); row["vsa_phase_sd"].append(sp)
            row["hash"].append(mh); row["hash_sd"].append(sh)
            row["dict_bytes"].append(dbytes)
            # significance: best VSA variant beats hash by >2pp AND >2 std errors
            for mv, sv in ((mc, sc), (mp, sp)):
                se = (sv**2 + sh**2) ** 0.5 / N_TRIAL**0.5
                if mv - mh > 0.02 and mv - mh > 2 * se:
                    survives = True
                    best_sig_margin = max(best_sig_margin, mv - mh)
            print("%-6d %7.1f+/-%4.1f %7.1f+/-%4.1f %7.1f+/-%4.1f %9d%s"
                  % (K, 100 * mc, 100 * sc, 100 * mp, 100 * sp, 100 * mh, 100 * sh,
                     dbytes, "  >B" if dbytes > B else ""))
        results["B%d" % B] = row
        print("")

    # best raw margin (any-point, any VSA encoding, across all budgets) for reporting
    best_raw = max(
        max((np.array(r["vsa_phase"]) - np.array(r["hash"])).max(),
            (np.array(r["vsa_cplx"]) - np.array(r["hash"])).max())
        for r in results.values())
    print("=" * 72)
    if survives:
        print("VERDICT: SURVIVES - a VSA encoding beat HashVote by >2pp AND >2 std "
              "errors at some (B,K) (best significant margin %+.1f pp). Worth the "
              "full build + a dedicated /code-review of the byte accounting."
              % (100 * best_sig_margin))
    else:
        print("VERDICT: FALSIFIED - HashVote (strong classical fixed-footprint KV, "
              "best-of-d) matches/beats BOTH VSA encodings at every equal-byte point "
              "(best raw VSA-minus-hash = %+.1f pp, not significant). The 'fixed-"
              "footprint superposition beats a dict' claim dies to a hash table; "
              "superposition is dominated (cf. Kleyko/Frady/Sommer 2022)."
              % (100 * best_raw))
    print("Note: dict is exact (recall 1.0) but its footprint GROWS with K past B; "
          "that is the only thing the fixed-footprint methods buy.")
    print("=" * 72)

    out = {"experiment": "killshot_b_superposition", "v_values": V_VALUES,
           "budgets": BUDGETS, "k_grid": K_GRID, "hash_d_choices": D_CHOICES,
           "n_trial": N_TRIAL, "survives": bool(survives),
           "best_significant_margin_pp": float(100 * best_sig_margin),
           "best_raw_margin_pp": float(100 * best_raw), "results": results}
    res_path = os.path.join(RES_DIR, 'killshot_b_superposition.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, len(BUDGETS), figsize=(11, 4.5), facecolor='#0a0a0a')
    if len(BUDGETS) == 1:
        axs = [axs]
    for ax, B in zip(axs, BUDGETS):
        r = results["B%d" % B]
        ax.set_facecolor('#111111')
        ax.errorbar(r["K"], [100 * x for x in r["vsa_cplx"]],
                    yerr=[100 * x for x in r["vsa_cplx_sd"]], fmt='o-',
                    color='#4fc3f7', label='VSA-cplx (8B/dim)', capsize=2)
        ax.errorbar(r["K"], [100 * x for x in r["vsa_phase"]],
                    yerr=[100 * x for x in r["vsa_phase_sd"]], fmt='d-',
                    color='#26c6da', label='VSA-phase (4B/dim)', capsize=2)
        ax.errorbar(r["K"], [100 * x for x in r["hash"]],
                    yerr=[100 * x for x in r["hash_sd"]], fmt='s-',
                    color='#ff7043', label='HashVote (best d)', capsize=2)
        ax.axhline(100.0 / V_VALUES, color='#555555', ls=':', label='chance')
        ax.set_xscale('log'); ax.set_xlabel('K facts stored', color='white')
        ax.set_ylabel('recall@1 %', color='white')
        ax.set_title('B=%d bytes' % B, color='white')
        ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=8)
    fig.suptitle('Kill-shot B: fixed-footprint superposition vs hash-vote at equal '
                 'bytes', color='white')
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'killshot_b_superposition.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    verdict = "SURVIVES" if survives else "FALSIFIED"
    entry = (
        "\n## Kill-shot B - fixed-footprint VSA superposition vs hash-vote "
        "(equal bytes)\n\n"
        "- Task: store K key->value-id facts (V=%d), retrieve by key, at a FIXED "
        "byte budget. VSA tested at BOTH 8 B/dim (complex bundle) and 4 B/dim "
        "(phase-only, charitable) vs HashVote (best d in %s hashed rows, majority "
        "vote) - the strong classical fixed-footprint KV (stores no keys). Paired "
        "RNG, %d trials, mean+/-std; SURVIVES needs >2pp AND >2 s.e.\n"
        "- Budgets %s B; K swept %s.\n"
        "- **VERDICT: %s.** Best significant VSA-minus-HashVote margin = %+.1f pp; "
        "best raw margin = %+.1f pp. %s\n"
        % (V_VALUES, D_CHOICES, N_TRIAL, BUDGETS, K_GRID, verdict,
           100 * best_sig_margin, 100 * best_raw,
           ("A VSA encoding significantly crossed above the classical baseline "
            "-> worth a full build." if survives else "HashVote (fixed bytes, no "
            "key store, best-of-d) matches/beats BOTH VSA encodings everywhere; the "
            "'a dict can't hold K facts in fixed bytes' claim dies to a hash table. "
            "Superposition is dominated, consistent with Kleyko/Frady/Sommer 2022.")))
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended kill-shot B entry to findings.md")


if __name__ == '__main__':
    main()
