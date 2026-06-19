"""
SAGE-Sphere - Stage 1 - Plain Fibonacci-sphere substrate
=========================================================
Goal: swap the cube grid for a sphere lattice and PROVE retrieval does not
regress. Substrate swap ONLY - no new capabilities.

What this run shows:
  1. RETRIEVAL (gate): exact cosine over the stored 768-D payloads. This is
     substrate-INDEPENDENT, so sphere == cube by construction -> no regression.
     Measured as top-1 self-retrieval (storage integrity) on 3000 words.
  2. POSITION SEMANTICS (the actual win): on the sphere, 3-D geodesic distance
     tracks 768-D cosine; on the cube it does not. Spearman correlation,
     head-to-head, same PCA-3D placement for both.
  3. COLLISIONS: how many words share a lattice point (controllable by n_points).
  4. POSITIONAL retrieval recall@10 vs exact - honest measure of how lossy the
     coarse 3-D routing is (top-3 PCA holds only ~8% variance; expected modest).

Loads cached embeddings from Stage 0 (data/embeddings_cache.npz). No Ollama.

Run:  python experiments/stage1_substrate.py
"""

import os
import sys
import json

import numpy as np
from scipy.stats import spearmanr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.sphere_substrate import SphereSubstrate, fibonacci_sphere  # noqa: E402

DATA_NPZ = os.path.join(ROOT, 'data', 'embeddings_cache.npz')
RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

N_POINTS  = 4096          # sphere lattice points (cube parity: 16^3 = 4096)
N_PAIRS   = 20000         # pairs for the correlation test
N_PROBE   = 500           # words probed for positional recall
SEED      = 42


def _fmt(r):
    """Format a correlation that may be None (NaN from all-tied ranks)."""
    return "nan" if r is None else "%+.3f" % r


def load_embeddings():
    if not os.path.exists(DATA_NPZ):
        print("ERROR: %s missing - run Stage 0 first." % DATA_NPZ)
        sys.exit(1)
    z = np.load(DATA_NPZ, allow_pickle=True)
    words = list(z['words'])
    embs = torch.tensor(np.asarray(z['embs'], dtype=np.float32))
    embs = F.normalize(embs, p=2, dim=1)
    print("Loaded %d words x %d dims from cache." % (embs.shape[0], embs.shape[1]))
    return words, embs


def cube_positions(n_points, device):
    """Cube baseline grid: meshgrid in [-1,1]^3 with ~n_points cells."""
    side = round(n_points ** (1.0 / 3.0))
    c = torch.linspace(-1, 1, side, device=device)
    xx, yy, zz = torch.meshgrid(c, c, c, indexing='ij')
    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1), side


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RES_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s\n" % device)

    words, embs = load_embeddings()
    embs = embs.to(device)
    n = embs.shape[0]

    # ---- build sphere substrate + place ----
    sub = SphereSubstrate(n_points=N_POINTS, payload_dim=embs.shape[1],
                          seed=SEED, device=device)
    sub.place(embs, words)
    coll = sub.collision_stats()
    print("\nCollisions: %d/%d points used, %d collided, max bucket %d, "
          "%.1f%% of items in a collision."
          % (coll['points_used'], coll['n_points'], coll['points_collided'],
             coll['max_bucket'], coll['pct_items_in_collision']))

    # ---- (1) RETRIEVAL: exact top-1 self-retrieval (storage integrity) ----
    # Exact 768-D cosine never touches grid positions, so it is identical on
    # cube or sphere BY CONSTRUCTION - this measures that placement preserved
    # the payloads, NOT a fabricated cube-vs-sphere horse race.
    sims = embs @ sub.payloads.T                       # (n, n), one matmul
    top1 = sims.argmax(dim=1)
    correct = (top1 == torch.arange(n, device=device)).sum().item()
    self_acc = 100.0 * correct / n
    # exact cosine to self is 1.0; a miss only happens if two words share an
    # identical vector (argmax tie) - report that separately so a <100% number
    # is not misread as a storage failure.
    n_dup = int((sims.max(dim=1).values >= 0.99999).sum().item() - correct)
    print("\nRETRIEVAL (storage integrity): top-1 self-retrieval = %.2f%%"
          % self_acc)
    print("  exact 768-D cosine is position-independent -> no regression vs "
          "cube by construction.")
    if n_dup > 0:
        print("  (%d duplicate-embedding argmax ties, not storage failures)"
              % n_dup)

    # neighbor examples (exclude self)
    examples = {}
    for w in ['paris', 'king', 'water', 'germany', 'happy']:
        if w in words:
            qi = words.index(w)
            sc, idx = sub.query_exact(embs[qi], k=4)
            nbrs = [words[j] for j in idx.tolist() if j != qi][:3]
            examples[w] = nbrs
    print("Neighbor examples (exact cosine):")
    for w, nb in examples.items():
        print("  %-9s -> %s" % (w, ', '.join(nb)))

    # ---- (2) POSITION SEMANTICS: does 3-D position track 768-D cosine? ----
    cube_pos_grid, side = cube_positions(N_POINTS, device)
    proj_raw = sub.project_raw(embs)                        # (n,3) PCA, computed ONCE
    proj_unit = F.normalize(proj_raw, p=2, dim=1)           # continuous, on S^2
    # cube baseline: SAME raw PCA coords into [-1,1]^3 then snap to the grid
    rng = proj_raw.abs().amax(dim=0, keepdim=True) + 1e-8
    proj_cube = (proj_raw / rng).clamp(-1, 1)
    cube_pt = torch.argmin(torch.cdist(proj_cube, cube_pos_grid), dim=1)
    sphere_pt = sub.item_point

    ii = np.random.randint(0, n, N_PAIRS)
    jj = np.random.randint(0, n, N_PAIRS)
    keep = ii != jj
    it = torch.tensor(ii[keep], device=device)
    jt = torch.tensor(jj[keep], device=device)

    cos768 = (embs[it] * embs[jt]).sum(1).cpu().numpy()
    # PRIMARY (clean): cosine of the continuous PCA-3D direction, pre-snap.
    # Isolates how much 768-D angular structure survives 768->3 with NO lattice
    # or metric confound.
    cont_sim = (proj_unit[it] * proj_unit[jt]).sum(1).cpu().numpy()
    # SNAPPED references (confounded - different metric AND lattice resolution,
    # so reported as references, not a controlled "sphere beats cube" claim):
    sphere_sim = (sub.positions[sphere_pt[it]] * sub.positions[sphere_pt[jt]]).sum(1).cpu().numpy()
    cube_sim = (-(cube_pos_grid[cube_pt[it]] - cube_pos_grid[cube_pt[jt]]).norm(dim=1)).cpu().numpy()

    def safe_r(a, b):
        r = spearmanr(a, b).correlation
        return float(r) if r == r else None            # None if NaN (all ties)

    r_cont = safe_r(cos768, cont_sim)
    r_sphere = safe_r(cos768, sphere_sim)
    r_cube = safe_r(cos768, cube_sim)
    print("\nPOSITION SEMANTICS (Spearman vs true 768-D cosine):")
    print("  PRIMARY continuous 3-D direction : r = %s" % _fmt(r_cont))
    print("  snapped sphere geodesic (ref)    : r = %s" % _fmt(r_sphere))
    print("  snapped cube euclidean  (ref)    : r = %s" % _fmt(r_cube))
    print("  NOTE: snapped sphere/cube differ in metric AND resolution -")
    print("        confounded; the continuous number is the clean result.")

    # ---- (3) POSITIONAL retrieval recall@10 vs exact (sphere) ----
    probe = np.random.choice(n, min(N_PROBE, n), replace=False)
    recalls = []
    for qi in probe:
        qi = int(qi)
        _, ex = sub.query_exact(embs[qi], k=11)            # 11 -> ~10 after self drop
        _, po = sub.query_positional(embs[qi], k=11, n_cells=32)
        ex_set = set(ex.tolist()) - {qi}                   # true neighbours only
        po_set = set(po.tolist()) - {qi}
        if ex_set:
            recalls.append(len(ex_set & po_set) / len(ex_set))
    pos_recall = 100.0 * float(np.mean(recalls)) if recalls else 0.0
    print("\nPOSITIONAL recall@10 vs exact neighbours (self excluded, 32 cells): "
          "%.1f%% (coarse 3-D routing is lossy by design)." % pos_recall)

    # ---- GATE ----
    # Brief's gate is "retrieval must not regress vs cube". Exact retrieval is
    # position-independent, so non-regression holds by construction; the
    # falsifiable check is that placement preserved the payloads (full-accuracy
    # self-retrieval). A corrupted placement drops this below 99%.
    gate_pass = self_acc >= 99.0
    print("\n" + "=" * 60)
    print("STAGE 1 GATE: storage integrity (self-retrieval) >= 99 percent")
    print("  self-retrieval = %.2f%% -> %s"
          % (self_acc, "PASS" if gate_pass else "FAIL"))
    print("  (no-regression vs cube holds by construction: exact cosine")
    print("   retrieval never uses grid positions.)")
    print("=" * 60)

    # ---- save JSON ----
    out = {
        "stage": 1, "n_words": n, "n_points": N_POINTS, "cube_side": side,
        "retrieval_top1_self_pct": self_acc,
        "duplicate_embed_ties": n_dup,
        "gate_pass": bool(gate_pass),
        "gate": "storage integrity (self-retrieval >= 99%); no-regression vs "
                "cube holds by construction (exact cosine ignores positions)",
        "spearman_continuous_3d": r_cont,
        "spearman_sphere_geodesic_snapped": r_sphere,
        "spearman_cube_euclidean_snapped": r_cube,
        "spearman_note": "snapped sphere/cube confounded by metric + lattice "
                         "resolution; continuous is the clean measure",
        "positional_recall_at10_pct": pos_recall,
        "collisions": coll,
        "neighbor_examples": examples,
    }
    res_path = os.path.join(RES_DIR, 'stage1_substrate.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # ---- plots (dark theme) ----
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 5), facecolor='#0a0a0a')

    ax1 = fig.add_subplot(1, 3, 1, projection='3d', facecolor='#111111')
    P = sub.positions[sphere_pt].cpu().numpy()
    col = proj_unit[:, 2].cpu().numpy()
    ax1.scatter(P[:, 0], P[:, 1], P[:, 2], c=col, cmap='cool', s=4, alpha=0.6)
    ax1.set_title('%d words placed on S^2' % n, color='white')
    ax1.set_xticklabels([]); ax1.set_yticklabels([]); ax1.set_zticklabels([])

    ax2 = fig.add_subplot(1, 3, 2, facecolor='#111111')
    samp = np.random.choice(len(cos768), min(4000, len(cos768)), replace=False)
    ax2.scatter(cos768[samp], cont_sim[samp], s=5, alpha=0.3, color='#4fc3f7')
    ax2.set_title('continuous 3-D direction vs true 768-D cosine\n(Spearman r=%s)'
                  % _fmt(r_cont), color='white')
    ax2.set_xlabel('true 768-D cosine', color='white')
    ax2.set_ylabel('PCA-3D direction cosine', color='white')

    ax3 = fig.add_subplot(1, 3, 3, facecolor='#111111')
    labels = ['continuous\n3-D', 'sphere\nsnap', 'cube\nsnap']
    vals = [r_cont, r_sphere, r_cube]
    pv = [v if v is not None else 0.0 for v in vals]
    bars = ax3.bar(labels, pv, color=['#4fc3f7', '#26c281', '#ff7043'])
    ax3.axhline(0, color='#888888', lw=0.8)
    ax3.set_title('position-vs-semantics correlation', color='white')
    ax3.set_ylabel('Spearman r vs 768-D cosine', color='white')
    for b, v in zip(bars, vals):
        yv = v if v is not None else 0.0
        ax3.text(b.get_x() + b.get_width() / 2, yv, _fmt(v),
                 ha='center', va='bottom' if yv >= 0 else 'top', color='white')

    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage1_substrate.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    # ---- findings entry ----
    entry = (
        "\n## Stage 1 - Fibonacci-sphere substrate\n\n"
        "- %d words placed on %d-point S^2 lattice (cube reference %d^3).\n"
        "- **GATE (storage integrity): top-1 self-retrieval = %.2f%% -> %s.** "
        "Exact 768-D cosine is position-independent, so no-regression vs the cube "
        "holds BY CONSTRUCTION (retrieval never uses grid coordinates); the "
        "falsifiable check is that placement preserved the payloads.\n"
        "- Position semantics (Spearman vs 768-D cosine): continuous PCA-3D "
        "direction r=%s (clean, PRIMARY); snapped sphere r=%s and snapped cube "
        "r=%s are confounded references (different metric + lattice resolution), "
        "NOT a controlled 'sphere beats cube' claim. The continuous number is how "
        "much 768-D angular structure survives the 768->3 projection.\n"
        "- Collisions: %d/%d points used, %d collided, max bucket %d, "
        "%.1f%% of items collided.\n"
        "- Positional recall@10 vs exact neighbours (self excluded): %.1f%% - "
        "coarse 3-D routing is lossy; retrieval uses the full 768-D payload, NOT "
        "the 3-D position.\n"
        "- DESIGN NOTE: grid is 3-D (visualizable), payload is 768-D (exact "
        "retrieval). place() snaps a gradient-free PCA 768->3 projection onto the "
        "nearest lattice point. FLAG for Stage 3: build the kNN graph in FULL "
        "768-D cosine, using 3-D positions only for viz/partition, since top-3 "
        "PCA captures only ~8%% of variance.\n"
        % (n, N_POINTS, side, self_acc, "PASS" if gate_pass else "FAIL",
           _fmt(r_cont), _fmt(r_sphere), _fmt(r_cube),
           coll['points_used'], coll['n_points'], coll['points_collided'],
           coll['max_bucket'], coll['pct_items_in_collision'], pos_recall)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 1 entry to findings.md")
    print("\nNext: Stage 2 (isotropy, REQUIRED per Stage 0 HIGH verdict).")


if __name__ == '__main__':
    main()
