"""
sage_sequence_cube_v5.py
========================
Author: Ivelin Likov

Fifth iteration. Root cause of v4 regression identified and fixed.

ROOT CAUSE OF V4 REGRESSION — Embedding-as-pointer destroys retrieval index
    Training pulls s_emb toward o_emb (60 epochs, alpha=0.15).
    After training, s_emb no longer resembles the original query vector.
    So _nearest_subject(query_vec) cannot find the correct s_idx —
    its embedding has been modified away from the query.
    The pointer mechanism destroys the retrieval index it depends on.

    Evidence: clustering ratio inverted to 0.95x (v3=1.16x, v4=0.95x).
    Same-sequence steps are now FARTHER apart than different sequences
    because training has scrambled the embedding organisation.

THE FIX — Separate retrieval index from association

    Two responsibilities, two mechanisms:

    RETRIEVAL INDEX (embeddings):
        Train s_emb to stay close to the query vector  → s_idx is findable
        Train o_emb to stay close to the next vector   → o_idx is findable
        Embeddings remain meaningful for cosine search

    ASSOCIATION (explicit dictionary):
        s_idx → o_idx stored directly
        Collision-free (v4's unique reservation preserved)
        This is the pointer — dictionary lookup, not embedding drift

    Query: find nearest s_idx (retrieval index) → look up dict → return o_idx
    Training: reinforce, not corrupt.

RESULT:
    The subject embedding points toward the query (findable)
    The object embedding points toward the next state (interpretable)
    The s→o mapping is stored explicitly (reliable)
    All three are consistent — no conflict.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys, os, json
from datetime import datetime

for path in ['.', os.path.dirname(__file__),
             os.path.join(os.path.dirname(__file__), '..'),
             '/mnt/project']:
    if os.path.exists(os.path.join(path, 'cube_core_torch.py')):
        sys.path.insert(0, path)
        break

from cube_core_torch import SpatialCubeTorch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SAGESequenceCubeV5:
    """
    Dedicated sequence/transition memory cube — clean two-mechanism design.

    RETRIEVAL: embeddings anchored to their input vectors (cosine search works)
    ASSOCIATION: explicit s_idx→o_idx dictionary (collision-free, reliable)

    store_transition(A, B):
        s_idx = nearest unassigned subject-half point to A
        o_idx = nearest unassigned object-half point to B
        Train s_emb toward A  (reinforce retrieval match)
        Train o_emb toward B  (reinforce retrieval match)
        Store dict[s_idx] = o_idx

    query_next(query):
        s_idx = nearest subject-half point to query  (retrieval index intact)
        o_idx = dict[s_idx]                          (explicit association)
        return label[o_idx], embedding[o_idx]

    Core SAGE thesis preserved: geometry computes, not weights.
    Subject/object spatial split preserved: SAGEDivided unchanged.
    MultiCube specialist: episodic cubes untouched.
    """

    def __init__(self, cube_size=32, embed_dim=64,
                 train_epochs=80, alpha=0.2, device=None):
        self.cube_size    = cube_size
        self.embed_dim    = embed_dim
        self.train_epochs = train_epochs
        self.alpha        = alpha

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.cube = SpatialCubeTorch(cube_size, embed_dim, device=str(self.device))

        # Half masks (positions never move)
        self._sub_mask = self.cube.positions[:, 0] < 0
        self._obj_mask = self.cube.positions[:, 0] >= 0
        self._sub_idx  = torch.where(self._sub_mask)[0]
        self._obj_idx  = torch.where(self._obj_mask)[0]

        # Collision-free reservation (from v4, still needed)
        self._reserved_sub = set()
        self._reserved_obj = set()

        # THE POINTER: explicit association dictionary
        # s_idx → o_idx  (each s_idx maps to exactly one o_idx)
        self.transition_dict = {}

        # Labels and ground truth
        self.labels       = {}
        self.ground_truth = []
        self.stored_pairs = []

        print(f"\n{'='*60}")
        print(f"  SAGESequenceCubeV5 — separated retrieval + association")
        print(f"  Cube:         {cube_size}³ = {cube_size**3:,} points")
        print(f"  Subject half: {len(self._sub_idx):,} points  (x < 0)")
        print(f"  Object  half: {len(self._obj_idx):,} points  (x ≥ 0)")
        print(f"  Epochs/trans: {train_epochs}   alpha: {alpha}")
        print(f"  Device:       {self.device}")
        print(f"{'='*60}\n")

    # ── Half-aware nearest neighbour ─────────────────────────────────────────

    def _nearest_in(self, vec: torch.Tensor,
                    indices: torch.Tensor,
                    exclude: set = None) -> tuple[int, float]:
        vec  = F.normalize(vec.to(self.device), p=2, dim=0)
        sims = self.cube.embeddings[indices] @ vec

        if exclude:
            for reserved in exclude:
                local = (indices == reserved).nonzero(as_tuple=True)[0]
                if len(local) > 0:
                    sims[local[0]] = -2.0

        best = torch.argmax(sims)
        return indices[best].item(), float(sims[best].item())

    def _nearest_subject(self, vec, exclude=None):
        return self._nearest_in(vec, self._sub_idx, exclude)

    def _nearest_object(self, vec, exclude=None):
        return self._nearest_in(vec, self._obj_idx, exclude)

    def _topk_object(self, vec: torch.Tensor, k: int = 5) -> list:
        vec  = F.normalize(vec.to(self.device), p=2, dim=0)
        sims = self.cube.embeddings[self._obj_idx] @ vec
        k    = min(k, len(self._obj_idx))
        vals, locs = torch.topk(sims, k)
        return [(self._obj_idx[locs[i]].item(), float(vals[i].item()))
                for i in range(k)]

    # ── Store a transition ────────────────────────────────────────────────────

    def store_transition(self, current_vec: torch.Tensor,
                         next_vec: torch.Tensor,
                         label_current: str = "",
                         label_next:    str = "") -> tuple[int, int]:
        cv = F.normalize(current_vec.to(self.device), p=2, dim=0)
        nv = F.normalize(next_vec.to(self.device),    p=2, dim=0)

        # Step 1: find unassigned points in each half
        s_idx, s_sim_before = self._nearest_subject(cv, exclude=self._reserved_sub)
        o_idx, o_sim_before = self._nearest_object(nv,  exclude=self._reserved_obj)

        # Step 2: reserve
        self._reserved_sub.add(s_idx)
        self._reserved_obj.add(o_idx)

        # Step 3: store explicit association — THE POINTER
        self.transition_dict[s_idx] = o_idx

        # Step 4: train embeddings to stay close to their INPUT vectors
        # Subject: reinforce s_emb toward current_vec  (not toward o_emb)
        # Object:  reinforce o_emb toward next_vec      (not toward s_emb)
        # This keeps the retrieval index intact.
        for _ in range(self.train_epochs):
            # Subject embedding → current state
            s_emb = self.cube.embeddings[s_idx].detach()
            self.cube.embeddings[s_idx] += self.alpha * (cv - s_emb)
            self.cube.embeddings[s_idx]  = F.normalize(
                self.cube.embeddings[s_idx], p=2, dim=0)

            # Object embedding → next state
            o_emb = self.cube.embeddings[o_idx].detach()
            self.cube.embeddings[o_idx] += self.alpha * (nv - o_emb)
            self.cube.embeddings[o_idx]  = F.normalize(
                self.cube.embeddings[o_idx], p=2, dim=0)

        # Measure alignment after training
        s_sim_after = float((self.cube.embeddings[s_idx] @ cv).item())
        o_sim_after = float((self.cube.embeddings[o_idx] @ nv).item())

        # Labels
        if label_current:
            self.labels[s_idx]      = f"[S]{label_current}"
            self.cube.labels[s_idx] = f"[S]{label_current}"
        if label_next:
            self.labels[o_idx]      = f"[O]{label_next}"
            self.cube.labels[o_idx] = f"[O]{label_next}"

        self.stored_pairs.append({
            's_idx':        s_idx,
            'o_idx':        o_idx,
            'label_cur':    label_current,
            'label_nxt':    label_next,
            's_sim_before': s_sim_before,
            's_sim_after':  s_sim_after,
            'o_sim_after':  o_sim_after,
        })
        return s_idx, o_idx

    def store_sequence(self, vecs: list, labels: list = None) -> list:
        labels = labels or [f"step_{i}" for i in range(len(vecs))]
        pairs  = []
        for i in range(len(vecs) - 1):
            s_idx, o_idx = self.store_transition(
                vecs[i], vecs[i+1],
                label_current=labels[i],
                label_next=labels[i+1])
            self.ground_truth.append({
                'query_vec':   vecs[i],
                'correct_lbl': labels[i+1],
                'query_lbl':   labels[i],
            })
            pairs.append((s_idx, o_idx, labels[i], labels[i+1]))
        return pairs

    # ── Query ─────────────────────────────────────────────────────────────────

    def query_next(self, query_vec: torch.Tensor,
                   top_k: int = 5) -> dict:
        """
        Two-step geometric lookup.
        Step 1: nearest subject point  (retrieval index — embedding anchored to query)
        Step 2: dictionary lookup      (explicit association — no embedding drift)
        """
        qv = F.normalize(query_vec.to(self.device), p=2, dim=0)

        # Step 1: retrieval — find nearest subject point
        s_idx, s_sim = self._nearest_subject(qv)

        # Step 2: association — explicit dictionary lookup
        if s_idx not in self.transition_dict:
            return {'found': False, 'subject_idx': s_idx, 'subject_sim': s_sim}

        o_idx   = self.transition_dict[s_idx]
        o_emb   = self.cube.embeddings[o_idx]
        o_label = self.labels.get(o_idx, f"point_{o_idx}")
        o_pos   = self.cube.positions[o_idx].cpu().numpy()
        o_sim   = float((o_emb @ qv).item())

        # Top-k object points (for top-5 accuracy metric)
        o_topk  = self._topk_object(o_emb, k=top_k)

        return {
            'found':        True,
            'subject_idx':  s_idx,
            'subject_sim':  float(s_sim),
            'object_idx':   o_idx,
            'object_label': o_label,
            'object_sim':   o_sim,
            'object_pos':   o_pos,
            'top_k': [
                {'idx':   idx,
                 'label': self.labels.get(idx, f"point_{idx}"),
                 'sim':   float(sim)}
                for idx, sim in o_topk
            ],
        }

    def rollout(self, start_vec: torch.Tensor, n_steps: int = 4) -> list:
        trajectory = []
        current    = start_vec.clone()
        for step in range(n_steps):
            result = self.query_next(current, top_k=1)
            if not result['found']:
                trajectory.append({'step': step, 'label': '?', 'found': False})
                break
            trajectory.append({
                'step':        step,
                'label':       result['object_label'],
                'object_idx':  result['object_idx'],
                'subject_sim': result['subject_sim'],
                'found':       True,
            })
            # Feed object embedding as next query
            current = self.cube.embeddings[result['object_idx']].detach()
        return trajectory


# ─── Sequence factory ─────────────────────────────────────────────────────────

def make_sequences(dim=64, n_seq=3, steps=5, seed=42):
    """
    Well-separated sequences. Orthogonal bases ensure large inter-sequence
    distances. Small per-step noise ensures intra-sequence coherence.
    """
    torch.manual_seed(seed)
    all_names = [
        ['takeoff',  'climb',    'cruise',   'descend',  'land'],
        ['patrol_a', 'patrol_b', 'patrol_c', 'patrol_d', 'patrol_e'],
        ['evade_1',  'evade_2',  'evade_3',  'recover',  'resume'],
    ]
    sequences, name_list = [], []
    for seq_id in range(n_seq):
        # Strictly orthogonal one-hot base + tiny noise
        base       = torch.zeros(dim)
        base[seq_id * (dim // n_seq)] = 1.0
        base       = F.normalize(base + torch.randn(dim) * 0.02, p=2, dim=0)
        vecs       = []
        for step in range(steps):
            # Small drift per step — stays close to base, distinct per step
            drift = base + torch.randn(dim) * 0.08
            vecs.append(F.normalize(drift, p=2, dim=0))
        sequences.append(vecs)
        name_list.append(all_names[seq_id])
    return sequences, name_list


# ─── Experiments ──────────────────────────────────────────────────────────────

def run_exp1(cube, sequences, names):
    print("\n" + "="*60)
    print("EXPERIMENT 1 — Transition Retrieval Accuracy")
    print("="*60)

    rows, hits1, hits5, total = [], 0, 0, 0

    for seq_id, (seq, lbls) in enumerate(zip(sequences, names)):
        for step in range(len(seq) - 1):
            result      = cube.query_next(seq[step], top_k=5)
            correct_lbl = lbls[step + 1]

            if not result['found']:
                rows.append(("✗", f"seq{seq_id}", lbls[step],
                             correct_lbl, "NOT_FOUND", "—", "—"))
                total += 1
                continue

            retrieved  = result['object_label']
            top5       = [t['label'] for t in result['top_k']]
            s_sim      = result['subject_sim']
            o_sim      = result['object_sim']

            hit1 = correct_lbl.lower() in retrieved.lower()
            hit5 = any(correct_lbl.lower() in l.lower() for l in top5)
            if hit1: hits1 += 1
            if hit5: hits5 += 1
            total += 1

            rows.append(("✓" if hit1 else ("~" if hit5 else "✗"),
                         f"seq{seq_id}", lbls[step], correct_lbl,
                         retrieved[:22], f"{s_sim:.3f}", f"{o_sim:.3f}"))

    print(f"\n  {'':3} {'Seq':5} {'Query':12} {'Expected':12} "
          f"{'Retrieved':24} {'SubSim':6} {'ObjSim':6}")
    print(f"  {'-'*75}")
    for r in rows:
        print(f"  {r[0]:3} {r[1]:5} {r[2]:12} {r[3]:12} "
              f"{r[4]:24} {r[5]:6} {r[6]:6}")

    acc1 = hits1 / total
    acc5 = hits5 / total
    print(f"\n  Top-1 accuracy: {hits1}/{total} = {acc1:.1%}")
    print(f"  Top-5 accuracy: {hits5}/{total} = {acc5:.1%}")
    print(f"  Random baseline:          {1/cube.cube.n_points:.5%}")
    return {'top1': acc1, 'top5': acc5,
            'hits1': hits1, 'hits5': hits5, 'total': total}


def run_exp2(cube, sequences, names):
    print("\n" + "="*60)
    print("EXPERIMENT 2 — Multi-step Rollout (pure geometry)")
    print("="*60)
    correct, total = 0, 0
    for seq_id, (seq, lbls) in enumerate(zip(sequences, names)):
        traj = cube.rollout(seq[0], n_steps=len(seq) - 1)
        print(f"\n  Seq {seq_id}: [{lbls[0]}]")
        print(f"  Expected: {' → '.join(lbls[1:])}")
        print(f"  Rollout:  ", end="")
        steps = []
        for t in traj:
            gt    = lbls[t['step'] + 1] if t['step'] + 1 < len(lbls) else "?"
            match = t['found'] and gt.lower() in t['label'].lower()
            if match: correct += 1
            total += 1
            steps.append(("✓" if match else "✗") + f"[{t['label']}]")
        print(" → ".join(steps))
    acc = correct / total if total > 0 else 0.0
    print(f"\n  Rollout accuracy: {correct}/{total} = {acc:.1%}")
    return {'correct': correct, 'total': total, 'accuracy': acc}


def run_exp3(cube, sequences):
    print("\n" + "="*60)
    print("EXPERIMENT 3 — Spatial Clustering")
    print("="*60)
    seq_pos = []
    for seq in sequences:
        positions = []
        for vec in seq[:-1]:
            s_idx, _ = cube._nearest_subject(vec)
            positions.append(cube.cube.positions[s_idx].cpu().numpy())
        seq_pos.append(np.array(positions))

    intra, inter = [], []
    for pos in seq_pos:
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                intra.append(np.linalg.norm(pos[i] - pos[j]))
    for s1 in range(len(seq_pos)):
        for s2 in range(s1+1, len(seq_pos)):
            for p1 in seq_pos[s1]:
                for p2 in seq_pos[s2]:
                    inter.append(np.linalg.norm(p1 - p2))

    intra_m = float(np.mean(intra)) if intra else 0.0
    inter_m = float(np.mean(inter)) if inter else 0.0
    ratio   = inter_m / intra_m if intra_m > 0 else 0.0
    verdict = ("✓ STRONG"   if ratio > 1.5 else
               "~ MODERATE" if ratio > 1.1 else "✗ WEAK")
    print(f"\n  Intra-sequence mean dist: {intra_m:.4f}")
    print(f"  Inter-sequence mean dist: {inter_m:.4f}")
    print(f"  Separation ratio:         {ratio:.2f}x  {verdict}")
    return {'intra': intra_m, 'inter': inter_m,
            'ratio': ratio, 'verdict': verdict}


def run_exp4(cube, sequences):
    print("\n" + "="*60)
    print("EXPERIMENT 4 — Anti-Forgetting After 300 Noise Transitions")
    print("="*60)

    def acc():
        h, t = 0, 0
        for gt in cube.ground_truth:
            r = cube.query_next(gt['query_vec'], top_k=5)
            if r['found'] and any(gt['correct_lbl'].lower() in e['label'].lower()
                                  for e in r['top_k']):
                h += 1
            t += 1
        return h / t if t > 0 else 0.0

    before = acc()
    torch.manual_seed(77)
    dim = cube.embed_dim
    for _ in range(300):
        a = F.normalize(torch.randn(dim), p=2, dim=0)
        b = F.normalize(torch.randn(dim), p=2, dim=0)
        cube.store_transition(a, b)
    after      = acc()
    forgetting = before - after
    verdict    = ("✓ EXCELLENT"  if forgetting < 0.05 else
                  "~ ACCEPTABLE" if forgetting < 0.20 else "✗ HIGH")

    print(f"\n  Top-5 before noise: {before:.1%}")
    print(f"  Top-5 after  noise: {after:.1%}")
    print(f"  Forgetting:         {forgetting:.1%}  {verdict}")
    return {'before': before, 'after': after,
            'forgetting': forgetting, 'verdict': verdict}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "█"*60)
    print("  SAGESequenceCubeV5 — separated retrieval + association")
    print("  Training reinforces retrieval. Dictionary holds the pointer.")
    print("█"*60)

    DIM, CUBE, EPOCHS, ALPHA = 64, 32, 80, 0.2

    cube = SAGESequenceCubeV5(cube_size=CUBE, embed_dim=DIM,
                               train_epochs=EPOCHS, alpha=ALPHA)
    sequences, names = make_sequences(dim=DIM)

    print("Storing sequences...")
    for seq_id, (seq, lbls) in enumerate(zip(sequences, names)):
        cube.store_sequence(seq, labels=lbls)
        print(f"  Seq {seq_id}: {' → '.join(lbls)}")

    print(f"\n  Transition diagnostics:")
    print(f"  {'Query':12} → {'Next':12}  "
          f"{'s_sim_before':12} {'s_sim_after':11} {'o_sim_after':11} {'collision':9}")
    print(f"  {'-'*70}")
    all_s = [p['s_idx'] for p in cube.stored_pairs]
    for p in cube.stored_pairs:
        collision = "⚠ COLLISION" if all_s.count(p['s_idx']) > 1 else "✓ unique"
        print(f"  {p['label_cur']:12} → {p['label_nxt']:12}  "
              f"{p['s_sim_before']:.3f}        "
              f"{p['s_sim_after']:.3f}       "
              f"{p['o_sim_after']:.3f}       {collision}")

    unique_s = len(set(p['s_idx'] for p in cube.stored_pairs))
    n_trans  = len(cube.stored_pairs)
    print(f"\n  Unique s_idx: {unique_s}/{n_trans}  "
          f"({'✓ no collisions' if unique_s == n_trans else '✗ collisions'})")
    print(f"  Dictionary entries: {len(cube.transition_dict)}")

    r1 = run_exp1(cube, sequences, names)
    r2 = run_exp2(cube, sequences, names)
    r3 = run_exp3(cube, sequences)
    r4 = run_exp4(cube, sequences)

    results = {
        'version': 'v5',
        'config': {'dim': DIM, 'cube': CUBE, 'epochs': EPOCHS, 'alpha': ALPHA},
        'exp1': r1, 'exp2': r2, 'exp3': r3, 'exp4': r4,
        'unique_subject_points': unique_s,
        'total_transitions': n_trans,
    }

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = os.path.join(OUTPUT_DIR, f"results_v5_{ts}.json")
    with open(fpath, 'w') as f:
        def fix(o):
            if isinstance(o, (np.float32, np.float64)): return float(o)
            if isinstance(o, (np.int32,   np.int64)):   return int(o)
            return str(o)
        json.dump(results, f, indent=2, default=fix)
    print(f"\n  Results saved → {fpath}")

    print("\n" + "█"*60)
    print("  FINAL SUMMARY")
    print("█"*60)
    print(f"""
  Exp 1 — Retrieval     Top-1: {r1['top1']:.1%}   Top-5: {r1['top5']:.1%}
  Exp 2 — Rollout       {r2['correct']}/{r2['total']} steps = {r2['accuracy']:.1%}
  Exp 3 — Clustering    separation = {r3['ratio']:.2f}x  {r3['verdict']}
  Exp 4 — Forgetting    {r4['forgetting']:.1%}  {r4['verdict']}
""")

    if r1['top1'] >= 0.6:
        print("  ✓ SEQUENCE CUBE CONFIRMED")
        print("    Geometric transition encoding validated.")
        print("    Ready to integrate as MultiCube specialist.")
        print("    Novel architectural claim — publish.")
    elif r1['top1'] >= 0.3:
        print("  ~ IMPROVING — close to threshold")
        print("    Check s_sim_after values: should be > 0.7")
        print("    If low: increase EPOCHS or ALPHA")
    else:
        print("  ✗ Check diagnostics above:")
        print("    s_sim_before → s_sim_after should INCREASE (training working)")
        print("    If s_sim_after < s_sim_before: embedding is drifting away")
        print("    If unique_s < n_trans: collision still occurring")

    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
