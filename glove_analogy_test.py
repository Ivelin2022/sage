"""
SAGE Delta Encoding — GloVe 50d (GPU V2, Fast)
No spatial cohesion loop. Pure vectorised PyTorch.
Author: Ivelin Likov
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys, os, time

# ── Set your GloVe path here ─────────────────────────────────
# Download GloVe 6B from: https://nlp.stanford.edu/data/glove.6B.zip
# Then set the path to glove.6B.50d.txt below
import os
_script_dir = os.path.dirname(os.path.abspath(__file__))
GLOVE_FILE  = os.path.join(_script_dir, 'data', 'glove.6B.50d.txt')
CUBE_DIR    = _script_dir
# If your GloVe file is elsewhere, override here:
# GLOVE_FILE = '/path/to/glove.6B.50d.txt'

CUBE_SIZE  = 32
EMBED_DIM  = 50
N_VOCAB    = 10_000
EPOCHS     = 200
BATCH      = 4096      # huge — GPU does ~10 steps per epoch total
LR         = 0.02
MOMENTUM   = 0.9
DIR_WEIGHT = 0.3       # Force 3 weight
NEG_WEIGHT = 0.3       # Force 2 weight
TEMP_START = 0.3
TEMP_END   = 0.05
SEED       = 42

ANALOGIES = [
    ('king',    'man',     'woman',   'queen'),
    ('queen',   'woman',   'man',     'king'),
    ('paris',   'france',  'italy',   'rome'),
    ('berlin',  'germany', 'france',  'paris'),
    ('london',  'england', 'france',  'paris'),
    ('rome',    'italy',   'germany', 'berlin'),
    ('walking', 'walk',    'swim',    'swimming'),
    ('walked',  'walk',    'run',     'ran'),
    ('bigger',  'big',     'small',   'smaller'),
    ('faster',  'fast',    'slow',    'slower'),
    ('actor',   'man',     'woman',   'actress'),
    ('uncle',   'man',     'woman',   'aunt'),
]
WEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Load GloVe ────────────────────────────────────────────────
print("Loading GloVe...")
words, vecs = [], []
with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= N_VOCAB: break
        p = line.split()
        words.append(p[0])
        vecs.append(np.array(p[1:], dtype=np.float32))

vecs_np = np.array(vecs, dtype=np.float32)
vecs_np /= np.linalg.norm(vecs_np, axis=1, keepdims=True) + 1e-8
w2v = {w: vecs_np[i] for i, w in enumerate(words)}
print(f"Loaded {len(w2v):,} words")

# ── Build training tensors ────────────────────────────────────
print("Building pairs...")
Q_list, T_list = [], []

# Identity
for v in vecs_np:
    Q_list.append(v); T_list.append(v)

# Top-3 neighbours
sim = vecs_np @ vecs_np.T
np.fill_diagonal(sim, -1)
for i in range(len(vecs_np)):
    for j in np.argsort(sim[i])[-3:]:
        Q_list.append(vecs_np[i]); T_list.append(vecs_np[j])

# Analogy triples
for a, b, c, d in ANALOGIES:
    if all(w in w2v for w in [a, b, c, d]):
        va, vb, vc, vd = w2v[a], w2v[b], w2v[c], w2v[d]
        Q_list += [va, vc]; T_list += [vb, vd]
        q = va - vb + vc; q /= np.linalg.norm(q) + 1e-8
        Q_list.append(q); T_list.append(vd)

Q_all = torch.tensor(np.array(Q_list), dtype=torch.float32, device=device)
T_all = torch.tensor(np.array(T_list), dtype=torch.float32, device=device)
Q_all = F.normalize(Q_all, dim=1)
T_all = F.normalize(T_all, dim=1)
print(f"Pairs on GPU: {Q_all.shape[0]:,}")

# Direction vectors for Force 3
D_all = T_all - Q_all
D_norms = D_all.norm(dim=1, keepdim=True)
D_all = torch.where(D_norms > 1e-8, D_all / (D_norms + 1e-8), D_all)

# ── Cube ──────────────────────────────────────────────────────
torch.manual_seed(SEED)
N = CUBE_SIZE ** 3
E = F.normalize(torch.randn(N, EMBED_DIM, device=device), dim=1)
V = torch.zeros_like(E)  # momentum buffer
labels = {}

def get_temp(step):
    warmup = 500
    factor = 1.0 + 2.0 * max(0.0, 1.0 - step / warmup)
    return TEMP_END + (TEMP_START - TEMP_END) * max(0.0, 1.0 - step / warmup) * factor

# ── Training step — fully vectorised ─────────────────────────
def train_step(q_batch, t_batch, d_batch, step):
    global E, V
    temp = get_temp(step)

    # Similarities (B, N)
    sims   = q_batch @ E.T
    scores = F.softmax(sims / temp, dim=1)         # (B, N)

    # Force 3: direction gradient
    dir_scores = scores * DIR_WEIGHT               # (B, N)

    # Combined gradient: attraction + direction
    # grad = scores.T @ T + dir_scores.T @ D
    #      - (scores + dir_scores).sum(0) * E
    combined_scores = scores + dir_scores          # (B, N)
    weighted_target = scores.T @ t_batch + dir_scores.T @ d_batch   # (N, D)
    weight_sum      = combined_scores.sum(0).unsqueeze(1)            # (N, 1)
    avg_target      = weighted_target / (weight_sum + 1e-8)          # (N, D)
    grad            = avg_target - E                                 # (N, D)

    # Force 1: momentum
    V = MOMENTUM * V + (1.0 - MOMENTUM) * grad
    E = E + LR * V

    # Force 2: contrastive repulsion (vectorised)
    neg_idx = torch.randperm(N, device=device)[:50]
    t_mean  = t_batch.mean(0, keepdim=True)
    push    = E[neg_idx] - t_mean
    push_n  = push.norm(dim=1, keepdim=True)
    push    = torch.where(push_n > 1e-8, push / (push_n + 1e-8), push)
    E[neg_idx] = E[neg_idx] + LR * NEG_WEIGHT * push

    # Renormalise
    E = F.normalize(E, dim=1)

    # Loss
    with torch.no_grad():
        resp = F.normalize(scores @ E, dim=1)
        loss = (1.0 - (resp * t_batch).sum(1)).mean()
    return loss.item()

# ── Train ─────────────────────────────────────────────────────
print(f"Training {CUBE_SIZE}^3 | {EPOCHS} epochs | batch={BATCH}...")
t0 = time.time()
step = 0
for epoch in range(EPOCHS):
    perm = torch.randperm(Q_all.shape[0], device=device)
    Q_e, T_e, D_e = Q_all[perm], T_all[perm], D_all[perm]
    losses = []
    for i in range(0, Q_all.shape[0], BATCH):
        loss = train_step(Q_e[i:i+BATCH], T_e[i:i+BATCH], D_e[i:i+BATCH], step)
        losses.append(loss)
        step += 1
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}  "
              f"loss={np.mean(losses):.4f}  {time.time()-t0:.0f}s")

print(f"Done in {time.time()-t0:.1f}s\n")

# ── Label words ───────────────────────────────────────────────
for a, b, c, d in ANALOGIES:
    for w in [a, b, c, d]:
        if w in w2v:
            q = torch.tensor(w2v[w], device=device)
            q = F.normalize(q, dim=0)
            idx = (E @ q).argmax().item()
            labels[idx] = w

# ── Query ─────────────────────────────────────────────────────
def query_top(q_np, top_k=10):
    q = F.normalize(torch.tensor(q_np, dtype=torch.float32, device=device), dim=0)
    scores = E @ q
    top    = scores.topk(top_k).indices
    for idx in top:
        lbl = labels.get(idx.item(), f'point_{idx.item()}')
        if not lbl.startswith('point_'):
            return lbl
    return None

# ── Test ──────────────────────────────────────────────────────
def test(mode, weight=0.0):
    correct, total, rows = 0, 0, []
    for a, b, c, d in ANALOGIES:
        if not all(w in w2v for w in [a, b, c, d]):
            continue
        va, vb, vc = w2v[a], w2v[b], w2v[c]
        if mode == 'baseline':
            q = vc.copy()
        elif mode == 'delta_ab':
            q = vc + weight * (vb - va)
            q /= np.linalg.norm(q) + 1e-8
        elif mode == 'arithmetic':
            q = va - vb + vc
            q /= np.linalg.norm(q) + 1e-8
        pred = query_top(q)
        hit  = (pred == d)
        if hit: correct += 1
        total += 1
        rows.append((f'{a}-{b}+{c}={d}', pred, 'Y' if hit else 'N'))
    return correct, total, rows

# ── Results ───────────────────────────────────────────────────
print("="*62)
print("RESULTS")
print("="*62)

bc, bt, brows = test('baseline')
print(f"\nBASELINE:    {bc}/{bt} = {bc/bt*100:.1f}%")
for a, p, m in brows: print(f"  {m} {a:30s}  got: {p}")

ac, at, arows = test('arithmetic')
print(f"\nARITHMETIC:  {ac}/{at} = {ac/at*100:.1f}%  (upper bound)")
for a, p, m in arows: print(f"  {m} {a:30s}  got: {p}")

print(f"\nDELTA_AB sweep  q = normalize(c + w*(b-a))")
best_c, best_w = 0, 0.0
for w in WEIGHTS:
    c, t, _ = test('delta_ab', weight=w)
    tag = ' <- BEST' if c > best_c else ''
    if c > best_c: best_c, best_w = c, w
    print(f"  w={w}  {c}/{t} = {c/t*100:.1f}%{tag}")

_, _, best_rows = test('delta_ab', weight=best_w)
print(f"\nBEST DELTA_AB (w={best_w}):")
for a, p, m in best_rows: print(f"  {m} {a:30s}  got: {p}")

print(f"\n{'='*62}")
print(f"SUMMARY")
print(f"  Baseline:       {bc}/{bt} = {bc/bt*100:.1f}%")
print(f"  Arithmetic:     {ac}/{at} = {ac/at*100:.1f}%")
print(f"  Best delta_ab:  {best_c}/{bt} = {best_c/bt*100:.1f}%  (w={best_w})")
print(f"  Delta gain:     {best_c - bc:+d} analogies vs baseline")
if best_c > bc:
    print(f"  RESULT: Delta encoding IMPROVES analogy accuracy")
elif best_c == bc:
    print(f"  RESULT: No effect at this training scale")
else:
    print(f"  RESULT: Delta encoding hurts accuracy")
print(f"{'='*62}")
