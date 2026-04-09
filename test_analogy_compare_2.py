"""
test_analogy_compare.py  (batched version)
==========================================
Compares 4 conditions on word analogy tasks:
  A: GloVe 50d  + Baseline V2
  B: GloVe 50d  + SAGEDivided
  C: nomic 768d + Baseline V2
  D: nomic 768d + SAGEDivided

Fully batched training -- all pairs in one matrix operation per epoch.
~100x faster than per-pair loop. No cube_core imports needed.

Expected runtime on RTX 4090: ~3-5 minutes total.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, json, time, requests

# ── Config ─────────────────────────────────────────────────────────────────────
# Update these paths to match your local setup
# GloVe: https://nlp.stanford.edu/projects/glove/ (glove.6B.zip, use 50d file)
# Analogy: https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt

_HERE = os.path.dirname(os.path.abspath(__file__))

GLOVE_PATH   = os.path.join(_HERE, 'data', 'glove.6B.50d.txt')
ANALOGY_PATH = os.path.join(_HERE, 'questions-words.txt')
CACHE_DIR    = os.path.join(_HERE, 'analogy_cache')
RESULTS_DIR  = os.path.join(_HERE, 'analogy_results')
OLLAMA_URL   = 'http://localhost:11434/api/embeddings'
OLLAMA_MODEL = 'nomic-embed-text'

TARGET_CATEGORIES = [
    'family',
    'capital-common-countries',
    'capital-world',
    'gram7-past-tense',
]

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
CUBE_SIZE    = 32
N_RUNS       = 3
ALPHA        = 0.1
ALPHA_DIR    = 0.05
TAU          = 0.1
EPOCHS_GLOVE = 500
EPOCHS_NOMIC = 300
TOP_K        = 10

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_glove(path, needed_words):
    needed = set(w.lower() for w in needed_words)
    vecs = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            w = parts[0].lower()
            if w in needed:
                vecs[w] = np.array(parts[1:], dtype=np.float32)
    for w in vecs:
        n = np.linalg.norm(vecs[w])
        if n > 1e-8:
            vecs[w] /= n
    print(f"  GloVe: {len(vecs):,}/{len(needed):,} words found")
    return vecs


def load_nomic(needed_words):
    cache_file = os.path.join(CACHE_DIR, 'nomic_cache.json')
    cache = {}
    if os.path.exists(cache_file):
        raw = json.load(open(cache_file))
        cache = {w: np.array(v, dtype=np.float32) for w, v in raw.items()}
        print(f"  nomic cache: {len(cache):,} words loaded")
    missing = [w for w in needed_words if w not in cache]
    if missing:
        print(f"  Embedding {len(missing):,} new words via Ollama...")
        t0 = time.time()
        for i, word in enumerate(missing):
            try:
                r = requests.post(OLLAMA_URL,
                    json={'model': OLLAMA_MODEL, 'prompt': word}, timeout=30)
                vec = np.array(r.json()['embedding'], dtype=np.float32)
                n = np.linalg.norm(vec)
                if n > 1e-8:
                    vec /= n
                cache[word] = vec
            except Exception as e:
                print(f"  Ollama error '{word}': {e}")
            if (i+1) % 50 == 0:
                eta = (time.time()-t0)/(i+1)*(len(missing)-i-1)
                print(f"    {i+1}/{len(missing)} -- ETA {eta:.0f}s")
                json.dump({w: v.tolist() for w, v in cache.items()},
                          open(cache_file, 'w'))
        json.dump({w: v.tolist() for w, v in cache.items()}, open(cache_file, 'w'))
        print(f"  Saved {len(cache):,} embeddings")
    vecs = {w: cache[w] for w in needed_words if w in cache}
    print(f"  nomic: {len(vecs):,}/{len(needed_words):,} words available")
    return vecs


def load_analogies(path, target_cats):
    categories = {}
    current = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith(':'):
                cat = line[2:]
                current = cat if any(t in cat for t in target_cats) else None
                if current:
                    categories[current] = []
            elif current and line:
                parts = line.split()
                if len(parts) == 4:
                    categories[current].append(tuple(parts))
    for cat, pairs in categories.items():
        print(f"  {cat}: {len(pairs):,} questions")
    return categories


def get_needed_words(categories):
    words = set()
    for analogies in categories.values():
        for quad in analogies:
            words.update(quad)
    return words


# ── Build tensors ──────────────────────────────────────────────────────────────

def build_pair_tensors(categories, vecs, device):
    seen = set()
    a_list, b_list = [], []
    for cat, analogies in categories.items():
        for a, b, c, d in analogies:
            if a in vecs and b in vecs and (a, b) not in seen:
                a_list.append(vecs[a]); b_list.append(vecs[b])
                seen.add((a, b))
            if c in vecs and d in vecs and (c, d) not in seen:
                a_list.append(vecs[c]); b_list.append(vecs[d])
                seen.add((c, d))
    A = torch.tensor(np.stack(a_list), device=device)
    B = torch.tensor(np.stack(b_list), device=device)
    D = F.normalize(B - A, dim=1)
    print(f"  Pairs: {A.shape[0]:,}  dim={A.shape[1]}")
    return A, B, D


def build_vocab_tensor(vecs, device):
    words = list(vecs.keys())
    V = F.normalize(torch.tensor(np.stack([vecs[w] for w in words]),
                                 device=device), dim=1)
    w2i = {w: i for i, w in enumerate(words)}
    return words, V, w2i


# ── Models ─────────────────────────────────────────────────────────────────────

class SAGEBaseline:
    """Full cube, batched Hebbian + direction training."""

    def __init__(self, cube_size, embed_dim, device):
        N = cube_size ** 3
        self.E = F.normalize(torch.randn(N, embed_dim, device=device), dim=1)
        self.device = device

    def train_epoch(self, A, B, D):
        # scores (P, N): how much each pair activates each cube point
        scores = torch.softmax((B @ self.E.T) / TAU, dim=1)
        # Hebbian pull toward B + direction push
        SB = scores.T @ B               # (N, D)
        Sw = scores.sum(0)              # (N,)
        SD = scores.T @ D               # (N, D)
        delta = ALPHA * (SB - Sw.unsqueeze(1) * self.E) + ALPHA_DIR * SD
        self.E = F.normalize(self.E + delta, dim=1)

    def query(self, a_t, b_t, c_t):
        q = F.normalize((b_t - a_t + c_t).unsqueeze(0), dim=1)
        scores = torch.softmax((self.E @ q.T.squeeze(1)) / TAU, dim=0)
        return F.normalize((self.E * scores.unsqueeze(1)).sum(0, keepdim=True), dim=1).squeeze(0)


class SAGEDivided:
    """Partitioned cube: x<0=subject, x>=0=object. Batched training."""

    def __init__(self, cube_size, embed_dim, device):
        N = cube_size ** 3
        self.E = F.normalize(torch.randn(N, embed_dim, device=device), dim=1)
        self.device = device
        coords = torch.linspace(-1, 1, cube_size)
        gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing='ij')
        pos = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1).to(device)
        self.s_idx = torch.where(pos[:, 0] < 0)[0]
        self.o_idx = torch.where(pos[:, 0] >= 0)[0]

    def train_epoch(self, A, B, D):
        # Subject half <- A embeddings + direction D
        Es = self.E[self.s_idx]
        scores_s = torch.softmax((A @ Es.T) / TAU, dim=1)
        SA  = scores_s.T @ A
        SwS = scores_s.sum(0)
        SDS = scores_s.T @ D
        delta_s = ALPHA * (SA - SwS.unsqueeze(1) * Es) + ALPHA_DIR * SDS
        self.E[self.s_idx] = F.normalize(Es + delta_s, dim=1)

        # Object half <- B embeddings + direction D
        Eo = self.E[self.o_idx]
        scores_o = torch.softmax((B @ Eo.T) / TAU, dim=1)
        SB  = scores_o.T @ B
        SwO = scores_o.sum(0)
        SDO = scores_o.T @ D
        delta_o = ALPHA * (SB - SwO.unsqueeze(1) * Eo) + ALPHA_DIR * SDO
        self.E[self.o_idx] = F.normalize(Eo + delta_o, dim=1)

    def query(self, a_t, b_t, c_t):
        q = F.normalize((b_t - a_t + c_t).unsqueeze(0), dim=1)
        scores = torch.softmax((self.E @ q.T.squeeze(1)) / TAU, dim=0)
        return F.normalize((self.E * scores.unsqueeze(1)).sum(0, keepdim=True), dim=1).squeeze(0)


# ── Training & evaluation ──────────────────────────────────────────────────────

def train(model, A, B, D, epochs):
    t0 = time.time()
    print(f"    Training {epochs} epochs...", end='', flush=True)
    for ep in range(epochs):
        idx = torch.randperm(A.shape[0], device=A.device)
        model.train_epoch(A[idx], B[idx], D[idx])
        if (ep+1) % 100 == 0:
            print(f" {ep+1}", end='', flush=True)
    print(f"  ({time.time()-t0:.1f}s)")


def evaluate(model, categories, vecs, vocab_words, V_norm, w2i):
    cat_results = {}
    total_correct, total_q = 0, 0
    for cat, analogies in categories.items():
        correct, n_valid = 0, 0
        for a, b, c, d in analogies:
            if not all(w in vecs for w in [a, b, c, d]):
                continue
            n_valid += 1
            a_t = torch.tensor(vecs[a], device=model.device)
            b_t = torch.tensor(vecs[b], device=model.device)
            c_t = torch.tensor(vecs[c], device=model.device)
            resp = model.query(a_t, b_t, c_t)
            sims = V_norm @ resp
            for exc in [a, b, c]:
                if exc in w2i:
                    sims[w2i[exc]] = -1.0
            top_words = [vocab_words[i] for i in torch.topk(sims, TOP_K).indices.tolist()]
            if d in top_words:
                correct += 1
        acc = correct / n_valid if n_valid > 0 else 0
        cat_results[cat] = (correct, n_valid, acc)
        total_correct += correct
        total_q += n_valid
    return cat_results, total_correct, total_q, total_correct/total_q if total_q else 0


def run_experiment(label, model_class, A, B, D, vecs, categories,
                   vocab_words, V_norm, w2i, epochs):
    print(f"\n  {'='*52}")
    print(f"  {label}")
    print(f"  {'='*52}")
    all_accs, final_cat = [], None
    for run in range(N_RUNS):
        print(f"  Run {run+1}/{N_RUNS}:", end=' ', flush=True)
        model = model_class(CUBE_SIZE, A.shape[1], DEVICE)
        train(model, A, B, D, epochs)
        cat_r, correct, total, acc = evaluate(
            model, categories, vecs, vocab_words, V_norm, w2i)
        print(f"    Overall: {correct}/{total} = {acc:.1%}")
        for cat, (c, t, a) in cat_r.items():
            print(f"      {cat:<42} {c:>4}/{t:<4} = {a:.1%}")
        all_accs.append(acc)
        if run == N_RUNS - 1:
            final_cat = cat_r
    mean_acc = float(np.mean(all_accs))
    std_acc  = float(np.std(all_accs))
    print(f"\n  Result: {mean_acc:.1%} +/- {std_acc:.1%}")
    # Save interim immediately
    with open(os.path.join(RESULTS_DIR, f'interim_{label[:1]}.txt'), 'w') as f:
        f.write(f"{label}: {mean_acc:.4f} +/- {std_acc:.4f}\n")
        if final_cat:
            for cat, (c, t, a) in final_cat.items():
                f.write(f"  {cat}: {c}/{t} = {a:.4f}\n")
    return mean_acc, std_acc, final_cat


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SAGE Analogy Comparison: GloVe vs nomic x Baseline vs SAGEDivided")
    print("=" * 70)
    print(f"Device: {DEVICE} | Cube: {CUBE_SIZE}^3 | tau={TAU} | "
          f"alpha={ALPHA} | alpha_dir={ALPHA_DIR}\n")

    print("Loading analogy benchmark...")
    categories = load_analogies(ANALOGY_PATH, TARGET_CATEGORIES)
    needed = get_needed_words(categories)
    print(f"Unique words: {len(needed):,}\n")

    # GloVe
    print("Loading GloVe 50d...")
    gv = load_glove(GLOVE_PATH, needed)
    cats_g = {cat: [q for q in analogies if all(w in gv for w in q)]
              for cat, analogies in categories.items()}
    print(f"GloVe valid: {sum(len(v) for v in cats_g.values()):,} questions")
    A_g, B_g, D_g = build_pair_tensors(cats_g, gv, DEVICE)
    vw_g, V_g, wi_g = build_vocab_tensor(gv, DEVICE)

    # nomic
    print("\nLoading nomic 768d...")
    nv = load_nomic(needed)
    cats_n = {cat: [q for q in analogies if all(w in nv for w in q)]
              for cat, analogies in categories.items()}
    print(f"nomic valid: {sum(len(v) for v in cats_n.values()):,} questions")
    A_n, B_n, D_n = build_pair_tensors(cats_n, nv, DEVICE)
    vw_n, V_n, wi_n = build_vocab_tensor(nv, DEVICE)

    results = {}
    results['A'] = run_experiment('A: GloVe 50d + Baseline V2',
        SAGEBaseline, A_g, B_g, D_g, gv, cats_g, vw_g, V_g, wi_g, EPOCHS_GLOVE)
    results['B'] = run_experiment('B: GloVe 50d + SAGEDivided',
        SAGEDivided, A_g, B_g, D_g, gv, cats_g, vw_g, V_g, wi_g, EPOCHS_GLOVE)
    results['C'] = run_experiment('C: nomic 768d + Baseline V2',
        SAGEBaseline, A_n, B_n, D_n, nv, cats_n, vw_n, V_n, wi_n, EPOCHS_NOMIC)
    results['D'] = run_experiment('D: nomic 768d + SAGEDivided',
        SAGEDivided, A_n, B_n, D_n, nv, cats_n, vw_n, V_n, wi_n, EPOCHS_NOMIC)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    lbls = {'A':'A: GloVe 50d  + Baseline V2','B':'B: GloVe 50d  + SAGEDivided',
            'C':'C: nomic 768d + Baseline V2','D':'D: nomic 768d + SAGEDivided'}
    for k, lbl in lbls.items():
        m, s, _ = results[k]
        print(f"  {lbl:<38}  {m:.1%} +/- {s:.1%}")

    a, b, c, d = [results[k][0] for k in 'ABCD']
    print(f"\n  SAGEDivided gain (GloVe):   {(b-a)*100:+.1f}pp")
    print(f"  SAGEDivided gain (nomic):   {(d-c)*100:+.1f}pp")
    print(f"  nomic vs GloVe (baseline):  {(c-a)*100:+.1f}pp")
    print(f"  nomic vs GloVe (divided):   {(d-b)*100:+.1f}pp")

    print("\n  Per-category (Exp D):")
    _, _, cat_d = results['D']
    if cat_d:
        for cat, (co, tot, acc) in sorted(cat_d.items(), key=lambda x: -x[1][2]):
            print(f"    {cat:<44} {co:>4}/{tot:<4} = {acc:.1%}")

    out = os.path.join(RESULTS_DIR, 'analogy_compare_results.txt')
    with open(out, 'w') as f:
        f.write("SAGE Analogy Comparison Results\n")
        f.write(f"Cube: {CUBE_SIZE}^3 | tau={TAU} | alpha={ALPHA} | alpha_dir={ALPHA_DIR}\n\n")
        for k, lbl in lbls.items():
            m, s, cat_r = results[k]
            f.write(f"{lbl}: {m:.4f} +/- {s:.4f}\n")
            if cat_r:
                for cat, (co, tot, acc) in cat_r.items():
                    f.write(f"  {cat}: {co}/{tot} = {acc:.4f}\n")
            f.write("\n")
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
