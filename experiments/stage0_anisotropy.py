"""
SAGE-Sphere - Stage 0 - Anisotropy Diagnostic (production version)
===================================================================
Adapted from the reference `anisotropy_diagnostic.py`. Differences:
  - Reuses the existing nomic_cache.json (358 analogy-benchmark words) and
    extends the vocab to ~3000 real words from GloVe's frequency-ordered list.
  - Caches all embeddings to data/embeddings_cache.npz (word -> 768-D vector).
  - Writes results/stage0_anisotropy.json and a dark-theme cosine histogram.
  - Appends a dated entry to findings.md.
  - utf-8 everywhere, ASCII-only prints, Agg backend, os.path.join.

Question: are nomic-embed-text embeddings anisotropic (crammed into a cone so
random word pairs have high cosine)? Verdict gates the rest of the build:
  mean|cos| < 0.15  -> LOW      -> Stage 2 (isotropy) optional
  0.15 - 0.40       -> MODERATE -> Stage 2 recommended
  > 0.40            -> HIGH     -> Stage 2 required; composition fix = traversal

Usage:  python experiments/stage0_anisotropy.py
Requires: Ollama running with nomic-embed-text pulled.
"""

import os
import sys
import json
import time
import urllib.request

import numpy as np

import matplotlib
matplotlib.use('Agg')                       # MUST precede pyplot
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Paths (all derived from this file's location - no hardcoded absolutes)
# ----------------------------------------------------------------------
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # sage_sphere/
NEWARCH  = os.path.abspath(os.path.join(ROOT, '..', '..'))               # new architecture/
DATA_DIR = os.path.join(ROOT, 'data')
RES_DIR  = os.path.join(ROOT, 'results')
OLD_CACHE = os.path.join(NEWARCH, 'analogy_cache', 'nomic_cache.json')
GLOVE     = os.path.join(NEWARCH, 'data', 'glove.6B.50d.txt')
CACHE_NPZ = os.path.join(DATA_DIR, 'embeddings_cache.npz')
FINDINGS  = os.path.join(ROOT, 'findings.md')

OLLAMA_URL    = 'http://localhost:11434/api/embed'   # batched endpoint
MODEL         = 'nomic-embed-text'
TARGET_VOCAB  = 3000
N_PAIRS       = 10000
BATCH_SIZE    = 128                                   # words per request
SEED          = 0


# ----------------------------------------------------------------------
# Embedding via Ollama (stdlib urllib - no requests dependency)
# Uses /api/embed which accepts a LIST of inputs in ONE request - ~40x
# faster than one HTTP call per word (per-request overhead dominates).
# ----------------------------------------------------------------------
def embed_batch(words):
    """Return list of float64 vectors (one per word), or None on failure."""
    payload = json.dumps({'model': MODEL, 'input': words}).encode('utf-8')
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            embs = json.loads(r.read().decode('utf-8')).get('embeddings')
        if embs is None or len(embs) != len(words):
            return None
        return [np.asarray(e, dtype=np.float64) for e in embs]
    except Exception as e:
        print("  ! batch embed failed (%d words): %s" % (len(words), e))
        return None


def load_seed_cache():
    """Load the existing 358-word nomic cache as {word: float64 vector}."""
    vecs = {}
    if os.path.exists(OLD_CACHE):
        with open(OLD_CACHE, encoding='utf-8') as f:
            raw = json.load(f)
        for w, v in raw.items():
            vecs[w] = np.asarray(v, dtype=np.float64)
        print("Seeded %d words from existing nomic_cache.json" % len(vecs))
    else:
        print("No existing nomic_cache.json found - embedding from scratch.")
    return vecs


def build_vocab(have):
    """Extend `have` (set of words) up to TARGET_VOCAB using GloVe frequency
    order. GloVe 6B is sorted most-frequent first; we keep clean alpha tokens."""
    vocab = list(have)
    if not os.path.exists(GLOVE):
        print("GloVe not found at %s - using cached vocab only (%d words)."
              % (GLOVE, len(vocab)))
        return vocab
    seen = set(have)
    with open(GLOVE, encoding='utf-8') as f:
        for line in f:
            if len(vocab) >= TARGET_VOCAB:
                break
            w = line.split(' ', 1)[0]
            if w in seen:
                continue
            if w.isalpha() and w.islower() and 2 <= len(w) <= 20:
                vocab.append(w)
                seen.add(w)
    print("Vocab target %d -> built %d words (cached + GloVe frequent)."
          % (TARGET_VOCAB, len(vocab)))
    return vocab


def main():
    np.random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    # --- assemble vocab and embeddings, reusing cache ---
    cache_only = '--cache-only' in sys.argv
    vecs = load_seed_cache()
    if cache_only:
        vocab = list(vecs.keys())
        print("Cache-only mode: using %d cached words, no new embedding." % len(vocab))
    else:
        vocab = build_vocab(set(vecs.keys()))

    missing = [w for w in vocab if w not in vecs]
    print("\nNeed to embed %d new words via %s (batch=%d) ..."
          % (len(missing), MODEL, BATCH_SIZE), flush=True)
    t0 = time.time()
    fails = 0
    done = 0
    for start in range(0, len(missing), BATCH_SIZE):
        chunk = missing[start:start + BATCH_SIZE]
        embs = embed_batch(chunk)
        if embs is None:                          # fall back word-by-word
            for w in chunk:
                one = embed_batch([w])
                if one:
                    vecs[w] = one[0]
                else:
                    fails += 1
        else:
            for w, e in zip(chunk, embs):
                vecs[w] = e
        done += len(chunk)
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        eta = (len(missing) - done) / rate if rate > 0 else 0
        print("  %d/%d  (%.0fs elapsed, %.1f w/s, ETA %.0fs, %d fails)"
              % (done, len(missing), el, rate, eta, fails), flush=True)

    words = [w for w in vocab if w in vecs]
    if len(words) < 10:
        print("ERROR: too few embeddings (%d). Is Ollama running with '%s'?"
              % (len(words), MODEL))
        sys.exit(1)

    # --- persist combined cache for later stages ---
    M = np.stack([vecs[w] for w in words])               # (N, 768) float64
    np.savez(CACHE_NPZ, words=np.array(words, dtype=object),
             embs=M.astype(np.float32))
    print("\nSaved %d embeddings -> %s" % (len(words), CACHE_NPZ))

    # --- norms (confirm / enforce unit-norm for cosine) ---
    norms = np.linalg.norm(M, axis=1)
    print("\nEmbeddings: %d words x %d dims" % (M.shape[0], M.shape[1]))
    print("Norm stats: mean=%.4f std=%.4f min=%.4f max=%.4f"
          % (norms.mean(), norms.std(), norms.min(), norms.max()))
    Mn = M / (norms[:, None] + 1e-12)

    # --- (1) random pairwise cosine ---
    idx = np.arange(len(words))
    a = np.random.choice(idx, size=N_PAIRS, replace=True)
    b = np.random.choice(idx, size=N_PAIRS, replace=True)
    keep = a != b
    a, b = a[keep], b[keep]
    cos = np.sum(Mn[a] * Mn[b], axis=1)
    mean_cos = float(cos.mean())
    mean_abs = float(np.abs(cos).mean())

    print("\n" + "=" * 60)
    print("RESULT 1 - random pairwise cosine similarity")
    print("=" * 60)
    print("  pairs sampled:   %d" % len(cos))
    print("  mean cosine:     %.4f   (isotropic ideal ~ 0)" % mean_cos)
    print("  mean |cosine|:   %.4f" % mean_abs)
    print("  std:             %.4f" % cos.std())
    print("  min / max:       %.4f / %.4f" % (cos.min(), cos.max()))

    # --- (2) directional collapse ---
    mean_vec_norm = float(np.linalg.norm(Mn.mean(axis=0)))
    centered = Mn - Mn.mean(axis=0)
    sample = centered if centered.shape[0] <= 2000 else centered[
        np.random.choice(centered.shape[0], 2000, replace=False)]
    try:
        s = np.linalg.svd(sample, compute_uv=False)
        var = s ** 2
        top1 = float(var[0] / var.sum())
        top5 = float(var[:5].sum() / var.sum())
    except Exception:
        top1 = top5 = float('nan')

    print("\n" + "=" * 60)
    print("RESULT 2 - directional collapse")
    print("=" * 60)
    print("  mean-vector norm: %.4f  (0=isotropic, ->1=all same way)"
          % mean_vec_norm)
    print("  top-1 PC var:     %.1f%%  (isotropic ~ %.3f%% per dim)"
          % (top1 * 100, 100.0 / M.shape[1]))
    print("  top-5 PC var:     %.1f%%" % (top5 * 100))

    # --- verdict ---
    if mean_abs < 0.15:
        level = "LOW"
        advice = ("Fairly isotropic. Stage 2 (isotropy) OPTIONAL; "
                  "sphere substrate alone helps more.")
    elif mean_abs < 0.40:
        level = "MODERATE"
        advice = ("Moderate anisotropy. Stage 2 (All-but-the-Top) RECOMMENDED "
                  "before placing on the sphere.")
    else:
        level = "HIGH"
        advice = ("Severe anisotropy - composition failure (8A/9C) confirmed to "
                  "live in the EMBEDDINGS. Stage 2 REQUIRED; sphere fixes the "
                  "metric, NOT composition; graph traversal is the fix.")

    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print("  Anisotropy level: %s  (mean |cos| = %.4f)" % (level, mean_abs))
    print("  -> %s" % advice)

    # --- save JSON ---
    out = {
        "stage": 0, "model": MODEL,
        "n_words": int(M.shape[0]), "n_dims": int(M.shape[1]),
        "n_pairs": int(len(cos)), "n_failed_embeds": int(fails),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "mean_cosine": mean_cos, "mean_abs_cosine": mean_abs,
        "cosine_std": float(cos.std()),
        "cosine_min": float(cos.min()), "cosine_max": float(cos.max()),
        "mean_vector_norm": mean_vec_norm,
        "top1_pc_variance": top1, "top5_pc_variance": top5,
        "anisotropy_level": level, "advice": advice,
    }
    res_path = os.path.join(RES_DIR, 'stage0_anisotropy.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # --- dark-theme cosine histogram ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    ax.hist(cos, bins=80, color='#4fc3f7', alpha=0.85, edgecolor='#0a0a0a')
    ax.axvline(0.0, color='#888888', ls='--', lw=1, label='isotropic ideal (0)')
    ax.axvline(mean_cos, color='#ff5252', ls='-', lw=2,
               label='mean cosine = %.3f' % mean_cos)
    ax.set_title('Stage 0 - random pairwise cosine  (%s anisotropy, mean|cos|=%.3f)'
                 % (level, mean_abs), color='white')
    ax.set_xlabel('cosine similarity', color='white')
    ax.set_ylabel('count', color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333')
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'stage0_cosine_hist.png')
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    # --- append findings.md entry ---
    entry = (
        "\n## Stage 0 - Anisotropy diagnostic\n\n"
        "- model: %s | words: %d | dims: %d | pairs: %d\n"
        "- mean cosine: %.4f | mean |cos|: %.4f | std: %.4f\n"
        "- mean-vector norm: %.4f | top-1 PC: %.1f%% | top-5 PC: %.1f%%\n"
        "- **VERDICT: %s** (mean|cos| = %.4f)\n"
        "- %s\n"
        % (MODEL, M.shape[0], M.shape[1], len(cos),
           mean_cos, mean_abs, cos.std(), mean_vec_norm,
           top1 * 100, top5 * 100, level, mean_abs, advice)
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended Stage 0 entry to findings.md")

    print("\nGATE: Stage 0 always passes (it parameterizes the build).")
    print("Next: Stage 1 - plain Fibonacci-sphere substrate.")


if __name__ == '__main__':
    main()
