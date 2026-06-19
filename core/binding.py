"""
core/binding.py - FHRR (Fourier Holographic Reduced Representations) binding.

Complex vector-symbolic algebra, gradient-free. The genuinely-geometric path to
STRUCTURED memory that flat retrieval and plain graphs cannot do:

  bind(a, b)     = a (x) b   elementwise complex multiply  (role-filler binding)
  unbind(c, a)   = c (x) a*   multiply by conjugate         (approximate inverse)
  bundle(vs)     = sum         superpose many facts into one vector
  cleanup(v, E)  = nearest entity in codebook E by complex cosine

Multi-hop chaining works ONLY with cleanup: each hop re-grounds the noisy unbind
result onto a clean stored entity (= the sphere substrate's nearest-neighbour
retrieval) before the next hop, so noise does not compound. This is the mechanism
greedy traversal lacked.

THESIS GUARD: numpy complex arithmetic only. No gradients, no training.
"""

import numpy as np


def random_phasors(n, dim, rng):
    """n random unit-magnitude complex vectors (e^{i theta}), shape (n, dim)."""
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(n, dim)))


def bind(a, b):
    """Role-filler binding: elementwise complex multiply (unit-magnitude in, out)."""
    return a * b


def unbind(c, a):
    """Approximate inverse: multiply by the conjugate (= inverse for phasors)."""
    return c * np.conj(a)


def bundle(vecs, axis=0):
    """Superpose (sum) a set of vectors into one memory vector."""
    return vecs.sum(axis=axis)


def hrr_bind(a, b):
    """HRR binding = circular convolution (for REAL vectors, e.g. embeddings).
    Works along the last axis with broadcasting; a (D,) with b (N, D) -> (N, D)."""
    n = a.shape[-1]
    return np.fft.irfft(np.fft.rfft(a, axis=-1) * np.fft.rfft(b, axis=-1),
                        n=n, axis=-1)


def hrr_unbind(c, a):
    """HRR unbinding = circular correlation: recovers b from conv(a, b) given a."""
    n = c.shape[-1]
    return np.fft.irfft(np.fft.rfft(c, axis=-1) * np.conj(np.fft.rfft(a, axis=-1)),
                        n=n, axis=-1)


def make_unitary(v):
    """Make real role vector(s) UNITARY (flat unit power spectrum) so HRR unbind
    is an exact inverse: |rfft(v)| == 1 per bin -> unbind(conv(v,b), v) == b
    (up to superposition cross-talk). Roles must be unitary or recovery is
    corrupted by a colored-noise kernel. Returns same shape as v."""
    n = v.shape[-1]
    F = np.fft.rfft(v, axis=-1)
    F = F / (np.abs(F) + 1e-12)
    return np.fft.irfft(F, n=n, axis=-1)


def cleanup(queries, codebook):
    """Snap noisy query vector(s) to the nearest codebook entity by complex cosine.
    queries: (D,) or (m, D) complex; codebook: (N, D) complex.
    returns: (m,) int indices of the nearest entity (scalar-friendly via atleast_2d).
    """
    q = np.atleast_2d(queries)
    sim = (q @ np.conj(codebook).T).real                    # Re<q, e>  (m, N)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    cn = np.linalg.norm(codebook, axis=1, keepdims=True).T
    sim = sim / (qn * cn + 1e-12)
    return np.argmax(sim, axis=1)
