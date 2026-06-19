"""
core/agent_memory.py - bounded associative memory for the AI-memory benchmark.

Gradient-free. The honest "memory for AI" question: against a vector DB, does a
bounded, self-consolidating, decaying memory retain more CURRENT facts per byte
on a long update stream?

  SAGEMemory(mode='flat') : VQ key-slots (D-dim) + recency value blend + decay
                            eviction. Consolidates repeated writes to the SAME
                            key into ONE slot (a vector DB stores each write).
  SAGEMemory(mode='grid') : addresses keys via a 3-D Fibonacci lattice instead
                            of D-dim slot-keys - the GEOMETRY TEST (predicted to
                            collide and lose, showing the 3-D part is not needed).
  VectorDB(mode=...)      : baseline store. 'unbounded' keeps every write (bloats);
                            'fifo' evicts the oldest write at a fixed budget.
                            Query is recency-correct (latest matching write wins).
"""

import numpy as np


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def fib3(n):
    """n golden-angle points on the unit 2-sphere (3-D), numpy."""
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    return _unit(np.stack([r * np.cos(th), r * np.sin(th), z], axis=1))


class SAGEMemory:
    def __init__(self, n_slots, dim, mode='flat', lr_key=0.3, lr_val=0.6,
                 decay=0.999, merge=0.5, proj=None):
        self.B = n_slots; self.dim = dim; self.mode = mode
        self.lr_key = lr_key; self.lr_val = lr_val
        self.decay = decay; self.merge = merge
        self.value = np.zeros((n_slots, dim))
        self.strength = np.zeros(n_slots)
        self.used = np.zeros(n_slots, dtype=bool)
        if mode == 'flat':
            self.cell_key = np.zeros((n_slots, dim))
        else:                                    # grid: fixed 3-D addresses
            self.proj = proj                     # (dim, 3) fixed random projection
            self.cell_pos = fib3(n_slots)        # (n_slots, 3)

    def _nearest(self, key):
        """Return (slot, similarity) of the slot this key addresses."""
        if self.mode == 'flat':
            if not self.used.any():
                return None, -1.0
            u = np.where(self.used)[0]
            sims = self.cell_key[u] @ _unit(key)
            j = int(np.argmax(sims))
            return int(u[j]), float(sims[j])
        p = _unit(key @ self.proj)               # key -> 3-D
        sims = self.cell_pos @ p
        j = int(np.argmax(sims))
        return j, float(sims[j])

    def write(self, key, val):
        key = _unit(key); val = _unit(val)
        self.strength[self.used] *= self.decay
        if self.mode == 'grid':
            slot, _ = self._nearest(key)
            if self.used[slot]:
                self.value[slot] = _unit((1 - self.lr_val) * self.value[slot]
                                         + self.lr_val * val)
            else:
                self.value[slot] = val; self.used[slot] = True
            self.strength[slot] += 1.0
            return
        slot, best = self._nearest(key)
        if slot is not None and best > self.merge:        # same key -> consolidate
            self.value[slot] = _unit((1 - self.lr_val) * self.value[slot]
                                     + self.lr_val * val)
            self.cell_key[slot] = _unit((1 - self.lr_key) * self.cell_key[slot]
                                        + self.lr_key * key)
            self.strength[slot] += 1.0
        else:                                             # new key -> allocate/evict
            slot = (int(np.where(~self.used)[0][0]) if not self.used.all()
                    else int(np.argmin(self.strength)))
            self.cell_key[slot] = key; self.value[slot] = val
            self.strength[slot] = 1.0; self.used[slot] = True

    def query(self, key):
        slot, _ = self._nearest(key)
        if slot is None or not self.used[slot]:
            return None
        return self.value[slot]

    def footprint(self):
        return int(self.used.sum())


class VectorDB:
    """Baseline store. mode='unbounded' (keep all) or 'fifo' (evict oldest at cap).
    Query returns the value of the MOST RECENT write whose key matches (recency-
    correct); falls back to the nearest key if none clears the match threshold."""
    def __init__(self, mode='unbounded', capacity=None, match=0.85):
        self.mode = mode; self.cap = capacity; self.match = match
        self.keys = []; self.vals = []

    def write(self, key, val):
        key = _unit(key); val = _unit(val)
        if self.mode == 'dedup' and self.keys:        # merge same key, mark recent
            sims = np.asarray(self.keys) @ key
            j = int(np.argmax(sims))
            if sims[j] > self.match:
                self.keys.pop(j); self.vals.pop(j)
                self.keys.append(key); self.vals.append(val)
                return
        self.keys.append(key); self.vals.append(val)
        if self.cap is not None and len(self.keys) > self.cap:
            self.keys.pop(0); self.vals.pop(0)

    def query(self, key):
        if not self.keys:
            return None
        sims = np.asarray(self.keys) @ _unit(key)
        match = np.where(sims > self.match)[0]
        if match.size:
            return self.vals[int(match[-1])]      # most recent matching write
        return self.vals[int(np.argmax(sims))]

    def footprint(self):
        return len(self.keys)
