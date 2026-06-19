"""
_sage_demo.py - shared library for the SAGE MVP demos.

Provides: Ollama embedding + chat (stdlib urllib, no extra deps), a faithful
text-storing SAGE bounded associative memory, and small terminal-UI helpers for
clean screen recording.

HONEST FRAMING (keep this in the demos): SAGE here is a *working* gradient-free
bounded associative memory = a vector store + merge/decay/evict. These demos show
it WORKING and integrated with an LLM. They do NOT claim it beats a vector DB -
the research showed it ties standard methods. The value on show is a functioning,
LLM-independent, continually-updated memory layer.

Requires Ollama running locally:  ollama serve
  embeddings:  ollama pull nomic-embed-text
  LLM (any):   ollama pull mistral     (or set SAGE_LLM=llama3.2 / phi3 / ...)
"""

import os
import sys
import json
import time
import urllib.request

import numpy as np

BASE = os.environ.get('SAGE_OLLAMA', 'http://localhost:11434')
EMBED_MODEL = 'nomic-embed-text'
LLM_MODEL = os.environ.get('SAGE_LLM', 'mistral')
EMBED_DIM = 768

_NOCOLOR = bool(os.environ.get('NO_COLOR'))


# ---- terminal UI (ASCII text; ANSI colour, auto-off via NO_COLOR) ----------
class C:
    R = '' if _NOCOLOR else '\033[0m'
    DIM = '' if _NOCOLOR else '\033[2m'
    B = '' if _NOCOLOR else '\033[1m'
    CYAN = '' if _NOCOLOR else '\033[96m'
    GREEN = '' if _NOCOLOR else '\033[92m'
    YELLOW = '' if _NOCOLOR else '\033[93m'
    RED = '' if _NOCOLOR else '\033[91m'
    MAG = '' if _NOCOLOR else '\033[95m'
    BLUE = '' if _NOCOLOR else '\033[94m'


def banner(title, sub=''):
    line = '=' * 72
    print('\n' + C.CYAN + line + C.R)
    print(C.CYAN + C.B + '  ' + title + C.R)
    if sub:
        print(C.DIM + '  ' + sub + C.R)
    print(C.CYAN + line + C.R)


def rule():
    print(C.DIM + '-' * 72 + C.R)


def badge(on):
    return (C.GREEN + '[LLM: ON ]' + C.R) if on else (C.RED + '[LLM: OFF]' + C.R)


def say(role, text, col):
    print('%s%-9s%s %s' % (col, role + ':', C.R, text))


# ---- Ollama -----------------------------------------------------------------
def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def embed(texts):
    """Embed a str or list[str] -> unit-norm np.float64 array (N, EMBED_DIM)."""
    one = isinstance(texts, str)
    items = [texts] if one else list(texts)
    payload = json.dumps({'model': EMBED_MODEL, 'input': items}).encode('utf-8')
    req = urllib.request.Request(BASE + '/api/embed', data=payload,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        embs = json.loads(r.read().decode('utf-8'))['embeddings']
    arr = _unit(np.asarray(embs, dtype=np.float64))
    return arr[0] if one else arr


def chat(system, user, history=None, timeout=120):
    """Single-turn chat with optional history. Returns (text, seconds).
    Raises on connection failure (caller decides how to fall back)."""
    msgs = [{'role': 'system', 'content': system}]
    for h in (history or []):
        msgs.append(h)
    msgs.append({'role': 'user', 'content': user})
    payload = json.dumps({'model': LLM_MODEL, 'messages': msgs,
                          'stream': False}).encode('utf-8')
    req = urllib.request.Request(BASE + '/api/chat', data=payload,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode('utf-8'))
    return out['message']['content'].strip(), time.time() - t0


def ollama_up():
    try:
        with urllib.request.urlopen(BASE + '/api/tags', timeout=5) as r:
            json.loads(r.read().decode('utf-8'))
        return True
    except Exception:
        return False


def require_ollama(need_llm=True):
    if not ollama_up():
        print(C.RED + "Ollama is not reachable at %s." % BASE + C.R)
        print("Start it with:  ollama serve")
        print("Pull models:    ollama pull %s" % EMBED_MODEL +
              (" ; ollama pull %s" % LLM_MODEL if need_llm else ""))
        sys.exit(1)


# ---- SAGE memory that stores TEXT (faithful to core/agent_memory.SAGEMemory) -
class SageMemory:
    """Gradient-free bounded associative memory storing text payloads.

    Mechanism (identical in spirit to core.agent_memory.SAGEMemory): cosine
    nearest-slot addressing; a write to an existing key (cos > merge) CONSOLIDATES
    into that slot (EMA on the key, replace text); a new key allocates a free slot
    or evicts the weakest (lowest strength); all strengths DECAY each write so
    stale memories fade. This is a vector store + merge/decay/evict - shown working,
    not claimed superior.
    """
    def __init__(self, n_slots=256, dim=EMBED_DIM, merge=0.62, lr_key=0.3,
                 decay=0.999):
        self.B = n_slots; self.dim = dim
        self.merge = merge; self.lr_key = lr_key; self.decay = decay
        self.key = np.zeros((n_slots, dim))
        self.text = [None] * n_slots
        self.strength = np.zeros(n_slots)
        self.used = np.zeros(n_slots, dtype=bool)

    def _nearest(self, k):
        if not self.used.any():
            return None, -1.0
        u = np.where(self.used)[0]
        sims = self.key[u] @ _unit(k)
        j = int(np.argmax(sims))
        return int(u[j]), float(sims[j])

    def write(self, k, text):
        """Store text under key vector k. Returns (action, slot, sim) for display."""
        k = _unit(k)
        self.strength[self.used] *= self.decay
        slot, best = self._nearest(k)
        if slot is not None and best > self.merge:            # consolidate same key
            self.key[slot] = _unit((1 - self.lr_key) * self.key[slot] + self.lr_key * k)
            self.text[slot] = text
            self.strength[slot] += 1.0
            return ('consolidate', slot, best)
        if not self.used.all():                               # free slot
            slot = int(np.where(~self.used)[0][0]); action = 'new'
        else:                                                 # evict weakest
            slot = int(np.argmin(self.strength)); action = 'evict'
        self.key[slot] = k; self.text[slot] = text
        self.strength[slot] = 1.0; self.used[slot] = True
        return (action, slot, best)

    def recall(self, k, topk=3, threshold=0.0):
        """Return up to topk (text, sim) above threshold, most-similar first."""
        if not self.used.any():
            return []
        u = np.where(self.used)[0]
        sims = self.key[u] @ _unit(k)
        order = np.argsort(-sims)[:topk]
        return [(self.text[int(u[o])], float(sims[o])) for o in order
                if sims[o] >= threshold]

    def footprint(self):
        return int(self.used.sum())

    def dump(self):
        return [(self.text[s], float(self.strength[s]))
                for s in np.argsort(-self.strength) if self.used[s]]

    # ---- persistence (so a demo "remembers across sessions") ----
    def save(self, path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        u = np.where(self.used)[0]
        np.savez(path, key=self.key[u], strength=self.strength[u],
                 text=np.array([self.text[s] for s in u], dtype=object),
                 cfg=np.array([self.B, self.dim, self.merge, self.lr_key,
                               self.decay], dtype=object))

    @classmethod
    def load(cls, path, n_slots=256):
        if not os.path.exists(path):
            return cls(n_slots=n_slots)
        z = np.load(path, allow_pickle=True)
        B, dim, merge, lr_key, decay = z['cfg']
        m = cls(n_slots=int(B), dim=int(dim), merge=float(merge),
                lr_key=float(lr_key), decay=float(decay))
        n = len(z['strength'])
        m.key[:n] = z['key']; m.strength[:n] = z['strength']
        for i, t in enumerate(z['text']):
            m.text[i] = str(t)
        m.used[:n] = True
        return m
