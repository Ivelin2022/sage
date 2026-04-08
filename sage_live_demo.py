"""
SAGE Live Demo — Multi-Cube System with Real Embeddings
=======================================================
Author: Ivelin Likov

A live interactive SAGE memory system using:
  - nomic-embed-text via Ollama for real 768d semantic embeddings
  - SpatialCubeV3Torch (with Langevin Force 5) for storage
  - MultiCube: 4 specialist cubes (facts, relations, context, goals)
  - Residual query chaining across cubes
  - NLerp parallel composition

Commands:
  store <cube> <text>    — store a fact in a specialist cube
                           cubes: facts / relations / context / goals
  query <text>           — query all cubes and compose results
  chain <text>           — sequential: query facts → use output to query relations
  status                 — show cube utilisation and stored concept count
  similar <text>         — find the top-5 most similar stored memories
  forget test            — insert noise and test forgetting
  quit                   — exit

Requires: ollama running locally with nomic-embed-text model
  ollama pull nomic-embed-text

Usage:
  python sage_live_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import math
from datetime import datetime

# ── Import V3 cube ───────────────────────────────────────────────────────────
try:
    from cube_core_v3_torch import SpatialCubeV3Torch as CubeClass
    CUBE_VERSION = 'V3 (Langevin)'
except ImportError:
    try:
        from cube_core_v2_torch import SpatialCubeV2Torch as CubeClass
        CUBE_VERSION = 'V2 (fallback)'
    except ImportError:
        print("ERROR: cube_core_v3_torch.py or cube_core_v2_torch.py not found.")
        print("Copy them to the same folder as this script.")
        sys.exit(1)

# ── Ollama embedding ─────────────────────────────────────────────────────────
try:
    import requests
    OLLAMA_URL = 'http://localhost:11434/api/embeddings'
    EMBED_MODEL = 'nomic-embed-text'

    def get_embedding(text: str) -> torch.Tensor:
        """Get nomic-embed-text embedding from Ollama."""
        resp = requests.post(OLLAMA_URL, json={
            'model': EMBED_MODEL,
            'prompt': text
        }, timeout=30)
        resp.raise_for_status()
        vec = torch.tensor(resp.json()['embedding'], dtype=torch.float32)
        return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)

    # Test connection
    _test = get_embedding('test')
    EMBED_DIM = len(_test)
    print(f"Ollama connected — {EMBED_MODEL} | dim={EMBED_DIM}")
    USE_OLLAMA = True

except Exception as e:
    print(f"Ollama not available ({e})")
    print("Falling back to random embeddings (64d) for testing.\n")
    EMBED_DIM = 64
    USE_OLLAMA = False

    def get_embedding(text: str) -> torch.Tensor:
        """Deterministic fake embedding based on text hash."""
        torch.manual_seed(hash(text) % (2**31))
        vec = torch.randn(EMBED_DIM)
        return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)


# ── Config ───────────────────────────────────────────────────────────────────
CUBE_SIZE    = 32     # 32³ = 32,768 points
LANGEVIN_T   = 0.01   # confirmed improvement in Test 7
ALPHA        = 0.05
TEMPERATURE  = 0.05
CHAIN_LAMBDA = 0.5    # mixing coefficient for sequential chaining

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Cube: {CUBE_VERSION} | Embed: {EMBED_DIM}d")


# ── MultiCube system ─────────────────────────────────────────────────────────

# Fixed seeds per cube — hash() is non-deterministic across Python sessions
CUBE_SEEDS = {'facts': 1001, 'relations': 1002, 'context': 1003, 'goals': 1004}
CUBE_NAMES = ['facts', 'relations', 'context', 'goals']
CUBE_DESCRIPTIONS = {
    'facts':     'Factual knowledge — what things are',
    'relations': 'Relational knowledge — how things connect',
    'context':   'Contextual memory — when and where',
    'goals':     'Goals and intentions — what to do',
}


class MultiCubeSAGE:
    def __init__(self):
        self.cubes = {}
        self.memories = {}  # cube_name -> list of {'text': str, 'vec': tensor, 'grid_idx': int}
        self._embed_cache = {}  # text -> tensor, avoids repeat Ollama calls

        for name in CUBE_NAMES:
            self.cubes[name] = CubeClass(
                cube_size=CUBE_SIZE,
                embed_dim=EMBED_DIM,
                seed=CUBE_SEEDS[name],
                device=str(device),
                langevin_T=LANGEVIN_T,
            ) if CUBE_VERSION.startswith('V3') else CubeClass(
                cube_size=CUBE_SIZE,
                embed_dim=EMBED_DIM,
                seed=CUBE_SEEDS[name],
                device=str(device),
            )
            self.memories[name] = []

        print(f"\nMultiCube ready — {len(CUBE_NAMES)} specialist cubes × {CUBE_SIZE}³ points")
        print(f"Total capacity: {len(CUBE_NAMES) * CUBE_SIZE**3:,} grid points\n")

    def store(self, cube_name: str, text: str, verbose=True) -> dict:
        """Store a text in a specialist cube."""
        if cube_name not in self.cubes:
            return {'error': f'Unknown cube: {cube_name}. Choose from: {CUBE_NAMES}'}

        # Deduplication: skip if exact text already stored in this cube
        existing = [m for m in self.memories[cube_name] if m['text'] == text]
        if existing:
            if verbose:
                print(f"  ↩ Already in [{cube_name}] | grid pos {existing[0]['grid_idx']}")
            return {'cube': cube_name, 'grid_idx': existing[0]['grid_idx'],
                    'util': len(self.memories[cube_name]) / self.cubes[cube_name].n_points * 100,
                    'duplicate': True}

        t0 = time.perf_counter()
        # Use cache: avoid re-embedding the same text twice in a session
        vec = self._embed_cache.get(text)
        if vec is None:
            vec = get_embedding(text)
            self._embed_cache[text] = vec
        embed_time = time.perf_counter() - t0

        cube = self.cubes[cube_name]
        pairs = [(vec, vec)]

        # Train for 50 steps — enough to push embedding to distinct grid position
        # 10 steps only achieves ~22% movement toward target at 768d;
        # 50 steps achieves ~72% — needed to avoid grid collisions
        for _ in range(50):
            cube.learn_batch(pairs, alpha=ALPHA, beta=0.002,
                             teach_directions=False, momentum=0.9, neg_weight=0.1)

        # Find where it landed — with collision detection
        sims = cube.embeddings @ vec.to(device)
        grid_idx = sims.argmax().item()

        # Collision detection: if this position already has a different memory,
        # find next best unoccupied position to avoid silent overwrites
        occupied = {m['grid_idx'] for m in self.memories[cube_name]}
        if grid_idx in occupied:
            # Find top-20 candidates, pick first unoccupied
            top20 = sims.topk(20).indices.tolist()
            for candidate in top20:
                if candidate not in occupied:
                    grid_idx = candidate
                    break
            else:
                # All top-20 occupied — train more steps to push to new position
                for _ in range(50):
                    cube.learn_batch(pairs, alpha=ALPHA * 2, beta=0.002,
                                     teach_directions=False, momentum=0.9, neg_weight=0.1)
                sims = cube.embeddings @ vec.to(device)
                grid_idx = sims.argmax().item()

        # Label the point
        label = text[:30] + ('...' if len(text) > 30 else '')
        cube.labels[grid_idx] = label

        self.memories[cube_name].append({
            'text': text,
            'vec': vec,
            'grid_idx': grid_idx,
            'time': datetime.now().isoformat(),
        })

        util = len(self.memories[cube_name]) / cube.n_points * 100

        if verbose:
            print(f"  ✓ Stored in [{cube_name}] | grid pos {grid_idx} | "
                  f"{util:.2f}% util | embed {embed_time*1000:.0f}ms")

        return {'cube': cube_name, 'grid_idx': grid_idx, 'util': util}

    def retrieve_from_cube(self, cube_name: str, query_vec: torch.Tensor) -> dict:
        """Retrieve from a single cube."""
        cube = self.cubes[cube_name]
        q = F.normalize(query_vec.unsqueeze(0), p=2, dim=1).squeeze(0).to(device)
        sims = cube.embeddings @ q
        scores = F.softmax(sims / TEMPERATURE, dim=0)
        response = F.normalize((scores.unsqueeze(1) * cube.embeddings).sum(0, keepdim=True),
                                p=2, dim=1).squeeze(0)
        top1_idx = sims.argmax().item()
        top1_score = sims.max().item()

        # Find best matching stored memory by comparing QUERY vec against stored vecs.
        # Using query_vec (not blended response) gives cleaner semantic match:
        # "which stored memory is most relevant to this query?"
        best_memory = None
        best_sim = -1
        for mem in self.memories[cube_name]:
            s = F.cosine_similarity(query_vec.unsqueeze(0),
                                    mem['vec'].unsqueeze(0)).item()
            if s > best_sim:
                best_sim = s
                best_memory = mem

        return {
            'response_vec': response,
            'top1_idx': top1_idx,
            'top1_score': top1_score,
            'best_memory': best_memory,
            'cosine': best_sim,
        }

    def query_all(self, text: str) -> dict:
        """
        Parallel composition — query all cubes simultaneously.
        Uses NLerp (confirmed best in Test 8C) to merge outputs.
        """
        query_vec = self._embed_cache.get(text) or get_embedding(text)
        results = {}
        response_vecs = []

        for name in CUBE_NAMES:
            if not self.memories[name]:
                results[name] = {'response_vec': None, 'top1_idx': None,
                                 'top1_score': None, 'best_memory': None, 'cosine': None}
                continue
            r = self.retrieve_from_cube(name, query_vec)
            results[name] = r
            response_vecs.append(r['response_vec'])

        if not response_vecs:
            return {'error': 'No memories stored yet.', 'results': {}}

        # NLerp composition — geodesic average
        composed = F.normalize(torch.stack(response_vecs).sum(0, keepdim=True),
                                p=2, dim=1).squeeze(0)

        # Find best matching memory across ALL cubes
        best_overall = None
        best_score = -1
        for name, r in results.items():
            if r['best_memory'] and r['cosine'] > best_score:
                best_score = r['cosine']
                best_overall = {'cube': name, **r['best_memory'], 'cosine': r['cosine']}

        return {
            'query': text,
            'composed_vec': composed,
            'per_cube': results,
            'best_overall': best_overall,
        }

    def chain_query(self, text: str) -> dict:
        """
        Sequential composition — residual query chaining.
        facts → (mix output into query) → relations
        Confirmed mechanism from Test 8A research.
        """
        query_vec = self._embed_cache.get(text) or get_embedding(text)

        steps = []

        # Step 1: Query facts cube
        r_facts = self.retrieve_from_cube('facts', query_vec)
        steps.append({'cube': 'facts', **r_facts})

        # Step 2: Mix facts output into query (residual chain)
        cos_sim = F.cosine_similarity(query_vec.unsqueeze(0),
                                       r_facts['response_vec'].cpu().unsqueeze(0)).item()
        effective_lambda = CHAIN_LAMBDA * max(0.0, cos_sim)
        chained_query = F.normalize(
            query_vec + effective_lambda * r_facts['response_vec'].cpu(),
            p=2, dim=0
        )

        # Step 3: Query relations with modified query
        r_relations = self.retrieve_from_cube('relations', chained_query)
        steps.append({'cube': 'relations (chained)', **r_relations})

        # Step 4: Mix into context
        cos_sim2 = F.cosine_similarity(chained_query.unsqueeze(0),
                                        r_relations['response_vec'].cpu().unsqueeze(0)).item()
        effective_lambda2 = CHAIN_LAMBDA * max(0.0, cos_sim2)
        chained_query2 = F.normalize(
            chained_query + effective_lambda2 * r_relations['response_vec'].cpu(),
            p=2, dim=0
        )

        r_context = self.retrieve_from_cube('context', chained_query2)
        steps.append({'cube': 'context (chained)', **r_context})

        return {
            'query': text,
            'chain_lambda': effective_lambda,
            'steps': steps,
        }

    def find_similar(self, text: str, top_k=5) -> list:
        """Find most similar stored memories across all cubes."""
        query_vec = self._embed_cache.get(text) or get_embedding(text)
        all_memories = []

        for name in CUBE_NAMES:
            for mem in self.memories[name]:
                sim = F.cosine_similarity(query_vec.unsqueeze(0),
                                           mem['vec'].unsqueeze(0)).item()
                all_memories.append({'cube': name, 'text': mem['text'],
                                     'similarity': sim})

        all_memories.sort(key=lambda x: x['similarity'], reverse=True)
        return all_memories[:top_k]

    def status(self) -> dict:
        """Show cube utilisation."""
        status = {}
        for name in CUBE_NAMES:
            n_stored = len(self.memories[name])
            util = n_stored / self.cubes[name].n_points * 100
            status[name] = {'stored': n_stored, 'util_pct': util,
                             'description': CUBE_DESCRIPTIONS[name]}
        return status

    def forget_test(self) -> dict:
        """Insert noise and measure forgetting."""
        # Get baseline
        before = {}
        for name in CUBE_NAMES:
            if not self.memories[name]:
                continue
            sims = []
            cube = self.cubes[name]
            for mem in self.memories[name][:5]:
                q = mem['vec'].to(device)
                s_all = cube.embeddings @ q
                scores = F.softmax(s_all / TEMPERATURE, dim=0)
                resp = F.normalize((scores.unsqueeze(1) * cube.embeddings).sum(0, keepdim=True),
                                    p=2, dim=1).squeeze(0)
                sims.append(F.cosine_similarity(resp.unsqueeze(0).cpu(),
                                                 mem['vec'].unsqueeze(0)).item())
            before[name] = float(np.mean(sims)) if sims else 0.0

        # Insert noise — 50 batches × 16 vectors = 800 total noise insertions
        for name in CUBE_NAMES:
            cube = self.cubes[name]
            for _ in range(50):
                noise = F.normalize(torch.randn(16, EMBED_DIM), p=2, dim=1)
                pairs = [(noise[i], noise[i]) for i in range(16)]
                cube.learn_batch(pairs, alpha=ALPHA, beta=0.002,
                                 teach_directions=False, momentum=0.9, neg_weight=0.1)

        # Measure after
        after = {}
        for name in CUBE_NAMES:
            if not self.memories[name]:
                continue
            sims = []
            cube = self.cubes[name]
            for mem in self.memories[name][:5]:
                q = mem['vec'].to(device)
                s_all = cube.embeddings @ q
                scores = F.softmax(s_all / TEMPERATURE, dim=0)
                resp = F.normalize((scores.unsqueeze(1) * cube.embeddings).sum(0, keepdim=True),
                                    p=2, dim=1).squeeze(0)
                sims.append(F.cosine_similarity(resp.unsqueeze(0).cpu(),
                                                 mem['vec'].unsqueeze(0)).item())
            after[name] = float(np.mean(sims)) if sims else 0.0

        return {'before': before, 'after': after,
                'forgetting': {k: before.get(k, 0) - after.get(k, 0)
                               for k in CUBE_NAMES}}


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_header():
    print("\n" + "="*60)
    print("  SAGE Live Demo — Multi-Cube Geometric Memory")
    print(f"  {CUBE_VERSION} | {EMBED_DIM}d embeddings | {device}")
    print("="*60)
    print("  Commands:")
    print("    store <cube> <text>    — store in facts/relations/context/goals")
    print("    query <text>           — parallel query all cubes (NLerp)")
    print("    chain <text>           — sequential chain: facts→relations→context")
    print("    similar <text>         — find top-5 similar memories")
    print("    status                 — cube utilisation")
    print("    forget test            — noise insertion forgetting test")
    print("    quit                   — exit")
    print("="*60 + "\n")


def run_demo():
    print_header()
    sage = MultiCubeSAGE()

    # Pre-load a few examples if no Ollama
    if not USE_OLLAMA:
        print("(No Ollama — storing demo facts with fake embeddings)\n")
        examples = [
            ('facts', 'Paris is the capital of France'),
            ('facts', 'The Eiffel Tower is in Paris'),
            ('relations', 'France is a country in Europe'),
            ('relations', 'The Seine river flows through Paris'),
            ('context', 'Napoleon built many monuments in Paris'),
            ('goals', 'Visit the Louvre museum in Paris'),
        ]
        for cube, text in examples:
            sage.store(cube, text, verbose=True)
        print()

    while True:
        try:
            line = input("sage> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        parts = line.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == 'quit':
            print("Bye.")
            break

        elif cmd == 'status':
            status = sage.status()
            print()
            for name, s in status.items():
                bar = '█' * int(s['util_pct'] / 2) + '░' * (50 - int(s['util_pct'] / 2))
                print(f"  [{name:<10}] {s['stored']:>4} memories | "
                      f"{s['util_pct']:>5.2f}% | {bar[:20]}")
                print(f"              {s['description']}")
            print()

        elif cmd == 'store':
            if len(parts) < 3:
                print("  Usage: store <cube> <text>")
                print(f"  Cubes: {', '.join(CUBE_NAMES)}")
                continue
            cube_name = parts[1].lower()
            text = parts[2]
            result = sage.store(cube_name, text)
            if 'error' in result:
                print(f"  Error: {result['error']}")

        elif cmd == 'query':
            if len(parts) < 2:
                print("  Usage: query <text>")
                continue
            text = ' '.join(parts[1:])
            print(f"\n  Querying: '{text}'")
            t0 = time.perf_counter()
            result = sage.query_all(text)
            elapsed = time.perf_counter() - t0

            if 'error' in result:
                print(f"  {result['error']}")
                continue

            print(f"\n  ── Per-cube results ({elapsed*1000:.0f}ms) ──")
            for name, r in result['per_cube'].items():
                if r['best_memory']:
                    print(f"  [{name:<10}] score={r['top1_score']:.3f} | "
                          f"cos={r['cosine']:.3f} | \"{r['best_memory']['text'][:50]}\"")
                else:
                    print(f"  [{name:<10}] no memories")

            if result['best_overall']:
                b = result['best_overall']
                print(f"\n  ★ Best match: [{b['cube']}] cos={b['cosine']:.3f}")
                print(f"    \"{b['text']}\"")
            print()

        elif cmd == 'chain':
            if len(parts) < 2:
                print("  Usage: chain <text>")
                continue
            text = ' '.join(parts[1:])
            print(f"\n  Chaining: '{text}'")
            t0 = time.perf_counter()
            result = sage.chain_query(text)
            elapsed = time.perf_counter() - t0

            print(f"\n  ── Chain steps ({elapsed*1000:.0f}ms, λ={result['chain_lambda']:.3f}) ──")
            for step in result['steps']:
                if step['best_memory']:
                    print(f"  → [{step['cube']:<20}] cos={step['cosine']:.3f} | "
                          f"\"{step['best_memory']['text'][:45]}\"")
                else:
                    print(f"  → [{step['cube']:<20}] no memories")
            print()

        elif cmd == 'similar':
            if len(parts) < 2:
                print("  Usage: similar <text>")
                continue
            text = ' '.join(parts[1:])
            print(f"\n  Finding similar to: '{text}'")
            matches = sage.find_similar(text)
            if not matches:
                print("  No memories stored yet.")
            for i, m in enumerate(matches, 1):
                print(f"  {i}. [{m['cube']:<10}] sim={m['similarity']:.4f} | \"{m['text']}\"")
            print()

        elif cmd == 'forget' and len(parts) > 1 and parts[1] == 'test':
            print("\n  Running forgetting test (800 noise insertions)...")
            result = sage.forget_test()
            print("\n  Before vs After noise:")
            any_stored = False
            for name in CUBE_NAMES:
                b = result['before'].get(name)
                a = result['after'].get(name)
                if b is not None:
                    any_stored = True
                    f = result['forgetting'].get(name, 0)
                    print(f"  [{name:<10}] before={b:.4f}  after={a:.4f}  "
                          f"forgetting={f:.4f} {'✓ robust' if f < 0.15 else '✗ degraded'}")
            if not any_stored:
                print("  Store some memories first.")
            print()

        else:
            print(f"  Unknown command: '{cmd}'")
            print("  Type 'quit' to exit or see header for commands.\n")


if __name__ == '__main__':
    run_demo()
