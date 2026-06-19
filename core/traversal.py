"""
core/traversal.py - graph traversal as the composition primitive (Stage 4).

The GO/NO-GO mechanism. A relation ("edge type") is represented gradient-free by
a PROTOTYPE DIRECTION: the mean of (emb[tail] - emb[head]) over TRAINING pairs of
that relation - averaging only, no learned parameters. Composition options:

  arithmetic_analogy : Method A  - single-pair parallelogram b-a+c, global nearest
  prototype_analogy  : Method A' - c + dir_R, global nearest  (averaged relation)
                       Method B  - c + dir_R, restricted to c's kNN neighbours
                                   (= follow the typed edge along the graph)
  multi_hop_traverse : chain typed hops, RE-GROUNDING on a real node each hop
  multi_hop_arithmetic: baseline - sum offsets, one global nearest (drifts)
  dijkstra           : typed shortest path over the labelled kNN graph

THESIS GUARD: tensor ops + averaging + a heap-based shortest path. No autograd,
no optimizer, no gradient descent anywhere.
"""

import heapq
import torch
import torch.nn.functional as F


def relation_prototype(pairs, emb):
    """Mean unit direction (tail - head) over TRAINING pairs of one relation.
    pairs: list of (head_idx, tail_idx). Returns (D,) unit vector or None."""
    if not pairs:
        return None
    h = torch.tensor([p[0] for p in pairs], device=emb.device)
    t = torch.tensor([p[1] for p in pairs], device=emb.device)
    return F.normalize((emb[t] - emb[h]).mean(0), p=2, dim=0)


def _mask_exclude(sims, pool, exclude, emb):
    if exclude:
        ex = torch.tensor(sorted({int(x) for x in exclude}), device=emb.device)
        sims = sims.masked_fill(torch.isin(pool, ex), -2.0)
    return sims


def arithmetic_analogy(a, b, c, emb, candidates=None, topk=5, exclude=()):
    """3CosAdd parallelogram (b - a + c). candidates=None -> global nearest
    (Method A); candidates=neighbour idx -> nearest within the kNN pool (A_knn,
    the same-pool control for Method B)."""
    v = F.normalize(emb[b] - emb[a] + emb[c], p=2, dim=0)
    pool = (torch.arange(emb.shape[0], device=emb.device)
            if candidates is None else candidates)
    sims = _mask_exclude(emb[pool] @ v, pool, exclude, emb)
    return pool[torch.topk(sims, min(topk, pool.shape[0])).indices]


def neighbors(c, emb, k):
    """Top-k cosine neighbours of node c (its kNN graph neighbourhood)."""
    sims = emb @ emb[c]
    sims[c] = -2.0
    return torch.topk(sims, min(k, emb.shape[0])).indices


def prototype_analogy(c, dir_R, emb, candidates=None, topk=5, exclude=()):
    """Method A' (candidates=None -> global) / Method B (candidates=neighbour idx).
    From c, rank targets by alignment of (target - c) with the relation prototype."""
    pool = (torch.arange(emb.shape[0], device=emb.device)
            if candidates is None else candidates)
    offs = F.normalize(emb[pool] - emb[c], p=2, dim=1)
    sims = _mask_exclude(offs @ dir_R, pool, exclude, emb)
    return pool[torch.topk(sims, min(topk, pool.shape[0])).indices]


def multi_hop_traverse(start, dirs, emb, k, exclude=()):
    """Chain typed hops, RE-GROUNDING on a real graph node each hop.
    dirs: list of relation prototypes. Returns the final landed node index."""
    cur = start
    visited = {start} | set(int(x) for x in exclude)
    for dir_R in dirs:
        cand = neighbors(cur, emb, k)
        offs = F.normalize(emb[cand] - emb[cur], p=2, dim=1)
        sims = offs @ dir_R
        bad = torch.tensor([int(x) in visited for x in cand.tolist()],
                           device=emb.device)
        if bool(bad.all()):
            break                       # dead end: every neighbour visited
        sims = sims.masked_fill(bad, -2.0)
        cur = int(cand[torch.argmax(sims)])
        visited.add(cur)
    return cur


def beam_traverse(start, dirs, emb, k, beam_width=5):
    """Multi-hop with a BEAM (soft commitment): keep the top `beam_width` partial
    paths by cumulative directional alignment, expanding within each node's kNN.
    Returns the best final node. Avoids the greedy single-hop error compounding
    that makes plain re-grounding fragile. Gradient-free."""
    beams = [(0.0, start, frozenset([start]))]      # (cum_score, node, visited)
    for dir_R in dirs:
        cands = []
        for score, cur, vis in beams:
            nb = neighbors(cur, emb, k)
            al = (F.normalize(emb[nb] - emb[cur], p=2, dim=1) @ dir_R)
            order = torch.argsort(al, descending=True)
            taken = 0
            for idx in order.tolist():
                node = int(nb[idx])
                if node in vis:
                    continue
                cands.append((score + float(al[idx]), node, vis | {node}))
                taken += 1
                if taken >= beam_width:
                    break
        if not cands:
            break
        cands.sort(key=lambda x: x[0], reverse=True)
        beams = cands[:beam_width]
    return beams[0][1] if beams else start


def multi_hop_arithmetic(start, dirs, emb, exclude=()):
    """Baseline multi-hop: sum the prototype offsets, ONE global nearest (drifts -
    no re-grounding on intermediate nodes)."""
    v = emb[start].clone()
    for dir_R in dirs:
        v = v + dir_R
    v = F.normalize(v, p=2, dim=0)
    sims = emb @ v
    for x in ({start} | set(int(x) for x in exclude)):
        sims[x] = -2.0
    return int(torch.argmax(sims))


def dijkstra(adj, start, allowed_types=None):
    """Typed shortest path over a labelled graph (gradient-free).
    adj: dict node -> list of (neighbour, weight, frozenset_of_relation_types).
    allowed_types: if given, only edges carrying one of these relation types may
    be traversed. Returns dist dict {node: cost} of reachable nodes.
    """
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w, types in adj.get(u, ()):
            if allowed_types is not None and types.isdisjoint(allowed_types):
                continue
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist
