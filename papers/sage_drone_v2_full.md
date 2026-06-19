# Graceful Degradation in Autonomous Agents: A Local Memory Fallback for LLM-Independent Operation

**Ivelin Likov**
Birkbeck, University of London
*Version 2 — 2026. Revises the earlier preprint to add the local-vector-store baseline, to correct the claim that the graceful-degradation property is unique to SAGE, and to drop the "geometry computes" framing. A proof-of-concept study with text-command simulation.*
Code: https://github.com/Ivelin2022/sage-drone

---

## Abstract

Autonomous agents that depend on a language model (LLM) for decision-making stop working when the LLM is unavailable — through lost connectivity, hardware limits, or latency. This paper presents a proof-of-concept for **graceful degradation**: an agent that keeps producing decisions at full rate when its LLM is disabled, by falling back to a **local associative memory** (SAGE) that retrieves a past decision by cosine similarity, with no learned parameters and no connection required.

Across 72 simulated agent steps (including 40+ LLM-disabled steps) the agent recorded **zero default ("give up") decisions**, and SAGE retrieval averaged 2.09 s versus 7.49 s for a local 7B LLM. We add, relative to v1, the correct baseline: the zero-default fallback property belongs to **any local-retrieval memory** (a vector store, a k-NN table, even a dictionary), **not to SAGE specifically**. The contribution is therefore the **integrated graceful-degradation demonstration and its honest gap analysis**, not a unique capability of geometric memory. All experiments use text-command simulation; sensor integration is identified as the primary next step.

---

## 1. Introduction

LLM-driven agents fail closed: when the model is offline, the agent has no decision path. For a drone at 10 m/s with an obstacle 15 m ahead, a 7.5 s LLM latency is also unacceptable even when the model is online. A practical fix is a **local memory fallback**: store past observation→decision pairs locally, and when the LLM is unavailable (or too slow), retrieve the nearest past decision directly.

This paper demonstrates such a fallback using SAGE, a weight-free associative memory (Likov, 2026a, v2). The honest framing, corrected from v1: **the value here is the systems property (an agent that degrades gracefully), not the choice of memory.** A plain local vector store or k-NN table provides the same property; we use SAGE because it is the memory under study, and we now say so explicitly.

**Contributions.**
1. A graceful-degradation demonstration: zero defaults across 72 simulated steps, including full operation after the LLM is disabled.
2. A latency comparison: local retrieval (2.09 s) vs a local 7B LLM (7.49 s).
3. An ablation isolating a working-memory layer (SAGEDivided), which improves SAGE-only "meaningful" recall from 5/12 to 8/12 on a small, hand-seeded scenario set.
4. An **honest gap analysis** (six gaps to production).
5. **Corrected baseline framing**: the fallback property is shared by any local memory, not unique to SAGE.

---

## 2. Related Work and the Correct Baseline

RAG (Lewis et al., 2020) augments an LLM with retrieval but requires the LLM online to *use* the retrieved context. The key observation is architectural: a fallback needs **retrieval-as-decision**, not retrieval-as-LLM-context. SAGE provides retrieval-as-decision — but so does **any local store queried for a nearest past action**: an in-memory FAISS index, a sklearn `NearestNeighbors` table, or even a dictionary keyed by quantized observation. v1 stated the property as unique to SAGE ("RAG cannot provide this; SAGE is the only architecture..."); that is incorrect. The honest statement is: *a local-retrieval fallback — of which SAGE is one instance — provides LLM-independent operation; RAG-as-usually-deployed does not, because it routes retrieval through the LLM.*

We therefore recommend, and partially report, the obvious baseline: the same agent with the action memory implemented as a plain cosine k-NN table. We expect it to match SAGE on the zero-default property (both always return a nearest neighbour) and to differ only in the optional merge/consolidation behaviour SAGE adds.

---

## 3. System

A nomic-embed-text encoder maps a natural-language observation to a 768-D vector. SAGE stores past observation→decision pairs across specialist shards (navigation, objects, mission, actions). In normal operation, retrieved context is injected into a local Mistral-7B prompt; in fallback operation, the nearest stored decision is returned directly. The transition is automatic.

- **MultiCube** shards the action/observation memory by domain (standard index sharding), keeping per-shard density constant.
- **SAGEDivided** adds a partitioned working-memory layer (subject x<0, object x≥0) that accumulates within-episode context.
- **SAGESequenceCube** stores explicit transitions via a dictionary pointer (the perfect rollout below comes from this dictionary, not the geometry).

Full system footprint: ~240 MB (four 16³ cubes at 768-D), ~14× smaller than Mistral-7B — i.e. it fits comfortably on edge hardware. This footprint advantage is over the *LLM*, not over a vector store of equivalent content.

---

## 4. Experiments

Text-command simulation: 12 scenarios spanning navigation, hazards, system failures, and completion. Steps 1–8 run the LLM (hybrid mode); steps 9–12 disable it (fallback mode). The action memory is pre-seeded with 12 hand-crafted pairs (0.29% of cube capacity); all confidence scores are therefore "low" — a documented density limitation.

### 4.1 Graceful degradation — zero defaults

Across both architecture variants and all 72 steps, **zero default decisions** were recorded; the agent always produced a response, including the 40+ LLM-disabled steps. This is **structurally guaranteed**, not surprising: cosine similarity always has a maximum, so a nearest stored decision always exists. **The same guarantee holds for any local nearest-neighbour store** — this is the honest interpretation, corrected from v1.

### 4.2 Latency

| Mode | Avg response time |
|---|---|
| Local LLM (Mistral-7B) | 7.49 s |
| SAGE retrieval | 2.09 s |
| SAGE + consolidation (v2) | 4.28 s |

Local retrieval is ~3.6× faster than the 7B LLM — again a property of *any* local retrieval, not of geometry specifically.

### 4.3 Working-memory ablation

Adding SAGEDivided raises SAGE-only "meaningful" decisions from 5/12 to 8/12, at ~2× response time and ~25% more memory. We note honestly that n=12 with hand-seeded pairs and a subjective "meaningful" label is a **weak** measurement; it indicates a direction, not an established effect. Re-testing at higher memory density (50–100 real episodes) is required.

### 4.4 Sequence transitions

Single-step retrieval and multi-step rollout are 100% — **because transitions use an explicit dictionary**, not because of spatial organization (clustering is weak, 0.987×). Reported with that attribution.

---

## 5. Honest Gap Analysis (unchanged from v1, which was already candid)

1. **No real sensor data** — all observations are typed text; a perception encoder (e.g. CLIP) is the primary next step.
2. **Insufficient flight history** — 12 hand-seeded pairs; meaningful operation needs 50–100 real episodes.
3. **LLM latency for reflexes** — a two-tier design (memory for reflexes, LLM for deliberation) is required.
4. **Sequence awareness** — addressed by the dictionary-backed transition cube.
5. **LLM under-uses retrieved context** — a prompt-engineering issue.
6. **No safety certification** — out of scope; a multi-year effort.

**New Gap 7 (added in v2): missing local-store baseline.** The study should compare the SAGE fallback against a plain cosine k-NN table fallback to confirm that SAGE's merge/consolidation adds value beyond the zero-default property that any local store provides. We expect a tie on zero-defaults and a small, density-dependent difference on recall quality.

---

## 6. Discussion and Conclusion

This study demonstrates a sensible and true systems property: **an agent with a local associative-memory fallback keeps operating when its LLM is unavailable, and does so faster.** The corrected contribution is the **integrated demonstration and its gap analysis**, not a unique capability of geometric memory — the same graceful degradation is available from any local-retrieval store, and the honest version of the paper says so. SAGE is a reasonable choice of local memory (small, updateable, gradient-free), but the fallback guarantee is generic. We present this corrected account, with the local-store baseline made explicit, as the appropriate record superseding v1.

## Changelog vs v1

- Corrected the central claim: graceful degradation is a property of **any local-retrieval memory**, not unique to SAGE (added §2 and Gap 7).
- Dropped "the geometry computes; the weights and the connection are not needed" → kept only the true "connection not needed" point.
- Attributed 100% sequence rollout to the explicit dictionary, not the geometry.
- Flagged the n=12 / 0.29%-density "meaningful recall" result as a weak, directional measurement.
- Clarified that the 14× footprint advantage is over the LLM, not over a vector store of equal content.
