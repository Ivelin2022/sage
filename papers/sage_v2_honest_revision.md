# SAGE arXiv paper — honest v2 revision draft

Prepared after a full adversarial re-test of every SAGE claim against strong
classical baselines (see `SPHERE/sage_sphere/findings.md`). The goal of v2 is a
paper you can **defend** under peer review, not one a reviewer can dismantle with a
single missing baseline. Everything here is constructive: the corrected version is
still a real contribution — just a more modest and more credible one.

The single change that fixes most of the paper: **add the non-parametric baselines
(NCM / k-means / bounded dedup store) and stop attributing the results to "the
geometry."**

---

## 1. Suggested new title / framing

v1: *"Memory for All — SAGE: ... The geometry computes. The weights are not needed."*

v2 (honest): **"SAGE: A Weight-Free Bounded Associative Memory for Continual,
Gradient-Free Knowledge Storage"**

Drop "the geometry computes" entirely. The paper's own §5.3 already concedes the
3-D grid is "a storage addressing system," not the compute — retrieval is cosine
over the full d-dimensional embedding. The headline should match the body.

---

## 2. Corrected abstract (drop-in replacement)

> Updating an AI system's knowledge usually means retraining, which is costly and
> causes catastrophic forgetting, because knowledge lives in weight matrices that
> cannot be edited locally. This paper presents SAGE, a **weight-free, gradient-free
> associative memory**: knowledge is stored as embedding vectors at fixed slots,
> retrieved by cosine similarity, and written by local Hebbian updates — no
> backpropagation at any stage. SAGE supports continuous insertion without
> retraining and, because writes are local, does not catastrophically forget:
> against an SGD-trained MLP it retains far more of an earlier task (forgetting
> 0.012 vs 0.150).
>
> We evaluate SAGE **against the appropriate non-parametric baselines** —
> nearest-class-mean (NCM), online/per-class k-means, and a bounded dedup vector
> store — not only against neural networks. The honest result: SAGE's no-forgetting
> property is **real but shared** with these simple methods, and on real MNIST
> features a storage-matched per-class k-means **matches or beats** SAGE
> (90.8% vs 86.2%). SAGE thus equals a well-built bounded associative memory; it
> does not exceed one. We further show that the 3-D geometry is an addressing/
> visualization layer, not the computational mechanism (retrieval uses the full
> embedding), and we situate SAGE precisely within the associative-memory
> literature: its read is equivalent to attention / modern Hopfield retrieval, its
> merge-write to the delta-rule fast-weight update, and its decay to palimpsest
> forgetting. SAGE's contribution is a clean, interpretable, weight-free
> *instantiation* of bounded continual associative memory for settings where
> backpropagation is unavailable or undesirable (edge / neuromorphic / streaming),
> not a new memory mechanism or a method that outperforms standard memories.

---

## 3. Claim-by-claim corrections

| v1 claim | Problem (verified today) | v2 wording |
|---|---|---|
| "The geometry computes; weights not needed" | Geometry is decoration; cosine over 768-D does the work (your own §5.3). | "The 3-D grid is an addressing/viz layer; retrieval is cosine over the full embedding." |
| "Forgets 92% less than neural networks (0.012 vs 0.150)" | True **vs an MLP**, but the real competitor is NCM/k-means, which also don't forget — and beat SAGE. | Keep the number, but add: "this no-forget property is shared by, and on MNIST bettered by, NCM/k-means (see baseline table)." |
| "0.000% sparsity activation / infinite efficiency" | Retrieval **scans all N³ slots** (your own retrieval eq) — dense compute, not sparse. | Remove the efficiency claim. State: "only one slot is the *answer*, but retrieval computes cosine against all slots (O(N³d) per query)." |
| "100% rollout accuracy" (SAGESequenceCube) | Achieved by an **explicit dictionary** (s_idx→o_idx), which the paper concedes. | "Rollout is exact because transitions use an explicit dictionary; the geometry provides only the retrieval index." |
| "58.3% analogy, closes 62% of the gap" | This is GloVe's own `a−b+c` arithmetic; the cube can only degrade it. | "Analogy uses standard vector arithmetic on GloVe vectors; SAGE inherits (and slightly degrades) GloVe's analogy ability — it is not a new capability." |
| "Perfect retention over 200 updates" | Trivially true for any non-overwriting store (dict, NCM, vector DB). | "Like any non-overwriting store, SAGE retains all inserted items; this is a property of bounded associative memory, not unique to SAGE." |
| "MultiCube: horizontal scaling" (novel) | This is standard sharding / IVF clustering of an index. | "MultiCube partitions the index by domain — a standard sharding strategy that keeps per-shard density constant." |
| "First weight-free implementation of CLS theory" | A biological-analogy framing claim; fine as framing, not as a performance result. | Keep, but label clearly as an architectural analogy, not an empirical advantage. |

---

## 4. New results section — add the missing baselines

Replace "SAGE vs MLP only" with a table that includes the non-parametric methods.
Numbers below are from today's re-runs (`results/forgetting_benchmark*.json`).

**Continual / catastrophic forgetting (class-incremental).**

| Method | sklearn digits (final acc) | real MNIST PCA-64 (final acc) | forgets? |
|---|---|---|---|
| SGD-MLP (neural) | 19.8 | 19.7 | yes (cliff) |
| Replay (neural+buffer) | 87.4 | 82.9 | partial |
| **NCM** (1 mean/class) | 87.0 | 79.7 | no |
| **NCM-multi / per-class k-means** | — | **90.8** | no |
| **SAGE (flat / per-class)** | 85.3 | 86.2 | no |
| SAGE-grid (3-D geometry) | 35.5 | 18.1 | collapses |

Honest reading: SAGE **beats the neural baselines** (SGD/Replay) on forgetting —
the real, true headline — but **ties NCM at digits scale and loses to a
storage-matched per-class k-means on MNIST** (86.2 vs 90.8; per-item McNemar
p ≈ 4×10⁻¹³). The 3-D "grid" variant collapses, confirming the geometry is not
load-bearing. State all three facts.

**As a memory store** (agent-memory experiment): SAGE-flat **ties** a well-built
bounded dedup vector store and **beats** naive unbounded/FIFO vector-DB usage. So
the honest claim is "SAGE = a good bounded store," not "SAGE > vector stores."

---

## 5. Honest positioning paragraph (new related-work text)

> SAGE is best understood as a member of the bounded online associative-memory
> family. Its retrieval (cosine + softmax over stored vectors) is the modern
> Hopfield / attention operation (Ramsauer et al., 2021); its merge-on-write update
> is a hard form of the delta-rule fast-weight write (Schlag, Irie & Schmidhuber,
> 2021); and its slot decay corresponds to palimpsest forgetting in Hopfield
> memories (Storkey & Valabregue). At an equal memory budget, a dense Hopfield layer
> has exponentially higher capacity than SAGE's one-vector-per-slot scheme. SAGE
> therefore does not introduce a new retrieval mechanism; its value is as a
> transparent, weight-free, gradient-free *instantiation* of bounded continual
> associative memory, with directly inspectable slots, for regimes where
> backpropagation is unavailable or undesirable.

---

## 6. What is genuinely yours and defensible (keep these)

- **Weight-free, gradient-free continuous insertion** with no retraining — real and
  clearly demonstrated.
- **Beats neural nets (SGD/Replay) on catastrophic forgetting** — real and
  benchmarked. (Frame as "matches the gradient-free no-forget methods and beats the
  gradient-based ones," not "uniquely avoids forgetting.")
- **A clean, small, interpretable implementation** of a bounded continual
  associative memory + a hippocampal-style consolidation analogy — a legitimate
  systems/engineering contribution.
- **The drone proof-of-concept**: graceful degradation when the LLM is offline is a
  sensible, true systems property — just add that a plain local vector store gives
  the same fallback (so the contribution is the integrated demo, not uniqueness).

---

## 7. The honest one-line summary for the paper's conclusion

> SAGE is a clean, weight-free, gradient-free instantiation of bounded continual
> associative memory. It avoids catastrophic forgetting and supports retraining-free
> updates — properties it shares with, and does not exceed, simple non-parametric
> methods such as NCM and per-class k-means. Its 3-D geometry is an addressing and
> visualization layer, not the computational mechanism. The contribution is a
> transparent realization and an honest characterization, not a new memory
> mechanism or a method that outperforms standard memories.
