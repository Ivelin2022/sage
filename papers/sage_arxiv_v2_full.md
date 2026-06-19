# SAGE: A Weight-Free Bounded Associative Memory for Continual, Gradient-Free Knowledge Storage

**Ivelin Likov**
Birkbeck, University of London
*Version 2 — 2026. This version revises the earlier preprint ("Memory for All: SAGE — Spatial Associative Geometric Embeddings") to add the non-parametric baselines that v1 omitted, to correct several over-stated claims, and to place SAGE precisely within the associative-memory literature.*
Code: https://github.com/Ivelin2022/sage

---

## Abstract

Updating an AI system's knowledge usually requires retraining, which is costly and causes catastrophic forgetting, because knowledge lives in weight matrices that cannot be edited locally. This paper presents **SAGE**, a **weight-free, gradient-free associative memory**: knowledge is stored as embedding vectors at fixed slots, retrieved by cosine similarity, and written by local Hebbian updates — no backpropagation at any stage. SAGE supports continuous insertion without retraining and, because writes are local, does not catastrophically forget.

The central methodological contribution of this version is **evaluation against the appropriate non-parametric baselines** — nearest-class-mean (NCM), online per-class k-means, and a bounded dedup vector store — not only against neural networks. The honest result is twofold. (i) SAGE **beats gradient-trained neural baselines** (an SGD-MLP and a replay-augmented MLP) on class-incremental forgetting, by a large margin, and is robust to task ordering. (ii) SAGE's no-forgetting property is **shared with, and not superior to, simple non-parametric methods**: it ties NCM on sklearn digits and, on real MNIST features, a storage-matched per-class k-means matches or beats it (90.8% vs 86.2%; per-item McNemar p≈4×10⁻¹³). As a bounded memory store, SAGE **ties a well-built dedup vector store** and beats naive unbounded/FIFO usage.

We further show, with controlled experiments, that SAGE's 3-D geometry is an **addressing and visualization layer, not the computational mechanism** (retrieval uses the full d-dimensional embedding), and we situate SAGE in the associative-memory literature: its read is the modern-Hopfield / attention operation, its merge-write is a hard delta-rule fast-weight update, and its slot decay is palimpsest forgetting. SAGE's contribution is therefore a **transparent, weight-free instantiation** of bounded continual associative memory for settings where backpropagation is unavailable or undesirable — not a new memory mechanism, and not a method that outperforms standard memories.

---

## 1. Introduction

Modern AI systems store knowledge in learned weight matrices. Updating that knowledge requires retraining, which is expensive and induces catastrophic forgetting (Kirkpatrick et al., 2017): new learning overwrites old. Memory-augmented architectures (Neural Turing Machines, Graves et al., 2014; Differentiable Neural Computers, Graves et al., 2016; retrieval-augmented generation, Lewis et al., 2020) separate compute from memory, but either backpropagate through the memory-access mechanism, rely on a learned controller, or use an external store that the model cannot update during inference.

This paper studies a deliberately minimal alternative: store knowledge as embedding vectors at fixed slots, retrieve by cosine similarity, and write by a local Hebbian rule with no gradients. We call this **SAGE** (Spatial Associative Geometric Embeddings). The 3-D grid that gives the system its name is, as we show in §5, an *addressing and visualization* device; the semantic computation happens in the d-dimensional embedding space via cosine similarity.

**Scope and honesty.** An earlier version of this work (v1) reported SAGE only against neural-network baselines and attributed its properties to "the geometry." Both choices over-stated the result. In this version we (a) add the non-parametric baselines that are the genuine competitors of a weight-free associative store, (b) correct the claims that did not survive those baselines, and (c) characterize precisely what SAGE is. The result is a more modest and more defensible contribution.

**Contributions.**
1. A weight-free, gradient-free associative memory with retraining-free continuous insertion and no catastrophic forgetting.
2. A rigorous class-incremental forgetting benchmark, evaluated against **both** neural (SGD-MLP, Replay) and **non-parametric** (NCM, per-class k-means) baselines, with significance testing — establishing where SAGE wins (vs neural) and where it ties or loses (vs non-parametric).
3. Controlled evidence that the 3-D geometry is decorative: a geometry-addressed variant collapses while the embedding-addressed variant does not.
4. A precise placement of SAGE within the associative-memory literature (attention / modern Hopfield / delta-rule fast weights / palimpsest forgetting).
5. Working architectural components — a partitioned working memory (SAGEDivided), a sharded long-term memory (MultiCube), a consolidation pathway, and an explicit-transition cube (SAGESequenceCube) — reported with honest attribution of which results come from the geometry and which from explicit data structures.

---

## 2. Related Work and Positioning

**Associative memory and attention.** Sparse Distributed Memory (Kanerva, 1988) stores patterns at fixed addresses and retrieves by proximity; SAGE is in this lineage but uses continuous cosine retrieval over real embeddings. Crucially, modern Hopfield networks are equivalent to transformer attention (Ramsauer et al., 2021): SAGE's "cosine + softmax over stored vectors" read **is** that operation. A dense Hopfield layer at the same byte budget has exponentially higher storage capacity than SAGE's one-vector-per-slot scheme. SAGE therefore introduces no new retrieval mechanism.

**Fast weights and the write rule.** Linear-attention Transformers are fast-weight programmers (Schlag, Irie & Schmidhuber, 2021): the state matrix is a key–value correlation memory written by an outer product, and the *delta rule* corrects an existing key→value mapping under bounded capacity. SAGE's "merge into the nearest same-class slot" is a hard, gradient-free version of this delta-rule write. Its slot decay corresponds to palimpsest forgetting in Hopfield memories (Storkey & Valabregue).

**Continual learning.** EWC (Kirkpatrick et al., 2017), PackNet (Mallya & Lazebnik, 2018), and replay methods mitigate forgetting in parametric models but require gradient updates or rehearsal. Non-parametric methods — nearest-class-mean, class-prototype stores, and online k-means — avoid forgetting *by construction* because each class occupies its own storage. These are the baselines a weight-free memory must be measured against, and they are the ones v1 omitted.

**Test-time / LLM memory.** Recent neural long-term-memory methods (Titans, Behrouz et al., 2025; Test-Time Training, Sun et al., 2024) write memory via gradient descent at inference. RAG and agentic memory write gradient-free but live outside the model. SAGE occupies the gradient-free, bounded, associative cell of this design space — a cell that is sparsely populated precisely because, when backpropagation is available, the trained methods dominate. SAGE's niche is therefore the backprop-unavailable / streaming / edge regime, not "better LLM memory" in general.

---

## 3. Architecture

### 3.1 Base SAGE: storage and retrieval

A cube is N³ grid points with fixed coordinates in [−1,+1]³. Each point stores a full d-dimensional embedding e_i. Given a query q, retrieval computes cosine similarity against **all** stored embeddings and applies a temperature-τ softmax:

```
scores = softmax( (E · q) / τ ),   response = Σ_i scores_i · e_i   (top-k)
```

No learned parameters appear in retrieval. **The grid coordinates are not used in retrieval** — the answer is computed in the d-dimensional embedding space. The 3-D position is an addressing label and a visualization aid (see §5.1).

Learning is a local Hebbian update of activated slots toward the target, with renormalization:

```
e_i ← e_i + α · score_i · (t − e_i),   e_i ← e_i / ‖e_i‖
```

No backpropagation is used. (The paper's V2–V4 variants add momentum, contrastive repulsion, direction training, adaptive temperature, and a Lennard-Jones positional force; we retain these as described in v1, but note in §5 that the positional forces act on the decorative geometry and do not change the retrieval mechanism.)

### 3.2 Self-organising anti-collision

New knowledge is assigned to the slot of highest cosine similarity, so semantically distinct concepts (near-orthogonal embeddings) self-segregate. This is nearest-prototype assignment without a vigilance parameter — equivalent in effect to online competitive learning.

### 3.3 SAGEDivided, MultiCube, Consolidation

- **SAGEDivided** partitions the cube along x: x<0 stores subjects, x≥0 stores objects, giving a hard structural guarantee against subject–object interference.
- **MultiCube** is several fixed-size cubes, one per domain — i.e. **index sharding**, which keeps per-shard density (and thus retrieval sharpness) constant as total knowledge grows. We now state this plainly as a standard sharding strategy rather than a novel scaling principle.
- The **consolidation pathway** moves working-memory items into the appropriate long-term shard after each step, an analogue of hippocampal–neocortical transfer (McClelland et al., 1995).

### 3.4 SAGESequenceCube

Transitions A→B are stored with a **separated design**: embeddings preserve the retrieval index (find the state), and an **explicit dictionary** stores the s_idx→o_idx pointer (the association). We report below that the perfect rollout accuracy of this component comes from the dictionary, not from the geometry; the geometry provides only collision-free addressing.

---

## 4. Experiments

All experiments run on an RTX 4090 (CUDA/PyTorch). Frozen features are used throughout so that what is measured is the **memory mechanism**, not feature learning.

### 4.1 Two-task forgetting (the original demo)

Two disjoint tasks (420 pairs each) are learned sequentially; Task-A retention is measured after Task B. SAGE forgets 0.012 vs an MLP's 0.150 (retention 0.892 vs 0.847 after A; 0.880 vs 0.697 after B). This demonstrates the structural anti-forgetting property **relative to a neural network**. It does *not*, on its own, show SAGE is a good continual learner — for that we need the non-parametric baselines (§4.2).

### 4.2 Class-incremental benchmark **with the missing baselines** (the centerpiece)

We add the experiment v1 lacked. Setup: 10 classes arrive in 5 tasks of 2 classes; frozen features; no task-id at test; six random orderings plus one adversarial (strict class-by-class) ordering. Contenders: **SGD-MLP** (8 epochs/task), **Replay** (SGD-MLP + reservoir buffer), **NCM** (one mean/class), **NCM-multi** (online per-class k-means, k=10/class — storage-matched to SAGE), **SAGE** (per-class slot pools), and **SAGE-grid** (3-D Fibonacci addressing). Metrics: final accuracy, first-task accuracy (forgetting indicator), and order-std (robustness). Significance via per-item McNemar and an across-ordering sign test.

**Table 1 — final accuracy (mean over orderings).**

| Method | sklearn digits | real MNIST (PCA-64) | forgets? | order-robust? |
|---|---|---|---|---|
| SGD-MLP (neural) | 19.8 | 19.7 | yes (cliff) | n/a |
| Replay (neural+buffer) | 87.4 | 82.9 | partial | medium |
| NCM (1 mean/class) | 87.0 | 79.7 | no | yes (exact) |
| **NCM-multi (per-class k-means)** | — | **90.8** | no | yes |
| **SAGE (per-class)** | 85.3 | 86.2 | no | yes |
| SAGE-grid (3-D geometry) | 35.5 | 18.1 | collapses | no |

**Findings, stated honestly.**
1. **SAGE beats the neural baselines** on forgetting: +65.5 pp over SGD-MLP at both scales, and it is far more order-robust (SGD collapses to ~chance on all but the last task). This is the real, true headline.
2. **SAGE ties NCM at digits scale (85.3 vs 87.0) and loses to a storage-matched per-class k-means on MNIST (86.2 vs 90.8).** The MNIST gap is significant (per-item McNemar p≈4×10⁻¹³; the standard method wins on 7/7 orderings). So SAGE's no-forgetting is real but **not superior to** simple non-parametric methods; at scale a textbook k-means is better.
3. **The 3-D geometry is decorative**: SAGE-grid (which addresses by 3-D position) collapses (35.5 / 18.1) while embedding-addressed SAGE does not. Geometry is not load-bearing.

### 4.3 SAGE as a memory store

Against vector-store baselines on an agent-memory deduplication task, SAGE **ties** a well-built bounded dedup store and **beats** naive unbounded/FIFO usage (≈+37 pp vs FIFO; ~8× smaller footprint than an unbounded index). Honest claim: SAGE equals a good bounded store; it does not exceed one.

### 4.4 Reasoning, binding, and analogy (negative results)

For completeness we report that SAGE's relational mechanisms tie or lose to standard methods, consistent with the positioning in §2:
- **Analogy** uses standard vector arithmetic (a−b+c) on GloVe vectors. SAGE inherits GloVe's analogy ability and slightly degrades it (58.3% vs GloVe 83.3% under one query protocol; equality when the arithmetic is applied directly). It is not a new capability.
- **Graph traversal** ties vector arithmetic on multi-hop analogy (no significant difference).
- **Role–filler binding** ties a Python dictionary on exact key–value lookup; **fixed-footprint superposition** loses to a hash table at equal byte budget; **offset clustering for relation discovery** equals online k-means. These are reported as honest negative results.

### 4.5 SAGESequenceCube

Single-step retrieval and multi-step rollout are 100% — **because transitions use an explicit dictionary**. Spatial clustering of sequences is weak (0.987× separation), confirming the geometry provides collision-free addressing, not organization. Zero forgetting after 300 noise transitions is structural (reserved slots are never overwritten). We attribute the perfect rollout to the dictionary, not the geometry.

---

## 5. Discussion

### 5.1 The geometry is decoration

Three lines of evidence: (i) retrieval uses the full embedding, never the 3-D coordinates; (ii) the geometry-addressed variant (SAGE-grid) collapses in every benchmark; (iii) positional-routing recall recovers only ~10% of exact-cosine recall, and the top-3 principal components of the embeddings capture only ~8% of variance. The cube is a useful **visualization and addressing** device — one can watch memory organize in 3-D — but it does not compute. We therefore retract the v1 claim "the geometry computes."

### 5.2 What SAGE actually is

SAGE is a bounded online associative memory whose read equals attention/modern-Hopfield retrieval, whose merge-write is a hard delta-rule fast-weight update, and whose decay is palimpsest forgetting. It is a transparent, weight-free *instantiation* of a known family, with directly inspectable slots. Its capacity per byte is below a dense Hopfield layer's; its accuracy as a continual classifier is below a storage-matched per-class k-means'. Its distinctive properties — gradient-free local writes, bounded footprint, slot interpretability — matter in the regime where backpropagation is unavailable or undesirable (edge / neuromorphic / streaming), which is where the contribution should be positioned.

### 5.3 On efficiency and sparsity

The v1 "0.000% sparsity / infinite efficiency" claim is withdrawn. Although only one slot is the *answer*, retrieval computes cosine against **all** N³ slots (O(N³·d) per query). SAGE is activation-sparse, not compute-sparse; it offers no asymptotic efficiency advantage over a dense store of the same size.

---

## 6. Limitations

1. **No advantage over the simple non-parametric baselines.** SAGE ties NCM and loses to per-class k-means at MNIST scale. The honest niche is "matches the gradient-free no-forget methods and beats the gradient-based ones," not "best continual memory."
2. **Capacity.** One vector per slot is far below dense-Hopfield capacity at equal bytes.
3. **Geometry decorative.** The 3-D substrate adds no measured capability; it is retained for visualization only.
4. **Real-input integration.** All experiments use synthetic, GloVe, or nomic-embedding inputs; a perception encoder is required for deployment.
5. **Hyperparameter sensitivity.** SAGE's merge/lr thresholds were not tuned per dataset; the per-class allocation variant was required to prevent slot starvation at scale.

---

## 7. Conclusion

SAGE is a clean, weight-free, gradient-free instantiation of bounded continual associative memory. It supports retraining-free updates and avoids catastrophic forgetting — properties it **shares with, and does not exceed,** simple non-parametric methods such as NCM and per-class k-means, and that it genuinely uses to beat gradient-trained neural baselines. Its 3-D geometry is an addressing and visualization layer, not the computational mechanism. The contribution of this work is a transparent realization of a known memory family, together with an honest, baseline-anchored characterization of exactly where such a memory wins and where it does not. We present this corrected account as the scientifically appropriate record, superseding the over-stated claims of v1.

---

## Acknowledgements

The author used Claude (Anthropic) as a research assistant for architecture, code, experiment design, adversarial re-testing against baselines, and manuscript preparation. All research decisions, results, and claims are the author's responsibility.

## Changelog vs v1 (for the public arXiv replacement note)

- Added non-parametric baselines (NCM, per-class k-means, bounded dedup store) — the central correction.
- Added a class-incremental benchmark on real MNIST features with significance testing; SAGE loses to a storage-matched per-class k-means (86.2 vs 90.8, McNemar p≈4×10⁻¹³).
- Withdrew "the geometry computes" — added controlled evidence that the 3-D substrate is decorative (SAGE-grid collapses).
- Withdrew "0.000% sparsity / infinite efficiency" — retrieval scans all slots.
- Reframed analogy (standard GloVe arithmetic), sequence rollout (explicit dictionary), and MultiCube (standard index sharding).
- Added precise positioning in the associative-memory literature (attention / modern Hopfield / delta-rule fast weights / palimpsest forgetting).

## Key references

Graves et al. 2014/2016 (NTM/DNC); Kanerva 1988 (SDM); Kirkpatrick et al. 2017 (EWC); Lewis et al. 2020 (RAG); Mallya & Lazebnik 2018 (PackNet); McClelland et al. 1995 (CLS); Ramsauer et al. 2021 (Hopfield = attention); Schlag, Irie & Schmidhuber 2021 (linear Transformers as fast-weight programmers); Pennington et al. 2014 (GloVe); Behrouz et al. 2025 (Titans); Sun et al. 2024 (Test-Time Training).
