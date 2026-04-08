# SAGE — Spatial Associative Geometric Embeddings

**Geometric memory architecture. Stores knowledge as coordinate positions in a 3D cube. No backpropagation.**

Paper: *Memory for All: SAGE — Spatial Associative Geometric Embeddings* (Likov, 2026)  
DOI: https://doi.org/10.5281/zenodo.19192937  
Companion paper (drone): https://github.com/Ivelin2022/sage-drone

---

## Core idea

Instead of storing knowledge in weight matrices, SAGE stores it as unit-norm embedding vectors at fixed grid points in a 32³ unit cube. Retrieval uses cosine similarity. Learning uses local Hebbian updates. No gradient computation at any stage.

> **The geometry computes. The weights are not needed.**

---

## Repository structure

### Core architecture

| File | Description |
|---|---|
| `cube_core_torch.py` | V1 — base architecture, PyTorch |
| `cube_core_v2_torch.py` | V2 — adds momentum, contrastive repulsion, direction training, adaptive temperature |
| `cube_core_v3_torch.py` | **V3 — adds Langevin dynamics (Force 6), +20% retrieval quality** |
| `cube_core.py` | V1 NumPy (reference implementation) |
| `sage_divided.py` | SAGEDivided — short-term working memory with fixed x=0 partition |
| `multicube.py` | MultiCube — horizontal scaling via specialist cubes |
| `sage_sequence_cube_v5.py` | SAGESequenceCube — explicit episodic transition memory |

### Live demo

| File | Description |
|---|---|
| `sage_live_demo.py` | Interactive multi-cube system with real 768d embeddings via Ollama |

### Experiments

| File | Description |
|---|---|
| `test_7_forces.py` | 9-force comparison: bio forces + Langevin + gyroscopic + pinning |
| `test_8a_query_chaining.py` | Sequential cross-cube composition via residual query chaining |
| `test_8b_shared_coords.py` | Shared grid coordinates (TEM-style O(1) association) |
| `test_8c_parallel_compose.py` | Parallel composition: NLerp vs Hopfield vs FHRR vs PoE |
| `run_tests_8abc.py` | Runner for tests 8A/8B/8C |
| `exp5_scaling_ablation.py` | Vertical vs horizontal scaling ablation |

### Supporting files

| File | Description |
|---|---|
| `trainer.py` / `trainer_torch.py` | Training loop + benchmarking |
| `run.py` / `run_torch.py` | Entry points |
| `visualiser.py` / `visualiser_torch.py` | 3D cube visualisation |

---

## Architecture versions

| Version | Forces | Key addition |
|---|---|---|
| V1 | Spatial cohesion | Base architecture |
| V2 | + Momentum, contrastive repulsion, direction training, adaptive temperature | 20% cluster cohesion improvement |
| V3 | + **Langevin dynamics** (Force 6) | **+20% cosine retrieval quality, negative forgetting under noise** |

---

## Key results

| Experiment | Result |
|---|---|
| Catastrophic forgetting | 92.7% less than neural network, 91.2% less than EWC |
| Continuous learning | Perfect retention across 200 sequential concept updates |
| Output sparsity | 0.000% — queries activate only the relevant region |
| Load sweep | 100% top-1 retrieval from 5% to 95% utilisation |
| Langevin dynamics (V3) | +20.1% cosine similarity over V2 baseline |
| Forgetting under noise (V3) | Negative — system improves under 800 noise insertions |
| Live demo (768d, 44 concepts) | Correct retrieval on 10/10 semantic queries |
| Chain: "evolution of computing" | brain → neurons → Ada Lovelace (0.74 cosine) |

---

## Running the live demo

Requires [Ollama](https://ollama.ai) running locally:

```bash
ollama pull nomic-embed-text
python sage_live_demo.py
```

Available commands:
```
store facts The Eiffel Tower is 330 metres tall
store relations Gustave Eiffel designed the Eiffel Tower
query European landmarks
chain history of France
similar scientific discoveries
forget test
```

Falls back to 64d random embeddings if Ollama is unavailable.

---

## Running the experiments

```bash
# Test 7 — force comparison (9 forces, ~13 min on RTX 4090)
python test_7_forces.py

# Tests 8A/8B/8C — cross-cube composition (~40 min)
python run_tests_8abc.py

# Dependencies
pip install torch scipy matplotlib requests
```

---

## Requirements

```
torch>=2.0
scipy
matplotlib
requests          # for Ollama API
numpy
```

---

## Citation

```bibtex
@article{likov2026sage,
  title  = {Memory for All: SAGE — Spatial Associative Geometric Embeddings},
  author = {Likov, Ivelin},
  year   = {2026},
  doi    = {10.5281/zenodo.19192937},
  url    = {https://github.com/Ivelin2022/sage}
}
```

---

Acknowledgement: Claude (Anthropic) was used as an AI research assistant throughout this work.  
All research decisions, results, and claims are the sole responsibility of the author.
