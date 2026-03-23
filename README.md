# SAGE: Spatial Associative Geometric Embeddings

**Memory for All** — a weight-free geometric memory architecture.

> *"The geometry computes. The weights are not needed."*

Paper: Memory for All: SAGE — Spatial Associative Geometric Embeddings (Likov, 2026)

**arXiv link will be added after patent filing and submission.**

GitHub: https://github.com/Ivelin2022/sage

## What is SAGE?

SAGE stores knowledge as coordinate positions in a 3D unit cube rather than in learned weight matrices. Retrieval uses cosine similarity. Learning uses local Hebbian updates. No backpropagation is required at any stage.

Key properties:
- 92% less catastrophic forgetting than neural networks (0.012 vs 0.150)
- Perfect retention across 200 sequential concept updates
- 0.000% sparsity activation — only the relevant region activates per query
- 58.3% analogy accuracy on GloVe 50d with V2 direction training
- Continuous learning during deployment without retraining

## Repository structure

```
cube_core.py           V1 base architecture (NumPy)
cube_core_torch.py     V1 GPU (PyTorch/CUDA) — Tables 1-2
cube_core_v2.py        V2: momentum + contrastive + direction training (NumPy)
cube_core_v2_torch.py  V2 GPU — Table 3 analogy experiments
cube_core_v4.py        V4: V2 + LJ gravity (NumPy)
comparison_v1_v4.py    Reproduces Table 4: architecture comparison
glove_delta_v2_fast.py Reproduces Table 6: GloVe analogy + delta encoding
requirements.txt       numpy, scipy, torch
```

## Quick start

```bash
pip install -r requirements.txt

# Baseline comparison (Table 1)
python run_torch.py

# Architecture comparison V1 vs V2 vs V4 (Table 4)
python comparison_v1_v4.py

# GloVe analogy test (Table 6) — edit GLOVE_FILE path first
python glove_delta_v2_fast.py
```

## Key results

| Experiment | Result |
|---|---|
| Catastrophic forgetting | 0.012 vs 0.150 (92% less) |
| Continuous learning | 1.0000 retention across 200 steps |
| Sparsity | 0.000% — all versions |
| Analogy V1 | 16.7% (2/12) GloVe 50d |
| Analogy V2 | 58.3% (7/12) GloVe 50d |
| GloVe baseline | 83.3% (10/12) |

## Citation

```bibtex
@article{likov2026sage,
  title  = {Memory for All: SAGE — Spatial Associative Geometric Embeddings},
  author = {Likov, Ivelin},
  year   = {2026},
  url    = {https://github.com/Ivelin2022/sage}
}
```

Acknowledgement: Claude (Anthropic) was used as an AI research assistant.
All research decisions and claims are the sole responsibility of the author.
