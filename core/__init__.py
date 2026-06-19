"""
SAGE-Sphere core package.

Gradient-free / Hebbian / cosine only. NO autograd, optimizers, or loss.backward
anywhere in this package - that is a thesis violation (see README).

Modules are added per stage:
  sphere_substrate.py  (Stage 1)  Fibonacci-sphere grid + placement
  hebbian.py           (Stage 1)  Hebbian write/update (port of Langevin Force 6)
  retrieval.py         (Stage 1)  cosine routing, top-k
  isotropy.py          (Stage 2)  All-but-the-Top preprocessing (conditional)
  knn_graph.py         (Stage 3)  kNN-relation graph + Hebbian edge counts
  traversal.py         (Stage 4)  Dijkstra / walk composition
  binding.py           (Stage 5)  two-sphere index/relation switchboard
  divided_sphere.py    (Stage 6)  hemisphere partition + Grover hooks
"""
