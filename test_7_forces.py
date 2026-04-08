"""
Test 7 — Bio Forces (200 epochs) + Novel Tier 3 Physics Forces
===============================================================
Author: Ivelin Likov

This test combines two pending experiments:

PART A — Bio forces at 200 epochs (Test 6D rerun at proper depth)
  Oja's rule, SLERP update, BCM rule, lateral inhibition, oscillatory coupling
  All showed no signal at 50 epochs — 200 epochs needed for meaningful comparison.

PART B — Novel Tier 3 physics forces (never built before)
  Gyroscopic precession: redirects updates perpendicular to stored memory axis
  Magnetic domain pinning: history-weighted directional protection
  Langevin dynamics: calibrated noise for density distribution

All forces operate gradient-free on unit-norm embeddings.
Baseline: standard V2 Hebbian update.

Metrics:
  - Top-1 retrieval accuracy (did it find the right concept?)
  - Mean cosine similarity (response quality)
  - Forgetting score (how much does noise overwrite stored memories?)
  - Direction consistency (how consistently are relational directions encoded?)

Runtime: ~2-3 hours on RTX 4090

Results: test_7_forces_results.json + test_7_forces.png
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ── Config ───────────────────────────────────────────────────────────────────
CUBE_SIZE    = 32
EMBED_DIM    = 64
N_CONCEPTS   = 8000   # ~25% cube utilisation — real density pressure
EPOCHS       = 200
BATCH_SIZE   = 512    # larger batch for GPU efficiency
ALPHA        = 0.02
TEMPERATURE  = 0.05
N_RUNS       = 3
N_NOISE      = 3000   # enough noise to stress-test forgetting protection
N_ANALOGY    = 100    # more analogy pairs for stable direction estimate

print(f"Cube: {CUBE_SIZE}³ | Embed: {EMBED_DIM}d | Concepts: {N_CONCEPTS} | Epochs: {EPOCHS}")
print(f"Runs: {N_RUNS} | Noise steps: {N_NOISE}")


# ── Force implementations ─────────────────────────────────────────────────────
# All forces modify the error signal before applying to embeddings.
# Input: error = target - e_i, e_i = current embedding, aux = per-point state
# Output: modified error vector

def standard_error(error, e_i, aux_i):
    """V2 baseline: standard Hebbian error."""
    return error


def oja_error(error, e_i, aux_i):
    """
    Oja's rule: e += alpha*(target - (target·e)*e)
    Adds a decay term that prevents collapse to a single direction.
    Natural online PCA on the unit sphere.
    """
    target = e_i + error  # recover target from error
    target = F.normalize(target.unsqueeze(0), p=2, dim=1).squeeze(0)
    oja_e  = target - torch.dot(target, e_i) * e_i
    return oja_e


def slerp_error(error, e_i, aux_i, alpha=ALPHA):
    """
    SLERP (Spherical Linear Interpolation) update.
    Moves along the great circle toward target instead of straight line.
    Preserves angular relationships better than chord interpolation.
    """
    target = F.normalize((e_i + error).unsqueeze(0), p=2, dim=1).squeeze(0)
    cos_theta = torch.dot(e_i, target).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    if sin_theta.abs() < 1e-6:
        return error  # parallel, use standard update
    slerp_target = (torch.sin((1 - alpha) * theta) / sin_theta) * e_i + \
                   (torch.sin(alpha * theta) / sin_theta) * target
    return slerp_target - e_i


def bcm_error(error, e_i, aux_i):
    """
    BCM (Bienenstock-Cooper-Munro) rule.
    Sliding threshold theta separates LTP (potentiation) from LTD (depression).
    Prevents runaway potentiation — homeostatic mechanism.
    aux_i = running average activation (theta)
    """
    act  = torch.dot(e_i, e_i + error)  # approximate current activation
    theta = aux_i.clamp(min=1e-3)        # sliding threshold
    sign  = 1.0 if act > theta else -0.1  # LTP vs LTD
    return sign * error


def lateral_inhibition_error(error, e_i, aux_i, strength=0.3):
    """
    Anti-Hebbian lateral inhibition: push nearby non-target points away.
    Modifies the update to subtract the projection onto similar directions.
    aux_i = running mean of nearby embeddings
    """
    if aux_i is None or aux_i.norm() < 1e-6:
        return error
    # Project out the component toward the neighbourhood mean
    nbr_hat = F.normalize(aux_i.unsqueeze(0), p=2, dim=1).squeeze(0)
    inhibited = error - strength * torch.dot(error, nbr_hat) * nbr_hat
    return inhibited


def oscillatory_error(error, e_i, aux_i, t=0, omega=0.1):
    """
    Oscillatory/resonance coupling: modulate update strength by phase.
    Similar concepts (co-activated) synchronize phases → stronger coupling.
    aux_i = phase of this grid point
    """
    if aux_i is None:
        return error
    phase = aux_i.item() if isinstance(aux_i, torch.Tensor) else aux_i
    modulation = 0.5 + 0.5 * math.cos(phase + omega * t)  # [0, 1]
    return modulation * error


def gyroscopic_error(error, e_i, aux_i):
    """
    Gyroscopic precession: removes component along spin axis (= current embedding).
    High-activation points "spin" faster → resist overwriting more.
    aux_i = activation count (scalar)

    Benjamin et al. (NeurIPS 2024): linear momentum HURTS forgetting.
    Precession redirects (doesn't reduce) update → learning capacity preserved.
    """
    activation_count = aux_i.item() if isinstance(aux_i, torch.Tensor) else float(aux_i)
    # Spin: saturating [0, 1) — higher activation = stronger precession
    spin = activation_count / (activation_count + 10.0)
    # Remove component along spin axis (= e_i)
    parallel  = torch.dot(error, e_i) * e_i
    perp_error = error - parallel
    # Blend: low activation → standard error; high activation → perpendicular only
    effective = (1.0 - spin) * error + spin * perp_error
    return effective


def pinning_error(error, e_i, aux_i, beta=0.5):
    """
    Magnetic domain pinning: project out component along pinning field.
    H = activation_history * e_i (stronger field for well-used memories).
    No Fisher matrix needed — direction projection only.

    Related to Synaptic Intelligence (Zenke 2017) but uses directional geometry
    instead of quadratic penalty.
    """
    activation_count = aux_i.item() if isinstance(aux_i, torch.Tensor) else float(aux_i)
    H  = activation_count * e_i  # pinning field
    H_norm2 = torch.dot(H, H) + 1e-8
    # Adaptive beta: stronger protection for well-established memories
    adaptive_beta = beta * math.tanh(activation_count / 10.0)
    pinned = error - adaptive_beta * (torch.dot(error, H) / H_norm2) * H
    return pinned


def langevin_error(error, e_i, aux_i, T=0.01):
    """
    Langevin dynamics: add calibrated tangent-space noise to Hebbian update.
    At equilibrium, samples Boltzmann distribution over Hebbian energy landscape.
    Proven theory: SGLD (Welling & Teh, ICML 2011).

    The tangent-space projection (removing radial component before renorm)
    is the novel element for unit-norm embedding spaces.
    """
    noise = torch.randn_like(e_i) * math.sqrt(2 * T)
    # Project noise to tangent plane at e_i (remove radial component)
    noise_tangent = noise - torch.dot(noise, e_i) * e_i
    return error + noise_tangent


# ── Minimal cube with pluggable force ─────────────────────────────────────────

class ForceCube:
    def __init__(self, cube_size, embed_dim, seed, force_fn, force_name):
        torch.manual_seed(seed)
        self.n     = cube_size ** 3
        self.dim   = embed_dim
        self.force = force_fn
        self.name  = force_name

        # R3 initialisation (adopted from Test 6A)
        self.embeddings = self._r3_init(self.n, embed_dim)

        # Per-point auxiliary state (activation count, phase, etc.)
        self.aux = torch.zeros(self.n, device=device)

        # For oscillatory: per-point phases
        self.phases = torch.rand(self.n, device=device) * 2 * math.pi

        # For lateral inhibition: neighbourhood mean (lazy, updated every 10 steps)
        self.nbr_means = torch.zeros(self.n, embed_dim, device=device)

        self.step = 0

    def _r3_init(self, n, d):
        """R3-sequence initialisation — vectorised."""
        phi3 = 1.2207440846  # plastic constant
        # seeds[i] = (0.5 + phi3^(i+1)) % 1.0
        exponents = torch.arange(1, d + 1, dtype=torch.float64)
        seeds = ((0.5 + phi3 ** exponents) % 1.0).float().to(device)  # (d,)
        # k = 0..n-1, i = 0..d-1:  val[k,i] = (0.5 + seeds[i]*(k+1)) % 1.0
        k_idx = torch.arange(1, n + 1, dtype=torch.float32, device=device)  # (n,)
        vecs  = (0.5 + seeds.unsqueeze(0) * k_idx.unsqueeze(1)) % 1.0  # (n, d)
        vecs  = vecs * 2 - 1  # scale to [-1, 1]
        return F.normalize(vecs, p=2, dim=1)

    def retrieve(self, q):
        q  = F.normalize(q.unsqueeze(0), p=2, dim=1).squeeze(0)
        s  = self.embeddings @ q
        sc = F.softmax(s / TEMPERATURE, dim=0)
        r  = (sc.unsqueeze(1) * self.embeddings).sum(0)
        return F.normalize(r.unsqueeze(0), p=2, dim=1).squeeze(0), s.argmax().item(), s

    def learn_batch(self, queries, targets):
        """Batch Hebbian update with pluggable force — vectorised."""
        Q = F.normalize(queries, p=2, dim=1)
        T = F.normalize(targets, p=2, dim=1)
        B = Q.shape[0]

        sims   = Q @ self.embeddings.T              # (B, N)
        scores = F.softmax(sims / 0.1, dim=1)       # (B, N)
        top_k  = min(20, self.n)
        top_s, top_i = torch.topk(scores, top_k, dim=1)  # (B, K)

        # Standard vectorised gradient: scores.T @ T  -  weight_sum * E
        # (identical to the V2 baseline)
        pos_grad   = scores.T @ T                        # (N, d)
        weight_sum = scores.sum(0).unsqueeze(1)          # (N, 1)
        base_grad  = pos_grad - weight_sum * self.embeddings  # (N, d)

        # Apply force modification per active point
        # Forces that need per-point state are applied as a correction on top
        # of the base gradient.  Forces that are state-free are applied batch-wise.
        if self.name == 'standard':
            mod_grad = base_grad

        elif self.name == 'oja':
            # Oja: replace standard error with (target - (target·e)*e)
            # pos_grad already contains scores.T @ T
            # Oja extra term: -scores.T @ ((T·E_activated)*E_activated)
            # Approximation: subtract (scores.T @ T)·E elementwise
            dot_te = (pos_grad * self.embeddings).sum(1, keepdim=True)  # (N,1)
            mod_grad = pos_grad - dot_te * self.embeddings - weight_sum * self.embeddings

        elif self.name == 'slerp':
            # SLERP: actual spherical linear interpolation toward target direction.
            # For each point e_i, compute the SLERP step toward the weighted target.
            # target direction per point = normalize(pos_grad / weight_sum)
            wt_safe = weight_sum.clamp(min=1e-8)
            t_dir   = F.normalize(pos_grad / wt_safe, p=2, dim=1)      # (N, d) target dirs
            cos_theta = (self.embeddings * t_dir).sum(1, keepdim=True).clamp(-1+1e-6, 1-1e-6)
            theta     = torch.acos(cos_theta)                            # (N, 1)
            sin_theta = torch.sin(theta).clamp(min=1e-6)
            # SLERP step: move ALPHA fraction of angle toward target
            a = ALPHA
            slerp_pos = (torch.sin((1-a)*theta)/sin_theta) * self.embeddings +                         (torch.sin(a*theta)/sin_theta) * t_dir
            mod_grad  = (slerp_pos - self.embeddings) * (weight_sum > 1e-6).float()

        elif self.name == 'bcm':
            # BCM: sign depends on instantaneous activation vs sliding threshold.
            # act = mean softmax score received this batch (proxy for firing rate).
            # theta = self.aux = cumulative score / step (running mean).
            act   = scores.mean(0).unsqueeze(1)              # (N,1) mean score this batch
            theta = (self.aux / (self.step + 1)).unsqueeze(1).clamp(min=1e-6)
            sign  = torch.where(act > theta,
                                torch.ones_like(theta),
                                torch.full_like(theta, -0.1))
            mod_grad = base_grad * sign

        elif self.name == 'lat_inhib':
            # Lateral inhibition: subtract component toward neighbourhood mean
            nbr_hat  = F.normalize(self.nbr_means, p=2, dim=1)       # (N, d)
            proj     = (base_grad * nbr_hat).sum(1, keepdim=True) * nbr_hat
            mod_grad = base_grad - 0.3 * proj

        elif self.name == 'oscillatory':
            # Oscillatory: modulate gradient by phase
            mod  = (0.5 + 0.5 * torch.cos(self.phases)).unsqueeze(1)  # (N,1)
            mod_grad = base_grad * mod

        elif self.name == 'gyroscopic':
            # Gyroscopic: remove component along spin axis (= e_i)
            # base_grad = (pos_grad - wt * E), so parallel = (base_grad · E)*E
            spin     = (self.aux / (self.aux + 10.0)).unsqueeze(1)         # (N,1)
            parallel = (base_grad * self.embeddings).sum(1, keepdim=True) * self.embeddings
            perp     = base_grad - parallel
            mod_grad = (1 - spin) * base_grad + spin * perp

        elif self.name == 'pinning':
            # Magnetic pinning: remove component along H = activation * e_i
            act_count = self.aux.unsqueeze(1)                              # (N,1)
            H         = act_count * self.embeddings                        # (N,d)
            H_norm2   = (H * H).sum(1, keepdim=True) + 1e-8              # (N,1)
            beta_a    = 0.5 * torch.tanh(self.aux / 10.0).unsqueeze(1)
            proj_H    = (base_grad * H).sum(1, keepdim=True) / H_norm2 * H
            mod_grad  = base_grad - beta_a * proj_H

        elif self.name == 'langevin':
            # Langevin: add tangent-space noise
            noise        = torch.randn_like(self.embeddings) * math.sqrt(2 * 0.01)
            noise_radial = (noise * self.embeddings).sum(1, keepdim=True) * self.embeddings
            noise_tang   = noise - noise_radial
            mod_grad     = base_grad + noise_tang * (weight_sum > 1e-6).float()

        else:
            mod_grad = base_grad

        # Apply update
        mask = weight_sum.squeeze() > 1e-6
        if mask.any():
            self.embeddings[mask] += ALPHA * mod_grad[mask] / (weight_sum[mask] + 1e-8)
            self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        # Update activation counts
        self.aux.scatter_add_(0, top_i.flatten(),
                              top_s.flatten())

        # Update phases for oscillatory
        if self.name == 'oscillatory':
            self.phases += 0.1

        # Update neighbourhood means for lateral inhibition (every 20 steps)
        if self.name == 'lat_inhib' and self.step % 20 == 0:
            for i in torch.where(mask)[0][:30]:
                sims_i = self.embeddings @ self.embeddings[i]
                nbrs   = sims_i.topk(6).indices[1:]
                self.nbr_means[i] = self.embeddings[nbrs].mean(0)

        self.step += 1

        # Loss
        resp = (top_s.unsqueeze(2) * self.embeddings[top_i]).sum(1)
        resp = F.normalize(resp, p=2, dim=1)
        return (1 - (resp * T).sum(1)).mean().item()


# ── Evaluation functions ──────────────────────────────────────────────────────

def eval_retrieval(cube, concepts, n_test=40):
    """Top-1 accuracy and mean cosine on held-out concepts."""
    test_c = concepts[-n_test:]
    correct, cosines = 0, []
    for i, c in enumerate(test_c):
        resp, _, _ = cube.retrieve(c)
        all_sims   = test_c @ resp
        if all_sims.argmax().item() == i:
            correct += 1
        cosines.append(F.cosine_similarity(resp.unsqueeze(0), c.unsqueeze(0)).item())
    return correct / n_test, float(np.mean(cosines))


def eval_forgetting(cube, stored_concepts, n_noise=N_NOISE):
    """
    Measure how much stored memories degrade after noise insertions.
    Lower = better (memories are more robust).
    """
    # Baseline cosine similarity
    before = []
    for c in stored_concepts[:20]:
        resp, _, _ = cube.retrieve(c)
        before.append(F.cosine_similarity(resp.unsqueeze(0), c.unsqueeze(0)).item())

    # Insert noise in batches for speed
    noise_batch = 64
    for i in range(0, n_noise, noise_batch):
        bs = min(noise_batch, n_noise - i)
        noise = F.normalize(torch.randn(bs, EMBED_DIM, device=device), p=2, dim=1)
        cube.learn_batch(noise, noise.clone())

    # After cosine similarity
    after = []
    for c in stored_concepts[:20]:
        resp, _, _ = cube.retrieve(c)
        after.append(F.cosine_similarity(resp.unsqueeze(0), c.unsqueeze(0)).item())

    forgetting = float(np.mean(before) - np.mean(after))
    return max(0.0, forgetting)


def eval_direction_consistency(cube, analogy_pairs):
    """
    Direction consistency: cos(dir_AB, dir_CD) for parallel analogy pairs.
    Higher = relational directions more consistently encoded.
    """
    consistencies = []
    for (a, b, c, d) in analogy_pairs:
        # Use top-1 stored embedding (not blended) for cleaner direction signal
        _, idx_a, _ = cube.retrieve(a)
        _, idx_b, _ = cube.retrieve(b)
        _, idx_c, _ = cube.retrieve(c)
        _, idx_d, _ = cube.retrieve(d)
        e_a = cube.embeddings[idx_a]
        e_b = cube.embeddings[idx_b]
        e_c = cube.embeddings[idx_c]
        e_d = cube.embeddings[idx_d]
        dir_ab = F.normalize((e_b - e_a).unsqueeze(0), p=2, dim=1).squeeze(0)
        dir_cd = F.normalize((e_d - e_c).unsqueeze(0), p=2, dim=1).squeeze(0)
        consistencies.append(F.cosine_similarity(dir_ab.unsqueeze(0), dir_cd.unsqueeze(0)).item())
    return float(np.mean(consistencies))


# ── Build force list ─────────────────────────────────────────────────────────

FORCES = [
    ('standard',     standard_error,     'V2 Baseline'),
    ('oja',          oja_error,          "Oja's Rule"),
    ('slerp',        slerp_error,        'SLERP Update'),
    ('bcm',          bcm_error,          'BCM Rule'),
    ('lat_inhib',    lateral_inhibition_error, 'Lateral Inhibition'),
    ('oscillatory',  oscillatory_error,  'Oscillatory Coupling'),
    ('gyroscopic',   gyroscopic_error,   'Gyroscopic Precession'),
    ('pinning',      pinning_error,      'Magnetic Pinning'),
    ('langevin',     langevin_error,     'Langevin Dynamics'),
]


# ── Main experiment ───────────────────────────────────────────────────────────

def run_one(seed, force_name, force_fn):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate concepts
    concepts = F.normalize(torch.randn(N_CONCEPTS, EMBED_DIM, device=device), p=2, dim=1)

    # Analogy pairs: concepts[0]:concepts[1] :: concepts[2]:concepts[3], etc.
    # Use a rotation to create parallel relational structure
    R = torch.linalg.qr(torch.randn(EMBED_DIM, EMBED_DIM, device=device))[0]
    analogy_pairs = []
    for i in range(0, min(N_ANALOGY * 4, N_CONCEPTS - 4), 4):
        a, b = concepts[i], F.normalize((concepts[i] @ R).unsqueeze(0), p=2, dim=1).squeeze(0)
        c, d = concepts[i+2], F.normalize((concepts[i+2] @ R).unsqueeze(0), p=2, dim=1).squeeze(0)
        analogy_pairs.append((a, b, c, d))

    # Build cube with this force
    cube = ForceCube(CUBE_SIZE, EMBED_DIM, seed, force_fn, force_name)

    # Train
    n_train = int(N_CONCEPTS * 0.8)
    train_c = concepts[:n_train]

    for ep in range(EPOCHS):
        perm  = torch.randperm(n_train, device=device)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            Q   = train_c[idx]
            cube.learn_batch(Q, Q.clone())

    # Evaluate
    top1, cosine = eval_retrieval(cube, concepts)
    forgetting   = eval_forgetting(cube, train_c[:30])
    dir_cons     = eval_direction_consistency(cube, analogy_pairs[:min(20, len(analogy_pairs))])

    return {
        'top1_acc':    top1,
        'cosine_mean': cosine,
        'forgetting':  forgetting,
        'dir_consist': dir_cons,
    }


# ── Run all forces ────────────────────────────────────────────────────────────

print("\n" + "="*65)
print("Test 7 — Bio Forces (200 epochs) + Novel Tier 3 Physics Forces")
print("="*65)

all_results = {name: {'top1': [], 'cosine': [], 'forgetting': [], 'dir': []}
               for name, _, _ in FORCES}

for run in range(N_RUNS):
    print(f"\n── Run {run+1}/{N_RUNS} ──")
    for force_name, force_fn, label in FORCES:
        t0 = time.perf_counter()
        r  = run_one(seed=42 + run * 100, force_name=force_name, force_fn=force_fn)
        t1 = time.perf_counter()
        all_results[force_name]['top1'].append(r['top1_acc'])
        all_results[force_name]['cosine'].append(r['cosine_mean'])
        all_results[force_name]['forgetting'].append(r['forgetting'])
        all_results[force_name]['dir'].append(r['dir_consist'])
        print(f"  {label:<24}: Top1={r['top1_acc']:.3f}  "
              f"Cos={r['cosine_mean']:.4f}  "
              f"Forg={r['forgetting']:.4f}  "
              f"Dir={r['dir_consist']:.4f}  "
              f"({t1-t0:.0f}s)")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY (mean ± std, 3 runs)")
print("="*65)
print(f"{'Force':<24} {'Top-1':>6} {'Cosine':>8} {'Forgetting':>10} {'Dir Cons':>9}")
print("-"*65)

baseline = all_results['standard']
base_top1 = float(np.mean(baseline['top1']))
base_forg = float(np.mean(baseline['forgetting']))
base_dir  = float(np.mean(baseline['dir']))

summary = {}
for force_name, force_fn, label in FORCES:
    r = all_results[force_name]
    top1 = float(np.mean(r['top1']))
    cos  = float(np.mean(r['cosine']))
    forg = float(np.mean(r['forgetting']))
    dir_ = float(np.mean(r['dir']))
    summary[force_name] = {'top1': top1, 'cosine': cos, 'forgetting': forg, 'dir': dir_,
                            'top1_std': float(np.std(r['top1'])),
                            'forgetting_std': float(np.std(r['forgetting']))}
    marker = " ←" if force_name == 'standard' else ""
    print(f"  {label:<22} {top1:>6.3f} {cos:>8.4f} {forg:>10.4f} {dir_:>9.4f}{marker}")

print("\nΔ vs baseline:")
for force_name, force_fn, label in FORCES:
    if force_name == 'standard':
        continue
    s = summary[force_name]
    print(f"  {label:<22}  Forg: {s['forgetting']-base_forg:+.4f}  "
          f"Dir: {s['dir']-base_dir:+.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
out_data = {
    'metadata': {
        'experiment': 'Test 7 — Bio Forces + Tier 3 Physics Forces',
        'run_date': datetime.now().isoformat(),
        'cube_size': CUBE_SIZE, 'embed_dim': EMBED_DIM,
        'n_concepts': N_CONCEPTS, 'epochs': EPOCHS,
        'n_runs': N_RUNS, 'n_noise': N_NOISE, 'device': str(device),
    },
    'summary': summary,
    'baseline': {k: float(np.mean(v)) for k, v in baseline.items()},
    'forces': {name: label for name, _, label in FORCES},
}
out_json = 'test_7_forces_results.json'
with open(out_json, 'w') as f:
    json.dump(out_data, f, indent=2)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Test 7 — Bio Forces (200 epochs) + Tier 3 Physics Forces\n'
             f'{CUBE_SIZE}³ cube | {EMBED_DIM}d | {EPOCHS} epochs | {N_RUNS} runs',
             fontsize=13, fontweight='bold')

labels    = [label for _, _, label in FORCES]
colors_tier1 = ['#888888'] + ['#3366CC'] * 5  # baseline grey, bio forces blue
colors_tier3 = ['#CC3333', '#2E8B57', '#FF8C00']  # novel forces warm
all_colors = colors_tier1 + colors_tier3

metrics = [
    ('top1',      'Top-1 Retrieval Accuracy', True),   # higher better
    ('cosine',    'Mean Cosine Similarity',   True),
    ('forgetting','Forgetting Score',          False),  # lower better
    ('dir',       'Direction Consistency',    True),
]

for ax, (key, title, higher_better) in zip(axes.flat, metrics):
    vals = [summary[name][key] for name, _, _ in FORCES]
    stds = [summary[name].get(f'{key}_std', 0.0) for name, _, _ in FORCES]
    bars = ax.bar(range(len(FORCES)), vals,
                  color=all_colors, alpha=0.85)
    ax.axhline(summary['standard'][key], color='black', linestyle='--',
               alpha=0.5, label='Baseline')
    ax.set_title(title + (' (↑ better)' if higher_better else ' (↓ better)'))
    ax.set_xticks(range(len(FORCES)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.legend(fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(color='#888888', label='V2 Baseline'),
    Patch(color='#3366CC', label='Bio forces (Part A)'),
    Patch(color='#CC3333', label='Gyroscopic (Part B)'),
    Patch(color='#2E8B57', label='Pinning (Part B)'),
    Patch(color='#FF8C00', label='Langevin (Part B)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           bbox_to_anchor=(0.5, -0.02), fontsize=9)

plt.tight_layout()
out_png = 'test_7_forces.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResults: {out_json}")
print(f"Plot:    {out_png}")
print("Done.")
