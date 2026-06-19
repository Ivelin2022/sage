"""
SAGE-Sphere - Forgetting + order-robustness benchmark (the never-run experiment)
=================================================================================
Tests SAGE's ONE plausibly-real structural edge: gradient-free local writes can't
catastrophically forget. Against the baseline it can actually beat - NEURAL nets
(SGD/replay) that DO forget - on class-incremental learning.

Setup: a 10-class image set with FROZEN features so we test the MEMORY mechanism,
not feature learning. Selectable dataset (argv[1], default 'digits'):
  digits : sklearn load_digits (1797 x 64 raw pixels)            -- the toy
  mnist  : real MNIST via fetch_openml, PCA-64 frozen features   -- the scale-up
The PCA extractor is fit on TRAIN ONLY (leakage-safe) -- a fixed, non-learned
backbone analogue, so the continual-learning test is purely the downstream memory.
Classes arrive in TASKS (2 classes each, 5 tasks), single pass, no task id at test.
Repeated over many class orderings incl. an adversarial (strict class-by-class) one.

Contenders:
  SGD-MLP   : MLP head, trains EPOCHS of mini-batch SGD PER TASK then the task data
              is gone (learns each task well, then FORGETS it - the honest standard
              CL forgetting curve: recent task right, first task wrong)
  Replay    : SGD-MLP + a small reservoir replay buffer (forgets LESS, costs memory)
  NCM       : nearest-class-mean (per-class centroid; structurally CAN'T forget)
  SAGE-flat : label-aware prototype memory (per-class slots; CAN'T forget)
  SAGE-grid : same but 3-D Fibonacci addressing (GEOMETRY arm - collisions corrupt
              labels across classes -> predicted to forget/lose, burying the geometry)

Metrics (mean +/- std over orderings): final overall accuracy; accuracy on the
FIRST task's classes after the full stream (the forgetting indicator); and the
std of final accuracy ACROSS orderings (order-robustness). Honest expectation:
SAGE/NCM join one cluster (don't forget, order-robust); SGD/Replay the other
(forget, order-fragile); SAGE-grid collapses (geometry). SAGE ties NCM - the win
is "joins the no-forget gradient-free cluster while crushing the neural methods".

Pure numpy + torch + sklearn. No Ollama.  python experiments/forgetting_benchmark.py
"""

import os
import sys
import json

import numpy as np
from scipy.stats import binomtest
from sklearn.datasets import load_digits, fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR  = os.path.join(ROOT, 'results')
FINDINGS = os.path.join(ROOT, 'findings.md')

N_CLASSES   = 10
PER_TASK    = 2            # classes per task -> 5 tasks
N_SLOTS     = 100          # SAGE budget (~10/class)
BUF         = 200          # replay reservoir size
EPOCHS      = 8            # SGD epochs PER TASK (so it actually learns each task,
BATCH       = 32           #   then forgets it -> an HONEST forgetting curve)
N_ORDERS    = 6            # random orderings (+ 1 adversarial)
SEED        = 42

# mnist scale-up knobs (ignored for 'digits')
MNIST_TRAIN_PER_CLASS = 2000   # subsample for tractable runtime (~10x digits)
MNIST_TEST_PER_CLASS  = 400
PCA_DIM               = 64     # frozen feature dim (matches digits D for a fair compare)


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def fib3(n):
    i = np.arange(n)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    return _unit(np.stack([r * np.cos(th), r * np.sin(th), z], axis=1))


# ---- classifiers -----------------------------------------------------------
class NCM:
    def __init__(self, dim, ncls):
        self.sum = np.zeros((ncls, dim)); self.cnt = np.zeros(ncls)

    def learn_task(self, X, y):                  # online single pass; order-invariant
        for i in range(len(y)):
            self.sum[y[i]] += X[i]; self.cnt[y[i]] += 1

    def classify(self, X):
        mean = _unit(self.sum / (self.cnt[:, None] + 1e-9))
        return np.argmax(_unit(X) @ mean.T, axis=1)


class NCMMulti:
    """STORAGE-MATCHED strong baseline for SAGE-pc: k prototypes/class via online
    per-class sequential k-means (running mean of the nearest same-class centroid).
    Same budget as SAGE-pc (k = n_slots//ncls). If SAGE-pc only TIES this, then
    SAGE's win over single-mean NCM is just 'more prototypes', not its mechanism."""
    def __init__(self, dim, n_slots, ncls=N_CLASSES):
        self.k = max(1, n_slots // ncls); self.ncls = ncls
        self.proto = np.zeros((ncls, self.k, dim)); self.cnt = np.zeros((ncls, self.k))

    def learn_task(self, X, y):                  # online single pass (order-dependent)
        for i in range(len(y)):
            c = int(y[i]); x = _unit(X[i]); cc = self.cnt[c]
            if (cc == 0).any():                  # fill an empty centroid first
                j = int(np.argmin(cc)); self.proto[c, j] = x; self.cnt[c, j] = 1
            else:                                # update nearest same-class centroid
                j = int(np.argmax(_unit(self.proto[c]) @ x)); self.cnt[c, j] += 1
                self.proto[c, j] += (x - self.proto[c, j]) / self.cnt[c, j]

    def classify(self, X):
        lab = np.repeat(np.arange(self.ncls), self.k)
        used = self.cnt.reshape(-1) > 0
        P = _unit(self.proto.reshape(self.ncls * self.k, -1))[used]
        sims = _unit(X) @ P.T
        return lab[used][np.argmax(sims, axis=1)]


class SAGEProto:
    """mode='flat'  : label-aware prototypes, GLOBAL argmin(cnt) eviction (the
                      original; a later class can evict an earlier class's slot).
       mode='grid'  : 3-D Fibonacci addressing (geometry arm; collisions corrupt).
       per_class=True: each class owns a fixed pool of n_slots//ncls slots and
                      evicts ONLY within its own pool. This removes the cross-class
                      starvation / primacy lock that /code-review traced as the
                      cause of SAGE-flat's MNIST collapse (global cnt never decays,
                      so early-task slots become un-evictable)."""
    def __init__(self, dim, n_slots, mode='flat', proj=None, merge=0.6, lr=0.3,
                 per_class=False, ncls=N_CLASSES):
        self.dim = dim; self.mode = mode; self.merge = merge; self.lr = lr
        self.key = np.zeros((n_slots, dim)); self.label = -np.ones(n_slots, dtype=int)
        self.used = np.zeros(n_slots, dtype=bool); self.cnt = np.zeros(n_slots)
        self.per_class = per_class; self.k_pc = n_slots // ncls   # slots/class
        if mode == 'grid':
            self.proj = proj; self.pos = fib3(n_slots)

    def write(self, x, y):
        x = _unit(x)
        if self.mode == 'grid':                      # 3-D address: cross-class collisions
            j = int(np.argmax(self.pos @ _unit(x @ self.proj)))
            if self.used[j]:
                self.key[j] = _unit((1 - self.lr) * self.key[j] + self.lr * x)
            else:
                self.key[j] = x; self.used[j] = True
            self.label[j] = y                        # last-write label -> collisions corrupt
            self.cnt[j] += 1
            return
        if self.per_class:                           # each class evicts only its own pool
            pool = np.arange(y * self.k_pc, (y + 1) * self.k_pc)
            pu = pool[self.used[pool]]
            if pu.size:
                ss = self.key[pu] @ x; jj = int(pu[int(np.argmax(ss))])
                if ss.max() > self.merge:
                    self.key[jj] = _unit((1 - self.lr) * self.key[jj] + self.lr * x)
                    self.cnt[jj] += 1
                    return
            free = pool[~self.used[pool]]
            slot = int(free[0]) if free.size else int(pool[int(np.argmin(self.cnt[pool]))])
            self.key[slot] = x; self.label[slot] = y
            self.used[slot] = True; self.cnt[slot] = 1
            return
        same = self.used & (self.label == y)         # merge only within same class
        if same.any():
            su = np.where(same)[0]
            ss = self.key[su] @ x; jj = int(su[int(np.argmax(ss))])
            if ss.max() > self.merge:
                self.key[jj] = _unit((1 - self.lr) * self.key[jj] + self.lr * x)
                self.cnt[jj] += 1
                return
        slot = (int(np.where(~self.used)[0][0]) if not self.used.all()
                else int(np.argmin(self.cnt)))
        self.key[slot] = x; self.label[slot] = y; self.used[slot] = True; self.cnt[slot] = 1

    def learn_task(self, X, y):                  # online single pass
        for i in range(len(y)):
            self.write(X[i], int(y[i]))

    def classify(self, X):
        u = np.where(self.used)[0]
        if u.size == 0:
            return np.zeros(X.shape[0], dtype=int)
        sims = _unit(X) @ self.key[u].T
        return self.label[u[np.argmax(sims, axis=1)]]


class SGDHead:
    """Standard CL protocol: trains EPOCHS of mini-batch SGD on each task's data
    (so it genuinely learns the task), then the task data is gone. Earlier tasks
    are overwritten -> honest catastrophic forgetting (recent task right, first
    task wrong). With replay=K it also trains on K buffered samples per batch."""
    def __init__(self, dim, ncls, lr=0.1, hidden=128, replay=0, seed=0):
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, ncls))
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.replay = replay
        self.bx = np.zeros((BUF, dim), np.float32); self.by = np.zeros(BUF, np.int64)
        self.nbuf = 0; self.seen = 0; self.rng = np.random.default_rng(seed)

    def _reservoir(self, x, yy):
        if self.nbuf < BUF:
            self.bx[self.nbuf] = x; self.by[self.nbuf] = yy; self.nbuf += 1
        else:
            j = int(self.rng.integers(0, self.seen + 1))
            if j < BUF:
                self.bx[j] = x; self.by[j] = yy
        self.seen += 1

    def learn_task(self, X, y):
        Xf = X.astype(np.float32); yl = y.astype(np.int64)
        for _ in range(EPOCHS):
            idx = self.rng.permutation(len(yl))
            for s in range(0, len(yl), BATCH):
                b = idx[s:s + BATCH]
                xb = torch.tensor(Xf[b]); yb = torch.tensor(yl[b])
                if self.replay and self.nbuf > 0:
                    r = self.rng.integers(0, self.nbuf, min(BATCH, self.nbuf))
                    xb = torch.cat([xb, torch.tensor(self.bx[r])])
                    yb = torch.cat([yb, torch.tensor(self.by[r])])
                self.opt.zero_grad()
                F.cross_entropy(self.net(xb), yb).backward()
                self.opt.step()
        if self.replay:
            for i in range(len(yl)):
                self._reservoir(Xf[i], int(yl[i]))

    def classify(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X.astype(np.float32))).argmax(1).numpy()


def load_dataset(name, seed):
    """Return (Xtr, Xte, ytr, yte, dim) of unit-norm frozen features, 10 classes.
    'digits' = raw 64-D pixels; 'mnist' = real MNIST -> PCA fit on TRAIN ONLY."""
    if name == 'digits':
        d = load_digits()
        X = _unit(d.data.astype(np.float64)); y = d.target.astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,
                                              random_state=seed, stratify=y)
        return Xtr, Xte, ytr, yte, X.shape[1]

    if name == 'mnist':
        print("Fetching MNIST via fetch_openml (cached after first run)...")
        # needs pandas (sklearn>=1.6 parser='auto' picks the pandas parser for
        # dense data); pip install pandas if this raises ImportError.
        mn = fetch_openml('mnist_784', version=1, as_frame=False)
        Xall = mn.data.astype(np.float64) / 255.0
        yall = mn.target.astype(int)
        rng = np.random.default_rng(seed)
        need = MNIST_TRAIN_PER_CLASS + MNIST_TEST_PER_CLASS
        tr_idx, te_idx = [], []                       # leakage-safe per-class split
        for c in range(N_CLASSES):
            ci = np.where(yall == c)[0]
            assert len(ci) >= need, (
                "class %d has only %d samples (<%d = train+test per class); "
                "lower MNIST_TRAIN/TEST_PER_CLASS" % (c, len(ci), need))
            rng.shuffle(ci)
            ntr = MNIST_TRAIN_PER_CLASS
            tr_idx.extend(ci[:ntr])
            te_idx.extend(ci[ntr:ntr + MNIST_TEST_PER_CLASS])
        tr_idx = np.array(tr_idx); te_idx = np.array(te_idx)
        pca = PCA(n_components=PCA_DIM, random_state=seed).fit(Xall[tr_idx])  # TRAIN only
        Xtr = _unit(pca.transform(Xall[tr_idx])); ytr = yall[tr_idx]
        Xte = _unit(pca.transform(Xall[te_idx])); yte = yall[te_idx]
        return Xtr, Xte, ytr, yte, PCA_DIM

    raise ValueError("unknown dataset '%s' (use 'digits' or 'mnist')" % name)


def _slot_census(model, yte, pred, name):
    """Print per-class used-slot count + per-class test recall for a SAGEProto.
    Confirms/refutes the primacy-starvation diagnosis: if early classes hog the
    slots and later classes have ~0, the global-eviction starvation is real."""
    lbl = model.label[model.used]
    print("  [%s] per-class slots / test-recall:" % name)
    line = "    "
    for c in range(N_CLASSES):
        cm = (yte == c)
        acc = 100.0 * np.mean(pred[cm] == c) if cm.any() else 0.0
        line += "c%d:%2ds/%4.0f%%  " % (c, int(np.sum(lbl == c)), acc)
        if c == 4:
            line += "\n    "
    print(line)
    print("    total used slots: %d / %d" % (int(model.used.sum()), model.key.shape[0]))


def run_ordering(order, Xtr, ytr, Xte, yte, dim, proj, seed, census=False):
    """Stream classes in TASKS (PER_TASK classes each) per `order`; each task is a
    training phase, then its data is gone. Returns per-method (final_acc, task0_acc).
    task0 = the FIRST task's classes (the forgetting indicator).
    census=True prints the per-class slot allocation for the SAGE flat-family."""
    task0 = set(order[:PER_TASK])
    tasks = [order[i:i + PER_TASK] for i in range(0, N_CLASSES, PER_TASK)]
    rng = np.random.default_rng(seed)            # per-ordering within-task shuffle
    # NOTE: SGD/Replay use a FIXED init seed (SEED) across orderings ON PURPOSE -
    # NCM/SAGE are deterministic given the data, so holding the neural init fixed
    # makes order-std measure PURELY the class-arrival-order effect for every
    # method (apples-to-apples), not init noise. The arrival order itself still
    # varies per ordering (the `order` arg + the per-ordering `rng` block shuffle).
    models = {
        "SGD-MLP":  SGDHead(dim, N_CLASSES, replay=0, seed=SEED),
        "Replay":   SGDHead(dim, N_CLASSES, replay=1, seed=SEED),
        "NCM":      NCM(dim, N_CLASSES),
        "NCM-multi": NCMMulti(dim, N_SLOTS),     # storage-matched strong baseline
        "SAGE-flat": SAGEProto(dim, N_SLOTS, mode='flat'),
        "SAGE-pc":  SAGEProto(dim, N_SLOTS, mode='flat', per_class=True),
        "SAGE-grid": SAGEProto(dim, N_SLOTS, mode='grid', proj=proj),
    }
    for task in tasks:
        mask = np.isin(ytr, task)
        Xt, yt = Xtr[mask], ytr[mask]
        p = rng.permutation(len(yt))
        Xt, yt = Xt[p], yt[p]                     # same task block for every model
        for m in models.values():
            m.learn_task(Xt, yt)
    out = {}
    t0_mask = np.isin(yte, list(task0))
    preds = {}
    for name, m in models.items():
        pred = m.classify(Xte); preds[name] = pred
        out[name] = (100.0 * np.mean(pred == yte),
                     100.0 * np.mean(pred[t0_mask] == yte[t0_mask]))
    if census:
        print("SLOT CENSUS (order head = first task = classes %s):"
              % sorted(task0))
        _slot_census(models["SAGE-flat"], yte, preds["SAGE-flat"], "SAGE-flat")
        _slot_census(models["SAGE-pc"], yte, preds["SAGE-pc"], "SAGE-pc")
        # McNemar exact test on the DECISIVE pair (SAGE-pc vs NCM-multi), per-item
        # on this ordering's test set. b = SAGE-pc right & NCM-multi wrong; c = the
        # reverse. Project rule (cf. Stage 4c): no verdict without a significance test.
        a = preds["SAGE-pc"] == yte; bm = preds["NCM-multi"] == yte
        b = int(np.sum(a & ~bm)); c = int(np.sum(~a & bm))
        pval = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
        fav = "NCM-multi" if c > b else ("SAGE-pc" if b > c else "neither")
        print("  McNemar SAGE-pc vs NCM-multi (this ordering, n_test=%d): "
              "discordant b(pc>nm)=%d c(nm>pc)=%d -> p=%.2e, favors %s"
              % (len(yte), b, c, pval, fav))
        print("")
    return out


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'digits'
    ds_label = {'digits': 'sklearn digits',
                'mnist': 'real MNIST (PCA-%d frozen feats)' % PCA_DIM}.get(dataset, dataset)
    Xtr, Xte, ytr, yte, dim = load_dataset(dataset, SEED)
    rng = np.random.default_rng(SEED)
    proj = rng.standard_normal((dim, 3))
    print("%s: %d train / %d test, D=%d, %d classes, %d tasks (%d classes each)."
          % (dataset.upper(), len(ytr), len(yte), dim, N_CLASSES,
             N_CLASSES // PER_TASK, PER_TASK))
    print("Class-incremental: SGD trains %d epochs/task (learns then forgets), "
          "NCM/SAGE single pass. %d random orderings + 1 adversarial.\n"
          % (EPOCHS, N_ORDERS))

    orders = [list(rng.permutation(N_CLASSES)) for _ in range(N_ORDERS)]
    orders.append(list(range(N_CLASSES)))            # adversarial: strict class order
    order_names = ["rand%d" % i for i in range(N_ORDERS)] + ["adversarial"]

    methods = ["SGD-MLP", "Replay", "NCM", "NCM-multi", "SAGE-flat", "SAGE-pc",
               "SAGE-grid"]
    final = {m: [] for m in methods}
    task0 = {m: [] for m in methods}
    per_order = []
    for oi, (od, nm) in enumerate(zip(orders, order_names)):
        res = run_ordering(od, Xtr, ytr, Xte, yte, dim, proj, SEED + oi,
                           census=(oi == 0))      # slot census on the first ordering
        per_order.append({"order": nm, **{k: v[0] for k, v in res.items()}})
        for m in methods:
            final[m].append(res[m][0]); task0[m].append(res[m][1])

    # Storage-budget disclosure so the cross-method comparison is honest: every
    # method is a BOUNDED memory (SAGE slots, Replay buffer, NCM one mean/class).
    # SAGE ties NCM only by spending ~N_SLOTS/N_CLASSES x NCM's storage, and
    # Replay's buffer is a FIXED small budget, not unbounded replay - both stated
    # explicitly below so neither baseline looks stronger/weaker than it is.
    stream = len(ytr)
    task_n = stream // (N_CLASSES // PER_TASK)
    print("Storage budgets (bounded-memory comparison): SAGE %d slots (~%d/class) "
          "| Replay %d buffer (~%d%% of a task's %d samples) | NCM 10 means."
          % (N_SLOTS, N_SLOTS // N_CLASSES, BUF, 100 * BUF // max(task_n, 1),
             task_n))
    print("(SAGE ties NCM at ~%dx NCM storage; Replay buffer is fixed, not "
          "unbounded - both are honest bounded-memory budgets.)\n"
          % (N_SLOTS // N_CLASSES))
    print("%-11s %14s %14s %12s" %
          ("method", "final acc", "first-task acc", "order-std"))
    print("%-11s %14s %14s %12s" %
          ("", "(mean+/-std)", "(forgetting)", "(robustness)"))
    rows = []
    for m in methods:
        fa = np.array(final[m]); t0 = np.array(task0[m])
        rows.append({"method": m, "final_mean": float(fa.mean()),
                     "final_std": float(fa.std()), "task0_mean": float(t0.mean()),
                     "task0_std": float(t0.std()),
                     "adversarial_final": float(fa[-1])})
        print("%-11s %6.1f +/- %4.1f %6.1f +/- %4.1f %11.1f"
              % (m, fa.mean(), fa.std(), t0.mean(), t0.std(), fa.std()))

    print("\n=== Adversarial (strict class-by-class) ordering - the forgetting "
          "stress test ===")
    print("%-11s %12s %14s" % ("method", "final acc", "first-task acc"))
    for m in methods:
        print("%-11s %11.1f%% %13.1f%%"
              % (m, final[m][-1], task0[m][-1]))

    def stat(m):
        return next(r for r in rows if r["method"] == m)

    def verdict(delta, tie=3.0):              # compute the label from the NUMBERS,
        if abs(delta) < tie:                  # never hard-code a conclusion (the old
            return "TIE"                       # template canned 'CRUSHES/~TIE' and
        return ("WIN +%.1f pp" % delta if delta > 0   # misfired when the result flipped)
                else "LOSS %.1f pp" % delta)

    sgd = stat("SGD-MLP"); ncm = stat("NCM"); rep = stat("Replay")
    ncmm = stat("NCM-multi")
    sgf = stat("SAGE-flat"); spc = stat("SAGE-pc"); sgg = stat("SAGE-grid")
    d_sgd = sgf["final_mean"] - sgd["final_mean"]
    d_ncm = sgf["final_mean"] - ncm["final_mean"]
    d_rep = sgf["final_mean"] - rep["final_mean"]
    d_grid = sgf["final_mean"] - sgg["final_mean"]
    d_pc_flat = spc["final_mean"] - sgf["final_mean"]   # does per-class fix help?
    d_pc_ncm = spc["final_mean"] - ncm["final_mean"]
    d_pc_ncmm = spc["final_mean"] - ncmm["final_mean"]  # THE decisive test
    d_ncmm_ncm = ncmm["final_mean"] - ncm["final_mean"] # is it just multi-proto?
    # across-ordering sign test: how often does NCM-multi beat SAGE-pc?
    nmwin = int(np.sum(np.array(final["NCM-multi"]) > np.array(final["SAGE-pc"])))
    n_ord = len(final["SAGE-pc"])
    sign_p = binomtest(min(nmwin, n_ord - nmwin), n_ord, 0.5).pvalue
    print("\n" + "=" * 72)
    print("SAGE-flat vs SGD-MLP (naive neural net): final %+.1f pp [%s], "
          "first-task %+.1f pp, order-std %.1f vs %.1f"
          % (d_sgd, verdict(d_sgd), sgf["task0_mean"] - sgd["task0_mean"],
             sgf["final_std"], sgd["final_std"]))
    print("SAGE-flat vs Replay (neural + buffer): final %+.1f pp [%s]"
          % (d_rep, verdict(d_rep)))
    print("SAGE-flat vs NCM (simple no-forget baseline): final %+.1f pp [%s]"
          % (d_ncm, verdict(d_ncm)))
    print("SAGE-flat vs SAGE-grid (does 3-D geometry help?): final %+.1f pp [%s]"
          % (d_grid, verdict(d_grid)))
    print("-" * 72)
    print("ALLOCATION TEST -- SAGE-pc (per-class pools) vs SAGE-flat (global "
          "eviction): final %+.1f pp [%s]  (big WIN => flat collapse WAS slot "
          "starvation)" % (d_pc_flat, verdict(d_pc_flat)))
    print("SAGE-pc vs NCM (single centroid, 10x less storage): final %+.1f pp [%s]"
          % (d_pc_ncm, verdict(d_pc_ncm)))
    print("NCM-multi vs NCM (is multi-prototype itself the win?): final %+.1f pp [%s]"
          % (d_ncmm_ncm, verdict(d_ncmm_ncm)))
    print("*** DECISIVE -- SAGE-pc vs NCM-multi (storage-MATCHED multi-proto "
          "baseline): final %+.1f pp [%s] ***" % (d_pc_ncmm, verdict(d_pc_ncmm)))
    print("  TIE => SAGE-pc == a well-built multi-prototype store (no mechanism "
          "edge); WIN => SAGE's mechanism genuinely beats k-means/class.")
    print("  SAGE-pc %.1f | NCM-multi %.1f | NCM %.1f | flat %.1f"
          % (spc["final_mean"], ncmm["final_mean"], ncm["final_mean"],
             sgf["final_mean"]))
    print("  SIGNIFICANCE: NCM-multi beats SAGE-pc on %d/%d orderings (sign-test "
          "p=%.3f); per-item McNemar printed in the SLOT CENSUS block above."
          % (nmwin, n_ord, sign_p))
    print("=" * 72)

    out = {"experiment": "forgetting_benchmark", "dataset": dataset,
           "dim": dim, "n_train": len(ytr), "n_test": len(yte),
           "n_classes": N_CLASSES, "per_task": PER_TASK,
           "n_orders": N_ORDERS + 1, "n_slots": N_SLOTS, "summary": rows,
           "per_order": per_order}
    suffix = '' if dataset == 'digits' else '_' + dataset
    res_path = os.path.join(RES_DIR, 'forgetting_benchmark%s.json' % suffix)
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print("\nSaved: %s" % res_path)

    # plot: final acc + first-task acc per method (mean +/- std)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0a0a0a')
    ax.set_facecolor('#111111')
    xc = np.arange(len(methods))
    fm = [stat(m)["final_mean"] for m in methods]
    fs = [stat(m)["final_std"] for m in methods]
    tm = [stat(m)["task0_mean"] for m in methods]
    ts = [stat(m)["task0_std"] for m in methods]
    ax.bar(xc - 0.2, fm, 0.38, yerr=fs, color='#4fc3f7', label='final accuracy',
           capsize=3)
    ax.bar(xc + 0.2, tm, 0.38, yerr=ts, color='#ff7043',
           label='first-task accuracy (forgetting)', capsize=3)
    ax.set_xticks(xc); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('accuracy %', color='white')
    ax.set_title('Class-incremental forgetting (%s): final vs first-task '
                 'accuracy per method' % dataset, color='white')
    ax.legend(facecolor='#111111', edgecolor='#333333', fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(RES_DIR, 'forgetting_benchmark%s.png' % suffix)
    fig.savefig(plot_path, dpi=130, facecolor='#0a0a0a')
    plt.close(fig)
    print("Saved: %s" % plot_path)

    entry = (
        "\n## Forgetting + order-robustness benchmark (continual learning) [%s]\n\n"
        "- %s, class-incremental (%d tasks x %d classes), SGD %d "
        "epochs/task (then forgets), %d orderings incl. adversarial. Frozen "
        "features (D=%d).\n"
        "- Final acc (mean+/-std) / first-task acc (forgetting): SGD-MLP "
        "%.1f/%.1f, Replay %.1f/%.1f, NCM %.1f/%.1f, NCM-multi %.1f/%.1f, "
        "SAGE-flat %.1f/%.1f, SAGE-pc %.1f/%.1f, SAGE-grid %.1f/%.1f.\n"
        "- SAGE-flat deltas (final acc, label computed from the numbers - no "
        "canned verdict): vs SGD-MLP %+.1f pp [%s]; vs Replay %+.1f pp [%s]; vs "
        "NCM %+.1f pp [%s]; vs SAGE-grid %+.1f pp [%s]. order-std (robustness): "
        "SAGE-flat %.1f vs NCM %.1f vs SGD %.1f.\n"
        "- ALLOCATION TEST: SAGE-pc (per-class slot pools, fixes the global-"
        "eviction starvation /code-review found) vs SAGE-flat %+.1f pp [%s]; "
        "vs NCM (1 mean/class, 10x less storage) %+.1f pp [%s].\n"
        "- DECISIVE (storage-matched): NCM-multi vs NCM %+.1f pp [%s] (is multi-"
        "prototype itself the win?); SAGE-pc vs NCM-multi %+.1f pp [%s] -- TIE => "
        "SAGE-pc == a well-built multi-prototype store (no mechanism edge); WIN => "
        "SAGE's mechanism beats k-means/class.\n"
        "- AUTO-SUMMARY (facts only). Honest interpretation written BY HAND after "
        "/code-review - the old template hard-coded a 'CRUSHES/ties-NCM' conclusion "
        "(true for digits, FALSE for mnist), removed. See the manual prose verdict.\n"
        % (dataset, ds_label, N_CLASSES // PER_TASK, PER_TASK, EPOCHS,
           N_ORDERS + 1, dim,
           sgd["final_mean"], sgd["task0_mean"], rep["final_mean"],
           rep["task0_mean"], ncm["final_mean"], ncm["task0_mean"],
           ncmm["final_mean"], ncmm["task0_mean"],
           sgf["final_mean"], sgf["task0_mean"], spc["final_mean"], spc["task0_mean"],
           sgg["final_mean"], sgg["task0_mean"],
           d_sgd, verdict(d_sgd), d_rep, verdict(d_rep), d_ncm, verdict(d_ncm),
           d_grid, verdict(d_grid),
           sgf["final_std"], ncm["final_std"], sgd["final_std"],
           d_pc_flat, verdict(d_pc_flat), d_pc_ncm, verdict(d_pc_ncm),
           d_ncmm_ncm, verdict(d_ncmm_ncm), d_pc_ncmm, verdict(d_pc_ncmm))
    )
    with open(FINDINGS, 'a', encoding='utf-8') as f:
        f.write(entry)
    print("Appended forgetting-benchmark entry to findings.md")


if __name__ == '__main__':
    main()
