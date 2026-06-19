# SAGE-Sphere - Findings Log

Running log, appended at the end of every stage. Negative results are recorded
honestly - a closed-negative (cf. Test 9E) is a valid outcome.

## Stage 0 environment (pre-run)

- Ollama up with `nomic-embed-text:latest` pulled (confirmed).
- Seed cache: `../analogy_cache/nomic_cache.json` = 358 real analogy-benchmark
  words (capitals, countries, family relations), 768-D nomic vectors.
- GloVe 6B 50d present (`../data/glove.6B.50d.txt`) - used to extend vocab to
  ~3000 frequency-ordered words.
- Reference script `../SAGE_SPHERE_*` / `anisotropy_diagnostic.py` reviewed:
  math correct; adapted into `experiments/stage0_anisotropy.py` (cache reuse,
  histogram plot, results/ + findings output, utf-8, Agg dark theme).
- NOTE for Stage 6: brief says "reuse existing Qiskit/AerSimulator oracle
  circuit" but no Qiskit circuit exists in the cube codebase - only the
  SAGEDivided geometry (x=0 partition). Stage 6 oracle is build-from-scratch.

## Stage 0 - Anisotropy diagnostic

- model: nomic-embed-text | words: 358 | dims: 768 | pairs: 9969
- mean cosine: 0.4237 | mean |cos|: 0.4237 | std: 0.0702
- mean-vector norm: 0.6528 | top-1 PC: 5.9% | top-5 PC: 16.3%
- **VERDICT: HIGH** (mean|cos| = 0.4237)
- Severe anisotropy - composition failure (8A/9C) confirmed to live in the EMBEDDINGS. Stage 2 REQUIRED; sphere fixes the metric, NOT composition; graph traversal is the fix.

## Stage 0 - Anisotropy diagnostic

- model: nomic-embed-text | words: 3000 | dims: 768 | pairs: 9995
- mean cosine: 0.4337 | mean |cos|: 0.4337 | std: 0.0547
- mean-vector norm: 0.6585 | top-1 PC: 2.3% | top-5 PC: 7.8%
- **VERDICT: HIGH** (mean|cos| = 0.4337)
- Severe anisotropy - composition failure (8A/9C) confirmed to live in the EMBEDDINGS. Stage 2 REQUIRED; sphere fixes the metric, NOT composition; graph traversal is the fix.

## Stage 1 - Fibonacci-sphere substrate

- 3000 words placed on 4096-point S^2 lattice (cube reference 16^3).
- **GATE (storage integrity): top-1 self-retrieval = 100.00% -> PASS.** Exact 768-D cosine is position-independent, so no-regression vs the cube holds BY CONSTRUCTION (retrieval never uses grid coordinates); the falsifiable check is that placement preserved the payloads.
- Position semantics (Spearman vs 768-D cosine): continuous PCA-3D direction r=+0.269 (clean, PRIMARY); snapped sphere r=+0.269 and snapped cube r=+0.230 are confounded references (different metric + lattice resolution), NOT a controlled 'sphere beats cube' claim. The continuous number is how much 768-D angular structure survives the 768->3 projection.
- Collisions: 1921/4096 points used, 729 collided, max bucket 9, 60.3% of items collided.
- Positional recall@10 vs exact neighbours (self excluded): 10.2% - coarse 3-D routing is lossy; retrieval uses the full 768-D payload, NOT the 3-D position.
- DESIGN NOTE: grid is 3-D (visualizable), payload is 768-D (exact retrieval). place() snaps a gradient-free PCA 768->3 projection onto the nearest lattice point. FLAG for Stage 3: build the kNN graph in FULL 768-D cosine, using 3-D positions only for viz/partition, since top-3 PCA captures only ~8% of variance.

## Stage 2 - Isotropy preprocessing (All-but-the-Top)

- Swept D in [0, 1, 2, 3, 5, 7, 10] on 3000 words (D=0 = untreated baseline).
- Raw mean|cos| = 0.4331. **Best D = 1 -> mean|cos| = 0.0480 (88.9% drop), neighbour Jaccard vs raw = 0.722, continuous-3D Spearman = +0.321.**
- **GATE: PASS** (require >=10% mean|cos| drop AND Jaccard >= 0.50).
- Re-run Stage 0 diagnostic on processed embeddings: mean|cos| 0.4331 -> 0.0480. Anisotropy reduced as expected.
- Downstream: continuous-3D correlation moved +0.269 -> +0.321 vs the Stage 1 raw baseline (+0.269); isotropy helps the 3-D projection.
- Saved processed embeddings to data/embeddings_isotropic.npz (D=1); downstream stages use these.
- NOTE: neighbour-Jaccard-vs-raw is a SCRAMBLE check (isotropy is meant to shift some neighbourhoods), NOT a quality score. Whether isotropy HELPS composition is decided at Stage 4 (analogy on raw vs isotropic); both embedding sets are kept (raw cache + isotropic).

## Stage 3 - kNN-relation graph

- Built in FULL 768-D cosine (per Stage 1 flag) on ISOTROPIC embeddings; raw compared. 3000 words, k in [10, 20, 50].
- **GATE: PASS** (>=70% within <=2 hops) - chosen isotropic k=10: 1-hop = 96.5%, <=2-hop = 98.5%, <=3-hop = 99.6%.
- Connectivity: giant component 100.0% of nodes, 1 components, mean degree 12.9.
- Edge labelling (Hebbian co-occurrence): 250/259 benchmark triples are DIRECT edges (96.5%), 226 labelled edges.
- Per-relation 1-hop coverage (chosen config):
    - capital-common-countries       23 pairs | 1hop 100% <=2hop 100%
    - capital-world                 116 pairs | 1hop 93% <=2hop 97%
    - family                         23 pairs | 1hop 96% <=2hop 100%
    - gram3-comparative              10 pairs | 1hop 100% <=2hop 100%
    - gram5-present-participle       15 pairs | 1hop 100% <=2hop 100%
    - gram6-nationality-adjective    22 pairs | 1hop 100% <=2hop 100%
    - gram7-past-tense               40 pairs | 1hop 100% <=2hop 100%
    - gram8-plural                   10 pairs | 1hop 100% <=2hop 100%
- Saved chosen kNN to data/stage3_knn.npz for Stage 4. NOTE: <=3-hop coverage is inflated by small-world connectivity; 1-hop (partner is a nearest neighbour) is the signal that matters for typed traversal.

## Stage 4 - Dijkstra traversal (GO/NO-GO)

- Leakage-safe: dir_R from TRAIN pairs only; analogy questions from TEST pairs (4 distinct words); multi-hop countries split train/test. K_CAND=50.
- **VERDICT (isotropic): NO-GO.** Gated on CLEAN same-pool comparisons.
- Analogy Hits@1 micro (isotropic): A arith-glob=85.1%, Aknn arith-kNN=85.5%, A' proto-glob=74.0%, B proto-kNN=85.2%. Macro B=90.9%. answer-in-kNN(c)=100.0%.
  - **CLEAN gate (B vs Aknn, same pool): -0.3** (relation mechanism, not pool size). Context brief (B vs A): +0.1.
- Analogy Hits@1 micro (raw): A=86.0%, Aknn=86.1%, A'=74.4%, B=85.1%.
- Multi-hop (isotropic): A arith=61.8%, Bg reground-global=38.2%, B reground-kNN=70.6%; hop-1 country=73.5%, cont-in-kNN=73.5%.
  - **CLEAN gate (Bg vs A, isolates re-grounding): -23.5.**
- Multi-hop (raw): A=58.8%, Bg=35.3%, B=64.7%.
- GATE: GO = re-grounding beats arith on multi-hop by >=5 AND B>=Aknn on analogy; NO-GO = loses by 5; else TIE. Stop and analyse honestly; Stage 1 substrate still stands.

## Stage 4b - Hybrid follow-up (multi-hop)

- Same leakage-safe split as Stage 4. Methods isolate restriction vs greedy commitment. K_CAND=50, beam=5.
- Multi-hop (isotropic): A arith-global=61.8%, A_r2 arith-2hop=61.8%, B greedy-kNN=70.6%, Beam beam-kNN=64.7%.
- Multi-hop (raw): A=58.8%, A_r2=58.8%, B=64.7%, Beam=70.6%.
- A_r2 - A = +0.0 (restriction on PLAIN arithmetic); Beam - B = -5.9 (soft vs greedy).
- **HYBRID MIXED: B (greedy traversal) is best (+8.8 vs A); restricted arithmetic alone does NOT help (A_r2 +0.0) - gain entangled with re-grounding**

## Stage 4c - Powered multi-hop (valid significance)

- 116 chains; ONE prototype on 35 train countries, held-out test on the rest (valid independence, unlike k-fold). 10000 bootstrap resamples.
- **VERDICT (isotropic, held-out n=81): TIE / NOT SIGNIFICANT (cannot reject A == B).**
- Isotropic held-out: A(arith)=55.6%, B(traversal)=55.6%, B-A=+0.0 pp; McNemar p=1.0000 (discordant A>B=18, B>A=18); bootstrap 95% CI [-13.6, +14.8] pp. Descriptive 5-fold CV: A=62.9%, B=52.6%.
- Raw held-out: A=66.7%, B=51.9%, B-A=-14.8 pp; McNemar p=0.0357; CI [-27.2, -2.5].
- McNemar tests discordant pairs; the bootstrap CI is on the marginal accuracy difference - distinct estimands, GO requires BOTH. Single-hop analogy remains a solid null (Stage 4: B~=Aknn~=85%).

## Binding memory - multi-hop (new direction after traversal NO-GO)

- FHRR bind/unbind + geometric cleanup (= sphere nearest-neighbour). Synthetic KB, N=500 entities, ideal random vectors. Swept D in [512, 1024, 2048], R in [1, 2, 4, 8, 16, 32].
- **PROMISING: cleanup-chained binding holds 2-hop up to R=32 facts/entity at D=2048 (2-hop=100.0% vs traversal 62%).**
- Mechanism: cleanup re-grounds each hop on a clean entity, so noise does NOT compound - the property greedy traversal lacked. 2-hop ~ (1-hop)^2.
- NOTE: this is the IDEAL case (near-orthogonal random vectors). Follow-up: repeat on the real isotropic embeddings (correlated -> lower capacity), and if capacity-limited, add a resonator-network cleanup.

## Binding memory on REAL embeddings (HRR, stress-tested)

- HRR circular-convolution binding, frequency-unitary roles, + cleanup. N=500, D=768, R up to 256. Three conditions: ideal (Gaussian), shuffled (column-permuted real = correlation removed, marginals matched), real.
- **Capacity (largest R with 2-hop >= 70%): real=32, shuffled=32, ideal=32 -> binding WORKS up to R=32 facts/entity on real embeddings.**
- Correlation cost (real vs SHUFFLED at R=32, matched marginals) = 2.4 pp.
- Honest framing: unitary roles whiten the unbinding noise, so correlation costs the CLEANUP step (disambiguating among correlated neighbours), NOT binding capacity; the real-vs-shuffled control isolates exactly that. R=1 recovery is algebraically exact (uninformative).

## Agent memory benchmark (bounded self-managing memory vs vector DB)

- Stream of 4000 fact-updates over 500 keys (avg 8.0 updates/key); query current value, fuzzy keys, D=768. Budget sweep [128, 256, 500, 1000].
- At budget B=500 (=#keys): SAGE-flat=100.0%, DB-dedup=100.0%, DB-fifo=63.2%, SAGE-grid=7.8%; DB-unbounded=100.0% at 4000 entries (vs SAGE's 500).
- **SAGE-flat vs DB-fifo (consolidation vs naive evict) = +36.8 pp; SAGE-flat vs DB-dedup (smart bounded store, exact last value) = +0.0 pp (near-TIE; dedup may slightly lead - SAGE has no edge over a good store); SAGE-flat vs SAGE-grid (3-D geometry) = +92.2 pp.**
- HONEST CONCLUSION: SAGE-flat beats NAIVE vector-DB usage (unbounded bloat / FIFO wasting budget on duplicate writes) but TIES a well-designed bounded dedup store (DB-dedup) - because SAGE-flat IS one. The 3-D geometry contributes NOTHING (SAGE-grid collapses via collisions). Defensible claim: the right DESIGN is bounded + self-consolidating + decaying; SAGE is a valid instance, the geometry is not load-bearing.

## STRATEGIC SYNTHESIS (2026-06-18) - what the whole arc established

- **Geometry (3-D/sphere) = decoration.** Positional routing recovers 10% of exact-cosine; 3-D holds ~8% variance; SAGE-grid memory 7.8% vs flat 100%. Confirmed vs baseline.
- **Reasoning = dead.** Traversal ties arithmetic (Stage 4c NO-GO, p=1.0); binding ties a dict.
- **Memory = a good DESIGN, not magic.** SAGE-flat == bounded dedup store (tie); beats naive vector-DB usage (+37pp vs FIFO, 8x less footprint than unbounded).
- **One untested real edge:** gradient-free no-forgetting - beats NEURAL nets (SGD/EWC) structurally, ties simple non-parametric (NCM/Hopfield). NEXT EXPERIMENT: the never-run forgetting + adversarial-ordering benchmark (the cube guide's "geometry proof" + "forgetting benchmark", finally run).
- **Process:** /code-review caught every overclaim (tautological gate, confounded comparisons, k-fold dependence, no-baseline demos) before it was believed. Keep doing that.

## Forgetting + order-robustness benchmark (continual learning) [canonical]

- sklearn digits, class-incremental (5 tasks x 2 classes), SGD 8 epochs/task (honest forgetting; an earlier strawman SGD that single-passed and collapsed to ~chance was caught by /code-review and fixed), 7 orderings incl. adversarial. Frozen features (D=64).
- Final acc (mean+/-std) / first-task acc (forgetting): SGD-MLP 19.8/0.0, Replay 87.4/81.8, NCM 87.0/84.5, SAGE-flat 85.3/84.1, SAGE-grid 35.5/6.2.
- SAGE-flat vs SGD-MLP (neural net): final +65.6 pp, first-task +84.1 pp, order-std 1.3 vs 0.2 (SAGE more robust). vs NCM (simple no-forget): -1.7 pp (~TIE). vs SAGE-grid (3-D geometry): +49.8 pp.
- HONEST: SAGE joins the no-forget, gradient-free, order-robust cluster (with NCM) and beats the NEURAL methods that forget - its one real structural edge, benchmarked. Ties NCM (no edge over the simplest no-forget method); geometry hurts (SAGE-grid). NOTE: SGD forgets as a CLIFF (all-but-last-task -> 0%), not a graded gradient.

## Forgetting + order-robustness benchmark (continual learning) [mnist] -- AUTHORITATIVE (consolidated; supersedes the 4 stale [mnist] appends)

- real MNIST via fetch_openml, PCA-64 frozen features (fit on TRAIN only = leakage-safe), 20000 train / 4000 test, class-incremental (5 tasks x 2 classes), 7 orderings incl. adversarial. SGD 8 epochs/task; NCM/NCM-multi/SAGE single pass. CODE-REVIEWED (the verdict AND the baselines).
- Final acc (mean+/-std) / first-task acc: SGD-MLP 19.7/0.0, Replay 82.9/77.3, NCM 79.7/79.2, NCM-multi 90.8/91.3, SAGE-flat 55.4/98.2, SAGE-pc 86.2/87.1, SAGE-grid 18.1/0.0. order-std (robustness): NCM 0.0, NCM-multi 0.3, SAGE-pc 0.4, Replay 1.7, SAGE-flat 6.0.
- THE ARC (three runs, each /code-reviewed):
  1. SAGE-flat collapsed to 55.4% (loses NCM -24.3pp). /code-review: BUG not ceiling -- global argmin(cnt) eviction with NO decay (the lib core/agent_memory.py:66 decays strength) -> early-task slots un-evictable -> later classes STARVED. SLOT CENSUS proved it: first task (classes 5,9) took 91/100 slots; classes 2,3,4 got ZERO slots / 0% recall.
  2. SAGE-pc (per-class slot pools = 10 slots/class, evict only within class) FIXED allocation -> 86.2% (+30.9 vs flat); census shows every class exactly 10 slots, balanced recall. So the flat collapse WAS cross-class slot starvation -- CONFIRMED.
  3. SAGE-pc beats single-mean NCM (+6.5) but at 10x its storage. The storage-MATCHED strong baseline NCM-multi (online per-class k-means, same 10 protos/class) = 90.8%, BEATING SAGE-pc.
- DECISIVE (storage-matched, significance-tested): NCM-multi BEATS SAGE-pc by 4.6pp, on 7/7 orderings (sign-test p=0.016), per-item McNemar p=3.98e-13 (discordant b=131 pc>nm, c=277 nm>pc). NCM-multi vs single-NCM = +11.1pp -> the gain over single-NCM is MULTI-PROTOTYPE-ness, NOT SAGE's mechanism. SAGE-pc also forgets MORE than NCM-multi (first-task 87.1 vs 91.3).
- HONEST VERDICT (SETTLED, code-reviewed): **the digits-scale "SAGE ties NCM" win does NOT survive at MNIST scale.** Consistent with the whole project -- SAGE = a good DESIGN, not magic: a well-built standard method (online per-class k-means / NCM-multi) BEATS it at equal storage, and SAGE's cosine-merge + EMA + count-eviction adds complexity that slightly HURTS vs plain k-means. WHAT SURVIVES (structural, NOT SAGE-specific): gradient-free, bounded, MULTI-prototype continual memory beats neural nets on catastrophic forgetting (SAGE-pc & NCM-multi crush SGD +66-71, beat Replay +3-8) and is order-robust -- but the BEST instance of that niche is plain per-class k-means, simpler AND better than SAGE. SAGE-flat (single global pool) and single-mean NCM both lose to the multi-proto methods because MNIST classes are multi-modal.
- Caveat: SAGE-pc's merge=0.6/lr=0.3 are untuned for MNIST; a sweep could narrow the 4.6pp gap, but 7/7 + McNemar p=4e-13 make a reversal-to-WIN unlikely. SAGE-grid (3-D geometry) collapses to 18.1% as in every prior benchmark -- geometry remains decoration.

## Kill-shot A - noisy multi-hop VSA bind-chain vs kNN-table cleanup chain

- Real nomic atoms (N=32, D=768), INDEPENDENT random unitary roles (fix: embedding-derived roles broke VSA to chance), random successor map, noisy vector cue each hop (sigma [0.0, 0.25, 0.5, 0.75, 1.0]), H in [1, 2, 4]. VSA sigma=0 1-hop recovery=96.9% (sanity). cleanup=kNN identical for both; VSA adds bind-superpose (one 6144-B bundle) vs explicit dict (128 B).
- **VERDICT: FALSIFIED.** Best VSA-minus-(kNN-table) margin = +0.4 pp across all (H,sigma). Explicit-table + kNN-cleanup matches/beats the VSA chain everywhere at smaller footprint; bind-superpose only adds crosstalk, cleanup=kNN is shared. The load-bearing-geometry claim dies. (Whitened column shows anisotropy is addressed by classical whitening, not VSA.)

## Kill-shot C - SAGE Hebbian Relation Bank vs MATCHED online spherical k-means (offset clustering)

- Label-free relation DISCOVERY: cluster (b-a) offsets of 259 word pairs into 8 gold relation types (nomic, D=768). ARI vs gold, 15-seed dist. Decisive baseline = online spherical k-means (the algorithm HebbBank IS); sklearn KMeans/MiniBatch/Ward for context.
- ARI mean+/-std: SAGE-bank 0.512+/-0.068, online-sph-kmeans 0.544+/-0.043, kmeans-euclid 0.460+/-0.064, minibatch 0.446+/-0.059, agglom 0.576.
- DECISIVE: SAGE-bank vs matched online-spherical, paired diff -0.032, 95% CI [-0.107, +0.061]. SAGE-bank vs best standard -0.064. Syn vs sem ARI 1.000 vs 0.274.
- **VERDICT: FALSIFIED.** SAGE-bank TIES the matched online-spherical k-means (it IS that algorithm); its edge over batch/euclidean k-means is just the online+spherical recipe, both standard. No SAGE mechanism -> NO-GO. ARI-syn >> ARI-sem matches DiffVec 2016 (offsets cluster morphology, not lexical semantics).

## Kill-shot B - fixed-footprint VSA superposition vs hash-vote (equal bytes)

- Task: store K key->value-id facts (V=256), retrieve by key, at a FIXED byte budget. VSA tested at BOTH 8 B/dim (complex bundle) and 4 B/dim (phase-only, charitable) vs HashVote (best d in [1, 2, 4, 8] hashed rows, majority vote) - the strong classical fixed-footprint KV (stores no keys). Paired RNG, 20 trials, mean+/-std; SURVIVES needs >2pp AND >2 s.e.
- Budgets [2048, 8192] B; K swept [5, 10, 20, 40, 80, 160, 320].
- **VERDICT: FALSIFIED.** Best significant VSA-minus-HashVote margin = -990.0 pp; best raw margin = +0.7 pp. HashVote (fixed bytes, no key store, best-of-d) matches/beats BOTH VSA encodings everywhere; the 'a dict can't hold K facts in fixed bytes' claim dies to a hash table. Superposition is dominated, consistent with Kleyko/Frady/Sommer 2022.

## SAGE investigation - CLOSING SUMMARY (all arms falsified, 2026-06-18)

Three "ranked fix" kill-shots, each at EQUAL resource vs its strongest classical baseline, all /code-reviewed. The review caught verdict-MANUFACTURING bugs in ALL THREE before any verdict was trusted: a fake FALSIFIED in A (embedding-correlated roles broke VSA to chance), a fake SURVIVES in C (weak batch-Euclidean baseline + single-run cherry-pick), and a multiple-comparisons/byte-handicap risk in B - all fixed, then re-run.

- **Kill-shot A (noisy multi-hop bind-chain): FALSIFIED.** With a FAIR VSA (independent random unitary roles, 96.9% recovery at sigma=0), the explicit-table + kNN-cleanup chain matches/beats it at EVERY (H,sigma); best VSA margin +0.4pp, at 48x smaller footprint. cleanup=kNN is shared; bind-superpose only ADDS crosstalk. Where geometry helps cleanup, classical WHITENING fixes it (raw 38.7% -> whitened 82.4% at H=4/sigma=0.25), not VSA.
- **Kill-shot B (fixed-footprint superposition): FALSIFIED.** HashVote (best-of-d, fixed bytes, stores no keys) crushes BOTH VSA encodings (charitable 4 B/dim phase-only AND honest 8 B/dim complex) past K~40; at K=320/B=2048 VSA 6.9-12.9% vs HashVote 87.6%. Best raw margin +0.7pp, not significant. The sqrt(K) crosstalk collapse = Kleyko/Frady/Sommer 2022.
- **Kill-shot C (Relation Bank): FALSIFIED.** SAGE-bank (ARI 0.512+/-0.068, 15 seeds) TIES the MATCHED online-spherical k-means (0.544+/-0.043; paired CI [-0.107,+0.061] straddles 0), and sits below agglomerative (0.576). SAGE-bank IS online spherical k-means; its edge over sklearn batch/Euclidean k-means is just the online+spherical recipe, both standard. Syntactic ARI 1.000 vs semantic 0.274 = textbook DiffVec-2016 (offsets cluster morphology, not lexical semantics).

**VERDICT FOR THE WHOLE SAGE PROGRAM:** every arm - 3-D geometry, reasoning/traversal, memory (dedup + continual forgetting), and now VSA binding / superposition / relation-discovery - TIES or LOSES to a well-built standard baseline once that baseline is actually constructed (dict, k-means/NCM-multi, hash-vote, kNN-table, online spherical k-means). SAGE = good engineering of standard ideas; it has NO benchmarked structural edge a classical method lacks. The genuine deliverable is the PROCESS: a disciplined, adversarial, baseline-anchored falsification of a gradient-free Hebbian/VSA memory, in which multi-agent /code-review caught EVERY overclaim - in BOTH directions (fake wins AND fake losses) - before it was believed. NEXT: write up the falsification arc; do not build further SAGE arms without a NEW structural hypothesis that NAMES what a classical method structurally cannot do (none of geometry/reasoning/memory/binding qualified).

## LLM-memory frame - structural niche hunt (5-agent research, 2026-06-18)

User reframe: "the memory is for AI / LLM models." Ran a 5-agent web-research brainstorm (LLM long-term memory landscape; gradient-free test-time niches; Hopfield/attention equivalence; neuromorphic cost model; adversarial triage). Convergent, cited result: **the LLM reframe opens NO new structural niche.**

- WHAT SAGE PROVABLY IS (citations): its READ is transformer attention (Ramsauer 2020, "Hopfield Networks is All You Need"); its merge-WRITE is the delta-rule fast-weight write (Schlag/Irie/Schmidhuber 2021, "Linear Transformers Are Secretly Fast Weight Programmers"); its decay is palimpsest Hopfield (Storkey). No new mechanism. At EQUAL bytes a dense modern-Hopfield layer has EXPONENTIAL capacity vs SAGE's linear-K slots -> SAGE is strictly dominated as a raw store. For LLM memory, SAGE = RAG with a merge-upsert instead of append.
- PROPERTY LATTICE FULLY COVERED: gradient-free+associative = RAG & attention/Hopfield; bounded+consolidating = online k-means / bounded dedup store (SAGE already TIES this, agent_memory + MNIST); continual gradient-free write = RAG. No orphaned cell. The closest-to-orphan (superimposed/key-free) is owned by modern Hopfield (better capacity).
- NICHES CHECKED, ALL COLLAPSE: edge/no-backprop -> a cache suffices; privacy/unlearning -> vector DB deletes rows too, and merge-for-no-raw-text DESTROYS clean per-fact deletion (properties fight); one-shot binding -> KV-cache IS gradient-free Hebbian binding (TTT=linear attention=fast weights, formal); continual personalization -> the already-banked win (beats GRADIENT methods, TIES prototype stores).
- HARDWARE: ties NCM. The "Hebbian write needs no division, NCM's mean does" premise is REFUTED - analog conductance decay implements running means for free. SAGE wins only over backprop memory, shared by all gradient-free methods.
- P(real structural LLM-memory niche) ~= 0.06. Skeptic's warning: do not gerrymander a niche by stacking qualifiers until only SAGE fits (the failure mode behind every prior overclaim). The matched-consolidation test has ALREADY been run twice (both ties).
- NET: no arm six. The deliverable is the falsification writeup, now strengthened by placing SAGE precisely in the associative-memory family (attention/Hopfield/fast-weights at lower capacity). Key papers: Titans 2501.00663; TTT 2407.04620; Hopfield-is-attention 2008.02217; Fast-Weight-Programmers 2102.11174; Larimar 2403.11901; Gated DeltaNet 2412.06464; in-memory associative memory Nat.Commun. 2026 (2505.12960); exact-gradient-AIMC 2406.12774.

## Papers assessment + honest v2 drafts (2026-06-18)

Reviewed the user's PUBLISHED v1 preprints in `PAPERS/` against the whole investigation. Outcome: v1 OVERCLAIMS and is missing the key baselines.
- **Main paper** (`sage_arxiv_paper_final.pdf`, "Memory for All"): headline "the geometry computes; weights not needed" CONTRADICTS its own discussion (geometry = addressing layer). Compares SAGE only to MLPs/neural nets, NEVER to NCM/k-means/dedup-store (the real competitors that tie/beat it). "0.000% sparsity / infinite efficiency" is misleading (retrieval scans all N^3 slots). "100% rollout" = an explicit dictionary (conceded). Analogy 58.3% = GloVe's own arithmetic, degraded by the cube. A reviewer rejects on the missing NCM baseline alone.
- **Drone paper** (`sage_drone_paper_final.pdf`): much healthier (explicit proof-of-concept + honest 6-gap analysis; graceful-degradation is structurally true). Fixable: "RAG can't / only SAGE can do this" is FALSE (any local vector-store fallback gives zero defaults) -> add the local-store baseline; drop "geometry computes".
- **`LLI-CAMS...Dreaming...pdf` is NOT the user's** (Cleilson Elias de Sousa, UFRJ) -- a separate memory-for-AI framework; leave it.
- DELIVERED honest v2 full manuscripts: `PAPERS/sage_arxiv_v2_full.md` (main, baseline-augmented Table 1 with today's numbers: SAGE beats neural nets, ties NCM, loses to per-class k-means 86.2 vs 90.8 McNemar p=4e-13; geometry decorative; sparsity withdrawn; positioned as attention/Hopfield/delta-rule family), `PAPERS/sage_drone_v2_full.md` (local-store baseline + Gap 7), and `PAPERS/sage_v2_honest_revision.md` (claim-by-claim corrections + arXiv changelog). The v2 = modest, defensible, integrity-preserving (a researcher who tested their own idea and reported it straight). Next: exact numbers from results/*.json, regenerate figures, convert to Word/LaTeX.
