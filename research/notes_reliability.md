# Low-compute VLM reliability project audit

Research cutoff: **14 September 2026**. This is a targeted primary-source audit, separate from the 300-record title/category screen in `corpus/screened_reliability.json`. The source ledger records what was actually read: abstract, targeted full-text sections, or official model/data page. Opening a paper is not counted as reading the entire paper. No experiments have been run and no novelty or publication outcome is guaranteed.

## Recommendation

Two candidates survive this audit as worthwhile pilots: **(1) distinguish repeated visual evidence from genuinely informative additional views**, and **(2) predict when a VLM should isolate images versus let them interact**. Both can use frozen 3B–8B models and small existing datasets. Candidate 2 has the clearest tension in published methods; candidate 1 has the cleanest controlled benchmark construction, but now has a direct conceptual precursor in text RAG. Neither should be described as an untouched topic.

The useful novelty standard is a sharply identified failure with causal controls, two independent model families, and a simple remedy that beats the strongest cheap baseline. A broad new hallucination benchmark, another crop/zoom policy, or another hidden-state confidence probe is unlikely to meet it.

## 1. Repeated pixels versus informative new views

**Research question.** When a multi-image VLM receives additional images, can it distinguish redundant presentations of evidence from views that actually resolve a visual uncertainty? Does repetition inflate confidence on incorrect answers after controlling for image position, input length, and any accuracy improvement from repeated processing?

**Concrete example.** A gallery contains a clear image with a red cup and a misleading, low-resolution crop of a blue cup. Repeating the misleading crop should not manufacture reliable support for the claim that every cup is blue. Conversely, a new close-up that reveals the true color of a previously ambiguous cup should be allowed to improve the answer and its confidence. The second case matters: blindly discarding everything derived from the same photograph can remove useful detail at a model's finite input resolution.

**Proposed contribution.** A small, carefully controlled benchmark of evidence dependence versus view complementarity, plus an inexpensive selector/aggregator that preserves newly visible detail while suppressing source repetition. The contribution is visual evidence accounting and calibrated behavior under controlled interventions. It is not the invention of deduplication or the general idea that copied sources are dependent.

### Closest work and what remains different

| Primary source | Existing contribution | Required distinction |
|---|---|---|
| [Counting Copies as Evidence](https://www.researchgate.net/publication/412230355_Counting_Copies_as_Evidence_Confidence_Inflation_from_Dependent_Evidence_in_Retrieval-Augmented_Generation_RAG), Aug 2026 position preprint | Proposes dependence-aware text-RAG aggregation, controlled provenance, independent-evidence counts, and confidence-inflation diagnostics. | Direct conceptual precursor. Cite it prominently. Show empirical visual phenomena, especially informative crops/new views that naive common-origin collapse would discard. |
| [MIMIC](https://aclanthology.org/2026.findings-acl.366/), ACL Findings 2026 | Controls image count, sequence length and the spread of object information; studies counting, listing, common and odd-one tasks. | These controls are established. Add a fixed-content replication axis and a positive control where a new view genuinely changes visual sufficiency. |
| [MIOH](https://arxiv.org/abs/2608.30653), Aug 2026 | Evaluates existence, counting, attributes and position across reasoning patterns and contextual/perceptual pressures. | Reuse its tasks, but isolate repeated-source dependence from merely increasing gallery size. |
| [CGC](https://arxiv.org/abs/2604.22498), Apr 2026 | Builds correlated original/focus/context crops with source-image grounding supervision and uses GRPO. | Cross-crop identity and source grounding are established. Focus on confidence and evidence accounting without GRPO. |
| [Utility-MRAG](https://aclanthology.org/2026.acl-long.1620/), ACL 2026 | Uses lightweight surrogate models to estimate downstream utility of candidate visual evidence without training. | Evidence selection alone is insufficient novelty. Include this as a comparator when selecting views. |
| [SoFA](https://arxiv.org/abs/2503.13792), [FOCUS](https://arxiv.org/abs/2508.13744), [delimiter scaling](https://arxiv.org/abs/2602.01984) | Existing order/leakage mitigation methods. | Show any repetition effect survives order randomization and at least one strong leakage remedy. |

**Method.** Start with exact hashing and perceptual hashing as deliberately strong cheap baselines. Then test a query-conditioned rule: group exact/near-duplicate views; keep a representative plus a view only if it provides new query-relevant detail, estimated using aligned visual patches or a small frozen evidence-utility model. Preserve explicit source and view IDs. Aggregate per-view predictions once per redundant cluster rather than treating each copy as independent support. For proof-of-concept, use deterministic set operations for existential/universal object claims instead of a second large generator.

**Data.** Construct 300–800 base galleries from [COCO](https://cocodataset.org/) with 2–4 source images. Add exact repeats, mild JPEG/resizing variants, and controlled crops. Use selected [NLVR2](https://lil.nlp.cornell.edu/nlvr/) statements or MIOH-style quantified claims for external validation. NLVR2 direct image access may require its form; COCO construction is the immediate fallback. Use native images/crops rather than diffusion editing. Manually verify absence labels and 100–200 difficult cases: an unannotated object is not automatically absent.

**Essential controls.**

- Fixed number of images, fixed visual-token budget, balanced source position, and random insertion order. Compare duplicates with equally numerous irrelevant images.
- A duplicated object image may legitimately change an image-count answer. Include explicit image-occurrence counting versus unique-source/set claims and do not score them with one invariance rule.
- Same-source complementary crops and genuinely new viewpoints must remain useful. Report the rate at which a remedy discards a necessary view.
- Confidence increases accompanied by better accuracy are not automatically failures. Report calibration, error-conditioned confidence changes, and selective risk, not only raw confidence inflation.
- Split by original source/gallery before generating variants; bootstrap confidence intervals by base gallery. Repeated variants are not independent examples.

**Baselines and metrics.** Vanilla concatenation; exact/perceptual deduplication; CLIP/MMR diversity selection; Utility-MRAG if implementation permits; per-image prediction with deterministic logic; one of delimiter scaling/SoFA/FOCUS; oracle source grouping as an upper bound. Main metrics: accuracy and paired answer consistency, false-positive rate on universal claims, Brier score, ECE as a secondary descriptive metric, risk–coverage curves, coverage at fixed empirical risk, necessary-view retention, and measured latency/peak memory. Avoid claiming a distribution-free guarantee from an empirical curve.

**Compute planning estimate, not measured.** A minimal study with two compact models, 300 galleries and a small condition grid should fit roughly **8–20 single-GPU hours**. A broader 3-family study plus ablations is approximately **25–80 GPU-hours**, with 24–48 GB VRAM, short answers and bounded image tokens. Begin with Qwen2.5-VL-3B or Qwen3-VL-4B; add an independent family rather than treating two Qwen sizes as independent replication. Quantize only if needed and evaluate confidence in the same precision used for calibration. These estimates depend strongly on actual hardware and inference implementation.

**48-hour pilot.** In the first work window, build 100–150 manually checked galleries with 0/1/3 repeats and one complementary-view condition. Run two model families with ≤8 generated tokens and cap the GPU portion at 2–4 hours. Continue if repetition causes a meaningful increase in error-conditioned confidence or at least a 5-point paired accuracy/false-assertion change that survives position controls, and a simple dependence-aware baseline recovers at least half of that gap while preserving complementary-view gains. Kill or downgrade if effects disappear under prompt clarification, fixed image count, or exact hashing; if the entire remedy is ordinary hash deduplication; or if the informative-view control cannot be labeled reliably. Thresholds are planning criteria, not statistical significance claims.

**Novelty assessment:** medium-to-high collision risk at the concept level; medium uncertainty about a defensible visual-specific empirical contribution. Good class project even if the remedy is simple, provided the benchmark and causal controls yield a reproducible finding. Do not market it as the first work on source dependence.

## 2. When should images interact?

**Research question.** Why do some successful multi-image methods encourage cross-image attention while others suppress it? Can the required evidence structure of a question predict which intervention helps, on the same image set and model?

**Concrete paired task.** On the same two photographs, ask “What color is the cup in image A?” and “Do the cups in A and B have the same color?” The first requires source isolation; the second requires integrating two observations. More difficult pairs add entity matching, view changes, or distracting near-duplicate objects. The hypothesis that attention policy should vary with evidence structure is to be tested, not assumed.

### Closest work and collision analysis

- [SoFA](https://arxiv.org/abs/2503.13792) interpolates between causal and bidirectional attention to reduce position bias.
- [CAPL](https://arxiv.org/abs/2603.07048) selectively enables inter-image interaction and trains preferences. Its method already uses fixed equal mixing with causal attention and alternating layers, and explicitly acknowledges that temporal/single-image tasks may need different structure. Therefore simply mixing masks, alternating layers, or saying “different tasks need different attention” is not novel.
- [Delimiter scaling](https://arxiv.org/abs/2602.01984) strengthens image separation through hidden-state scaling without extra decoding. [FOCUS](https://arxiv.org/abs/2508.13744) is an existing focused inference/decoding method. Both are essential isolation comparators.
- [MIMIC](https://aclanthology.org/2026.findings-acl.366/) already finds that blocking inter-image visual attention in deeper layers can help and that early-layer blocking can hurt. A repeat of its layer sweep is not a paper contribution.
- [RSCD/MVH-Bench](https://arxiv.org/abs/2603.23934) targets reference/view confusion with contrastive attention masking. [AVAM](https://arxiv.org/abs/2508.17860) already performs adaptive visual anchoring and compression. “Adaptive attention” by itself is too broad a claim.

**Proposed contribution.** Establish a reproducible interaction between question evidence structure and the sign of an intervention's benefit. Then train a tiny query-only router over two or three **existing frozen policies**. The strongest result would be that a policy learned on one task/domain predicts which intervention helps on unseen tasks, while outperforming the single best fixed policy under matched latency and without hurting either task group. The primary finding may be explanatory even if the router gains are modest.

**Method.** Evaluate baseline, a separation intervention (delimiter scaling or deep-layer block masks), and an integration intervention (SoFA-style mixing) offline. Fit logistic regression or a tiny MLP from a frozen text embedding of the question plus cheap gallery metadata to choose among these policies. Keep the large model frozen. A query-only rule avoids an extra VLM prefill; if routing uses VLM hidden states and requires rerunning the prompt, charge that cost. Compare explicit keyword/template routing, learned routing, random routing with matched route frequencies, the globally best policy chosen on validation, a fixed mixture, and an oracle chosen using correctness. Do not use gold task labels at test time except in an oracle analysis.

**Data.** 300–600 galleries with matched isolation/integration questions; build easy controlled pairs from COCO and harder grounding pairs from public multi-image benchmarks. Hold out entire question template families and source images. Add MIMIC or MIOH tasks as external tests. Include tasks whose answer genuinely depends on multiple images, not just a disguised single-image query. To avoid confounding, compare attention policies within one backbone first, then replicate on a second architecture with compatible hooks.

**Metrics.** Accuracy by evidence-structure group, paired effect of each policy, router regret to oracle, worst-group accuracy, position sensitivity, source attribution accuracy where labels exist, and end-to-end latency/memory. The apparent benefit of a router must exceed the gap obtained by simply choosing the best fixed hyperparameter on validation. Report uncertainty over base galleries and distinguish exploratory discoveries from held-out confirmation.

**Compute planning estimate, not measured.** Approximately **10–25 GPU-hours** for a minimum frozen-model experiment; **30–90 GPU-hours** for three families, policy sweeps and held-out evaluation. A 24 GB card should be the initial target with a 3B–4B model and capped image tokens. Full attention materialization can dominate memory and latency; use it only for small diagnostic batches, and benchmark deployable paths separately. Mask-based variants may lose optimized attention kernels; no zero-overhead claim until timed.

**48-hour pilot.** Create 100 paired-question galleries. On one small backbone, compare vanilla, modest delimiter scaling, and one literature-based cross-image attention policy. Continue only if policy preferences vary systematically with task structure and the oracle policy has at least a 3-point advantage over the best fixed policy. Validate the interaction on a second backbone before building a router. Kill if the same fixed intervention wins nearly everywhere, if prompt-only routing performs as well as a learned method with no further insight, or if the gain is entirely caused by an attention implementation mismatch. A null result is a useful course report but is a weaker paper pitch.

**Novelty assessment:** medium collision risk and medium uncertainty. This is a narrower and more defensible target than proposing another mask, but prior work has already discussed the tradeoff. The contribution must be a predictive, independently validated explanation, not a literature juxtaposition.

## 3. Reserve direction: reliability when the inference configuration changes

An operationally relevant question for a crowded cluster is whether a confidence/abstention policy calibrated on one precision, resolution or image budget remains valid after switching configurations. However, this is **not a top recommendation** after the collision audit. [Bigger or Cheaper?](https://arxiv.org/abs/2607.24440) directly studies quantization, degradation and uncertainty. [BEFORE THE FLIP](https://arxiv.org/abs/2609.06922) studies hidden score/grounding shifts and finds no robust benefit from per-question precision allocation. [BCEA](https://arxiv.org/abs/2606.16667) explicitly recalibrates the complete acquisition policy and permits sequential stopping inside a fixed pipeline.

A narrow surviving variant would audit **configuration switching chosen from input difficulty**, with fixed-policy calibration versus route-conditional calibration and strict configuration provenance. But this risks being an application of standard conditional calibration, and adjacent CLIP ensemble work on route-conditional conformal validity has also appeared. It needs a more exhaustive statistics novelty search before being sold as a paper. Minimal class study: 2 models × 2 precisions × 2 input budgets, cached scores, 500 base questions, 8–25 estimated GPU-hours, bootstrap risk–coverage by source. A 48-hour pilot should stop if the effect is fully fixed by ordinary per-configuration temperature/threshold refitting. This is a practical fallback, not a claim of new calibration theory.

## Popular ideas already covered

| Tempting proposal | Closest primary-source collision |
|---|---|
| Crop/zoom again when uncertain and then abstain | [BCEA](https://arxiv.org/abs/2606.16667), including post-acquisition risk calibration and structured flips |
| Predict hallucination with a tiny hidden-state classifier | [HALP](https://arxiv.org/abs/2603.05465); [TruthLens](https://arxiv.org/abs/2608.05616) exposes truthfulness at the output head |
| Learn an abstention direction for missing visual evidence | [TRAPSBench](https://arxiv.org/abs/2608.13167), including matched pairs, linear probes and activation steering |
| Separate perception confidence from reasoning confidence | [VL-Calibration](https://arxiv.org/abs/2604.09529) |
| Show image removal or corruption exposes textual shortcuts | [Treble Counterfactual VLMs](https://arxiv.org/abs/2503.06169), [causal sufficiency/necessity](https://arxiv.org/abs/2508.04182), and many earlier contrastive-decoding papers |
| Build a general multi-image hallucination benchmark | [MIHBench](https://arxiv.org/abs/2508.00726), [MIOH](https://arxiv.org/abs/2608.30653), [MVH-Bench](https://arxiv.org/abs/2603.23934), [MIMIC](https://aclanthology.org/2026.findings-acl.366/) |
| Show shorter answers reduce hallucination without better grounding | [Does Playing it Safe Count as Faithfulness?](https://arxiv.org/abs/2609.01888), 1 Sep 2026 |
| Give arbitrary names to visual entities to help comparison | [VLMs Need Words](https://arxiv.org/abs/2604.02486) |

## Broad title screen record

I read all 300 assigned title/category rows, IDs 1–300. Dispositions are **251 core VLM, 47 adjacent, 2 explicitly nonvisual**. This is title screening only. General optimization, pure vision, visual generation/world modeling and ambiguous foundation-model entries were treated as adjacent when their title did not establish visual-language understanding. The two exclusions were speech generation/editing and textual web-application generation. Row 187 has a trailing formatting fragment but an interpretable title, so it is not unparseable.

The broad scan independently surfaced TruthLens, HALP, VL-Calibration and VLMs Need Words; their abstracts were subsequently verified on arXiv and entered separately in the source ledger. Other useful title-only leads include MultiHaystack, Evidence-RL, MVI-Bench, GRACE, Tone Matters, Image-LoRA and QAPruner. They are leads, not fully audited nearest neighbors. The screen metadata deliberately does not upgrade the remaining 296 entries to abstract or full-text review.
