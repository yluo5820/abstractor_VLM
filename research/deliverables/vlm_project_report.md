# Low-Compute VLM Research Project Ideas

## Recommendation

**The strongest starting point is a controlled study of whether visual-token compression preserves object recognition while damaging spatial relationships.** Start with a frozen 3B model, a few hundred paired scenes, three compatible pruning baselines, and a small token-restoration experiment. If the mechanism survives strong controls, add a cheap inference-time correction. This has a plausible path from a useful class project to a research paper without training a foundation model.

The easiest alternative to prototype is **repeated visual evidence versus genuinely new visual evidence**. It needs ordinary multi-image inference rather than model-internal modifications. Its publication risk is higher: dependent evidence is already discussed in text RAG, and multi-image position and aggregation failures are established. The valuable question is specifically when repeated pixels, complementary crops, and new viewpoints should change an answer or its confidence.

The rankings below are judgments about the combination of feasibility, experimental clarity, and potential contribution. They are not measured outcomes or guarantees of novelty. All proposed methods remain hypotheses. No model experiments have been run for this report.

| Rank | Project question | Estimated course study | Main obstacle |
|---|---|---|---|
| 1 | Does token pruning preserve objects but lose their spatial relationships? | 25–70 GPU-hours | Must distinguish the effect from existing spatial coverage and position-bias fixes |
| 2 | Do repeated images distort confidence or evidence attribution? | 25–80 GPU-hours | Must go beyond ordinary deduplication and existing text-RAG dependence analysis |
| 3 | When should images interact, and when should their evidence stay separate? | 30–90 GPU-hours | Needs a real crossover between policies and a useful query-based predictor |
| 4 | When is retrying a GUI action justified by the available evidence? | 32–80 GPU-hours | Environment engineering and close prior work on active verification |
| 5 | Can compression respect separate error budgets for fragile evidence groups? | 20–60 GPU-hours for one model | Calibration alone is incremental; useful savings may disappear |
| 6 | Which one missing video frame would resolve an uncertain answer? | 24–72 GPU-hours | Active frame selection is crowded; controlled ambiguity must add insight |
| 7 | When a visual fact changes, which derived memories must be invalidated? | 16–48 GPU-hours | Existing memory and truth-maintenance methods may solve it already |

These are total single-GPU hours across sequential jobs, not simultaneous GPU requirements. They assume short answers and bounded image sizes, include a contingency allowance, and exclude queue time. Treat them as planning ranges until a 100-example timing run measures actual throughput. The core versions require no VLM weight training and no paid API. A second architecture or large natural-image validation can increase the budget.

## Course fit and timing

The public syllabus assigns 50% to the research project and requests a concrete gap, hypothesis, method, datasets, metrics, baselines, feasibility assessment, and timeline. It does not state that a benchmark record is required. A controlled analysis plus a tested intervention fits these public expectations; acceptance and grading remain the instructor’s decision. Proposal slides are due September 16, with presentations September 17. The midterm is October 15; code and audit materials freeze November 17. The rebuttal is due November 24, interviews run November 30–December 4, and the final report deadline is on Canvas. [1](https://faculty.cc.gatech.edu/~zk15/teaching/AY2026_cs8803vlm_fall/index.html)

The course also requires team ownership of the research decisions and interpretation, attribution of external resources, and disclosure of material AI assistance in `AGENT_USAGE.md`. This report is a brainstorming and literature resource for the team to evaluate. [1](https://faculty.cc.gatech.edu/~zk15/teaching/AY2026_cs8803vlm_fall/index.html)

PACE is the Georgia Tech computing program; ICE is its instructional cluster. Public hardware descriptions do not establish the GPUs or hours available to this particular course account. The plan therefore assumes intermittent access to one 24–48 GB GPU and roughly 50–150 total GPU-hours, with a viable one-model result before spending the upper end. [2](https://pace.gatech.edu/ice-cluster/)

## Evidence coverage and limits

The broad screen contains **1,220 distinct arXiv-linked catalogue records** from six successfully retrieved research lists. Every record received a title-and-category screen: 1,002 were judged directly relevant to VLMs or visual multimodal methods, 199 adjacent, 17 nonvisual, and 2 unusable or suspicious. The accompanying index records every disposition and the catalogue provenance. The counts describe bibliographic records, not independently verified canonical paper metadata.

**This is not a claim to have fully read 1,000 papers.** Titles and supplied categories were screened at scale; selected abstracts and methods/evaluation sections received closer inspection. The focused audit covers 65 distinct research/position papers beyond the broad title screen; 36 include selected full-text section reads. Other inspections use abstracts, introductions, method excerpts, or author project descriptions. The detailed primary-source ledgers specify the depth paper by paper. The arXiv search API was rate-limited, so the broad inventory uses curated bibliographies; important novelty judgments use primary papers and official project pages. Catalogue errors, abbreviated titles, and possible cross-ID duplicates remain limitations. A thousand full-paper critical reviews have not been completed.

The research cutoff is September 14, 2026. Recent preprints were included, with their status kept distinct from verified conference publications. This is a broad, purposive landscape screen with a focused novelty audit, not a systematic review with exhaustive database coverage. Failure to find an exact predecessor is evidence to investigate a project, not proof that it is new. The most important negative result of the audit is that many appealing generic ideas already have close 2026 predecessors.

## 1. Spatial information under token compression

**Research question:** At a fixed inference budget, can a VLM still identify the objects and their attributes while losing the information needed to relate them spatially?

For example, a model may correctly identify a red cube and a blue sphere, yet become unreliable about which is left of the other after tokens are removed. The project would test whether the failure is attributable to losing contextualized spatial information at locations outside the object regions, rather than merely failing to recognize the objects.

**Closest prior work.** FastV established attention-based visual-token pruning. FEATHER already demonstrated that early pruning can disproportionately harm localization and addressed positional selection bias and spatial coverage. PosPrune is another spatial-preservation comparator. Cui et al. showed that spatial binding information can be globally distributed across vision representations, including background locations. None of these facts should be claimed as discoveries of the project. The candidate contribution is a controlled test of how deployed compression changes the balance between object identity and relational information, followed by a correction justified by that test. [3](https://arxiv.org/abs/2403.06764) [4](https://web.stanford.edu/~markendo/projects/feather) [5](https://ojs.aaai.org/index.php/AAAI/article/view/38020) [6](https://arxiv.org/abs/2603.22278)

**Minimal experiment.** Use 200 base scene pairs for the bounded course study; expand to 400–600 pairs only after the pilot. Render matched versions with two objects swapped horizontally or vertically while preserving their identities and attributes. Ask both an identity/attribute question and a relation question. Vary spacing, position, distractors, texture, and object count. Split by base scene, layout family, and object combination so that near-identical variants do not leak between development and evaluation.

Begin with full inference, uniform retention, and one attention-based pruner. Add FEATHER or another compatible spatial-coverage baseline before drawing a novelty conclusion. Choose two retention budgets on development data, initially exploring 25%, 50%, and 75% only on the pilot subset. Run a small natural-image check using What’sUp and an appropriate GQA spatial subset. What’sUp already offers photographs that vary spatial relations while holding object identities fixed; creating such pairs is not itself novel. [7](https://arxiv.org/abs/2310.19785) [8](https://arxiv.org/abs/1902.09506)

**Mechanism experiment.** Fix the retained object tokens and the total token count. Exchange the remaining slots among direction-aligned background locations, uniform background locations, salience-selected locations, and matched random locations. Preserve the original multimodal position IDs, including text positions, and verify that 100% retention reproduces full inference. Match answer spaces and add nonspatial binding controls of similar baseline difficulty. Ground-truth object masks are permissible for an oracle diagnosis, but the proposed deployed method must not consume gold masks or answers.

Perform same-layer token restoration on a bounded subset: does restoring the putatively relevant spatial information recover relation answers more than restoring a matched number of control tokens? A token at a background coordinate is contextualized by the vision encoder and may contain information about the entire scene. Consequently, an observed effect would concern information represented at those token locations; it would not prove that raw background pixels intrinsically encode the relation. Attention maps or linear probes alone do not establish the causal explanation. Include content replacement at fixed coordinates, with matched layer, token count, and donor magnitude; evaluate published pruning schedules separately from controlled same-layer interventions. A claim about background pixels would require a separate pixel intervention followed by rerunning the encoder.

**Possible small method.** Reserve a modest, question-conditioned spatial-coverage quota in an existing pruner. A horizontal-relation question could retain dispersed evidence across the horizontal extent, with a query-agnostic coverage policy as the direct comparator. This is a hypothesis, not a recommended architecture before the mechanism test succeeds. If uniform retention or FEATHER explains the improvement, the new-method claim fails even if the empirical analysis remains useful.

**Metrics and controls.** Report ordinary accuracy, accuracy on both members of each swapped pair, and the compression-induced drop on relation questions relative to identity questions. Also report destructive flips: cases where full inference is correct and compressed inference is wrong. A separate full-correct subset helps diagnose lost capability, but must not replace all-example accuracy. Bootstrap over original scenes, not over correlated variants. Include label-balanced prompts and a text-only control to test language shortcuts.

Compare both equal-token and equal-latency operating points. Record image preprocessing, vision encoding, token selection, prefill, decoding, peak memory, and total latency. Preserve identical attention backends where possible. Recent work explicitly studies when token reduction fails to yield end-to-end speedups; lower token counts alone are insufficient evidence of efficiency. [9](https://arxiv.org/abs/2608.03649)

**Compute and implementation.** Start with Qwen2.5-VL-3B. An official checkpoint exists; model selection should follow compatibility with the chosen baseline code, rather than forcing several unsupported ports. Use short deterministic answers and capped pixels. A later replication on a second architecture, such as an InternVL3.5 4B variant, makes the paper argument stronger. The public model cards establish availability, not measured memory fit for this configuration. [10](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) [11](https://huggingface.co/OpenGVLab/InternVL3_5-4B-MPO)

A concrete course cap is 200 scene pairs × 2 images × 2 question types, plus 400 natural-image queries: 1,200 queries total. Full inference plus three compressed policies at two budgets gives 8,400 main calls, followed by a bounded 200-case intervention study. Budget 25–70 GPU-hours for that scope; a larger two-architecture paper package may require 80–150 hours or more. Reserve 4–8 GPU-hours for the pilot. Avoid saving all-layer activations for every image. If the required interventions disable fused attention or require a larger backbone, re-estimate before scaling.

**Two-day decision.** First run 100 layout pairs with full, uniform, and one pruning policy. Continue if the full model answers both members correctly on at least roughly 70% of pairs, the relation-specific compression drop is about 8 percentage points larger than the identity drop at a useful budget, and matched restoration shows a meaningful advantage over random restoration. These are practical go/no-go thresholds, not significance claims from a small sample.

**Paper threshold.** A publishable version would need a replicated content-versus-relation dissociation, a convincing mechanism beyond positional bias, natural-image transfer, and a useful quality/latency frontier against strong pruning baselines. A one-model diagnosis with a careful negative result can still be a substantive course deliverable. The publication risk is moderate to high; the hypothesis is precise enough to fail early.

## 2. Repeated pixels versus new visual evidence

**Research question:** How do answers, confidence, and evidence attribution respond when a multi-image prompt contains more copies of the same information, compared with complementary visual information?

Construct sets containing an informative image and distractors. Insert exact copies, rescaled copies, or overlapping crops while keeping the answer fixed for questions such as object existence or a specified attribute. Contrast these with complementary crops that reveal previously hidden details and genuine new views. Add explicit image-count tasks where adding a copy really should change the answer. A system that always ignores duplicates would fail those controls.

**Novelty boundary.** MIMIC already analyzes multi-image aggregation and separates image-count effects from sequence length. SoFA studies position bias; MIHBench and MIOH study multi-image hallucinations. An August 2026 text-RAG position paper, *Counting Copies as Evidence*, explicitly proposes confidence-inflation diagnostics and dependence-aware aggregation. CGC also already trains source-aware attribution across original, focus, and context crops. The project must cite these predecessors; correlated crops or attribution alone are not a new contribution. Its possible contribution is the visual distinction between redundant pixels and complementary views, plus tested error/calibration consequences. [12](https://arxiv.org/abs/2604.22498) [13](https://aclanthology.org/2026.findings-acl.366/) [14](https://arxiv.org/abs/2503.13792) [15](https://arxiv.org/abs/2508.00726) [16](https://arxiv.org/abs/2608.30653) [17](https://www.researchgate.net/publication/412230355_Counting_Copies_as_Evidence_Confidence_Inflation_from_Dependent_Evidence_in_Retrieval-Augmented_Generation_RAG)

**Design.** Start with 200–300 base evidence sets and a fixed menu of transformations. Split original image sources and galleries before creating variants, then bootstrap by base gallery; repeated copies are not independent calibration or test examples. Counterbalance image order and the position of the informative image. Include length-matched distractor controls, single-image repeat-computation controls, and a stable image-ID convention. For claims about source dependence, use known source IDs in construction and separate the oracle-provenance condition from an estimated-provenance condition. Similar-looking images are not necessarily copies of the same source.

Measure answer flips, source attribution errors, confidence shifts, Brier score or negative log likelihood for constrained answers, and risk at matched coverage. A confidence increase is not automatically harmful: repeated processing might improve correctness. The key test is whether calibration worsens after accounting for accuracy, or whether unsupported answers become more frequent. Distinguish duplicate-source hallucination from positional distraction and from genuinely informative additional pixels.

**Minimal intervention.** Compare unmodified inference, an explicit source-ID prompt, exact-hash deduplication, perceptual deduplication, and a simple source-cluster-aware selection policy. Preserve complementary regions rather than deleting every similar crop. Include one compatible existing leakage intervention, such as delimiter scaling or SoFA, to distinguish source dependence from generic image interference. A utility-based visual evidence selector is an important additional comparator because training-free evidence selection itself already exists. [18](https://aclanthology.org/2026.acl-long.1620/)

The core study can use constrained questions on public annotated images and procedurally generated diagrams, with Qwen3-VL-4B and a second multi-image model. A minimal two-model study can target 8–20 GPU-hours; a larger study with ablations should reserve 25–80 GPU-hours; actual multi-image throughput controls the estimate. Use 2–6 GPU-hours for an initial 100-set screen. No model weights need updating. [19](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)

**Two-day decision.** Continue only if the effect survives position/length controls in two models and is more consequential than a harmless confidence shift. Require a simple mitigation to improve a real error or calibration metric while retaining gains from complementary crops. Abandon the method claim if exact deduplication solves the whole problem or if the phenomenon disappears after controlling actual accuracy. A stronger paper would establish when visual redundancy is harmful, neutral, or useful; “duplicates are bad” is too simple.

## 3. Choosing when images should share information

**Research question:** Can the question predict whether a multi-image VLM should encourage cross-image integration or preserve image-specific evidence?

Use exactly the same image set for two question types. An isolation question asks for an attribute in image B; an integration question asks what is common across images A and B, or which object changes. This removes image-set difficulty as an explanation for differences between attention policies.

**Closest work and proposed distinction.** SoFA and CAPL promote cross-image interaction to address position bias. FOCUS, image-delimiter scaling, and MIMIC include mechanisms that restrict or reorganize information flow. Their results occur under different models and tasks, so their prescriptions should not be presented as an established contradiction. The project would test whether a reproducible task-dependent crossover appears under a shared evaluation. Fixed mixing and layer alternation inspired by CAPL are important baselines for a router. Full CAPL also includes preference training, so frozen-mask adaptations must be labeled CAPL-inspired rather than reproductions. The same distinction applies to using MIMIC’s attention rule without its trained model. [14](https://arxiv.org/abs/2503.13792) [20](https://arxiv.org/abs/2603.07048) [21](https://arxiv.org/abs/2508.13744) [22](https://arxiv.org/abs/2602.01984) [13](https://aclanthology.org/2026.findings-acl.366/)

**Study.** Create about 300–500 matched image sets with isolation, attribute-binding, and integration questions. Compare the unchanged model, two compatible existing policies, a fixed mixture, random routing, and an oracle that selects the better policy per example. Evaluate pairwise accuracy, wrong-image attribution, cross-image hallucination, worst-group accuracy, and inference overhead. Hold out question paraphrases and an entire task template so the router cannot simply memorize superficial wording.

Only after an oracle routing advantage appears should the team fit a tiny classifier to question text or use a transparent task rule. Freeze it before final evaluation. Use the same policy implementations and image processing for all comparisons. Masking may not be compatible with every fused-attention backend, so implementation fit is a serious early gate.

**Budget and pilot.** Plan 30–90 GPU-hours for a broader study of approximately 10,000–20,000 short evaluations, with a 3–8 GPU-hour initial study. The pilot should demonstrate a crossover in the same model and image sets, an oracle margin worth recovering, and a question-only router that improves over the best fixed mixture on held-out templates. If the oracle advantage is negligible or a fixed policy dominates both task groups, stop. The possible paper is a predictive account of intervention choice, not another generic attention mask.

## 4. Evidence-aware GUI retries

**Research question:** Does an agent decide to retry because an action actually failed, or because the screen did not visibly change?

In a local notes interface, an Add action may successfully append a note without refreshing the visible panel. A dropped action can leave the same screen. Retrying creates a duplicate in the first case and fixes the second. Conversely, an animation may occur even when the write failed. This yields a useful 2×2 design: actual success/failure crossed with visible change/no change.

VeriGUI trains verification and recovery, VisCritic uses before/after visual state comparisons, and Interactive Reward Agent already checks persistent environment state beyond screenshots. The new contribution cannot be “verify actions” or “inspect hidden state.” It would have to be the sequential retry decision under ambiguous evidence, including the different consequences of idempotent actions and duplicate-producing actions. [23](https://arxiv.org/abs/2604.05477) [24](https://arxiv.org/abs/2606.24525) [25](https://arxiv.org/abs/2607.25904)

**Study.** Build 200–300 independent episodes, each with four controlled variants. Keep application logs as evaluator-only ground truth. Compare direct screenshot judgment, an expected-effect prompt, a pixel-difference rule, always retry, always inspect, and a budget-matched IRA-style verifier. A proposed gate acquires one read-only observation when evidence is insufficient and the retry could duplicate a write. Every policy must have the same observation access; the proposed method cannot read privileged success flags.

Pixel-identical cases with identical observable histories are not ordinary binary classification examples: success is unidentifiable from those inputs. Grade recognition of insufficient evidence separately. Include cases with subtle but usable evidence and a small natural application slice, because a toy impossibility demonstration alone is weak research.

Report duplicate-write rate, missed failures, completion after at most two extra decisions, and observations per episode. Bootstrap by base episode. The practical target is fewer duplicate actions at matched completion rate and observation cost. The main expense is reliable environment construction, not GPU training.

**Budget and pilot.** Forty episodes × four variants, two small models, and three policies produces roughly 1,300–1,600 decisions including follow-ups: plan 4–10 GPU-hours. The full study is about 10,000–18,000 calls and 32–80 GPU-hours. Continue if screen-change bias appears across models and the gate improves the duplicate/completion/cost tradeoff. Stop if an ordinary expected-effect prompt or always-inspect achieves the same frontier, or if released IRA data already cover the exact factorial retry problem.

## 5. Separate compression error budgets for fragile evidence

**Research question:** Can a compression policy save real time while keeping additional errors small in each important input group, rather than merely preserving average benchmark accuracy?

Define compression harm as an example where the full model answers correctly and the compressed model answers incorrectly. Report this separately from beneficial flips; the latter should not cancel destructive changes when evaluating capability preservation. A 2% compression-harm target is not a claim of 98% absolute accuracy.

ET-Prune already allocates tokens using evidence cues, CARES predicts sufficient resolution, and LazyMCoT uses calibrated difficulty routing. General risk-control techniques also exist. The possible contribution is an evaluation of group-specific harm, including calibration cost and failure under distribution shift, with a transparent policy that saves measured time. [26](https://arxiv.org/abs/2608.01979) [27](https://arxiv.org/abs/2510.19496) [28](https://arxiv.org/abs/2606.16158) [29](https://arxiv.org/abs/2208.02814)

Use a few predeclared groups, such as small text, dense text, spatial relations, and other questions. Deployment groups must be observable without gold answers. Cache predictions for a finite menu of resolution or pruning policies. Choose a fast policy only when a one-sided error bound satisfies the target in every group, correcting for selection across groups and candidate policies. Keep development, calibration, and final test images separate. Error bounds require the stated independent-sampling assumptions and a matching calibration/test population; a shifted-dataset result is an empirical stress test, not protection against arbitrary distribution shift. Correctness need not improve monotonically with visual budget, so do not apply a monotonicity-dependent guarantee without checking its conditions.

Estimate 20–60 GPU-hours for a 3B study, or 40–100 with a second model. A 300–500 image pilot should first show an oracle saving of roughly 20% at an acceptable harm level. Then check whether a cheap predictor preserves useful savings after its overhead. Sparse calibration groups may force full inference; report that result rather than suppress uncertainty. This is feasible as a course analysis but has high publication novelty risk unless the group failure is important, reproducible, and recoverable.

## Two lower-priority alternatives

**One missing video frame.** Give a model sparse frames that leave a question ambiguous, then let it buy one additional timestamp. Compare post-reveal accuracy and risk reduction per extra frame against random, midpoint, motion-based, and existing query-aware selection. VirtueBench already measures answerability under sampling, TRAPSBench studies epistemic restraint, and VideoAgent already performs active video search. A defensible extension would isolate the quality of choosing a discriminating observation under the same initial information and the same frame/call budget. If a motion-based selector scans otherwise unseen frames, count those accesses, feature passes, and preprocessing costs; equal final frame counts alone are not equal information. Timestamp selection must have usable cues in the allowed observation; an oracle choice without such cues is only an upper bound. The pilot is 50 paired scenarios; the full study is 200–400 short clips, with a 24–72 GPU-hour planning budget. Stop if a midpoint or cheap sampler is effectively as good, or if the model cannot answer with the oracle frame. [30](https://arxiv.org/abs/2603.07071) [31](https://arxiv.org/abs/2608.13167) [32](https://arxiv.org/abs/2403.10517)

**Invalidating derived visual memory.** A screenshot update changes one primitive fact, such as seats occupied at a table. Does the memory update that fact but preserve an obsolete conclusion such as which table has most empty seats? Compare append-only summaries, last-write-wins records, full refresh, and evidence-linked invalidation at the same memory and extraction budget. WorldMemArena and MemLens already evaluate memory updates, so the narrow angle is propagation to derived conclusions and preservation of unaffected facts. Use 150–300 short sequences, a shared frozen extractor, and oracle-extraction controls; plan 16–48 GPU-hours. Stop if ordinary record updates plus recomputation achieve the same result. This is a backup, with substantial novelty risk from existing memory and truth-maintenance work. [33](https://arxiv.org/abs/2605.29341) [34](https://arxiv.org/abs/2605.14906)

## Compute plan and execution gates

**Prefer bounded inference over open-ended training or rollouts.** A 3B/4B frozen checkpoint, short answers, one GPU, and cached outputs make the work resumable when ICE is crowded. Use a fixed image/token cap. A 4B model fitting in memory does not imply that high-resolution multi-image inference or activation capture will fit too. Keep quantized and unquantized configurations separate in the analysis because quantization can change the phenomenon being studied.

For each configuration, time 100 representative examples after warmup. Estimate total allocated GPU time as the sum of calls × measured seconds per call ÷ 3,600, plus setup, intervention runs, and a 30–100% rerun allowance. For illustration, 12,000 calls at 2 seconds each cost 6.7 GPU-hours before overhead; at 8 seconds they cost 26.7 hours. At 20 seconds they cost 66.7 hours. These arithmetic scenarios explain why a nominally small model is not a reliable cost estimate by itself.

Use roughly 30–60 minute independently restartable batches, recording completed examples before exiting. Cache decoded images, question variants, predictions, selected token indices, and timings. Render synthetic scenes, prepare annotations, fit small probes, and bootstrap statistics on CPU while waiting for a GPU. Avoid full-corpus attention dumps. Pin checkpoints, preprocessing, precision, package versions, data splits, prompts, and decoding settings before the final sweep.

| Stage | Minimum evidence before spending more |
|---|---|
| Before proposal | Read the closest papers; state one falsifiable gap and a small pilot; do not imply preliminary results exist |
| First two working days with GPU access | One end-to-end baseline, valid labels, measured throughput, and a keep/stop decision |
| First two project weeks | Stable data construction, baseline reproduction, source-level split, and a replicated initial effect |
| Midterm preparation | Locked primary metrics and baselines; at least one defensible result; a clear limitation or failure case |
| Final experiment period | Second model or external data validation, matched-cost comparisons, ablations, uncertainty estimates |
| Before code freeze | Reproducible commands, complete attribution and AI-use disclosure, immutable result configuration |

If the cluster budget collapses to under 20 hours, retain one model, one principal dataset, and the controlled analysis. Projects 2 or 7 have the easiest implementation fallback. If 100–150 hours become available, prioritize a second architecture and an external validation set rather than adding a large fine-tuning phase. State-of-the-art performance would be a welcome result within a clearly defined cost regime; it should not be a prerequisite for completing the course project.

## Reading priorities and rejected generic pitches

For the recommended project, read **Cui et al., FEATHER, FastV, PosPrune, and the end-to-end break-even study** before writing the novelty claim. Then inspect What’sUp examples and reproduce one compatible pruning baseline. The decisive question is whether the proposed relation-specific mechanism remains after FEATHER-style coverage and position controls. [6](https://arxiv.org/abs/2603.22278) [4](https://web.stanford.edu/~markendo/projects/feather) [3](https://arxiv.org/abs/2403.06764) [5](https://ojs.aaai.org/index.php/AAAI/article/view/38020) [9](https://arxiv.org/abs/2608.03649) [7](https://arxiv.org/abs/2310.19785)

For the easiest multi-image alternative, read **MIMIC, Counting Copies as Evidence, SoFA, and MIOH**. For the GUI alternative, read **VeriGUI and IRA** before implementing an environment. These short reading sets are more valuable for committing to a project than another undirected sweep of titles.

| Generic pitch | Why it was not promoted |
|---|---|
| Zoom before abstaining | BCEA already combines budgeted visual evidence acquisition and calibrated abstention. [35](https://arxiv.org/abs/2606.16667) |
| Protect small OCR regions during pruning | ET-Prune already targets question-conditioned evidence and dynamic budgets. [26](https://arxiv.org/abs/2608.01979) |
| Route difficult examples to more visual reasoning | CARES and LazyMCoT occupy close versions. [27](https://arxiv.org/abs/2510.19496) [28](https://arxiv.org/abs/2606.16158) |
| Restore compressed tokens for later questions | SparseVILA and recent cache-refresh work already address related multi-turn needs. [36](https://arxiv.org/abs/2510.17777) [37](https://arxiv.org/abs/2609.05821) |
| Recheck a stale screen before acting | Existing TOCTOU and pixel-versus-structure studies have direct proposals. [38](https://arxiv.org/abs/2604.18860) [39](https://arxiv.org/abs/2607.04334) |
| Reverse video order and test consistency | CRPO/DyBench already use related counterfactual constructions. [40](https://arxiv.org/abs/2605.21988) |
| Train a small hallucination probe | HALP already targets detection before token generation. [41](https://arxiv.org/abs/2603.05465) |

The recommended decision is to pilot Project 1 first, with Project 2 as the implementation fallback. Commit to a final contribution only after the baseline and mechanism checks, and let a failed pilot eliminate an idea quickly. A careful finding that narrows when a popular method works can be more defensible than a small average-score increase with an unclear cause.


## Sources

Numbered references correspond to citations in the report. Full inspection depths and additional audited sources are in the accompanying primary-source ledger. Dates are first-public dates or the recorded publication date; updated versions and unverified venue claims are distinguished in the ledger.

1. Georgia Institute of Technology, Zsolt Kira. [CS 8803-VLM: Vision-Language Models & Multimodal Agents, Fall 2026](https://faculty.cc.gatech.edu/~zk15/teaching/AY2026_cs8803vlm_fall/index.html). Fall 2026; accessed 2026-09-14.

2. Georgia Tech PACE. [ICE Cluster](https://pace.gatech.edu/ice-cluster/). accessed 2026-09-14.

3. Chen, Liang, Zhao, Haozhe, Liu, Tianyu et al.. [An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models](https://arxiv.org/abs/2403.06764). 2024-03-11.

4. Mark Endo, Xiaohan Wang, Serena Yeung-Levy. [Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](https://web.stanford.edu/~markendo/projects/feather). 2025.

5. AAAI. [PosPrune: Visual Token Pruning with Positional Bias Correction for Efficient Large Vision-Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/38020). 2026-03-14.

6. Cui, Kelly, Prakash, Nikhil, Messica, Shoval et al.. [The Dual Mechanisms of Spatial Variable Binding in Vision-Language Models](https://arxiv.org/abs/2603.22278). 2026-03-23.

7. Amita Kamath, Jack Hessel, Kai-Wei Chang. [What’s “up” with vision-language models? Investigating their struggle with spatial reasoning](https://arxiv.org/abs/2310.19785). 2023-10-30.

8. Drew A. Hudson, Christopher D. Manning. [GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506). 2019.

9. Dou, Hao, Tian, Ruiwen. [When Do Fewer Visual Tokens Accelerate Multimodal Inference? A Break-Even Study Across Decision Locations and Hardware](https://arxiv.org/abs/2608.03649). 2026-08-04.

10. Qwen team. [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). official model card; accessed 2026-09-14.

11. OpenGVLab. [InternVL3.5-4B-MPO model card and family specifications](https://huggingface.co/OpenGVLab/InternVL3_5-4B-MPO). accessed 2026-09-14.

12. Lihao Zheng, Zhenwei Shao, Yu Zhou et al.. [CGC: Compositional Grounded Contrast for Fine-Grained Multi-Image Understanding](https://arxiv.org/abs/2604.22498). 2026-04-24.

13. Anurag Das, Adrian Bulat, Alberto Baldrati et al.. [More Images, More Problems? A Controlled Analysis of VLM Failure Modes.](https://aclanthology.org/2026.findings-acl.366/). 2026-07.

14. Xinyu Tian, Shu Zou, Zhaoyuan Yang et al.. [Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](https://arxiv.org/abs/2503.13792). 2025-03-18.

15. arXiv. [MIHBench: Benchmarking and Mitigating Multi-Image Hallucinations in Multimodal Large Language Models](https://arxiv.org/abs/2508.00726). 2025-08.

16. Joonki Min, Chaeyun Kim, Hyungwook Choi et al.. [Fine-Grained Multi-Image Object Hallucination Benchmark](https://arxiv.org/abs/2608.30653). 2026-08-31.

17. Irwan Rahadi. [Counting Copies as Evidence: Confidence Inflation from Dependent Evidence in Retrieval-Augmented Generation (RAG)](https://www.researchgate.net/publication/412230355_Counting_Copies_as_Evidence_Confidence_Inflation_from_Dependent_Evidence_in_Retrieval-Augmented_Generation_RAG). 2026-08.

18. Weiqing Luo, Zongye Hu, Xiao Wang et al.. [Utility-Oriented Visual Evidence Selection for Multimodal Retrieval-Augmented Generation](https://aclanthology.org/2026.acl-long.1620/). 2026-07.

19. Qwen team. [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). official model card; accessed 2026-09-14.

20. Xiaochen Yang, Hao Fang, Jiawei Kong et al.. [Looking Back and Forth: Cross-Image Attention Calibration and Attentive Preference Learning for Multi-Image Hallucination Mitigation](https://arxiv.org/abs/2603.07048). 2026-03-07.

21. arXiv. [Mitigating Cross-Image Information Leakage in LVLMs for Multi-Image Tasks](https://arxiv.org/abs/2508.13744). 2025-08.

22. Minyoung Lee, Yeji Park, Dongjun Hwang et al.. [Enhancing Multi-Image Understanding through Delimiter Token Scaling](https://arxiv.org/abs/2602.01984). 2026-02-02.

23. Zhang, Yuzhe, Xue, Xianwei, Wu, Xingyong et al.. [Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](https://arxiv.org/abs/2604.05477). 2026-04-07.

24. Qian, Jiachen. [VisCritic: Visual State Comparison as Process Reward for GUI Agents](https://arxiv.org/abs/2606.24525). 2026-06-23.

25. Shi, Chenrui, Wu, Yuwei, Liu, Yang et al.. [Interactive Reward Agent: GUI Task Evaluation via Environment-State Verification](https://arxiv.org/abs/2607.25904). 2026-07-28.

26. Ding, Zizhong, Li, Junxian, Liu, Kai et al.. [ET-Prune: Evidence-Aware Dynamic Budgeting for Visual Token Pruning in Text-Rich MLLMs](https://arxiv.org/abs/2608.01979). 2026-08-03.

27. Kimhi, Moshe, Shabtay, Nimrod, Giryes, Raja et al.. [CARES: Context-Aware Resolution Selector for VLMs](https://arxiv.org/abs/2510.19496). 2025-10-22.

28. Wang, Yifan, Li, Peiming, Li, Shiyu et al.. [Focus When Necessary: Adaptive Routing and Collaborative Grounding for Training-Free Visual Grounding](https://arxiv.org/abs/2606.16158). 2026-06-15.

29. Angelopoulos, Anastasios N., Bates, Stephen, Fisch, Adam et al.. [Conformal Risk Control](https://arxiv.org/abs/2208.02814). 2022-08-04.

30. Yu, Xueqing, Li, Bohan, Li, Yan et al.. [VirtueBench: Evaluating Trustworthiness under Uncertainty in Long Video Understanding](https://arxiv.org/abs/2603.07071). 2026-03-07.

31. Fnu Pramono, John Cai, Sourabh Kulkarni. [TRAPSBench: Vision-Language Models Encode but Fail to Express Epistemic Restraint](https://arxiv.org/abs/2608.13167). 2026-08-13.

32. Xiaohan Wang, Yuhui Zhang, Orr Zohar et al.. [VideoAgent: Long-form Video Understanding with Large Language Model as Agent](https://arxiv.org/abs/2403.10517). 2024-03-15.

33. Liu, Chengzhi, Yang, Yuzhe, Pu, Sophia Xiao et al.. [WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction](https://arxiv.org/abs/2605.29341). 2026-05-28.

34. Ren, Xiyu, Wang, Zhaowei, Du, Yiming et al.. [MemLens: Benchmarking Multimodal Long-Term Memory in Large Vision-Language Models](https://arxiv.org/abs/2605.14906). 2026-05-14.

35. Jian Xu, Yanning Wu, Delu Zeng et al.. [Look Again Before You Abstain: Budgeted Conformal Evidence Acquisition for Reliable Vision-Language Models](https://arxiv.org/abs/2606.16667). 2026-06-15.

36. Khaki, Samir, Guo, Junxian, Tang, Jiaming et al.. [SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](https://arxiv.org/abs/2510.17777). 2025-10-20.

37. Chen, Pengan, Zheng, Kaisheng, Hong, Liang et al.. [CONDUIT: A Unified Residual-Stream Restoration Framework for KV Cache Reuse in Vision-Language Models](https://arxiv.org/abs/2609.05821). 2026-09-05.

38. Xu, Wenpeng. [Temporal UI State Inconsistency in Desktop GUI Agents: Formalizing and Defending Against TOCTOU Attacks on Computer-Use Agents](https://arxiv.org/abs/2604.18860). 2026-04-20.

39. Zhang, Guijia, Chen, Yuxun, Qi, Yuheng et al.. [Do GUI Agents Believe Their Eyes? Diagnosing State-Belief Reliance on Pixels versus Structure](https://arxiv.org/abs/2607.04334). 2026-07-05.

40. Du, Dazhao, Liu, Jian, Qin, Jialong et al.. [Learning Spatiotemporal Sensitivity in Video LLMs via Counterfactual Reinforcement Learning](https://arxiv.org/abs/2605.21988). 2026-05-21.

41. Sai Akhil Kogilathota, Sripadha Vallabha E G, Luzhe Sun et al.. [HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token](https://arxiv.org/abs/2603.05465). 2026-03-05.

