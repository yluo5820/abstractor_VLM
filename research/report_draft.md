# Low-Compute VLM Research Project Ideas

## Recommendation

**The strongest starting point is a controlled study of whether visual-token compression preserves object recognition while damaging spatial relationships.** Start with a frozen 3B model, a few hundred paired scenes, three compatible pruning baselines, and a small token-restoration experiment. If the mechanism survives strong controls, add a cheap inference-time correction. This has a plausible path from a useful class project to a research paper without training a foundation model.

The easiest alternative to prototype is **repeated visual evidence versus genuinely new visual evidence**. It needs ordinary multi-image inference rather than model-internal modifications. Its publication risk is higher: dependent evidence is already discussed in text RAG, and multi-image position and aggregation failures are established. The valuable question is specifically when repeated pixels, complementary crops, and new viewpoints should change an answer or its confidence.

The rankings below are judgments about the combination of feasibility, experimental clarity, and potential contribution. They are not measured outcomes or guarantees of novelty. All proposed methods remain hypotheses. No model experiments have been run for this report.

| Rank | Project question | Estimated full study | Main obstacle |
|---|---|---|---|
| 1 | Does token pruning preserve objects but lose their spatial relationships? | 25–70 GPU-hours | Must distinguish the effect from existing spatial coverage and position-bias fixes |
| 2 | Do repeated images distort confidence or evidence attribution? | 20–60 GPU-hours | Must go beyond ordinary deduplication and existing text-RAG dependence analysis |
| 3 | When should images interact, and when should their evidence stay separate? | 25–70 GPU-hours | Needs a real crossover between policies and a useful query-based predictor |
| 4 | When is retrying a GUI action justified by the available evidence? | 32–80 GPU-hours | Environment engineering and close prior work on active verification |
| 5 | Can compression respect separate error budgets for fragile evidence groups? | 20–60 GPU-hours for one model | Calibration alone is incremental; useful savings may disappear |
| 6 | Which one missing video frame would resolve an uncertain answer? | 24–72 GPU-hours | Active frame selection is crowded; controlled ambiguity must add insight |
| 7 | When a visual fact changes, which derived memories must be invalidated? | 16–48 GPU-hours | Existing memory and truth-maintenance methods may solve it already |

These are total single-GPU hours across sequential jobs, not simultaneous GPU requirements. They assume short answers and bounded image sizes, include a contingency allowance, and exclude queue time. Treat them as planning ranges until a 100-example timing run measures actual throughput. The core versions require no VLM weight training and no paid API. A second architecture or large natural-image validation can increase the budget.

## Course fit and timing

The public syllabus assigns 50% to the research project and requests a concrete gap, hypothesis, method, datasets, metrics, baselines, feasibility assessment, and timeline. It does not state that a benchmark record is required. A controlled analysis plus a tested intervention fits these public expectations; acceptance and grading remain the instructor’s decision. Proposal slides are due September 16, with presentations September 17. The midterm is October 15; code and audit materials freeze November 17. The rebuttal is due November 24, interviews run November 30–December 4, and the final report deadline is on Canvas. [@course]

The course also requires team ownership of the research decisions and interpretation, attribution of external resources, and disclosure of material AI assistance in `AGENT_USAGE.md`. This report is a brainstorming and literature resource for the team to evaluate. [@course]

PACE is the Georgia Tech computing program; ICE is its instructional cluster. Public hardware descriptions do not establish the GPUs or hours available to this particular course account. The plan therefore assumes intermittent access to one 24–48 GB GPU and roughly 50–150 total GPU-hours, with a viable one-model result before spending the upper end. [@ice]

## Evidence coverage and limits

The broad screen contains **1,220 distinct arXiv-linked catalogue records** from six successfully retrieved research lists. Every record received a title-and-category screen: 1,002 were judged directly relevant to VLMs or visual multimodal methods, 198 adjacent, 14 nonvisual, and 6 unusable or suspicious. The accompanying index records every disposition and the catalogue provenance. The counts describe bibliographic records, not independently verified canonical paper metadata.

**This is not a claim to have fully read 1,000 papers.** Titles and supplied categories were screened at scale; selected abstracts and methods/evaluation sections received closer inspection. The detailed primary-source ledgers specify the depth paper by paper. The arXiv search API was rate-limited, so the broad inventory uses curated bibliographies; important novelty judgments use primary papers and official project pages. Catalogue errors, abbreviated titles, and possible cross-ID duplicates remain limitations. A thousand full-paper critical reviews have not been completed.

The research cutoff is September 14, 2026. Recent preprints were included, with their status kept distinct from verified conference publications. This is a broad, purposive landscape screen with a focused novelty audit, not a systematic review with exhaustive database coverage. Failure to find an exact predecessor is evidence to investigate a project, not proof that it is new. The most important negative result of the audit is that many appealing generic ideas already have close 2026 predecessors.

## 1. Spatial information under token compression

**Research question:** At a fixed inference budget, can a VLM still identify the objects and their attributes while losing the information needed to relate them spatially?

For example, a model may correctly identify a red cube and a blue sphere, yet become unreliable about which is left of the other after tokens are removed. The project would test whether the failure is attributable to losing contextualized spatial information at locations outside the object regions, rather than merely failing to recognize the objects.

**Closest prior work.** FastV established attention-based visual-token pruning. FEATHER already demonstrated that early pruning can disproportionately harm localization and addressed positional selection bias and spatial coverage. PosPrune is another spatial-preservation comparator. Cui et al. showed that spatial binding information can be globally distributed across vision representations, including background locations. None of these facts should be claimed as discoveries of the project. The candidate contribution is a controlled test of how deployed compression changes the balance between object identity and relational information, followed by a correction justified by that test. [@fastv] [@feather] [@posprune] [@spatialbinding]

**Minimal experiment.** Use approximately 400–600 base layouts. Render matched versions with two objects swapped horizontally or vertically while preserving their identities and attributes. Ask both an identity/attribute question and a relation question. Vary spacing, position, distractors, texture, and object count. Split by base scene, layout family, and object combination so that near-identical variants do not leak between development and evaluation.

Begin with full inference, uniform retention, and one attention-based pruner. Add FEATHER or another compatible spatial-coverage baseline before drawing a novelty conclusion. Use 25%, 50%, and 75% retention as exploratory settings, with the final operating point chosen on development data. Run a small natural-image check using What’sUp and an appropriate GQA spatial subset. What’sUp already offers photographs that vary spatial relations while holding object identities fixed; creating such pairs is not itself novel. [@whatsup] [@gqa]

**Mechanism experiment.** Fix the retained object tokens and the total token count. Exchange the remaining slots among direction-aligned background locations, uniform background locations, salience-selected locations, and matched random locations. Preserve the original spatial position IDs. Ground-truth object masks are permissible for an oracle diagnosis, but the proposed deployed method must not consume gold masks or answers.

Perform same-layer token restoration on a bounded subset: does restoring the putatively relevant spatial information recover relation answers more than restoring a matched number of control tokens? A token at a background coordinate is contextualized by the vision encoder and may contain information about the entire scene. Consequently, an observed effect would concern information represented at those token locations; it would not prove that raw background pixels intrinsically encode the relation. Attention maps or linear probes alone do not establish the causal explanation.

**Possible small method.** Reserve a modest, question-conditioned spatial-coverage quota in an existing pruner. A horizontal-relation question could retain dispersed evidence across the horizontal extent, with a query-agnostic coverage policy as the direct comparator. This is a hypothesis, not a recommended architecture before the mechanism test succeeds. If uniform retention or FEATHER explains the improvement, the new-method claim fails even if the empirical analysis remains useful.

**Metrics and controls.** Report ordinary accuracy, accuracy on both members of each swapped pair, and the compression-induced drop on relation questions relative to identity questions. Also report destructive flips: cases where full inference is correct and compressed inference is wrong. A separate full-correct subset helps diagnose lost capability, but must not replace all-example accuracy. Bootstrap over original scenes, not over correlated variants. Include label-balanced prompts and a text-only control to test language shortcuts.

Compare both equal-token and equal-latency operating points. Record image preprocessing, vision encoding, token selection, prefill, decoding, peak memory, and total latency. Preserve identical attention backends where possible. Recent work explicitly studies when token reduction fails to yield end-to-end speedups; lower token counts alone are insufficient evidence of efficiency. [@breakeven]

**Compute and implementation.** Start with Qwen2.5-VL-3B. An official checkpoint exists; model selection should follow compatibility with the chosen baseline code, rather than forcing several unsupported ports. Use short deterministic answers and capped pixels. A later replication on a second architecture, such as an InternVL3.5 4B variant, makes the paper argument stronger. The public model cards establish availability, not measured memory fit for this configuration. [@qwen25] [@internvl]

Budget about 8,000–15,000 main short calls plus a limited activation-intervention subset: 25–70 GPU-hours as an initial planning envelope. Reserve 4–8 GPU-hours for the pilot. Avoid saving all-layer activations for every image. If the required interventions disable fused attention or require a larger backbone, re-estimate before scaling.

**Two-day decision.** First run 100 layout pairs with full, uniform, and one pruning policy. Continue if the full model answers both members correctly on at least roughly 70% of pairs, the relation-specific compression drop is about 8 percentage points larger than the identity drop at a useful budget, and matched restoration shows a meaningful advantage over random restoration. These are practical go/no-go thresholds, not significance claims from a small sample.

**Paper threshold.** A publishable version would need a replicated content-versus-relation dissociation, a convincing mechanism beyond positional bias, natural-image transfer, and a useful quality/latency frontier against strong pruning baselines. A one-model diagnosis with a careful negative result can still be a substantive course deliverable. The publication risk is moderate to high; the hypothesis is precise enough to fail early.

## 2. Repeated pixels versus new visual evidence

**Research question:** How do answers, confidence, and evidence attribution respond when a multi-image prompt contains more copies of the same information, compared with complementary visual information?

Construct sets containing an informative image and distractors. Insert exact copies, rescaled copies, or overlapping crops while keeping the answer fixed for questions such as object existence or a specified attribute. Contrast these with complementary crops that reveal previously hidden details and genuine new views. Add explicit image-count tasks where adding a copy really should change the answer. A system that always ignores duplicates would fail those controls.

**Novelty boundary.** MIMIC already analyzes multi-image aggregation and separates image-count effects from sequence length. SoFA studies position bias; MIHBench and MIOH study multi-image hallucinations. An August 2026 text-RAG position paper, *Counting Copies as Evidence*, explicitly proposes confidence-inflation diagnostics and dependence-aware aggregation. CGC also already trains source-aware attribution across original, focus, and context crops. The project must cite these predecessors; correlated crops or attribution alone are not a new contribution. Its possible contribution is the visual distinction between redundant pixels and complementary views, plus tested error/calibration consequences. [@cgc] [@mimic] [@sofa] [@mihbench] [@mioh] [@countingcopies]

**Design.** Start with 200–300 base evidence sets and a fixed menu of transformations. Split original image sources and galleries before creating variants, then bootstrap by base gallery; repeated copies are not independent calibration or test examples. Counterbalance image order and the position of the informative image. Include length-matched distractor controls, single-image repeat-computation controls, and a stable image-ID convention. For claims about source dependence, use known source IDs in construction and separate the oracle-provenance condition from an estimated-provenance condition. Similar-looking images are not necessarily copies of the same source.

Measure answer flips, source attribution errors, confidence shifts, Brier score or negative log likelihood for constrained answers, and risk at matched coverage. A confidence increase is not automatically harmful: repeated processing might improve correctness. The key test is whether calibration worsens after accounting for accuracy, or whether unsupported answers become more frequent. Distinguish duplicate-source hallucination from positional distraction and from genuinely informative additional pixels.

**Minimal intervention.** Compare unmodified inference, an explicit source-ID prompt, exact-hash deduplication, perceptual deduplication, and a simple source-cluster-aware selection policy. Preserve complementary regions rather than deleting every similar crop. Include one compatible existing leakage intervention, such as delimiter scaling or SoFA, to distinguish source dependence from generic image interference. A utility-based visual evidence selector is an important additional comparator because training-free evidence selection itself already exists. [@utilitymrag]

The core study can use constrained questions on public annotated images and procedurally generated diagrams, with Qwen3-VL-4B and a second multi-image model. About 8,000–16,000 bounded calls yields a planning budget of 20–60 GPU-hours; actual multi-image throughput controls the estimate. Use 2–6 GPU-hours for an initial 100-set screen. No model weights need updating. [@qwen3]

**Two-day decision.** Continue only if the effect survives position/length controls in two models and is more consequential than a harmless confidence shift. Require a simple mitigation to improve a real error or calibration metric while retaining gains from complementary crops. Abandon the method claim if exact deduplication solves the whole problem or if the phenomenon disappears after controlling actual accuracy. A stronger paper would establish when visual redundancy is harmful, neutral, or useful; “duplicates are bad” is too simple.

## 3. Choosing when images should share information

**Research question:** Can the question predict whether a multi-image VLM should encourage cross-image integration or preserve image-specific evidence?

Use exactly the same image set for two question types. An isolation question asks for an attribute in image B; an integration question asks what is common across images A and B, or which object changes. This removes image-set difficulty as an explanation for differences between attention policies.

**Closest work and proposed distinction.** SoFA and CAPL promote cross-image interaction to address position bias. FOCUS, image-delimiter scaling, and MIMIC include mechanisms that restrict or reorganize information flow. Their results occur under different models and tasks, so their prescriptions should not be presented as an established contradiction. The project would test whether a reproducible task-dependent crossover appears under a shared evaluation. Fixed mixing and layer alternation inspired by CAPL are important baselines for a router. Full CAPL also includes preference training, so frozen-mask adaptations must be labeled CAPL-inspired rather than reproductions. The same distinction applies to using MIMIC’s attention rule without its trained model. [@sofa] [@capl] [@focus] [@delimiter] [@mimic]

**Study.** Create about 300–500 matched image sets with isolation, attribute-binding, and integration questions. Compare the unchanged model, two compatible existing policies, a fixed mixture, random routing, and an oracle that selects the better policy per example. Evaluate pairwise accuracy, wrong-image attribution, cross-image hallucination, worst-group accuracy, and inference overhead. Hold out question paraphrases and an entire task template so the router cannot simply memorize superficial wording.

Only after an oracle routing advantage appears should the team fit a tiny classifier to question text or use a transparent task rule. Freeze it before final evaluation. Use the same policy implementations and image processing for all comparisons. Masking may not be compatible with every fused-attention backend, so implementation fit is a serious early gate.

**Budget and pilot.** Plan 25–70 GPU-hours for 10,000–20,000 short evaluations, with a 3–8 GPU-hour initial study. The pilot should demonstrate a crossover in the same model and image sets, an oracle margin worth recovering, and a question-only router that improves over the best fixed mixture on held-out templates. If the oracle advantage is negligible or a fixed policy dominates both task groups, stop. The possible paper is a predictive account of intervention choice, not another generic attention mask.

## 4. Evidence-aware GUI retries

**Research question:** Does an agent decide to retry because an action actually failed, or because the screen did not visibly change?

In a local notes interface, an Add action may successfully append a note without refreshing the visible panel. A dropped action can leave the same screen. Retrying creates a duplicate in the first case and fixes the second. Conversely, an animation may occur even when the write failed. This yields a useful 2×2 design: actual success/failure crossed with visible change/no change.

VeriGUI trains verification and recovery, VisCritic uses before/after visual state comparisons, and Interactive Reward Agent already checks persistent environment state beyond screenshots. The new contribution cannot be “verify actions” or “inspect hidden state.” It would have to be the sequential retry decision under ambiguous evidence, including the different consequences of idempotent actions and duplicate-producing actions. [@verigui] [@viscritic] [@ira]

**Study.** Build 200–300 independent episodes, each with four controlled variants. Keep application logs as evaluator-only ground truth. Compare direct screenshot judgment, an expected-effect prompt, a pixel-difference rule, always retry, always inspect, and a budget-matched IRA-style verifier. A proposed gate acquires one read-only observation when evidence is insufficient and the retry could duplicate a write. Every policy must have the same observation access; the proposed method cannot read privileged success flags.

Pixel-identical cases with identical observable histories are not ordinary binary classification examples: success is unidentifiable from those inputs. Grade recognition of insufficient evidence separately. Include cases with subtle but usable evidence and a small natural application slice, because a toy impossibility demonstration alone is weak research.

Report duplicate-write rate, missed failures, completion after at most two extra decisions, and observations per episode. Bootstrap by base episode. The practical target is fewer duplicate actions at matched completion rate and observation cost. The main expense is reliable environment construction, not GPU training.

**Budget and pilot.** Forty episodes × four variants, two small models, and three policies produces roughly 1,300–1,600 decisions including follow-ups: plan 4–10 GPU-hours. The full study is about 10,000–18,000 calls and 32–80 GPU-hours. Continue if screen-change bias appears across models and the gate improves the duplicate/completion/cost tradeoff. Stop if an ordinary expected-effect prompt or always-inspect achieves the same frontier, or if released IRA data already cover the exact factorial retry problem.

## 5. Separate compression error budgets for fragile evidence

**Research question:** Can a compression policy save real time while keeping additional errors small in each important input group, rather than merely preserving average benchmark accuracy?

Define compression harm as an example where the full model answers correctly and the compressed model answers incorrectly. Report this separately from beneficial flips; the latter should not cancel destructive changes when evaluating capability preservation. A 2% compression-harm target is not a claim of 98% absolute accuracy.

ET-Prune already allocates tokens using evidence cues, CARES predicts sufficient resolution, and LazyMCoT uses calibrated difficulty routing. General risk-control techniques also exist. The possible contribution is an evaluation of group-specific harm, including calibration cost and failure under distribution shift, with a transparent policy that saves measured time. [@etprune] [@cares] [@lazymcot] [@crc]

Use a few predeclared groups, such as small text, dense text, spatial relations, and other questions. Deployment groups must be observable without gold answers. Cache predictions for a finite menu of resolution or pruning policies. Choose a fast policy only when a one-sided error bound satisfies the target in every group, correcting for selection across groups and candidate policies. Keep development, calibration, and final test images separate. Error bounds require the stated independent-sampling assumptions and a matching calibration/test population; a shifted-dataset result is an empirical stress test, not protection against arbitrary distribution shift. Correctness need not improve monotonically with visual budget, so do not apply a monotonicity-dependent guarantee without checking its conditions.

Estimate 20–60 GPU-hours for a 3B study, or 40–100 with a second model. A 300–500 image pilot should first show an oracle saving of roughly 20% at an acceptable harm level. Then check whether a cheap predictor preserves useful savings after its overhead. Sparse calibration groups may force full inference; report that result rather than suppress uncertainty. This is feasible as a course analysis but has high publication novelty risk unless the group failure is important, reproducible, and recoverable.

## Two lower-priority alternatives

**One missing video frame.** Give a model sparse frames that leave a question ambiguous, then let it buy one additional timestamp. Compare post-reveal accuracy and risk reduction per extra frame against random, midpoint, motion-based, and existing query-aware selection. VirtueBench already measures answerability under sampling, TRAPSBench studies epistemic restraint, and VideoAgent already performs active video search. A defensible extension would isolate the quality of choosing a discriminating observation under the same initial information and the same frame/call budget. If a motion-based selector scans otherwise unseen frames, count those accesses, feature passes, and preprocessing costs; equal final frame counts alone are not equal information. Timestamp selection must have usable cues in the allowed observation; an oracle choice without such cues is only an upper bound. The pilot is 50 paired scenarios; the full study is 200–400 short clips, with a 24–72 GPU-hour planning budget. Stop if a midpoint or cheap sampler is effectively as good, or if the model cannot answer with the oracle frame. [@virtuebench] [@traps] [@videoagent]

**Invalidating derived visual memory.** A screenshot update changes one primitive fact, such as seats occupied at a table. Does the memory update that fact but preserve an obsolete conclusion such as which table has most empty seats? Compare append-only summaries, last-write-wins records, full refresh, and evidence-linked invalidation at the same memory and extraction budget. WorldMemArena and MemLens already evaluate memory updates, so the narrow angle is propagation to derived conclusions and preservation of unaffected facts. Use 150–300 short sequences, a shared frozen extractor, and oracle-extraction controls; plan 16–48 GPU-hours. Stop if ordinary record updates plus recomputation achieve the same result. This is a backup, with substantial novelty risk from existing memory and truth-maintenance work. [@worldmem] [@memlens]

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

For the recommended project, read **Cui et al., FEATHER, FastV, PosPrune, and the end-to-end break-even study** before writing the novelty claim. Then inspect What’sUp examples and reproduce one compatible pruning baseline. The decisive question is whether the proposed relation-specific mechanism remains after FEATHER-style coverage and position controls. [@spatialbinding] [@feather] [@fastv] [@posprune] [@breakeven] [@whatsup]

For the easiest multi-image alternative, read **MIMIC, Counting Copies as Evidence, SoFA, and MIOH**. For the GUI alternative, read **VeriGUI and IRA** before implementing an environment. These short reading sets are more valuable for committing to a project than another undirected sweep of titles.

| Generic pitch | Why it was not promoted |
|---|---|
| Zoom before abstaining | BCEA already combines budgeted visual evidence acquisition and calibrated abstention. [@bcea] |
| Protect small OCR regions during pruning | ET-Prune already targets question-conditioned evidence and dynamic budgets. [@etprune] |
| Route difficult examples to more visual reasoning | CARES and LazyMCoT occupy close versions. [@cares] [@lazymcot] |
| Restore compressed tokens for later questions | SparseVILA and recent cache-refresh work already address related multi-turn needs. [@sparsevila] [@conduit] |
| Recheck a stale screen before acting | Existing TOCTOU and pixel-versus-structure studies have direct proposals. [@toctou] [@pixels] |
| Reverse video order and test consistency | CRPO/DyBench already use related counterfactual constructions. [@crpo] |
| Train a small hallucination probe | HALP already targets detection before token generation. [@halp] |

The recommended decision is to pilot Project 1 first, with Project 2 as the implementation fallback. Commit to a final contribution only after the baseline and mechanism checks, and let a failed pilot eliminate an idea quickly. A careful finding that narrows when a popular method works can be more defensible than a small average-score increase with an unclear cause.
