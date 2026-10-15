# Two project blueprints: geometry at multimodal interfaces

16 September 2026 · Proposed research, not experimental findings

## Decision in brief

Project 1 studies **where a learned vision–language computation stops working during adaptation**. Project 2 studies **why an action-learning update damages the pretrained computation**. Both can start from released checkpoints. They address different causal questions and should initially be separate projects.

Project 1 best matches an interest in representation geometry. Project 2 offers a more direct optimization problem, but needs more careful model plumbing. In both, the theoretical identities below are background or elementary derivations. A paper contribution requires new, reproducible empirical evidence or a substantial theorem beyond those identities.

### Updated novelty assessment

A closer paper identified during this expansion, [Hidden Forgetting in Continual Multimodal Learning](https://arxiv.org/abs/2607.02020), already studies evidence-use drift despite retained accuracy and proposes reliance preservation through channel interventions. This materially narrows Project 1: discovering hidden grounding drift or adding a reliance-preservation loss would overlap heavily. The proposed contribution must instead discriminate and causally localize **loss of an expressed interaction versus a mismatch between a retained representation and its downstream reader**.

The [PID study](https://arxiv.org/abs/2603.29676) already analyzes fusion across models, layers, and training. Project 2 must engage [Knowledge Insulation](https://arxiv.org/abs/2505.23705), [AEGIS](https://arxiv.org/abs/2604.16067), [Flatness Preserves Instruction Following](https://arxiv.org/abs/2606.23641), and older [Fisher-based forgetting protection](https://arxiv.org/abs/1612.00796). These are baseline obligations, not evidence of novelty.

## Project 1 — Where does cross-modal binding fail during adaptation?

### 1. Research question and useful outcome

A small domain-adapted VLM may still recognize the relevant objects and parse an instruction, while selecting the wrong object when those capabilities must be combined. Is the adapted model failing to construct the useful joint representation, or does a usable representation remain while later computation reads it incorrectly?

This distinction matters for training decisions. A failure in forming the interaction suggests changing fusion or supervision. A failure in reading retained information suggests that rebuilding the vision encoder is unnecessary. A method is optional: the central contribution could be an experimentally validated account of which failure occurs, where, and under which adaptation conditions.

The hypothesis is conditional. We do not assume all forgetting has this form, all interactions are low-rank, or all geometry survives adaptation.

### 2. Build an evaluation where the interaction is identifiable

For each diagnostic scene, create two visual states x_s and two questions q_t, with s,t in {-1,+1}. Index them so the correct binary answer is y=st. A basic example asks for the side of a red or blue object in two arrangements with their positions exchanged.

All four combinations must be valid and answerable. This is a factorial intervention on visual arrangement and linguistic request; it is not removal of an entire modality. Include three families:

| Family | Visual factor | Question factor | Target |
|---|---|---|---|
| Attribute binding | Which object has an attribute or position | Which attribute is requested | Correct candidate/side |
| Relation binding | Arrangement of multiple objects | Which relation is requested | Correct object |
| Property selection | Distribution of properties across objects | Which object is requested | Correct property |

Use rendered examples to verify causal control, followed by audited natural images. [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) offers scene graphs; [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) offers controlled reasoning data. Neither automatically provides the exact required quartets. Natural-image variants need validation. Where mirroring is appropriate, exclude images with text, asymmetric symbols, or ambiguous labels; include ordinary non-mirrored examples to detect transformation artifacts.

Split by original image/scene before making variants. Split scene groups again into calibration, validation, and final evaluation. Keep target labels, object combinations, and answer-token identities balanced. Report an unfiltered test and a subset the original model answered correctly, with the subset selected once before adaptation.

There is a simple motivation from experimental design. On the four cells, the basis [1,s,t,st] can separate constant, visual, linguistic, and interaction effects. If training sees only s=t, then both 1 and st equal 1: the constant and interaction terms are unidentifiable from those observations. Both visual states and both questions may be individually represented in the data, yet their composition is not identified. This is a toy counterexample to equating marginal coverage with interaction coverage; it is not a theorem about every VLM's optimization.

### 3. Define the mathematical object precisely

At a fixed layer and token position after modalities have interacted, let h_st be the hidden state. Define

\[
\mu=\tfrac14\sum_{s,t}h_{st},\quad
a=\tfrac14\sum_{s,t}s h_{st},\quad
b=\tfrac14\sum_{s,t}t h_{st},\quad
c=\tfrac14\sum_{s,t}st h_{st}.
\]

Then, exactly on this four-cell design,

\[
h_{st}=\mu+s a+t b+st c.
\]

This is a vector-valued factorial decomposition. c measures interaction *expressed in these coordinates at this location*. A zero c does not prove the underlying variables are absent: a nonlinear downstream operation could still combine separately represented s and t. A nonzero c does not establish correct reasoning. This statistic is also not mutual information or PID synergy.

At the final normalized hidden state, suppose two fixed single-token answers have logit difference f_st=w^T h_st+b_0. Their balanced signed-margin average is

\[
m=\tfrac14\sum_{s,t}st f_{st}=w^T c
=\|w\|\,\|c\|\cos\angle(w,c).
\]

This is exact for that output head, before any nonlinear logit postprocessing. It gives an operational decomposition into interaction magnitude and alignment with the answer readout. If the output embeddings are frozen, w is fixed. At earlier layers, use the actual remaining network and local directional responses rather than pretending that the final unembedding is a calibrated reader there.

The mean margin does not replace four-cell accuracy: one very confident correct answer can mask other errors. Report cellwise margins, minimum signed margin, all-four-correct rate, and ordinary per-question accuracy together.

### 4. Competing mechanisms and predictions

| Explanation | Predicted observation | Discriminating follow-up |
|---|---|---|
| Basic perception deteriorates | Object/attribute access falls too | Recognition behavior and matched probes |
| The interaction is less expressed | Task-aligned c shrinks in later layers | Layerwise contrasts and restoration of the relevant earlier computation |
| A reader becomes mismatched | Refit probes recover performance; fixed old probes/readout perform worse | Tiny reader repair and selective downstream restoration |
| Attention routing changes | Task-relevant visual contributions change at a specific operation | Controlled query/key/value or attention-output patches |
| Response format changes | Forced-choice margins are stable but free-form scoring falls | Fixed answer format and parser-free logit evaluation |

The rows are not mutually exclusive. Report mixed cases and cases where the measurements cannot distinguish them. Changes in a heatmap are not sufficient evidence of routing causality.

For representation probes, use the same capacity and training budget before and after adaptation. Compare both an old probe evaluated on new states and a newly fitted probe with disjoint training examples. Use linear and small nonlinear probes. The latter can distinguish loss of linear accessibility from a stronger loss of practical decodability, but neither proves information-theoretic erasure.

In image-prefix causal architectures, visual tokens before the question cannot contain an interaction with that later question. Measure answer/instruction positions and relevant fusion operations. Holding an encoder frozen also makes its raw features constant; that is a control, not an empirical discovery of preserved competence.

### 5. A staged experimental design

**Pilot.** Start with one released model, for example [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct), and 100–200 quartets. Check base competence, answer-token mapping, and the decomposition. Measure actual throughput before selecting the full grid. A second family such as [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) is a replication target.

**Adaptation.** Use 500–2,000 examples from a concrete narrow domain of referring/attribute questions. Begin with an ordinary domain subset. After measuring its correlations, construct a comparison with better scene–instruction coverage. Match example count, output format, training steps, and optimization settings; document distribution changes rather than claiming every marginal is matched. In the binary toy design, changing interaction coverage can itself change answer priors, so those factors need separate controls or a richer design.

Freeze the vision encoder and original output head initially; train modest LoRA adapters in the language/fusion computation. This isolates a meaningful question while permitting encoder-feature caching. Inspect representative early, middle, and late checkpoints. A minimal core is two adaptation conditions times three seeds: six short training runs. Add the second model on the decisive comparison after the pilot. Do not sweep every rank, layer, dataset, and learning rate.

**Geometry.** On calibration examples, form a matrix of interaction vectors across scenes. Fit candidate low-dimensional subspaces and test their out-of-sample usefulness. Apparent low rank with few examples is not evidence: compare against matched random/content contrasts and held-out explained variation. Test whether interaction diagnostics predict failures beyond initial margin, task difficulty, global feature drift, and single-channel reliance.

**Causal tests.** Identify a small layer range and computation using calibration data. Restore only a selected component from the original model on held-out cases. Include same-model patches, random subspaces, norm-matched changes, and different-example donor controls. Keep the downstream input structure valid. Cross-checkpoint patches can fail because of interface incompatibility, not because the patched site is irrelevant. Conversely, patching a full final answer representation can simply inject the answer. Treat that as a ceiling, not mechanistic proof.

**Repair test.** Only after a mechanism emerges, test the corresponding small repair: reader adaptation, restoring selected attention parameters, or an interaction-preservation penalty. Compare with ordinary replay, feature anchoring, and output distillation using exactly the same examples. If four variants are used in one method, its comparator gets those four variants too. Add a reliance-preservation comparator because of the newly identified closest paper. A full reproduction should be labelled separately from a simplified implementation.

### 6. What the paper would need to show

The strongest story would be: a practical adaptation regime produces a reproducible dissociation; the analysis localizes a particular failure in the interaction-to-reader path; a targeted intervention changes behavior as predicted; and the mechanism transfers to a second task family/model.

Three useful figures would be: (1) adaptation progress versus recognition, component accessibility, and joint grounding; (2) depth versus interaction magnitude, output accessibility, and intervention effect; (3) selective restoration compared with matched controls and generic repair.

The story fails if basic perception or response formatting explains the errors, the effect only exists on contrived scenes, or the geometric account predicts nothing beyond existing reliance diagnostics. A low-rank intervention that restores answers without explaining its specificity is a preliminary result.

### 7. Boundaries of possible theory

Global geometry can look stable while a small task-critical component changes. For instance, add many unchanged high-variance nuisance coordinates to h and let the answer depend on one interaction coordinate. Removing that coordinate can have a small relative effect on global geometry and a large effect on the task.

However, exact pointwise feature preservation would also preserve c. In fact, if d_st is the feature change on each cell,

\[
\|\Delta c\|\leq\tfrac14\sum_{s,t}\|d_{st}\|
\leq\sqrt{\tfrac14\sum_{s,t}\|d_{st}\|^2}.
\]

Thus the defensible claim concerns coarse/average diagnostics, insufficient coverage, and small task-relevant components. It is false to say that arbitrary strong feature preservation can never preserve interactions. With a fixed head, |Δm|≤||w|| ||Δc||; these elementary bounds can guide the measurements, but do not alone constitute a theory paper.

## Project 2 — Which action updates interfere with pretrained semantics?

### 1. Sharpen the question

The central question is whether semantic damage from action learning is explained by the size/rank of an update, by action-expert maturity, or by its orientation relative to directions that change pretrained predictions. The useful outcome is a diagnosis that predicts when backbone adaptation is worthwhile.

Keep the continuous action objective and head architecture fixed initially. Comparing CE with regression at the outset changes too many factors. Likewise, a new gradient projection method is not the core novelty; related methods already exist.

### 2. Define semantic change independently of action loss

Let θ denote the shared backbone parameters and φ the action expert. Around a reference checkpoint θ_ref, define on an anchor distribution A

\[
R(\theta)=\mathbb E_{u\sim A}
\operatorname{KL}\big(p_{\theta_{ref}}(\cdot\mid u)\|p_\theta(\cdot\mid u)\big).
\]

At θ_ref, R=0 and ∇R=0. Under ordinary differentiability conditions,

\[
R(\theta_{ref}+\delta)\approx\tfrac12\delta^T F_A\delta,
\]

where F_A is the model Fisher corresponding to the chosen output distribution and anchor inputs. This is not an empirical squared-label-gradient matrix by definition. An approximation using such a matrix needs separate validation.

The anchors may use fixed short answer prefixes to control cost. If only first-token or binary-answer distributions are measured, state that scope explicitly; it does not establish preservation of all language behavior. Evaluate actual held-out VQA accuracy and conditional grounding as separate outcomes, because KL drift can also reflect beneficial corrections.

For a nonstationary labelled retention loss, retain the linear term in its Taylor expansion. For later training checkpoints, distinguish a local reference reset from accumulated drift relative to the original model. Local Fisher predictions need not add exactly along a long trajectory.

### 3. The geometric object: update orientation, not rank alone

For stochastic applied backbone updates δ, let M=E[δδ^T] be their uncentered second-moment matrix. Then the local prediction is

\[
\mathbb E[R]\approx\tfrac12\operatorname{tr}(F_A M).
\]

This suggests measuring where action updates put their energy relative to semantically sensitive directions. Two update distributions can have the same norm and spectrum but different tr(F_A M).

For a two-dimensional example, take F_A=diag(100,1). An update εe_1 and an update εe_2 have equal norm; each deterministic second-moment matrix has rank one. Their predicted semantic changes are 50ε² and 0.5ε². Rank and size are identical, yet the prediction differs by a factor of 100.

This is standard matrix geometry, not a new theorem. Its potential value is in showing that real action-learning updates follow this account after the obvious confounds are controlled. Matrix rank of one reshaped weight gradient, rank of a LoRA update, and rank of M are different objects and must not be conflated.

### 4. Expert maturity gives a mechanistic connection

For an action head a_hat=H_φ(h_θ), backpropagation contains the factor

\[
g_\theta=J_{h,\theta}^T J_{H,h}^T\nabla_{\hat a}\ell_a.
\]

Head training changes both its residual errors and the Jacobian that transports them to the backbone. Therefore, warming an expert can change the direction, variance, and magnitude of backbone updates even when the backbone is unchanged. The hypothesis is that one or more of these changes explains semantic interference.

For a simple linear head W and squared error, g_h=W^T(Wh-a). That supplies a tractable toy model of how head maturity changes the signal. It does not imply every random head is harmful or every warm head is safe. A head can be well trained on shortcut-rich data and still send damaging updates.

### 5. A model setup that does not confound architecture with forgetting

For the controlled study, keep an intact pretrained small VLM and its language evaluation path. Attach a modest continuous action head/expert and train it on a small real demonstration subset. This trains a new head; it does not pretrain a new vision or language foundation model. Preserve an unchanged reference checkpoint for evaluation. All experimental arms begin with exactly the same backbone, processor, and semantic evaluator.

[SmolVLA's official configuration](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py) defaults to expert-only training and specifies a shortened VLM. Its [implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py) explicitly slices language layers when requested. These defaults mean native SmolVLA should not be assumed to provide an unchanged full VLM for VQA measurement. It is suitable for a later policy replication, but requires a documented semantic evaluation path and an actual gradient-flow check.

One possible audit path shares the policy's updated prefix weights with the original VLM's frozen remaining layers and language head, using the original VLM forward procedure for semantic evaluation. Validate this path against the original model at step zero and label it a restored VLM evaluator, not native-policy VQA. Alternatively, retain all layers in the controlled system from the beginning. If the baseline cannot perform the semantic tasks, it cannot meaningfully demonstrate semantic forgetting.

### 6. Single-update causal experiment

Make action-head snapshots at initialization, after a short warmup, and after longer warmup, always with the backbone frozen during snapshot construction. Use the same training split; do not define maturity from held-out test performance.

At each maturity level, collect candidate backbone updates from fixed action minibatches. Control action normalization, noise/timestep sampling for flow matching, batching, and optimizer state. Record actual applied updates, not merely raw gradients. Reparameterizing LoRA factors can change their Euclidean norms without changing effective weights; hold parameterization fixed and also report effective weight changes.

For each candidate:

1. Predict semantic change on calibration anchors.
2. Apply the backbone update to a disposable copy of the same reference state, holding the head fixed.
3. Measure semantic change and labelled retention on disjoint evaluation anchors.
4. Measure the action-loss improvement attributable to the backbone update with the head fixed.
5. Restore model and optimizer state before the next candidate.

Repeat at several small update scales to test the approximation's range. Compare raw updates, equal-norm updates, and matched action-progress subsets. Warmup costs count in any efficiency claim. At a single chosen weight block, controlled rotations of a low-rank update can preserve its singular values and norm while changing its direction; label these as diagnostic interventions, not legitimate training-gradient alternatives.

You do not need to form a full Fisher or Hessian. Start with directional finite differences of KL on a small calibration set; subsequently compare model-Jacobian or Hessian-vector estimates if worthwhile. Using the same examples both to estimate curvature and declare predictive success would be circular. Compare against norm, gradient spectrum, a diagonal Fisher approximation, and first-order labelled-loss prediction.

### 7. Distinguish the explanations

| Hypothesis | Evidence that would support it | Necessary control |
|---|---|---|
| Magnitude dominates | Maturity-related damage differences largely vanish after rescaling | Actual applied update norm, per-module scale |
| Expert maturity changes harmful directions | Warm and cold experts differ after equal-norm comparison | Same backbone, head architecture, minibatches, action normalization |
| Orientation is predictive | Directional Fisher/KL prediction transfers to unseen updates better than norm/rank | Disjoint calibration/evaluation anchors and held-out head snapshots |
| Data shortcuts dominate | Mature experts on narrow data still harm conditional instruction following | Separate data-coverage experiment, beyond loss reduction |
| Only local geometry matters initially | One-step predictions work but fail later | Measure approximation error and avoid extrapolating a local result |

Prediction metrics include rank correlation and error on held-out updates; benefit–damage curves should show action improvement against semantic degradation. A change metric alone must not be advertised as a classifier of harmful forgetting.

### 8. Validate over training trajectories

After the one-step experiment, run a compact comparison: joint adaptation from the start, fixed expert warmup, a smaller backbone learning rate, expert-only adaptation, and one established protection method. Three seeds are a reasonable target for the decisive comparisons. Use repeated trajectories and checkpoint evaluations as dependent measurements, not independent experimental replicates.

An optional new intervention could allow backbone updates only when the diagnostic predicts favorable action benefit versus semantic cost. It must beat simple warmup and learning-rate schedules on the same data and include diagnostic overhead. Otherwise the analysis may still be useful, but the method claim is weak.

A faithful Knowledge Insulation baseline includes its backbone action-supervision design. A stop-gradient-only condition is not a reproduction. [EWC](https://arxiv.org/abs/1612.00796), [SAM](https://arxiv.org/abs/2010.01412), and [gradient surgery](https://arxiv.org/abs/2001.06782) are existing ideas; compare or explicitly delimit them rather than presenting Fisher protection or gradient orthogonalization as new.

Split demonstration data by trajectory. Hold out task layouts and instructions when feasible. If claiming improved policies, run closed-loop simulation and report success with uncertainty over initial states. A small offline study can support a training-dynamics finding, but not general robotic control claims. Only compatible, validated checkpoints should be evaluated on [LIBERO-CF](https://github.com/yuffish/LIBERO-CF); a base checkpoint is not automatically ready for that benchmark.

### 9. What a successful paper would look like

The story could establish that a commonly used size/rank explanation is incomplete, identify how expert maturation changes the update distribution, and predict when protection or backbone adaptation is useful. Three central figures would show: equal-norm update damage by maturity; predicted versus observed semantic change on unseen updates; and action benefit versus semantic cost for practical training schedules.

The project weakens if a smaller learning rate explains everything, the diagnostic only predicts the data used to estimate it, or all improvements come from additional warmup compute. Showing that randomly resetting a head hurts is not enough. A negative result can refine training advice but needs a consequential, well-supported conclusion.

## Shared scope, compute, and course schedule

The figures below are allocations to constrain the project, not measured runtimes or promises of GPU availability. A representative benchmark on the allocated hardware must determine the final grid.

| Work | Project 1 allocation | Project 2 allocation |
|---|---:|---:|
| Pilot and implementation validation | 8 GPU-hours | 12 GPU-hours |
| Core adaptation / update experiments | 28 | 44 |
| Diagnostics and causal controls | 18 | 24 |
| Decisive replication | 10 | 16 |
| Reserve | 16 | 24 |
| Initial ceiling | 80 | 120 |

Robotics rollout evaluation is additional until measured. No multi-GPU requirement is assumed. Prefer one 24–48 GB GPU, small batches, short answer scoring, and resumable jobs. For Project 1, cache frozen encoder features and selected hidden positions rather than every token of every layer. For Project 2, reduce the number of candidate updates and calibration anchors before reducing essential controls.

Avoid full pretraining, a large benchmark sweep, many simultaneous hyperparameter axes, or a custom simulator in the core. Parameter-efficient adaptation still requires backward computation through downstream frozen layers; budget accordingly.

**Week 1:** establish the phenomenon and validate measurements. **Weeks 2–3:** complete the main controlled comparison. **Weeks 4–5:** localize or test the mechanism causally. **Weeks 6–7:** replicate decisive findings and evaluate a repair only if warranted. **Final weeks:** freeze implementation, audit splits, rerun necessary checks, and write limitations alongside results.

The two ideas can eventually connect: Project 2's semantic anchors could include Project 1's conditional grounding tests. That should be a later extension, after one central hypothesis has produced evidence. A class project should not depend on completing both programs.
