# Paper-level directions: modality geometry and the VLM interface

Research follow-up · 15 September 2026

**16 September update:** see [the expanded experiment blueprints](two_project_experiment_blueprints.md). A closer paper, [Hidden Forgetting in Continual Multimodal Learning](https://arxiv.org/abs/2607.02020), substantially narrows the novelty of Direction 1. The expansion refocuses it on causal discrimination between interaction formation and downstream reading, and corrects the assumption that native SmolVLA necessarily retains an unchanged full VLM evaluation path.

## Recommendation and scope

The strongest fit is **Does adaptation preserve the modalities but damage their interaction?** It connects a practical training problem to a geometric investigation, supports experiments on existing VLMs, and leaves a natural VLA extension. If training dynamics is more interesting than representation analysis, choose **What actually makes action gradients damage a pretrained VLM?** The third option, **Which alignment measurements characterize a usable interface?**, is attractive mathematically but faces the greatest overlap with recent work.

Here, geometry means the geometry of representations, readouts, and parameter updates. Spatial reasoning is one useful controlled test, not the definition of the research topic.

All proposed findings below are hypotheses. No experiments have been run. This is a focused follow-up to the earlier literature ledger, not another claim of reviewing 1,000 full papers. The linked papers provide problem evidence and novelty boundaries; neither citations nor a search establish that a proposed contribution is unpublished.

The course's public [project requirements](https://faculty.cc.gatech.edu/~zk15/teaching/AY2026_cs8803vlm_fall/index.html) support a hypothesis-driven research project with baselines, reproducibility, and evidence. Proposal slides are due September 16; the proposal can honestly present competing hypotheses and a pilot plan. It should not promise a finding or a theorem that has not been established.

## How to study the interface without pretraining

For a modular VLM, write the computation as

\[
z=E(x),\qquad v=P(z),\qquad p(y\mid x,q)=D(v,q).
\]

Study an existing encoder E, connector P, and decoder D. Extract features, compare controlled input pairs, intervene on activations, and conduct short adaptation experiments on a copy of the released checkpoint. None of these requires learning visual or linguistic capabilities from scratch.

There are three distinct questions: what information a representation contains; what the existing decoder uses; and what restricted adaptation can make the decoder use. A linear probe answers only a restricted version of the first. A trained connector does not, by itself, demonstrate that the decoder can exploit everything it transmits.

A connector can expand feature dimension. Therefore, neither a small parameter count nor the word “projector” proves information destruction. Conversely, high-dimensional features can retain a variable while making its extraction difficult for a particular decoder.

The affordable claim is about **existing interfaces and local adaptation**. Claims about how the interface originally emerged during foundation pretraining require suitable training checkpoints or new pretraining experiments. We should not make those claims from endpoint comparisons.

Freezing the vision encoder permits caching its features. If P is trained, gradients still pass through the frozen language decoder, so language activations cannot generally be cached for training P. “Few trainable parameters” does not imply negligible GPU cost.

## 1. Does adaptation preserve the modalities but damage their interaction?

**Working title:** Interaction Forgetting in Vision–Language Adaptation.

### Actual problem and proposed explanation

Adapting a general model to a small domain can improve the training task while weakening instruction-dependent grounding. In robotics this can mean selecting the familiar object even when the instruction specifies another one. [LIBERO-CF](https://arxiv.org/abs/2602.17659) documents counterfactual language-following failures; [Flatness Preserves Instruction Following](https://arxiv.org/abs/2606.23641) studies their relationship to fine-tuning and proposes SAM. The existence of instruction blindness is already established and cannot be our contribution.

Our narrower hypothesis is that some adaptation failures primarily damage the *conditional interaction* between modalities, while separately measured visual content and language content remain accessible. Average accuracy, marginal feature alignment, and individual concept probes may then miss the damaged computation.

This is stronger than “models ignore language.” It asks what changes during adaptation, why common preservation diagnostics can miss it, and whether preserving that specific computation prevents the failure.

### A concrete experiment that exposes the distinction

Use a scene with two objects and two instructions. All four combinations must be valid, with known labels:

| Scene | Ask for red object's side | Ask for blue object's side |
|---|---|---|
| Red left, blue right | Left | Right |
| Red right, blue left | Right | Left |

A vision-only rule cannot solve both questions. A language-only rule cannot solve both scenes. A model must bind the requested attribute to its position. This is a diagnostic example, not the whole benchmark.

For a fixed scalar output score, such as the left-versus-right logit difference, compute

\[
\Delta f=f(x_1,q_1)-f(x_1,q_2)-f(x_2,q_1)+f(x_2,q_2).
\]

An additive score f(x,q)=a(x)+b(q) has zero mixed difference. Nonzero mixed difference shows nonadditivity in the chosen score, **not necessarily correct grounding**. Evaluate its sign against the task, the four individual answers, and its change relative to the original checkpoint. Softmax saturation and arbitrary answer-token choices require controls; a logit difference avoids some probability-scale artifacts.

The mathematical starting point is ordinary functional ANOVA: under a product evaluation distribution, a square-integrable score decomposes into a constant, a vision term, a language term, and an interaction term. Adding a zero-mean product a(x)b(q) can alter the interaction without changing either conditional marginal mean. This demonstrates a limitation of marginal diagnostics. It is standard mathematics, not a novel theorem.

### Where the geometric contribution could come from

Construct the analogous mixed differences of hidden states at matched positions after vision and language have interacted. Across scenes, assemble these vectors into a matrix and examine their singular directions, their change during adaptation, and their effect on the output. Compare this interaction-specific geometry with global feature geometry on the same examples.

For architectures that place all image tokens before the instruction under a causal mask, those image-token states cannot depend on later instruction tokens. Their interaction difference is zero by construction. Measure the relevant computation at instruction/answer positions or in the cross-modal attention output instead. A language-independent connector can still shape the visual information available to that later computation.

The proposed empirical findings are:

1. Adaptation weakens task-relevant interaction before it substantially harms independently measured post-fusion visual and language accessibility.
2. A small set of interaction directions or attention pathways predicts which scene–instruction combinations fail, beyond baseline accuracy and global drift.
3. Restoring the identified computation from the original checkpoint selectively restores grounding; equal-size random interventions do not.

These three findings would support a coherent analysis paper. A generic interaction score, a collection of heatmaps, or a new regularizer alone would not.

### Experiments and controls

Start with [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) and replicate on [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). Fix image resolution, visual-token budget, prompt template, answer format, and precision. Use existing checkpoints; do not replace their vision towers.

Use a small [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) diagnostic subset or controlled renders to verify the measurement, then construct audited natural-image tests from [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html) scene graphs. GQA does not automatically supply valid four-cell counterfactuals: pairs need construction and verification. Attribute-to-location, relation-to-object, and object-to-property queries provide different interaction families. A result limited to left/right toy scenes is insufficient.

Create short adaptation trajectories using small domain subsets. First measure correlations in the actual adaptation data; deliberately biased versus balanced subsets are a controlled follow-up. Hold dataset size and training budget fixed. Save checkpoints before, during, and after adaptation. Split by original scene and object combination so paired variants cannot leak across train and test.

Track the adaptation task, ordinary VQA retention, four-cell correctness, and interaction geometry. Frozen-encoder probes staying constant is an architectural consequence, not evidence for the hypothesis. The interesting preservation test is post-fusion content accessibility with held-out linear and small nonlinear probes, as well as actual recognition behavior.

Select intervention directions and layers on calibration scenes. Evaluate activation patching on separate scenes and include random subspaces, norm-matched changes, and patches that restore comparable global feature distance. Use correctly functioning original-checkpoint examples for restoration tests, while reporting both that subset and the unfiltered evaluation population.

An optional remedy is to preserve a small set of validated interaction contrasts during adaptation. Compare it with equal-budget ordinary replay, balanced-data sampling, output distillation on the *same four examples*, feature anchoring, and SAM. If an interaction penalty merely gets more examples or more forward passes, the comparison is uninformative. Preserve only interactions the original model performs correctly; otherwise the method can preserve mistakes.

### Novelty boundary and decision rule

[PID analysis of LVLMs](https://arxiv.org/abs/2603.29676) already measures unique, redundant, and synergistic information and follows fusion across training stages. Our finite differences are not a PID estimator. The proposed distinction is controlled *degradation and causal restoration* of a conditional computation during small-data adaptation, with marginal diagnostics deliberately held accountable.

[Semantic Anchoring](https://arxiv.org/abs/2607.13597) already separates shared and private action representations; [CounterAlign](https://arxiv.org/abs/2608.21740) already uses counterfactual instruction supervision. Neither “shared/private” nor “add counterfactual pairs” is a sufficient novelty claim. LIBERO-CF and SAM are direct problem and baseline references, not peripheral citations.

Continue if the failure appears on natural-image tests in more than one model, interaction measures add explanatory value, and selective restoration survives controls. Stop or revise if failures are explained by ordinary recognition loss, if global drift predicts equally well, or if only artificially biased toy training produces the effect. A negative finding can be useful, but its publication value depends on overturning a meaningful expectation with strong evidence.

**Compute plan:** allocate about 5–10 GPU-hours to the pilot and cap the initial VLM study at 40–80 GPU-hours, on one intermittent 24–48 GB GPU. These are planning allocations, not measured runtimes. Start with 500–2,000 adaptation examples, short answers, and three seeds for the selected comparisons. Reduce the experimental grid after the pilot rather than running every combination. Keep a VLA replication optional until the VLM result is convincing.

## 2. What actually makes action gradients damage a pretrained VLM?

**Working title:** Initialization, Scale, or Geometry? Disentangling Negative Transfer at the VLM–Action Interface.

### Actual problem and competing explanations

[Knowledge Insulation](https://arxiv.org/abs/2505.23705) shows that naïvely coupling a continuous action expert to a pretrained VLM can harm training and knowledge transfer. Its recipe includes stopping action-expert gradients and giving the backbone discrete-action supervision. It is not simply “freeze the entire VLM.”

Several explanations can fit the observed degradation: an immature action head sends unhelpful gradients; action updates are too large; gradients move along directions important to pretrained behavior; or narrow demonstrations encourage visual shortcuts. [AEGIS](https://arxiv.org/abs/2604.16067) advances a spectral gradient mismatch account and uses geometric protection. [Flatness Preserves Instruction Following](https://arxiv.org/abs/2606.23641) provides another optimization-based account. These explanations should be separated experimentally rather than treated as interchangeable facts.

The paper question is **which mechanism dominates, under what conditions, and whether those conditions predict when backbone adaptation is beneficial**. A new gradient gate would be optional supporting evidence.

### Geometry with a testable consequence

For semantic-retention loss L_s and an actual small parameter update δ, Taylor expansion gives

\[
L_s(\theta+\delta)-L_s(\theta)
\approx g_s^\top\delta+\tfrac12\delta^\top H_s\delta.
\]

For simple SGD on action loss, δ=-ηg_a. For Adam or clipping, measure the actual optimizer update instead of substituting a raw gradient.

Near a stationary point of the retention objective, the linear term can be small while the quadratic term is not. Consequently, a gradient cosine can look harmless although the update damages behavior. A model-output KL anchor offers a related local Fisher approximation, but an empirical Fisher is not universally the Hessian of a ground-truth retention loss.

This is established local optimization mathematics. The contribution would be its controlled validation at the action interface: does directional curvature predict measured semantic damage after controlling update size and expert maturity? Do those predictions distinguish necessary adaptation from avoidable damage?

Do not assume regression gradients are inherently low-rank or CE gradients inherently high-rank. For a simple linear layer, a single-example gradient is an outer product under both losses. Batch construction, token count, output dimensionality, and head architecture affect observed rank. This makes matching those factors essential.

### A bounded design

Use an accessible small VLA such as [SmolVLA](https://huggingface.co/lerobot/smolvla_base), after checking that its implementation exposes the relevant backbone and expert gradients. Start from the same backbone in every condition. Warm an expert on the training split with the backbone frozen; compare it with a newly initialized expert of the same architecture. Include that warmup cost in total compute.

Cross expert maturity with update size control. Compare the resulting updates at equal actual norm and at matched action-loss progress; neither comparison alone answers every question. Evaluate semantic retention, conditional instruction following, and action learning. Add a controlled data-coverage comparison after the main factorial experiment, rather than immediately multiplying every factor into a huge grid.

Single-update interventions from identical checkpoints provide a cheap causal test: predict damage, apply the update to a copy, measure damage, restore the checkpoint, and repeat. Finite directional second differences of the retention loss can estimate curvature without forming a full Hessian. Use independent calibration and evaluation examples.

Follow with a small number of short training trajectories to test whether the one-step account survives cumulative adaptation. Baselines should include plain joint adaptation, head-only training, a fixed head-warmup schedule, learning-rate/clipping controls, and a preservation method such as SAM or anchoring. A faithful Knowledge Insulation comparison also needs its discrete-action objective; call a stop-gradient-only experiment an ablation, not a reproduction.

Comparing discrete CE with continuous regression is a possible extension, but changes to discretization and head design confound a claim about the loss alone. First establish the result with the same continuous loss and head architecture. Only then use a carefully matched toy setting to separate loss parameterization from action information.

For robotics evaluation, use a small held-out task subset and feasible counterfactual instructions from [LIBERO-CF](https://github.com/yuffish/LIBERO-CF). Split demonstrations by trajectory, not adjacent frames. Offline action error supports a claim about prediction and training dynamics; it does not establish control performance. A paper claiming better policies requires closed-loop evaluation with uncertainty over episodes.

### What would make this a paper?

A result such as “update magnitude and head maturity explain the apparent rank effect in one regime, while semantic directional curvature explains another” would refine practical training advice. It could explain when to warm the head, when to protect the backbone, and when those measures merely inhibit useful adaptation. Merely showing that stopping gradients helps would repeat existing work.

Continue if a controlled effect persists after matching scale and data, predicts unseen updates or checkpoints, and produces a different practical decision from simple warmup or smaller learning rates. Abandon a spectral novelty claim if it disappears when batch/token count is matched.

**Compute plan:** 8–15 GPU-hours for single-step and short-trajectory pilots; roughly 60–120 GPU-hours allocated to a focused study, with simulation evaluation separately budgeted after measuring throughput. This has more implementation and evaluation risk than Direction 1. No real robot or foundation-model pretraining is required for the scoped analysis, but a small offline study should not be presented as a general robotics result.

## 3. Which alignment measurements characterize a usable interface?

**Working title:** Alignment Is Task- and Decoder-Dependent: A Controlled Audit of VLM Interface Geometry.

### Problem and hypothesis

Researchers use modality gaps, representation similarity, principal components, and projector spectra to explain success or failure. Yet two interfaces can have similar global alignment while differing in the small directions needed for a particular question. Conversely, a change of coordinates can alter a geometric statistic without changing the function of the composed system.

The proposed study asks **when a geometry measurement supports a functional claim**, and whether measurement failures lead to wrong adaptation decisions. The goal is a validated set of conditions, counterexamples, and recommendations, rather than another universal alignment score.

### Theory and experiments

At a connector boundary, introduce an invertible transformation A and its exact inverse before the unchanged downstream computation. The composed function is unchanged, while some measurements of the intermediate representation change. This is an instrumentation control; it does not imply arbitrary basis changes are symmetries of an unmodified transformer. Layer norms, position encoding, and residual paths must not be silently ignored.

This control exposes coordinate-sensitive claims, but is insufficient by itself: a measurement tied to a specific trained parameterization may still be useful. Next test naturally occurring adaptation checkpoints and legal, small, task-relevant representation interventions.

For a visual change δv and fixed question q, a local decoder approximation gives

\[
\delta s\approx J_D(v,q)\,\delta v.
\]

Directions equally large in Euclidean feature space can have different effects on a fixed answer score. Distinguish information present in E(x), information accessible after P, and information used by D. Use held-out probes of matched capacity, norm-controlled interventions, and actual decoder outputs. Do not infer information destruction solely from failed linear decoding.

Compare global geometry, task-conditioned geometry, and behavioral evidence on recognition, relational queries, and fine detail. A useful result would show exactly when a common metric gives a wrong explanation or recommends a wrong restricted adaptation, then validate the corrected interpretation across models.

### Why this is the riskiest novelty choice

[Rethinking Model Selection Through Gromov–Wasserstein Distance](https://arxiv.org/abs/2605.01325) already connects cross-modal structural similarity to encoder selection and validates it with extensive full VLM training. [Layer Card](https://arxiv.org/abs/2602.04019) already develops a projected-residual view of where to place adapters. [Fisher-Projected Distillation](https://arxiv.org/abs/2608.01263) already uses a locally reachable visual-output space. [Gap-Init](https://arxiv.org/abs/2602.01522) already uses modality geometry to initialize extreme low-rank adaptation.

Consequently, “predict training from geometry,” “use the decoder Jacobian,” and “choose LoRA layers geometrically” are not fresh standalone pitches. [When Language Overwrites Vision](https://arxiv.org/abs/2605.08245) and [The Dominance of Text Space](https://aclanthology.org/2026.acl-long.1699/) also occupy the territory of over-alignment and preserving projector geometry. A generic modality-gap critique would face substantial overlap.

Proceed only if the pilot finds a concrete, consequential discrepancy the existing accounts do not explain. A coordinate-change demonstration or a renamed metric alone would not justify a paper. This direction can use frozen models for most experiments, with a few adapter updates for causal validation.

**Compute plan:** 5–10 GPU-hours for a measurement pilot; 25–60 GPU-hours for a focused audit. Lower training cost comes with higher conceptual and novelty risk.

## A practical first-week decision

Prioritize Direction 1. Build 100–200 verified scene–instruction quartets and first confirm that the chosen base model can solve enough of them. Test the mixed-difference measurement on known additive and interactive functions, then on the actual model. Run two small adaptation conditions from the same checkpoint and inspect whether recognition, instruction interpretation, and joint grounding separate.

The first deliverable should be one trustworthy figure with measured uncertainty: adaptation progress versus ordinary retention and conditional grounding. Its outcome decides whether further geometric localization is warranted. Do not build a large method before establishing the phenomenon.

If the effect is absent but action-interface training is appealing, Direction 2 is a coherent alternative with a different central hypothesis. Direction 3 is best pursued only after finding a specific failure of an existing geometric explanation.

All GPU-hour figures above are proposed allocations, not hardware guarantees. Benchmark a representative forward pass, backward pass, and evaluation job on the actual assigned GPU; include feature extraction, interventions, failed runs, and checkpoint evaluation in the budget. Jobs should be short and resumable for a crowded cluster. Publication prospects depend on the evidence, not on reaching a predetermined benchmark score.
