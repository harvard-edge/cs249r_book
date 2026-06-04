# Narrative Audit: Volumes 1 and 2

Worktree: `/Users/VJ/GitHub/MLSysBook-narrative-audit-vol1-vol2`

Branch: `codex/narrative-audit-vol1-vol2`

Date: 2026-06-03

Source lens: Systems Approach, "Textbooks in Tokenland", 2026-06-01:
https://systemsapproach.org/2026/06/01/textbooks-in-tokenland/

## Review Lens

The audit used this test: a textbook should provide perspective, build a mental model, motivate problems before mechanisms, choose details rather than act like an encyclopedia, and avoid prose that merely enumerates what an LLM could list.

Lists, tables, checklists, and reference appendices were not treated as problems by themselves. A section was flagged only when the list structure replaced the narrative work: no motivating design question, weak causal transitions, repeated taxonomies, or too many adjacent technique catalogs.

## Active Implementation Todo

### Commit organization

- [x] Use task/pass-oriented commits going forward rather than one commit per chapter when several chapters share the same edit pattern. Keep chapter-local commits only when the task is genuinely isolated to one chapter.
- [x] At the `517d071d22` checkpoint, decided not to rewrite the existing long post-merge history while macro-flow and agent findings are still active. A retrospective task-level rebase would be high-conflict and could obscure which earlier preservation/rule fixes came from which pass.
- [ ] Before any future history regrouping, create a safety branch at the pre-regroup HEAD and verify the worktree is clean. Do not rewrite the current chain while uncommitted chapter edits or active agent findings are still being integrated.
- [x] Prepare a review grouping map for the existing commits by pass so the branch remains understandable without a risky squash:
  - `dev` sync and conflict preservation: merge commit `ebb14bcf16` plus follow-up concrete-anchor restorations after the merge.
  - Progressive disclosure: commits whose subjects start with `Stage concepts progressively`, `Defer`, `Gloss`, or `Pace`, across Volume 1 and Volume 2.
  - Narrative/paragraph flow: commits whose subjects start with `Improve`, `Smooth`, `Fuse`, `Collapse`, `Clarify`, `Tighten`, `Reframe`, or `Group`, including the current macro-flow task commit `517d071d22`.
  - Pedagogical preservation: commits whose subjects start with `Restore`, `Keep`, or `Expand`, used when an earlier narrative edit over-compressed a useful example, sequence, or concrete anchor.
  - MLSysIM/LEGO/numeric-source work: commits whose subjects start with `Source`, `Use`, `Normalize`, `Name`, `Type`, or `Compute`, especially formatter/output-name and source-of-truth repairs.
  - Rules/emphasis/fallacy conventions: commits whose subjects start with `Demote`, `Mark`, `Normalize`, `Rephrase`, `Fix`, or `Restore ... fallacy labels`, covering `.claude/rules` compliance and chapter-architecture exceptions.
  - Macro subsection-flow pass: task commits from this point forward should name the chapter or pass and bundle only one coherent triage unit.
- [ ] Keep worktree progression linear: finish this narrative-audit worktree, merge its branch into local `dev` in `/Users/VJ/GitHub/MLSysBook`, then create the next task worktree from the updated local `dev` rather than allowing separate worktrees to splinter.

### Progressive-disclosure pass

- [x] Merge local `dev` into the worktree branch and preserve dev-side improvements.
- [x] Volume 1 chapters 1-16 edited and committed one chapter at a time.
- [x] Volume 2 progressive-disclosure agents completed for chapters 1-17.
- [x] Apply Volume 2 progressive-disclosure/narrative findings one chapter at a time.
  - [x] Volume 2 introduction edited and committed.
  - [x] Volume 2 compute infrastructure edited and committed.
  - [x] Volume 2 network fabrics edited and committed.
  - [x] Volume 2 data storage edited and committed.
  - [x] Volume 2 distributed training edited and committed.
  - [x] Volume 2 collective communication edited and committed.
  - [x] Volume 2 fault tolerance edited and committed.
  - [x] Volume 2 fleet orchestration edited and committed.
  - [x] Volume 2 deployment principles edited and committed.
  - [x] Volume 2 performance engineering edited and committed.
  - [x] Volume 2 inference edited and committed.
  - [x] Volume 2 edge intelligence edited and committed.
  - [x] Volume 2 operations at scale edited and committed.
  - [x] Volume 2 responsible fleet principles edited and committed.
  - [x] Volume 2 security and privacy edited and committed.
  - [x] Volume 2 robust AI edited and committed.
  - [x] Volume 2 sustainable AI edited and committed.
  - [x] Volume 2 responsible AI edited and committed.
  - [x] Volume 2 conclusion edited and committed.

### Dedicated follow-up passes

- [x] Full `.claude/rules` read: reviewed the complete rules inventory before resuming edits, with emphasis on `emphasis.md`, `book-prose.md`, `prose-craft.md`, `landing-fixes.md`, `fmt.md`, `math.md`, `lego-units.md`, `mlsysim.md`, `callouts.md`, `cross-references.md`, `bib-check.md`, `footnotes.md`, and MIT Press capitalization/style rules.
- [ ] Paragraph-shape pass across all edited Volume 1 and Volume 2 chapters: identify adjacent one- or two-sentence paragraphs that are really one idea, one transition, or one causal step, and fuse them into developed paragraphs when that improves student understanding. Preserve short paragraphs when they create deliberate pacing, introduce a sharp contrast, or isolate an important takeaway.
  - [x] `vol2/introduction.qmd`: fused scale-moment, sustainability, and D·A·M-to-C³ projection paragraphs where the split separated claim from evidence. Resolved in commit `628dbb6ca1`.
  - [x] `vol2/compute_infrastructure.qmd`: fused TPU trajectory and ring-vs-NVSwitch causal paragraphs while preserving concrete hardware anchors. Resolved in commit `ce55582658`.
  - [x] `vol2/network_fabrics.qmd`: fused the alpha-beta crossover anchor with the design-strategy implication. Resolved in commit `b20288dcd1`.
  - [x] `vol2/data_storage.qmd`: fused the feature-store/model-registry deferral with the retrieval-infrastructure handoff so the transition explains what belongs in storage versus operations. Resolved in commit `f1a996f782`.
  - [x] `vol2/distributed_training.qmd`: fused the three distributed-training trigger paragraphs into one escalating argument and collapsed an unused one-paragraph convergence signpost heading. Resolved in commit `a324d32b4e`.
  - [x] `vol2/collective_communication.qmd`: fused the quantization setup with its fidelity question and collapsed sparse-gradient H4 labels that were only one-paragraph continuations. Resolved in commit `fa3796cf8d`.
  - [x] `vol2/fault_tolerance.qmd`: fused checkpoint atomicity prose, collapsed debugging fragments, and restored compact diagnostic maps where symptom-to-action structure teaches better than dense prose. Resolved in commit `534f4edf77`.
  - [x] `vol2/fleet_orchestration.qmd`: collapsed an unused one-paragraph scheduler-challenge heading into the roadmap so the chapter flows directly into distributed scheduling complexity. Resolved in commit `fe38cb4508`.
  - [x] `vol2/performance_engineering.qmd`: fused online-softmax setup with its global-information constraint, localized a Volume 1 dependency phrase, and collapsed an unused one-paragraph MoE architecture heading while preserving the referenced KV-cache heading. Resolved in commit `dd559a4665`.
  - [x] `vol2/inference.qmd`: fused scheduling/topology openers with first cases, moved the MoE transition after the routing example, and restored a compact `Given` setup in the heterogeneous routing example. Resolved in commit `e3468a20f1`.
  - [x] `vol2/edge_intelligence.qmd`: fused the decentralized-coordination question with its three-phase setup and collapsed an unreferenced layer-selection subheading while preserving the profiling steps. Resolved in commit `56603c72f5`.
  - [x] `vol2/ops_scale.qmd`: fused debt and shadow-deployment fragments, collapsed monitoring H4s, and converted glossary-like efficiency/versioning bullets into action-oriented tables. Resolved in commit `b416926757`.
  - [x] `vol2/security_privacy.qmd`: fused the comparative-properties setup with the rationale for deferring the full privacy-technique table. Resolved in commit `4cf02052c4`.
  - [x] `vol2/sustainable_ai.qmd`: fused lifecycle/inference crossover transitions and collapsed an unreferenced resource-consumption heading while preserving the TinyML optimization sequence. Resolved in commit `38b072a0c9`.
  - [x] `vol2/responsible_ai.qmd`: fused the privacy-to-safety handoff so the transition carries the system-property contrast in one paragraph. Resolved in commit `a876b906a4`.
  - [x] `vol1/training.qmd`: fused the optimizer definition with the systems-implication sentence so the section enters as one teaching move. Resolved in commit `7ee6629f72`.
  - [x] `vol1/nn_computation.qmd`: fused depth-to-mechanics and autodiff-to-activation-storage transitions so the local paragraphs carry one continuous argument. Resolved in commit `e4e3789871`.
  - [x] `vol1/nn_architectures.qmd`: fused the hardware-mapping opener with the MLP/Tensor Core example while preserving concrete hardware anchors. Resolved in commit `3868ff3360`.
  - [x] `vol1/frameworks.qmd`: fused the distributed-execution deferral with the GPT-3 placement preview and collapsed unreferenced one-paragraph JAX subheads into a continuous functional-programming narrative. Resolved in commit `c5474cae59`.
  - [x] `vol1/data_selection.qmd`: fused the compute-optimal-frontier setup with the scaling-law diagnostic rationale while preserving the nearby paced takeaway and diagnostic bullets. Resolved in commit `a5528cf3f5`.
  - [x] `vol1/model_compression.qmd`: collapsed a thin sparsity hardware-support heading under structured sparse patterns and fused the toolchain-boundary handoff with visualization-based validation. Resolved in commit `c44bb310e0`.
  - [x] `vol1/ml_ops.qmd`: fused Oura dataset scale evidence and model-evaluation/MLOps implication paragraphs so the case study reads causally rather than as adjacent notes. Resolved in commit `dbdd43b604`.
  - [x] `vol2/robust_ai.qmd`: restored a compact variable list for the FGSM equation where symbol decoding is clearer than dense prose. Resolved in commit `829cea3b10`.
  - [x] `vol1/model_serving.qmd`: converted node-level optimization boundaries into a compact diagnostic map tied to profiler evidence. Resolved in commit `862d0868bd`.
  - [x] `vol1/benchmarking.qmd`: converted the LLM benchmark roster into a failure-oriented decision table with explicit score limits. Resolved in commit `fd3befbf21`.
  - [x] `vol2/ops_scale.qmd`: collapsed four one-paragraph platform-abstraction level subheads into a single referenced decision table. Resolved in commit `19a9abe88a`.
  - [x] `vol1/ml_ops.qmd`: collapsed single-paragraph optimization-framework and operational-maturity labels into their following narrative sections. Resolved in commit `e35b682e95`.
  - [x] `vol1/model_serving.qmd`: collapsed an unreferenced queuing-fundamentals overview heading into the Little's Law setup. Resolved in commit `c44719b8c2`.
  - [x] `vol2/ops_scale.qmd`: collapsed the generic underlying-principle H4 into the model-type operations closing transition. Resolved in commit `40abb558aa`.
  - [x] `vol2/inference.qmd`: collapsed a late fallacies-internal summary heading so the resource bottleneck hierarchy lands as a final pitfall payoff before the chapter summary. Resolved in commit `c29a297f38`.
  - [x] `vol2/robust_ai.qmd`: collapsed the unreferenced intuitive-understanding H4 so the X-ray distribution-shift example opens the existing concept-drift section directly. Resolved in commit `31f83c05c7`.
  - [x] `vol2/performance_engineering.qmd`: folded the one-paragraph KV-cache compression bridge into the weight/activation quantization section and retargeted the later cache-calculation reference to the surviving section. Resolved in commit `004a2ef5fe`.
  - [x] `vol2/edge_intelligence.qmd`: collapsed the biological-neural-efficiency H4 by moving the power-efficiency footnote to the first 20 W comparison and removing the duplicate opener. Resolved in commit `1d438f84f3`.
  - [x] `vol2/security_privacy.qmd`: collapsed redundant layered-defense and comparative-properties microsections while preserving the defense-stack explanation and the cross-method privacy comparison bridge. Resolved in commit `0fa63ec482`.
  - [x] `vol1/frameworks.qmd`: collapsed the execution-flow H5 into the TorchInductor cache paragraph so the compile pipeline runs directly into graph-break diagnostics. Resolved in commit `88b87546df`.
  - [x] `vol1/hw_acceleration.qmd`: folded the architectural-integration bridge into the precision-to-interconnect transition and aligned the concept metadata with the surviving interconnect section. Resolved in commit `0b5aa7c8c2`.
  - [x] `vol1/ml_ops.qmd`: collapsed the upstream-dependency-health H5 into the monitoring-stack follow-through while preserving the database-migration freshness example. Resolved in commit `d11c36d2f7`.
  - [x] `vol1/model_serving.qmd`: collapsed runtime configuration into the runtime-selection close so precision selection follows without a one-paragraph stop. Resolved in commit `584dcee0e5`.
  - [x] `vol1/training.qmd`: collapsed the mixed-precision practical-considerations heading into the cost/quality follow-through while preserving FP16/BF16 caveats. Resolved in commit `b4895b4ec1`.
  - [x] `vol2/introduction.qmd`: collapsed three governance H4s into a continuous threat-regulation-responsibility argument under the existing governance section. Resolved in commit `39dc46942c`.
  - [x] `vol2/inference.qmd`: collapsed the generic multi-region architecture-patterns bridge into the single-region limitations paragraph while preserving the concrete Pattern 1/2/3 sections. Resolved in commit `f991d6cf5b`.
  - [x] `vol2/ops_scale.qmd`: collapsed self-service guardrails and backfill challenge/best-practice labels while preserving the controls and incremental migration pattern. Resolved in commit `4e4fb119d7`.
  - [x] `vol2/sustainable_ai.qmd`: collapsed edge-measurement and sustainable-edge-pattern bridge headings, retargeting the later MLPerf Tiny cross-reference to the concrete hardware-monitor section. Resolved in commit `f013830615`.
- [ ] Content-preservation pass across compressed edits: verify that narrative tightening did not remove the reason a mechanism matters, the conditions under which it applies, or the limitations students need to understand.
  - [x] Ptolemy Volume 2 preservation findings resolved and committed:
    - `compute_infrastructure.qmd`: restored MoE scale/topology anchor from MLSysIM-backed Mixtral data.
    - `fault_tolerance.qmd`: restored image-classification fallback cascade scale ladder.
    - `edge_intelligence.qmd`: restored Krum and robust-aggregator assumptions.
    - `collective_communication.qmd`: restored vendor-backend performance caveat.
  - [x] Jason Volume 1 preservation findings integrated:
    - [x] `introduction.qmd`: restore Machine-axis roofline/ridge-point quantitative reason and fuse appendix referral with the following paragraph. Resolved in commit `9e383d0f82`.
    - [x] `data_engineering.qmd`: restore embedding-table mechanism for high-cardinality IDs.
    - [x] `ml_systems.qmd`: restore MobileNetV2 depthwise-separable-convolution mechanism.
    - [x] `frameworks.qmd`: restore fused-attention HBM-traffic and wall-clock scale anchor.
    - [x] `ml_workflow.qmd`: restore MobileNetV2 mechanism, lighthouse labels/thresholds, and validation target examples.
    - [x] `nn_computation.qmd`: restore BatchNorm synchronization, batch-size sensitivity, degradation range, and LayerNorm contrast.
  - [x] Meitner structure/emphasis findings integrated:
    - [x] `appendix_dam.qmd`: demote D.A.M axis body bold and restore sparse diagnostic scaffolding where useful.
    - [x] `model_compression.qmd`: rewrite body-prose bold paragraph starter for compound scaling.
    - [x] `fault_tolerance.qmd`: restore the transient/permanent/intermittent taxonomy as plain prose plus sparse structure.
    - [x] `ml_systems.qmd`: change noncanonical `Key Insight` callout label to `Systems insight`.
    - [x] `security_privacy.qmd`: restore defense-stack sequence in the callout where stepwise form teaches better than prose.
    - [x] `ops_scale.qmd`: restore planning-methodology checklist/sequence where it is pedagogically useful.
    - [x] `sustainable_ai.qmd`: restore grouped TinyML optimization bullets where scan structure carries the lesson.
    - [x] `benchmarking.qmd`: restore sparse LLM-metric taxonomy where dense prose obscures the metric roles.
  - [x] Arendt LEGO/MLSysIM findings integrated:
    - [x] `model_serving.qmd`: fix Stable Diffusion parameter/checkpoint-size source and formatter precision.
    - [x] `fault_tolerance.qmd`: move reusable checkpoint/model archetype and reliability scenario constants into MLSysIM or clearly mark one-off assumptions. Resolved in commit `96bdc62dfc`.
    - [x] `performance_engineering.qmd`: compute KV-cache memory and batch-size anchors in LEGO from MLSysIM inputs. Resolved in commit `5c5c616ffb`.
    - [x] `inference.qmd`: fix prose-style percent output comments and source mixed H100/A100 serving scenario values. Resolved in commit `7f0a9c745c`.
    - [x] `edge_intelligence.qmd`: source device-class range endpoints from MLSysIM and add references for empirical claims. Resolved in commit `c617e837ca`.
- [x] Header-depth pass: reviewed third-level and deeper one-paragraph headings through the initial agents plus residual Volume 1/Volume 2 agents; collapsed only headings that acted as labels rather than teaching structures, and preserved worked-example, taxonomy, benchmark, criterion, dashboard, and step headings where scan structure teaches.
- [x] Sparse bullet-list pass: kept bullets/lists only where they carry lookup, diagnostic, sequence, or worked-example structure better than prose; prior findings are integrated in `534f4edf77`, `829cea3b10`, `862d0868bd`, `fd3befbf21`, `b416926757`, `4d2ef26afc`, `7c68f82d35`, `b4ca459867`, and related preservation commits.
- [x] Fallacies/pitfalls convention pass: audited branch-touched `## Fallacies and Pitfalls` sections and restored the required `**Fallacy**:` / `**Pitfall**:` labels followed by italicized misconception statements where narrative edits collapsed them into ordinary prose. `.claude/rules/chapter-architecture.md` treats this as a chapter-architecture requirement and therefore as an explicit exception to the general body-prose emphasis caution. Resolved in commits `b45cc76e9c`, `c74219aaad`, `4b088faa46`, `b6667d8b8b`, `2a033880f4`, `28e998b133`, `1725efc211`, `26d5c6ec75`, and `2621dd1e2c`.
- [x] Re-audit changed examples/callouts and restore stepwise sequences when they teach a process better than prose; verified the GPT-2 data-pipeline example remains a staged trace and the heterogeneous routing example keeps a compact `Given` setup.
- [x] Emphasis pass: checked added/changed bold and italic use against `.claude/rules/emphasis.md`; branch-touched AP4 labels, triple bold-italic, qualitative bold, and bolded computed-result scans are clean.
  - [x] `vol1/ml_workflow.qmd`: normalized cloud/edge option labels inside the deployment-economics notebook so the colon sits outside the bold span and the labels read as parallel enumeration scaffolds. Resolved in commit `572cc95d5b`.
  - [x] `vol2/introduction.qmd`: demoted fallacy/pitfall statement text from triple bold-italic to italics, preserving only the allowed structural labels in bold. Resolved in commit `e3ccdaf73b`.
  - [x] `vol2/introduction.qmd`: earlier emphasis pass had rephrased fallacy/pitfall body labels into ordinary prose, but the later fallacies/pitfalls convention pass superseded that interpretation because `.claude/rules/chapter-architecture.md` explicitly requires `**Fallacy**:` / `**Pitfall**:` labels in these sections. Restored in commit `4b088faa46`.
  - [x] `vol2/responsible_ai.qmd`: added the required index co-location for the first-definition bold term `sociotechnical dynamics`. Resolved in commit `5ca3c11aa9`.
  - [x] `vol2/data_storage.qmd`: marked the synthetic-data P1 terms `Data Wall`, `provenance chain`, and `model collapse` as definition index entries. Resolved in commit `b700c443e3`.
  - [x] `vol2/robust_ai.qmd`: removed premature bold from a preview mention of spectral-signature defenses and kept the later sanitization paragraph as the first definition. Resolved in commit `3ba17a9831`.
  - [x] `vol1/data_engineering.qmd`: marked feeding-problem, flow-rate, and feeding-tax P1 terms as definition index entries. Resolved in commit `3cb502b9f9`.
  - [x] `vol2/network_fabrics.qmd`: marked SR-IOV and virtual functions as definition index entries at their first body definitions. Resolved in commit `b865cc8fdb`.
  - [x] `vol2/ops_scale.qmd`: demoted e-commerce example component names from body bold while preserving index entries. Resolved in commit `62ad8e5e24`.
  - [x] `vol1/backmatter/appendix_data.qmd`: removed AP4-style row/columnar body labels while preserving the paired storage-format contrast. Resolved in commit `a14d0eee77`.
  - [x] `vol1/backmatter/appendix_machine.qmd`: removed AP4-style GEMM/ReLU body labels while preserving the roofline comparison. Resolved in commit `41accc9dc2`.
  - [x] `vol2/security_privacy.qmd`: removed AP4-style `Example` body labels after the security/privacy definitions. Resolved in commit `c8de160a4f`.
- [x] Full `.claude/rules` compliance pass: rules reread is complete; branch-diff scans for colon-inside-bold, triple bold-italic outside definition callouts, qualitative bold, bolded computed prose outputs, learning-objective markup, and AP4 body labels are clean. `./book/binder check code --scope lego-units`, `./book/binder check registry --scope sources`, `python3 book/tools/audit/audit_mlsysim_drift.py`, `./book/binder check refs --scope inline`, `./book/binder check markup`, `./book/binder check prose`, and `./book/binder check footnotes` pass. Targeted guardrail tests pass with coverage disabled: `pytest --no-cov book/tests/test_footnote_caps.py book/tests/test_mlsysim_registry_coverage.py mlsysim/tests/test_hardware.py mlsysim/tests/test_ops_registry.py mlsysim/tests/test_system_registry.py mlsysim/tests/test_units_registry.py`.
  - [ ] Bibliography caveat: `./book/binder check bib --scope key-content` still fails on existing baseline key/year and institution-key issues in the two main reference files. The branch-touched new/changed entries do not introduce the hard key/year failures, but the global check is not green yet.
- [ ] Edited-paragraph flow pass: reread each changed section paragraph by paragraph, checking that the paragraph before sets up the current paragraph, the current paragraph advances the point, and the next paragraph follows naturally instead of reading like an isolated local edit.
  - [x] `vol1/introduction.qmd`: removed a redundant abstract opener before the five-pillar failure-chain narrative. Resolved in commit `a97ef73957`.
  - [x] `vol1/data_engineering.qmd`: folded the SATA caveat into the data-supply principle so it does not sit as a stranded note. Resolved in commit `ca2ae54fe0`.
  - [x] `vol1/training.qmd`: collapsed the thin mixed-precision roles H4 into the surrounding precision narrative. Resolved in commit `d1943e4262`.
  - [x] `vol2/fault_tolerance.qmd`: aligned the checkpoint coordination parent anchor with its visible heading to avoid duplicate distributed-checkpointing hierarchy. Resolved in commit `cfcd3fbb57`.
  - [x] `vol1/model_compression.qmd`: removed repeated distillation-vs-pruning limitations before the comparison table. Resolved in commit `79e141d6c4`.
  - [x] `vol1/benchmarking.qmd`: split failures/interference from reproducibility so the benchmark threats do not blur together. Resolved in commit `c8308f4026`.
  - [x] `vol1/hw_acceleration.qmd`: grouped the multi-chip scaling boundary taxonomy into package/node and datacenter/wafer-scale paragraphs. Resolved in commit `669cc696c9`.
  - [x] `vol1/model_serving.qmd`: moved the Little's Law concrete example after the equation and separated runtime selection from runtime configuration. Resolved in commit `5cbe9c271e`.
  - [x] `vol1/ml_ops.qmd`: separated continuous-retraining validation risk from strategy selection and removed repeated maturity-roadmap setup. Resolved in commit `bcbb1f9f35`.
  - [x] `vol1/responsible_engr.qmd`: split US sectoral examples from the cross-domain capability generalization. Resolved in commit `4b67959506`.
  - [x] `vol2/robust_ai.qmd`: retitled the adversarial-defense parent heading so detection, mitigation, and evaluation read as one workflow rather than as children of certified defenses. Resolved in commit `b7682a8a88`.
  - [x] `vol2/ops_scale.qmd`: collapsed the label-only feature monitoring H4 into the operational-integration narrative. Resolved in commit `f884c2561c`.
  - [x] `vol1/frameworks.qmd`: fused the single-node stream-overlap handoff so the synchronization lesson flows into correctness without a redundant distributed-training bridge.
  - [x] `vol1/hw_acceleration.qmd`: fused the precision-to-architectural-integration transition and kept the index entry on the term itself.
  - [x] `vol1/ml_ops.qmd`: fused the cost-aware automation close with the principles-summary setup.
  - [x] `vol1/training.qmd`: fused the `autocast` explanation with the hardware-dependent precision-policy transition.
  - [x] `vol1/responsible_engr.qmd`: fused the flaw-of-averages table takeaway with subgroup-selection guidance.
  - [x] `vol2/edge_intelligence.qmd`: fused the TinyTrain runtime motivation and grouped sparse-update tradeoffs into one developed paragraph.
  - [x] `vol2/ops_scale.qmd`: folded `Freshness tracking at scale` into operational integration after checking the surrounding Feature Store Operations section; point-in-time subheads remain because they carry a worked leakage example, a SQL listing, and a storage formula.
- [ ] Macro subsection-flow pass: after the edited-paragraph pass, inspect every changed chapter at the `##` and `###` level so the audit does not overfit to local paragraph repairs. Use prose-line counts excluding code, TikZ, math-heavy examples, and listing/table material as heuristics for spotting sections that may be too fragmented, too compressed, or still stitched together.
  - [ ] `ops_scale.qmd`: use the Feature Store Operations section as the first full-file diff audit for the pattern the user flagged: one local heading may be fixed while adjacent sibling subheads still act as label-only breaks. Review all branch changes in the file for this creep before moving on.
  - [x] Parallel agent wave 1 YAML read-only audits completed and closed: `ops_scale.qmd`, `fault_tolerance.qmd`, `inference.qmd`, `edge_intelligence.qmd`, `ml_ops.qmd`, and `training.qmd`.
  - [ ] Wave 1 triage queue:
    - [x] `ops_scale.qmd`: debt taxonomy, organizational practices, model-type operations diversity, platform-team justification, shadow deployment/traffic replay, edge fleet, CI/CD patterns by model type, nonlinear effects, and point-in-time leakage H4. Resolved in commit `517d071d22` after converting label-only subsections into measurement maps, comparative prose, and control tables while preserving concrete thresholds, cycle times, and failure examples.
    - [x] `fault_tolerance.qmd`: data-debugging paragraph scoped under numerical debugging, model-specific serving fault-tolerance catalog, case-study synthesis scoped under DeepSpeed, centralized-checkpointing fragments, and degradation-monitoring inventory. Resolved in commit `1ebb6a6465` by reframing centralized checkpointing as a bottleneck argument, adding serving/degradation action maps, moving data debugging to its own scope, and giving the case-study synthesis its own heading.
    - [x] `inference.qmd`: Purpose one-paragraph rule, serving hierarchy taxonomy flow, and sibling-option balance in scheduling/topology. Resolved in commit `ae28ef1c02` by restoring the one-paragraph Purpose convention and turning the serving hierarchy from four standalone level notes into one level-to-level argument.
    - [x] `edge_intelligence.qmd`: duplicated motivations/benefits H3s, alternative-approaches scope drift, structured-update H4 imbalance, bio-inspired learning placement, and engineering-challenges checklist interruption. Resolved by removing the duplicate benefits heading, giving knowledge transfer its own subsection, retitling the bio-inspired scope as data-efficient continual adaptation, and converting the late integration list into an engineering control table.
    - [x] `ml_ops.qmd`: production-debt hierarchy/anchor mismatch, A/B testing guidance compressed transition, and case-study lesson structure after Oura/ClinAIOps. Resolved by retargeting the production-debt anchor and updating the downstream cross-reference, turning the debt cases into two causal pairs, clarifying the A/B setup-to-decision flow, and adding a case-study synthesis before fallacies.
    - [x] `training.qmd`: lone `Training loop` H4 under `Architectural overview`. Resolved by removing the unreferenced H4 so the loop explanation remains part of the architectural overview rather than a one-child substructure.
  - [x] Wave 2 YAML audit findings received for `frameworks.qmd`, `hw_acceleration.qmd`, `model_serving.qmd`, `model_compression.qmd`, `benchmarking.qmd`, and `responsible_engr.qmd`; triage after the active `ops_scale.qmd` task commit.
  - [ ] Wave 2 triage queue:
    - [x] `frameworks.qmd`: autograd extensibility reads like an API tour, synchronization H5/H6 labels are shallow, core-operations order does not match the promised abstraction sequence, JAX H4 scope drift, TensorFlow selection criteria imbalance, training-step phase imbalance, and fallacies/pitfalls checklist accumulation. Resolved by removing shallow synchronization H6 labels, promoting parameter structures and distributed execution contexts out from under data loading, aligning the core-operations prose/caption with the actual section order, and collapsing the unreferenced JAX one-child H4 into the JAX profile. Existing autograd examples and fallacy/pitfall labels were preserved because they remain pedagogically useful and rule-compliant.
    - [ ] `hw_acceleration.qmd`: combinatorial-complexity hierarchy is unbalanced, runtime-support bridge needs a serving anchor, and multi-chip scaling should preserve package/node/cluster/wafer sequence.
    - [ ] `model_serving.qmd`: Little's Law scope drift, multi-server stitched prose, tail-tolerant technique taxonomy over-compressed, runtime ecosystem/configuration orphaning, profiling diagnostic compression, and generic node-to-factory bridge.
    - [ ] `model_compression.qmd`: repeated quantization-energy thread, PTQ/QAT label-only H5s, dynamic schemes catalog drift, sparsity utilization scope drift, late selection/comparison imbalance, and diagnostics misplaced under hardware-specific libraries.
    - [ ] `benchmarking.qmd`: training evaluation promised reproducibility but under-delivers, inference evaluation setup/sequence mismatch, MLPerf synthesis under narrow heading, and LLM table handoff weakened.
    - [ ] `responsible_engr.qmd`: fairness-accuracy bridge arrives too early, regulatory H4s remain law catalog, and ethical deployment checkpoint/final bridge is scoped too narrowly under US sectoral regulation.
- [ ] Whole-narrative flow pass: after local edits, reread each changed chapter as a chapter-level argument, not as a list of paragraphs. For every edited section, check the paragraph-before/current-paragraph/paragraph-after transition; for every chapter, check that the opening problem, mechanisms, examples, fallacies, summary, and chapter connection form one continuous textbook narrative for a learner moving through the book.
- [ ] Engineering-chapter identity audit: after macro-flow cleanup and local rule/source checks, launch parallel chapter agents with a standard rubric asking whether each chapter reads like an ML systems engineering chapter rather than a policy/category survey. The agents should answer yes/no with evidence and propose targeted improvements only where needed.
  - Rubric: Does the chapter ground categories in systems constraints, failure modes, measurements, implementation mechanisms, back-of-the-envelope math, concrete hardware/model/service examples, or LEGO/MLSysIM-backed quantities where those help? Does it avoid adding numbers merely for texture? Does each engineering anchor advance the chapter narrative rather than narrowing the chapter into an isolated anecdote?
  - Scope: Run this especially on operational/responsibility/governance chapters, but include all edited Volume 1 and Volume 2 chapters so the standard is consistent across the book.
  - Placement: Do this before the ordered progressive-disclosure audit and before the final Tokenland litmus, because any engineering-anchor repairs may change what later chapters can assume.
- [x] Retail-example continuity check: verified the foundational-principles callback in `ml_ops.qmd` is introduced by the preceding retail-drift scenario; the later fallacy example is self-contained and does not depend on that callback.
- [x] LEGO output-name pass: check all changed `_str` exports for conventions such as `_pct_str`, `_pp_str`, `_gb_str`, `_gb_per_s_str`, `_qps_str`, and scenario-specific names.
  - [x] `vol2/edge_intelligence.qmd`: normalized MIPS outputs to `fmt_rate(..., "MIPS")` so the rate formatter owns the rendered label. Resolved in commit `5deefce7e2`.
  - [x] `vol2/fault_tolerance.qmd`: renamed the branch-added checkpoint-table MTBF export from `mtbf_h_str` to `mtbf_hr_str` and updated the table reference. Resolved in commit `4ef0083a5a`.
  - [x] `vol1/introduction.qmd`: replaced bare million-scale accelerator-day exports with `fmt_count(..., scale="M", scale_style="word", label="accelerator-day")` and updated prose/table references. Resolved in commit `6c5f8ea356`.
  - [x] `vol1/ml_ops.qmd`: renamed the Oura study recording-duration export from `recording_hours_str` to `recording_hr_str` while preserving the closed `fmt_time(..., hour)` output. Resolved in commit `bea21aa250`.
  - [x] `vol2/edge_intelligence.qmd`: normalized branch-added TOPS outputs to `fmt_ops_rate(..., unit=TOPS)` so operation throughput uses the domain formatter. Resolved in commit `2d4555d25f`.
  - [x] `vol1/backmatter/appendix_machine.qmd`: changed branch-touched fixed-unit `fmt_qty` calls to explicit `unit=` arguments. Resolved in commit `da68472dfd`.
  - [x] `vol1/data_engineering.qmd`: moved the storage-bandwidth example's bandwidth and FLOP-rate outputs to typed domain formatters and explicit units. Resolved in commit `2ef1d7651c`.
  - [x] `vol1/introduction.qmd`: moved the AI Moment TFLOP/s outputs to `fmt_flop_rate(..., unit=TFLOPs / second)`. Resolved in commit `cbe980181b`.
  - [x] `vol1/ml_systems.qmd`: moved the ResNet cloud compute and memory-bandwidth outputs to `fmt_flop_rate` and `fmt_bandwidth` with explicit units. Resolved in commit `23347a8d5d`.
  - [x] `vol1/model_serving.qmd`: changed Stable Diffusion load-time `fmt_time` calls to explicit `unit=second` while preserving MLSysIM-backed checkpoint/load anchors. Resolved in commit `cf1aca2321`.
  - [x] `vol2/fault_tolerance.qmd`: changed checkpoint-overhead table `fmt_time` calls to explicit `unit=` arguments. Resolved in commit `d7e05fb677`.
  - [x] `vol1/ml_ops.qmd`: changed the Oura recording-duration `fmt_time` call to explicit `unit=hour`. Resolved in commit `77e14ba7af`.
  - [x] `vol1/ml_systems.qmd`: sourced compound TPU v4 Pod and DGX Spark compute-envelope strings from MLSysIM-backed values, and moved adjacent fixed-unit threshold outputs to typed formatters with explicit scenario-assumption inputs. Resolved in commit `942f0abc2f`.
- [x] MLSysIM source-of-truth pass: replace scenario-only constants with MLSysIM-backed models/devices when a reusable concept exists, including diffusion-model checkpoint size, variable edge-device/ring scenarios, and the fault-tolerance memory-bandwidth protection table.
  - [x] Stable Diffusion loading examples use `Models.GenerativeVision.StableDiffusion_v1_5`, `ReferenceStats.ModelLoading`, and `Systems.Storage.LocalNvmeGen3` rather than local model-size/load-time constants.
  - [x] Oura Ring validation and operational-accuracy examples use `ReferenceStats.OuraSleepStudy`.
  - [x] Fault-tolerance checkpoint archetypes use `ReferenceStats.CheckpointArchetypes`.
  - [x] Edge device-spectrum examples use `ReferenceStats.EdgeDeviceSpectrum`; the variable wearable/phone/voice-assistant tier profile now uses `ReferenceStats.EdgeAdaptationTierProfile`. Resolved in commit `513621ffdd`.
  - [x] Registry/source checks passed: `./book/binder check code --scope lego-units`, `./book/binder check registry --scope sources`, `python3 book/tools/audit/audit_mlsysim_drift.py`, and pre-commit for the changed edge/source files.
- [ ] Numeric-anchor pass: preserve concrete model, hardware, and threshold numbers when they teach scale better than abstract labels, and restore them where over-generalization made examples less useful, including the cloud/edge compute-envelope edits in `ml_systems.qmd` (initial `ml_systems.qmd` restore committed; full all-diff pass still pending).
  - [x] `vol2/edge_intelligence.qmd`: moved wearable, keyboard-memory, phone-tier, voice-assistant fleet, replay-buffer, LoRA rank, and LoRA update anchors into MLSysIM-backed outputs while preserving the concrete numbers. Resolved in commit `513621ffdd`.
  - [x] `vol1/data_engineering.qmd`: restored A100/A100-peak feeding-problem and H100 serialization-pitfall anchors through typed LEGO outputs. Resolved in commit `6fe29435cd`.
  - [x] `vol1/ml_systems.qmd`: restored the H100 NVLink hardware-layer bandwidth anchor through a typed LEGO output. Resolved in commit `72af3ce299`.
  - [x] `vol1/data_engineering.qmd`: restored DLRM, ResNet, GPT, and A100 anchors that had become generic while preserving the improved explanation. Resolved in commit `5a7b3ff40b`.
  - [x] `vol2/compute_infrastructure.qmd`: restored TPU v5p, TB-scale embedding-table, Stable Diffusion denoising-step, Gaudi RoCE-bandwidth, and advanced-packaging anchors while sourcing reusable Gaudi bandwidth from MLSysIM. Resolved in commit `aaf11454bb`.
  - [x] `vol2/fault_tolerance.qmd`: restored the ViT-Large checkpoint/serving-cascade anchor where the vision example had become generic. Resolved in commit `aa7243b0de`.
  - [x] `vol2/inference.qmd`: restored circuit-breaker 50 percent and 30-second recovery anchors through a reusable MLSysIM serving profile. Resolved in commit `b9fb7b4c4d`.
  - [x] `vol2/ops_scale.qmd`: restored 10,000-token RAG context cost, 1 percent sampling, and 500-feature pipeline monitoring anchors while preserving narrative flow. Resolved in commit `215decfbc6`.
  - [x] `vol2/security_privacy.qmd`: restored model-extraction quota ladder, anomaly-score rate-reduction arithmetic, and extraction-pricing break-even through local LEGO outputs. Resolved in commit `0ba1354450`.
  - [x] `vol2/sustainable_ai.qmd`: restored TinyML SRAM, harvested-energy, BNN energy/accuracy, pruning/quantization, and MCUNet 256 KB anchors inside the narrative sequence. Resolved in commit `6516554438`.
  - [x] `vol1/backmatter/appendix_dam.qmd`: restored the A100-to-H100 compute-wall scale-up anchor inside the D.A.M. case-study narrative. Resolved in commit `2c8b58b01d`.
  - [x] `vol1/data_engineering.qmd`: restored the KWS INT8 inference choice plus 8$\times$ A100 ResNet-50 timing/cost anchors. Resolved in commit `ab4c5de2bf`.
  - [x] `vol1/frameworks.qmd`: restored the FP32-to-INT8 memory and 2--4$\times$ throughput precision anchor. Resolved in commit `5aba925d33`.
  - [x] `vol1/introduction.qmd`: restored A100-class GPT-4 scale wording and the checked ResNet-50 batch-one A100 roofline anchor with typed LEGO outputs. Resolved in commit `b18bde6883`.
  - [x] `vol1/ml_systems.qmd`: restored A100, TPU v4, GB10 Grace Blackwell, PFLOP/s--EFLOP/s, A100 energy, TinyML quantization, and NPU throughput anchors. Resolved in commit `3bd8f2af11`.
  - [x] `vol1/ml_workflow.qmd`: restored NVIDIA Jetson-class deployment, ResNet-50/DLRM/TinyML archetype, and PSI/KS threshold anchors. Resolved in commit `10ce7a9641`.
  - [x] `vol1/nn_architectures.qmd`: restored INT8/INT4 quantization and concrete tile-size/Tensor Core hardware anchors. Resolved in commit `d799dae645`.
  - [x] `vol1/nn_computation.qmd`: restored H100 MAC, TPU v1 INT8/TDP, parameter-memory multiplier, A100/H100/Edge TPU/Jetson power-tier, INT8 quantization, and A100/H100 ridge-point anchors. Resolved in commit `9d16756d57`.
- [ ] Reference-needs pass on edited sections; if new sources are needed, stage in a dedicated `.bib`, run BetterBib, then copy to the main references file.
- [ ] Precommit/rule pass for footnote capitalization consistency when definition terms appear.
- [ ] Ordered concept-map audit: read Volume 1 and Volume 2 as separate books in chapter order. Volume 2 may assume Volume 1 knowledge and the appendices, but should not assume concepts introduced later in Volume 2.
- [ ] Cross-chapter progressive-flow pass: read Volume 1 and Volume 2 independently in book order, treating each chapter as allowed to assume only prior chapters in that volume. For later chapters, verify that language naturally builds on already disclosed concepts rather than importing future vocabulary or unearned abstractions.
- [ ] Parallel-agent flow audit: use independent chapter-level agents where they help evaluate paragraph-to-paragraph flow, chapter argument, and progressive disclosure. The master thread remains responsible for consolidating findings, respecting `.claude/rules`, and making edits.
- [ ] Final litmus test: reread the Systems Approach "Textbooks in Tokenland" post and check the revised chapters against that lens.
- [ ] Last pre-build path, after all narrative/rules/progressive-flow work: read `.claude/_reviews/algorithm_pass/ALGORITHM_PASS.md`, launch parallel agents over the guidance, and decide which code listings should be converted into algorithm presentation. For accepted conversions, update the algorithm text and add proper algorithmic citation/reference keys through the dedicated-bib/BetterBib workflow so the algorithm blends with the text rather than reading like an implementation listing.
- [ ] Dedicated index-term placement audit after the algorithm pass, likely in a separate worktree: inspect changed and surrounding prose to verify `\index{...}` entries land on the right first-use or concept-use location rather than drifting after paragraph fusions, heading collapses, or algorithm/listing rewrites.

## Executive Summary

Both volumes mostly have a textbook spine. The strongest chapters start from a constraint or failure mode, then derive mechanisms. The weaker passages usually occur in the middle of chapters, where a good framing paragraph gives way to a survey of techniques, vendors, metrics, laws, defenses, or operational practices.

The highest-priority rewrite target is Volume 2 `ops_scale.qmd`, especially the platform-scale middle and late sections. The second highest-priority target is Volume 2 `fault_tolerance.qmd`, where several sections drift into generic reliability, software engineering, or observability catalogs. In Volume 1, the most visible issues are `ml_systems.qmd`, `data_engineering.qmd`, `model_compression.qmd`, `training.qmd`, and later `ml_ops.qmd` case-study/principle mappings.

## Coverage Confirmation

All ten audit slices completed and were consolidated here:

- Volume 1 Part I: foundations opener, introduction, ML systems, ML workflow, data engineering.
- Volume 1 Part II: build opener, neural-network computation, neural-network architectures, frameworks, training.
- Volume 1 Part III: optimization opener, data selection, model compression, hardware acceleration, benchmarking.
- Volume 1 Part IV: deployment opener, model serving, MLOps, responsible engineering, conclusion.
- Volume 1 appendices: D.A.M, data, algorithm, machine, assumptions.
- Volume 2 Part I: fleet opener, introduction, compute infrastructure, network fabrics, data storage.
- Volume 2 Part II: distributed ML opener, distributed training, collective communication, fault tolerance, fleet orchestration.
- Volume 2 Part III: deployment opener, performance engineering, inference, edge intelligence, ops at scale.
- Volume 2 Part IV: responsible fleet opener, security/privacy, robust AI, sustainable AI, responsible AI, conclusion.
- Volume 2 appendices: D.A.M, C3, fleet, communication, reliability, inference, assumptions.

Frontmatter, references, and glossaries were intentionally out of scope for the narrative/laundry-list audit because they are not chapter narrative prose. No audit slice was left pending.

## Highest Priority Findings

### Volume 2: Ops at Scale

1. `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:4740`
   `Feature versioning and lineage` through freshness tracking reads stitched together: versioning, lineage, backfills, quality, validation, incident response, monitoring, and freshness.
   Rewrite around one feature failure lifecycle, such as a schema change causing training-serving skew, then organize mechanisms as invariants: temporal correctness, semantic stability, freshness, quality gates, and recovery.

2. `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:4915`
   `Organizational patterns` leans into generic centralized/embedded/hybrid pros, cons, and selection matrices.
   Recast around the systems boundary: when shared platform invariants outweigh team autonomy. Use one evolving organization as the through-line, then keep the comparison table as reference.

3. `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:3731`
   `Abstraction levels` presents four levels as a catalog.
   Rewrite around the abstraction frontier: what operators stop letting teams configure directly as model count, risk, and cost grow. The levels then become consequences of scaling pressure.

4. `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:1668`
   `Infrastructure planning methodology` opens with consecutive checklists before a concrete planning problem.
   Move the 175B planning example earlier and let it drive the section: model size plus deadline implies accelerator count, then network, power, facility, cost, and schedule.

5. `book/quarto/contents/vol2/ops_scale/ops_scale.qmd:5241`
   `Runbook development` and `Post-incident review` interrupt the outage story with raw operational templates.
   Rewrite as an incident narrative from detection to attribution to mitigation to review, explaining why each field matters. Keep templates as reference material after the narrative.

### Volume 2: Fault Tolerance

1. `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:1394`
   `Software fault detection and prevention` starts from a good ML-specific corruption premise, then turns into a generic checklist: unit tests, linting, CI/CD, containers, regression diagrams.
   Rewrite around one concrete corruption path, such as preprocessing truncating prompts or a data-loader race corrupting batches, then show which lifecycle gates catch it.

2. `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:925`
   `Hardware Fault Taxonomy` broadens into a hardware reliability encyclopedia.
   Recast as an operational decision tree: did the fault crash, silently corrupt, or recur under load? Carry two or three ML-specific failures through detection, mitigation, and recovery cost.

3. `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:3405`
   `Graceful Degradation` follows a strong outage scenario with adjacent lists of dimensions, fallback types, triggers, metrics, and thresholds.
   Turn the e-commerce outage into the organizing case: define the SLO and quality budget, then walk through model fallback, feature fallback, load shedding, monitoring, and recovery hysteresis.

4. `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:3141`
   `Model-specific training fault tolerance` and `model-specific serving fault tolerance` read like mini-catalogs of LLMs, recommendation, vision, and scientific workloads.
   Replace the mini-survey with a comparison organized by the state variable at risk: optimizer state, curriculum state, embedding freshness, augmentation/random state, simulator/search state, KV cache, or feature freshness.

5. `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd:3564`
   `Observability pillars` reads like observability 101.
   Lead with the cascade diagnosis case, then introduce metrics, traces, and logs as evidence types that isolate the root cause. Trim generic log-level and tracing definitions or make every example ML-specific.

## Volume 2 Chapter Findings

### Introduction and Fleet

1. `book/quarto/contents/vol2/introduction/introduction.qmd:1716`
   `Foundational Concepts` stacks too many organizing frames: Fleet Stack, roadmap, AI Triad, Five-Pillar Framework, Rosetta Stone, six systems principles, and archetypes.
   Choose one dominant spine, likely Fleet Stack plus C^3. Demote Five-Pillar/Rosetta/six-principles material to a compact reference map or sidebar. Use archetypes as recurring cases, not another concept list.

2. `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:1828`
   `Accelerator selection for ML workloads` moves through LLM training, inference, recommendation, vision/diffusion, MoE, and multimodal workloads as a taxonomy.
   Open with a concrete selection problem, use roofline-style reasoning to identify the binding constraint, then walk one or two archetypes in prose. Put remaining variants in a compact table.

3. `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:2521`
   `Alternative node architectures` reads like a vendor-by-vendor tour.
   Organize around design choices before vendors: crossbar vs mesh, HBM-rich package vs balanced node, integrated NIC vs external fabric. Use vendor systems as examples of those choices.

4. `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd:3993`
   `Emerging Infrastructure Technologies` risks becoming a future-tech list.
   Make each technology answer the same question: which wall does it move, what new wall appears, and what fleet-design decision changes?

Network and storage chapters were generally strong. Their lists and tables mostly serve mental models or reference needs.

### Deployment at Scale

`performance_engineering.qmd`, `inference.qmd`, and `edge_intelligence.qmd` mostly passed the narrative test. The main issue in this slice is `ops_scale.qmd`, covered above.

Lower-priority note:

- `book/quarto/contents/vol2/inference/inference.qmd:4368`
  `Circuit breakers and backpressure` is useful, but state-machine snippets and transition lists briefly dominate. A short overload narrative before the mechanics would make the pattern feel less pasted in.

### Responsible Fleet

1. `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd:3637`
   `Lifecycle-aware development methodologies` through `MLPerf sustainability benchmarks` flattens into a survey of pruning, quantization, distillation, TinyML methods, BNNs, MCUNet, Once-for-All, ProxylessNAS, and benchmark metrics.
   Rewrite around the measured bottleneck: memory movement, serving volume, edge battery, grid carbon, or lifecycle carbon. Keep named methods in compact reference boxes.

2. `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1788`
   `Hardware-Level Security Vulnerabilities` lists hardware bugs, physical attacks, fault injection, side channels, leaky interfaces, counterfeit hardware, and supply chain risks.
   Rewrite as a deployment decision model: processor isolation, physical access, side-channel observability, and provenance.

3. `book/quarto/contents/vol2/robust_ai/robust_ai.qmd:1258`
   `Attack categories and mechanisms` names FGSM, PGD, JSMA, C&W, EAD, transfer attacks, and physical attacks in taxonomic order.
   Rewrite around attacker access and cost: gradient access, query-only access, surrogate transfer, and physical sensor manipulation.

4. `book/quarto/contents/vol2/robust_ai/robust_ai.qmd:2187`
   `Data Poisoning Defenses` becomes a sequence of method families.
   Rewrite as a supply-chain control story: data ingress, representation audit, training objective, provenance, and post-training validation.

5. `book/quarto/contents/vol2/security_privacy/security_privacy.qmd:1208`
   `Defenses against model extraction` is close to an implementation checklist.
   Rewrite as an API information-budget argument: each query leaks bits, so defenses reduce bits per query, reduce query volume, raise attack cost, or detect systematic probing.

## Volume 1 Chapter Findings

### Foundations

1. `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:1439`
   Cloud, edge, mobile, and TinyML decomposition figures repeatedly use the same buckets: characteristics, benefits, challenges, examples.
   Rewrite around binding constraint -> architectural consequence -> failure mode if ignored, using latency, bandwidth, battery, and energy examples as the spine.

2. `book/quarto/contents/vol1/ml_systems/ml_systems.qmd:644`
   `Workload archetypes` and lighthouse workloads are introduced as a roster, and the fourth archetype is separated by intervening material.
   Derive the archetypes directly from iron-law terms, keep all four together, then use one worked diagnosis before relegating the full roster to reference support.

3. `book/quarto/contents/vol1/introduction/introduction.qmd:3241`
   `The five engineering disciplines` becomes five consecutive pillar descriptions.
   Use one production failure chain that crosses data, training, deployment, operations, and governance, then introduce the five pillars as ownership boundaries revealed by that chain.

4. `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:510`
   `Four foundational pillars` settles into quality, reliability, scalability, and governance definitions.
   Drive the section through one concrete KWS data decision and show how that single decision stresses all four pillars.

5. `book/quarto/contents/vol1/data_engineering/data_engineering.qmd:1410`
   `Data Acquisition` reads as a tour of options: existing datasets, scraping, crowdsourcing, synthetic data.
   Reframe around a coverage gap or distribution mismatch, then show each acquisition strategy as an attempted solution with a specific tradeoff.

### Development

1. `book/quarto/contents/vol1/nn_computation/nn_computation.qmd:5092`
   `Memory and computational resources` becomes nested resource inventory.
   Rewrite around the core idea: inference has persistent weights plus a rolling activation buffer. Keep totals in prose, move layer-by-layer counts into a compact table or parenthetical, and connect to batching, reuse, and quantization.

2. `book/quarto/contents/vol1/training/training.qmd:2541`
   `GPT-2 language model data pipeline` reads like a stitched pipeline checklist.
   Rewrite as a bottleneck story: can CPU tokenization keep the GPU fed? Follow one batch through timed gates and keep only details that change the bottleneck.

3. `book/quarto/contents/vol1/training/training.qmd:5554`
   `GPT-2 optimization on V100` collapses into headings, bullets, profile snippets, and a final table.
   Rewrite as a diagnostic loop: what failed, what measurement revealed it, what intervention changed, and what bottleneck appeared next.

### Optimization

1. `book/quarto/contents/vol1/model_compression/model_compression.qmd:5389`
   `Adaptive computation methods` is the clearest list-like passage in this slice.
   Frame it as an adaptive compute control loop: measure confidence or context, route work, pay routing overhead, then preserve batching/hardware utilization. Use one running example and put the taxonomy in a compact table.

2. `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd:4413`
   `Hardware Mapping` stacks mapping aspects, placement, memory allocation, and combinatorial complexity before intuition.
   Start with the convolution loop-ordering/data-movement example, show why one mapping is better, then derive placement, allocation, and scheduling as consequences.

3. `book/quarto/contents/vol1/model_compression/model_compression.qmd:5073`
   `Architectural Efficiency` drifts toward a survey of complementary approaches.
   Anchor in a profiling trace where theoretical FLOP reduction fails to produce wall-clock speedup, then introduce techniques as fixes for specific bottlenecks.

4. `book/quarto/contents/vol1/benchmarking/benchmarking.qmd:4593`
   `Large language model benchmarks` reads like a benchmark/metric catalog.
   Organize around one production failure: a model scores well on a public benchmark but fails deployment because generation cost, calibration/safety, or contamination was not measured.

### Deployment

1. `book/quarto/contents/vol1/model_serving/model_serving.qmd:4270`
   `Node-Level Optimization` becomes a catalog of serving-node techniques.
   Recast around a bottleneck-driven workflow using one profile trace: GPU idle gaps, tiny kernels, cold starts, CPU-bound small models, and the interventions each symptom implies.

2. `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:3562`
   `Case Studies` through Oura/ClinAIOps principle summaries repeat the same five principles in a cross-case table and again inside each case.
   Keep one summary table, then make each case explain its governing constraint. Oura: energy, privacy, weak ground truth, telemetry, OTA. ClinAIOps: clinical accountability, monitoring, feedback, governance.

3. `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:154`
   `Foundational principles` briefly feels like a framework inventory.
   Let the retail drift scenario carry the section. Introduce reproducibility, separation, consistency, observable degradation, and cost-aware automation as successive controls needed to diagnose and prevent the same failure.

4. `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd:1417`
   `The regulatory landscape` reads like a compact compliance survey.
   Anchor in one concrete obligation, such as a denied loan applicant requesting explanation or contestability. Use laws as forces that require audit logs, human review paths, data lineage, deletion capability, and incident response.

## Appendix Findings

Appendices can legitimately be more reference-like. The findings below are lower priority unless the goal is to make appendices read more like mini-lessons.

### Volume 1 Appendices

1. `book/quarto/contents/vol1/backmatter/appendix_dam.qmd:173`
   `D.A.M Case Studies` uses repeated `Symptom / Diagnosis / The fix` worksheets.
   Rewrite each as a short incident narrative: observed metric, false lead, dominant iron-law term, intervention, and why it changes that term.

2. `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:520`
   Amdahl/Gustafson examples enumerate calculations without building the fixed-work vs scaled-work model.
   Rewrite as paired scenarios: fixed training step with 5 percent serial overhead versus using added hardware to train on more data.

3. `book/quarto/contents/vol1/backmatter/appendix_data.qmd:350`
   `The algebra of data` reads like a SQL glossary.
   Rewrite around one feature pipeline: filter rows, project columns, join labels, and show what bytes are skipped or shuffled.

4. `book/quarto/contents/vol1/backmatter/appendix_machine.qmd:1122`
   `Bandwidth vs. latency` uses nested bullets for arithmetic.
   Rewrite as a compare/contrast paragraph and add the durable crossover model where `D_vol / BW` equals latency.

5. `book/quarto/contents/vol1/backmatter/appendix_data.qmd:653`
   `Information theory for systems` stitches definitions of entropy, information density, and SNR.
   Start from the operational question: why can more data or compute stop helping?

### Volume 2 Appendices

1. `book/quarto/contents/vol2/backmatter/appendix_fleet.qmd:58`
   `Foundations recap` through `Numbers Every Fleet Engineer Should Know` stacks recap bullets, quick-reference tables, invariants, and number tables.
   Lead with a concrete fleet design question: why does a 1,024-GPU plan not deliver 1,024 GPUs of useful work?

2. `book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd:673`
   `Capacity Planning Assumptions` presents MFU, scaling efficiency, and overhead budgets as separate buckets.
   Open with the effective-throughput equation, then walk one cluster example before the tables.

3. `book/quarto/contents/vol2/backmatter/appendix_communication.qmd:248`
   AllGather, ReduceScatter, AllToAll, and Broadcast repeat a definition/formula/implication pattern.
   Group them by communication contract: replicate state, partition reduced state, personalized exchange, and root distribution.

4. `book/quarto/contents/vol2/backmatter/appendix_assumptions.qmd:504`
   `Sustainability Assumptions` moves through PUE, WUE, carbon intensity, and rack density as separate categories.
   Frame around one 10 MW or 10,000-GPU facility scenario.

5. `book/quarto/contents/vol2/backmatter/appendix_inference.qmd:311`
   `Decision framework: Batch size selection given SLA` is useful but checklist-like.
   Recast as a decision tree: rule out instability, choose between utilization headroom and latency, then validate tail latency.

## Strong Narrative Anchors

These sections passed the audit especially well and can serve as local models for rewrites:

- `book/quarto/contents/vol1/parts/foundations_principles.qmd:3`
  Compact perspective-setter; gives a mental model rather than a catalog.
- `book/quarto/contents/vol1/ml_workflow/ml_workflow.qmd:50`
  Rural clinic / diabetic retinopathy thread motivates workflow mechanisms through a real constraint.
- `book/quarto/contents/vol1/parts/build_principles.qmd:3`
  Iron law and silicon contract frame the part as perspective, not topic list.
- `book/quarto/contents/vol1/nn_architectures/nn_architectures.qmd:2382`
  Turns RNN limitations into the motivation for attention.
- `book/quarto/contents/vol1/parts/optimize_principles.qmd:3`
  Pareto and D.A.M framing does real explanatory work before mechanisms appear.
- `book/quarto/contents/vol1/data_selection/data_selection.qmd:28`
  The data wall gives the chapter a problem-driven throughline.
- `book/quarto/contents/vol1/parts/deploy_principles.qmd:3`
  Deployment is framed as the place where correct code can still fail as a system.
- `book/quarto/contents/vol1/ml_ops/ml_ops.qmd:27`
  "Perfectly available and perfectly wrong" gives the chapter a real problem before mechanisms.
- `book/quarto/contents/vol2/parts/fleet_principles.qmd:1`
  Invariants build from physical limits toward system-level consequences.
- `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd:233`
  Motivates the ML networking inversion before mechanisms.
- `book/quarto/contents/vol2/parts/distributed_ml_principles.qmd:3`
  Strong part-level framing around communication tax, reliability tax, and overhead conservation.
- `book/quarto/contents/vol2/collective_communication/collective_communication.qmd:835`
  Good problem-to-mechanism progression from naive bottlenecks to lower bounds and algorithm choice.
- `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd:79`
  Memory wall, iron law, and efficiency frontier give readers a durable model before mechanisms.
- `book/quarto/contents/vol2/parts/responsible_fleet_principles.qmd:3`
  Strong part opener; names invariants and connects chapters without becoming a glossary.
- `book/quarto/contents/vol2/robust_ai/robust_ai.qmd:27`
  Silent-failure model motivates robustness before mechanisms.
- `book/quarto/contents/vol2/backmatter/appendix_reliability.qmd:161`
  Turns rare individual failures into continuous fleet failure, then motivates checkpointing and recovery.

## Rewrite Pattern

For most flagged sections, the repair pattern is consistent:

1. Start with one concrete system question or failure.
2. Identify the binding constraint or state variable at risk.
3. Walk through the causal chain.
4. Introduce mechanisms only as answers to that chain.
5. Keep tables, lists, and taxonomies as summaries after the narrative rather than as the narrative.

## Rules-Pass Verification Log

- Footnote capitalization precommit TODO: satisfied. The footnote group in
  `.pre-commit-config.yaml` dispatches through `./book/binder check footnotes`;
  `book/cli/commands/validate.py` includes `capitalization` as a default footnote
  scope; `./book/binder check footnotes` passes. The direct checker also reports
  `OK: footnote capitalization checks passed.` Unit assertions pass with
  `pytest -q --no-cov book/tests/test_footnote_caps.py`; the plain pytest
  command is blocked only by the repository-wide coverage gate.
- Bibliography workflow/rules pass: `./book/binder check bib`,
  `./book/binder check refs`, and `python3 book/tools/scripts/check_bib_qmd_integrity.py`
  all pass for the branch state. A read-only BetterBib scan of the main
  bibliography files reports broad legacy/project-style findings, so no broad
  BetterBib cleanup was applied. One actionable staging-workflow issue was fixed:
  the EU AI Act entry is now a standard `@misc` in both volume bibliographies
  with a corporate author and official EUR-Lex URL, committed as `2c5abb8a1c`.
- Emphasis/prose cleanup follow-up: removed non-structural inline bold from the
  fleet orchestration "limping switch" callout (`de5951e0a4`) and tightened a
  vague leverage construction in the inference summary (`3f3b8a0d46`). Both
  files passed their file-level pre-commit checks before commit.
- Rules-pass prose cleanup follow-up: tightened branch-added wording in
  `data_selection` (`8a0e688cf5`), `model_compression` (`227e25c11b`), and
  `ops_scale` (`2da1b955d3`). The edits removed a vague "highest leverage first"
  phrase, replaced section-meta prose with causal prose, and normalized a
  leakage example to structural bold labels. Each chapter passed its file-level
  pre-commit checks before commit.
- Visible "leverage" wording follow-up: changed the `ml_workflow` figure title
  and alt text from "Workflow Automation Leverage" / "leverage effect" to
  "Workflow Automation Returns" / "super-linear returns" while leaving the
  original internal cross-reference id stable (`059a5742d5`). A follow-up renamed
  the internal figure and chunk ids to `returns` as well (`dc5691646f`), after
  reference/orphan checks passed in file-level pre-commit.
- Contraction cleanup: expanded a non-rendered LEGO docstring contraction in the
  Volume 1 introduction (`eea0137861`). File-level pre-commit passed.
- Source-of-truth/formatting cleanup: converted the Volume 1 `ml_systems`
  deployment-threshold table from raw unit-bearing `MarkdownStr` literals to
  typed Pint quantities formatted with `fmt_flop_rate`, `fmt_bandwidth`,
  `fmt_ops_rate`, and `fmt_power` (`6511b9865c`). The values remain scenario
  assumptions, now labeled as such in LOAD. File-level pre-commit passed.
