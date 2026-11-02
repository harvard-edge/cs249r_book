# Labs development review · Volumes I–IV

The authoring and independent concept-review pass covers the complete four-volume lab route. The labs now share a chapter-owned investigation structure, continuous project narratives, and full-page book navigation. The integrated checks and final browser follow-ups are complete below. This report describes the current implementation and its evidence limits; it is not a classroom learning evaluation or permission to deploy a system.

## North star

Students should learn to make a conditional systems decision: identify a constraint, predict a consequence, change a controlled condition, inspect what happens, explain the improvement and its competing cost, and state what would reverse their choice.

The book explains concepts in a static form. TinyTorch asks students to build mechanisms. Labs provide a place to explore consequences without rebuilding the whole system. StaffML rehearses interview reasoning. A useful lab submission connects a choice to evidence and assumptions. Clicking through controls, reaching a preferred number, or satisfying an interface completion gate does not establish mastery.

## Repeatable student experience

Every opening exposes the project situation, chapter-specific learning objectives, complete investigation route, expected discussion artifact, prerequisites, and next question. Parts group the concepts the chapter owns. Each part introduces a small decision, asks for a prediction, provides a controlled comparison, reveals consequences immediately, and asks for an explanation and reversal condition. Progressive disclosure opens evidence after a prediction; the goals and route stay visible from the beginning. Incorrect predictions remain useful and do not block investigation.

Live controls update the current evidence. Pinning or capturing retains a comparison; saving a discussion is an explicit action. Changed controls, predictions, relevant context, or interpretation make affected saved evidence stale. An unchanged baseline is not an intervention; orientation can separately rehearse evidence handling. Exports distinguish incomplete work from an assembled discussion. Where storage is session-only, the interface says so and asks students to download their work. Local storage is not shared submission or cross-device synchronization.

An instructor can assign a chapter and its lab, then ask students to submit the downloaded discussion. The discussion should identify what changed and what stayed fixed, explain the mechanism, compare an alternative, name missing evidence, and defend a reversal condition. Students may use an LLM, but their artifact still needs to refer to the actual investigation evidence. No automated claim of authorship or mastery is made.

## Inventory and continuing narratives

| Volume | Notebooks, including orientation | Investigation parts | Continuing project |
|:--|--:|--:|:--|
| 1 | 17 | 80 | Activity ring, mobile photo organizer, road-object perception, or cloud recommender |
| 2 | 18 | 87 | A fleet engineering team assembling a deployment dossier |
| 3 | 21 | 84 | A repair agent progressing from a task contract to governed effects, evaluation, learning, and economics |
| 4 | 19 | 76 | A warehouse mobile manipulator progressing from a causal boundary to a defended release decision |
| Total | 75 | 327 | |

Volume I keeps TinyML, MobileML, EdgeML, and CloudML as different system scales and projects. Orientation selects the project; subsequent labs restore that preference. Supplied analytical or behavioral references are identified where they differ from the project. A reference result does not establish that the student's ring, phone, vehicle, or recommender is ready to deploy.

Volume II now starts with Lab 00: choose or restore a fleet project, rehearse prediction and reactive feedback, capture a detached comparison, explicitly save and read back a setup record, and download a backup. Lab 01 remains the technical introduction. Volume II now starts with Lab 00: choose or restore a fleet project, rehearse prediction and reactive feedback, capture a detached comparison, explicitly save and read back a setup record, and download a backup. Lab 01 remains the technical introduction. Volume II extends the design problem to coupled infrastructure, communications, training, recovery, scheduling, serving, operations, security, sustainability, and responsibility. Its fleet references differ by workload. The capstone carries prior scoped discussions and separately replays recognized analytical resource records. Its later experiments explicitly use an instructor-supplied fleet configuration; those parameters are not silently assembled from prior student choices. Neither a saved dossier nor a passing instructor case grants project deployment clearance.

Volume III follows the repair project through bounded requests, context and memory, tools and authority, durable execution, recovery, evaluation, operations, learning, delegation, and economics. Research, remediation, and support are bounded transfer framings. The controlled repair fixture does not become an independently executed or calibrated workload when a student switches projects.

Volume IV follows the warehouse mobile manipulator through body, computation, sensing, memory, intent, planning, enforcement, placement, intervention, verification, release, responsibility, and a transfer to a new cell. The narrative preserves unresolved physical evidence. Browser studies compare analytical references and constructed traces; the separate supervised hardware studio retains its own equipment, evidence, and staff qualification requirements.

## Single source and presentation

Chapter concept maps own the concepts and source sections. Lab concept files under `volN/concepts/` bind those concepts to student goals, parts, comparisons, evidence artifacts, limits, and narrative handoffs. `tools/build_learning_paths.py` compiles the packaged learning contracts; `tools/build_lab_inventory.py` generates the site inventory. The chapter route below is a generated review snapshot of those contracts, not another authoring source.

MLSysIM owns reusable scenario inputs, physical calculations, finite traces, study outputs, and replay mechanisms. Shared lab helpers own presentation and evidence handling. A maintained notebook supplies the track-dependent experience; separate notebooks are not authored for each device scale. Canonical fixtures carry declared assumptions and provenance rather than replacing missing evidence with apparent precision.

The catalog and labs use the shared Quarto brand, shield, menus, theme persistence, fonts, and light/dark styles. Each lab is a native full-page Marimo application without an outer lab iframe. Contextual navigation supplies the chapter reading and previous/next lab. The notebook occupies the remaining viewport; tables retain horizontal scrolling where their evidence cannot fit on a phone. Shared presentation changes apply across volumes instead of introducing a separate lab design system.

## Completed depth improvements

| Volume | Mechanisms now available | Interpretation boundary |
|:--|:--|:--|
| I | Annotation agreement versus correctness, independent-reference labor and adjudication; changing/stable feature freshness; memory-feasible service comparisons; representation collisions; exact differentiation and regeneration; matched-precision MFU; fixed-compute model/data allocation; quantization metadata; static/continuous decode admission; operations version/evidence recovery; proxy versus independently audited benefit; selective-label feedback; matched-service lifecycle accounting; cause-specific synthesis and retesting. | Finite cases and analytical references do not predict learned project quality or measured hardware timing. |
| II | Evolving error-feedback residuals; RDMA/PFC and retry traces; fixed-batch update equivalence; recovery and elastic correctness gates; scheduler fencing/leases and resize economics; profiling and useful-work accounting; canonical iteration/prefill and state-routing references; aggregation survival/release checks; operational time joins and alert obligations; finite poisoning/extraction and layered defenses; water/scarcity accounting; fairness uncertainty, selective feedback and additive-state unlearning; complete discussion dossier and analytical output replay. | Necessary mechanisms and explicit assumptions do not establish production protocol correctness, cryptographic assurance, convergence, or future incident rates. |
| III | Chapter-specific predictions and explanations; meaningful baseline/policy contrasts; guarded operation replay; uncertain-effect settlement; finite grader/reference comparison; KV allocation and paused-state accounting; exact-prefix work/reuse accounting with unknown latency and billing; bounded context, learning, delegation, and full-cohort economics traces. | Scripted branch outcomes are constructed counterexamples, not new model samples. Qualitative learning traces do not simulate optimizer updates or prove scaling benefits. |
| IV | Current ramp and limiting; coherent mailbox reads and independent expiry clocks; watchdog progress; finite multimodal action/timing comparisons; capture geometry and bounded clock effects; residual condition matching; delayed scalar estimation and covariance diagnostics; four-state occupancy and query/storage budgets; work tanks and series compliance; quintic blend/rate/stopping-reserve comparisons; subgroup intervals and separately assumed delays; staffing queues; gap-matched campaigns and conditional economics. | These are declared reference models and evidence reviews. None validates a sensor, braking plant, installed controller, physical station, or release permit. |

## Independent review and fixes

Parallel authors used the chapters, chapter concept maps, shared template, and course north star. Separate agents reviewed concept ownership, progressive disclosure, causal comparisons, misleading scope claims, prediction and evidence freshness, numeric boundaries, and student workflows. These were simulated expert, professor, and classroom perspectives. No actual student or professor feedback was collected.

The red-team cycle produced concrete corrections rather than a blanket completion score. Examples include suppressing throughput for memory-infeasible designs, retaining quantization metadata, separating agreement from correctness, preserving exact differentiation dependencies, rejecting baseline-only comparison completion, making prediction and pinned-baseline changes invalidate saved evidence, separating rollout coefficients from causal blame, and preventing pending external liabilities from becoming free budget.

Cross-volume review also corrected contradictory context-capacity premises, stale claims about unavailable cost arithmetic, mislabeled constructed evidence, and promised instruments that exceeded the implementation. Volume IV's zero-margin verification case now agrees with its nonnegative-margin oracle. Spatial Memory Part C now promises the actual scalar original-age/republish/predict/new-capture expiry comparison; it does not claim a frame-tree, general error-growth, or spatial erosion instrument. Capstone replay checks distinguish booleans from JSON numbers and reject malformed schema types while preserving legitimate owning-chapter snapshots.

Independent computational review checked bounded counterexamples and numerical invariants, including scalar updates against rational arithmetic, Wilson score boundaries, queue-tail inversion, and work conservation with a tank reserve. Source review is separate from browser verification, and neither is described as a real classroom study.

## Model limits that remain deliberate

**Volume I.** Numerical estimates depend on stated analytical efficiencies, feasible allocation, supported precision, and workload assumptions. Annotation and feedback studies operate on supplied finite records; no learned accuracy, empirical drift rate, or field-calibrated annotator model follows. The fixed-compute language reference does not predict a ring or vehicle model, and repeated-data learning gains remain unavailable. Decode batching assumes supplied iteration timing and prepared prompts; it does not model output-token KV growth or calibrated hardware execution. Operations version checks do not replay complete request outputs or cache contents. Lifecycle accounting compares matched declared obligations and finite behavior; it does not estimate currency TCO or full embodied emissions. A proposed behavioral repair does not supply missing execution evidence.

**Volume II.** Finite network, coordination, recovery, scheduling, and protocol traces demonstrate declared mechanisms and correctness obligations. They are not implementations or proofs of production NICs, distributed consensus, cryptography, physical fencing services, or empirical convergence. Workload and service models retain explicit scope; goodput does not imply useful retention of arbitrary partial outputs. Authorized-output leakage still needs empirical audit. Exact unlearning is confined to an additive toy state, not a deep-model certificate. Statistical audit bounds require the disclosed independent representative-count assumptions. Normative fairness, incident authority, privacy, and governance obligations remain human decisions. The capstone distinguishes inherited discussion evidence, replay-compatible reference outputs, and its separate instructor case.

**Volume III.** Most investigations compare versioned authored traces or deterministic checks over supplied records. Outcomes are scoped to their stated task contract and predicates; counts do not become empirical probabilities or benchmark rates. Prefix/token records are declared references, not tokenizer or provider-billing calibration. Learning comparisons audit conditioning, masks, normalization, preference admission, reward scope, and provenance; they do not compute trained quality, gradients, optimizer updates, or population gains. Delegation records retain boundaries and integration work without inferring numerical speedup. Nonrepair projects remain transfer exercises.

**Volume IV.** Geometry is a constructed planar reference; residual order statistics are not population error guarantees. Scalar independent-noise updates with known motion do not constitute a multivariate estimator or calibration campaign; single NEES/NIS values are diagnostics. Current response uses an ideal zero-speed electrical reference. Publication and watchdog traces do not qualify concurrent firmware, memory barriers, wiring, or WCET. Occupancy storage/read counts do not establish query latency. The energy tank assumes realized damping and ideal scalar work intervals; compliance does not establish available spring travel, contact safety, or stability. Blend studies use ideal fixed-endpoint tracking and sampled stopping references; between-tick behavior, terminal smoothing, and unmatched controller dynamics remain unqualified. Wilson intervals are not simultaneous subgroup guarantees; rates do not determine detection delays. Staffing queues rely on stated stationary assumptions and cannot promise physical response deadlines. Campaign payback is hypothetical and conditional on actual successful closure and separate approval; funding a campaign never closes a physical premise.

Across all volumes, actual learning outcomes, realistic completion time, assistive-technology accessibility, and sustained offline or dependency-CDN failure behavior need separate validation. A focused human classroom pilot is the next educational validation step.

## Final integrated verification

The original development pass covered 74 notebooks and 323 parts. The later Volume II onboarding addition is verified separately below. Each default-project part was exercised in actual browser Python, using a meaningful controlled comparison or an explicit orientation, invariant, ledger, or normative workflow criterion. This is not every track/control permutation or a classroom learning evaluation.

| Verification layer | Final status and evidence |
|:--|:--|
| Canonical contracts, chapter coverage, and generated inventory | All 74 chapter contracts / 323 parts pass the source ownership audit; generated contracts and inventory are current. |
| Full labs suite | 1,740 passed; 89 skipped; 160 expected failures; no failures or errors. `/tmp/mlsysbook-completion-labs-tests.xml` |
| Full MLSysIM suite | 3,498 passed; 19 skipped; no failures or errors. `/tmp/mlsysbook-completion-mlsysim-final-tests.xml` |
| Strict notebooks and native execution | All 74 strict checks pass. All 74 native notebooks / 323 parts were traversed; final browser checks and documented follow-ups close the runtime issues and older comparator flags. |
| Complete site and package freshness | Full build exported all 74 application routes. Subsequent notebook-only corrections were re-exported; current code bodies match every export after normalizing Marimo's rewritten cell interfaces. Both wheels match source, and all 306 served wheel copies match their canonical packages. |
| Desktop/mobile visuals | Opening and investigation views were inspected at 1440px and 390px by shard reviewers; the parent inspected the all-lab overview sheets and final corrective screenshots. Shared Quarto menus, theme persistence, full-page geometry, compact part tabs, bounded selectors, and internal keyboard table scrolling passed targeted checks. |
| Actual built-WASM matrix | All 74 labs / 323 parts have closed statuses. The original 317 generic passes and six explicit follow-up closures are preserved separately in the consolidated report; no raw failures were silently overwritten. |
| Evidence and discussion workflows | Representative explicit capture, save, edit, stale-state, resave, reload, and actual download paths passed. Generic capture attempts across the matrix are not described as complete student submissions. Native persistence checks used isolated temporary ledgers. |

The six raw Volume II flags were closed with fresh evidence: Lab 06 C/D/E after correcting an exact-zero duration formatter crash; Lab 11C's selected replay consequence; Lab 16E's normative ownership/capture gates with unchanged mechanical metrics; and Lab 17D's selected failed-constraint consequences. A separate Introduction comparison confirms that more independent devices do not pool memory, while reducing per-device state can change feasibility.

Final visual corrections include explicit selected/rejected consequences, nonzero adaptive units, neutral feedback before a decision, version-specific evidence refresh, readable chart legends and labels, and responsive case selectors. Long repeated concept names no longer expand the Volume III/IV tab strip: compact Part A–D labels retain complete concept names in the route and active heading. Fresh tab-click, keyboard, and focus checks keep the page aligned; wide evidence tables scroll within their own regions.

Evidence index:

- Consolidated actual-browser results, preserved raw reports, supplemental witnesses, source hashes, and package hashes: `/tmp/mlsysbook-final-browser-matrix/report.json`.
- All-lab visual inventory: `/tmp/mlsysbook-final-visual-all74/manifest.json` (39 sheets, 296 linked overview images; individual part originals remain available).
- Shared navigation/theme check: `/tmp/mlsysbook-final-shared-chrome/result.json`; browser-Python representative checks and draft downloads: `/tmp/mlsysbook-completion-wasm-chrome/`.
- Complete build: `/tmp/mlsysbook-completion-site-build.log`; exported code freshness: `/tmp/mlsysbook-completion-export-freshness.json`.

The excluded tests remain exclusions, not passing coverage. Constructed traces, analytical estimates, and hypothetical budgets retain their displayed limitations. These checks do not establish real student learning, assistive-technology accessibility, empirical project performance, or physical qualification.

The work remains in the `MLSysBook-labs-tradeoffs` worktree on `feat/labs-tradeoffs`. Local preview: http://localhost:29418/ . Changes remain local; nothing was published or pushed.

## Narrative and engagement follow-up · October 22

The student-facing pass now covers all 75 labs. Each opening uses the same H2, **Your scenario**, before the learning goals and complete part route. Volume I displays the selected MLSysIM project episode; Volume II uses the chapter-owned lab narrative and keeps the deployment context connected to its model-preparation references. Volumes III and IV expose their existing chapter scenes through shared companion helpers. Lab 00 welcomes students into the continuing project, explains how the investigations work, and introduces the next chapter. Repeated opening goal/next-step strips were removed where the learning map already supplies those connections.

Six real photographs illustrate the ring, phone, autonomous vehicle, cloud service, software repair assistant, and mobile manipulator. The helper loads optimized images from the installed package, with alt text and expandable source, creator, license, and modification credits. The photos illustrate the setting; model assumptions and results still come from their existing computation layer. The attribution source is `mlsysbook_labs/data/photos/manifest.json`.

Every closing exposes **Discuss and submit** before completion. Student language replaces setup-record and internal evidence-handling jargon. Download readiness is explicit and uses the shared page width. Volume III now offers a readable Markdown discussion alongside JSON evidence. Volume I/II discussions lead with the project, recommendation, reasoning, and evidence; versions and raw results follow in source details. A download is described as a personal copy, not an import or automatic submission. Printing instructions name a Markdown viewer.

### External review and iteration

Actual Claude CLI reviews used first-party **claude-opus-5-5** for all four volumes, followed by a second review of the revised template. Actual Antigravity (`agy`) reviews selected **gemini-4-argon-high** and covered all 75 labs through initial and successful completion responses. Inputs included canonical concept maps, notebook opening/ending excerpts, the full-volume context, and access to relevant chapter/source files. This was an expert narrative review, not a full independent technical audit of every chapter or human classroom feedback.

Applied findings include bringing the scenario ahead of caveats and goals, centralizing meaningful project photos, removing repeated opening/closing prompts, keeping submission instructions visible, providing a readable discussion download, separating student reasoning from administrative metadata, and clarifying save/download/print behavior. Stale concurrent-source findings and proposals for new numerical studies were checked rather than adopted automatically. Reports and the exact coverage manifest are under `/tmp/mlsysbook-narrative-external-review/`; the second review is `claude-iteration2.md`.

### Verification for this pass

- All 75 notebooks pass strict checks; canonical ownership, generated learning contracts, and landing inventory checks report no errors or staleness. Calculations were not changed by this narrative pass.
- The actual native browser sweep covers all 75 initial-project openings and closings at 1440px and 390px: 150 viewport checks and 300 screenshots. The parent and reviewer collectively inspected all 19 contact sheets. A later 35-lab Volume I/II sweep and final readiness-geometry witnesses cover the subsequent copy/layout refinements. Original reports remain intact: `/tmp/mlsysbook-all75-narrative-qa/latest-evidence.json`.
- Exported browser Python checks cover I00/02, II00/01, III00/09, and IV00/08, including real photos, credits, shared book navigation, responsive layout, and representative actual downloads. I00's four context/photo changes and save/download/reload pass; II00 saves/restores Cloud and carries it into II01. III/IV readable Markdown downloads pass. Final III checks also verify the volume accent in dark mode. The last actual-WASM readiness alignment witness passes on desktop and mobile (`/tmp/mlsysbook-vol1-final-alignment-wasm/report.json`).
- The full suite recorded 1,786 passed, 113 skipped, and 160 expected failures, plus one browser persistence initialization timeout. That unchanged persistence test passed on an isolated retry (`/tmp/mlsysbook-narrative-browser-test-retry.xml`). The full result remains preserved at `/tmp/mlsysbook-narrative-final-tests.xml`; the final focused UI suite passes 883 tests with 8 skips (`/tmp/mlsysbook-narrative-ui-final-tests.xml`).
- The full site exports all 75 routes. Final notice-only notebook exports and the helper package were refreshed and installed across the preview. Browser wheels match their source, and packaged image assets are present.

Cold browser initialization can be slow. III00 reproducibly logged a bootstrap RPC timeout, recovered, rendered, and downloaded its discussion successfully; this is preserved in `/tmp/mlsysbook-vol34-narrative-wasm-final/summary.md`. It is not counted as a clean cold-start experience. A human classroom pilot and broad assistive-technology validation remain separate work.

## Onboarding and landing follow-up · October 22

All four volumes now begin with Lab 00. Volume II adds a dedicated four-part orientation without renumbering or replacing its Chapter 1 introduction. The current canonical inventory contains 75 notebooks and 327 parts. Orientation rehearses the interface and owns no textbook concepts or numerical performance claims.

The landing page now puts four equal volume choices first, each with an explicit Start Lab 00 link and an expandable canonical route. Saving and local execution details open on demand. Shared Quarto navigation, typography, and volume accent colors remain in use. Navigation points Volume II to onboarding, and a regression test rejects an inventory whose volume lacks Lab 00.

The full updated labs suite passes: 1,776 passed, 113 skipped, and 160 expected failures, with no failures or errors (`/tmp/mlsysbook-onboarding-labs-tests.xml`). The ownership audit reports zero errors. The complete site build exports 75 routes; the final orientation revision was separately re-exported and the helper wheel rebuilt and synchronized. Source/package freshness checks pass.

Desktop (1440px) and phone (390px) landing checks pass, including expanded Volume II route order and no horizontal overflow (`/tmp/mlsysbook-onboarding-landing/result.json`). Native Volume II onboarding passes actual Cloud track selection, prediction, reactive view change, detached capture, explicit save and read-back, JSON download, and restoration after reload; screenshots and download are in `/tmp/v2-onboarding-qa/`. Those persistence tests use an isolated temporary ledger. The exported WASM orientation independently passes the same Cloud selection, prediction, reactive comparison, capture, save/read-back, actual JSON download, reload restoration, and full-route flow (`/tmp/v2-onboarding-wasm.log`, screenshots and download in `/tmp/v2-onboarding-wasm/`). The parent inspected its full-page book header and mobile rendering. The earlier all-lab browser matrix remains evidence for its original 74 notebooks; it does not retrospectively include the new onboarding lab.

## Chapter-by-chapter review route

Generated from `mlsysbook_labs/data/learning_paths/volN.json`. Update the canonical concept maps and regenerate contracts before refreshing this snapshot. Each entry gives the chapter goal, part questions, and onward connection; the notebook exposes the full evidence artifact and limitations. Companion design plans remain under `plans/volN/`.

### Volume 1

**00 · [Welcome to your deployment project](http://localhost:29418/vol1/lab_00_introduction/)**

Choose a deployment project and practice predicting, comparing, and saving your work.

- A · Choose your deployment project: Which mission and system boundary do you want to investigate?
- B · Rehearse the investigation: Which action changes the live case, and which retains a comparison?
- C · Keep your work: What has actually been stored, and what can you download?
- D · Start the first investigation: What have you practiced, and what will you investigate next?

**Next:** The system runs. Is it useful? — Choose the next data, algorithm, or machine intervention using separate evidence of behavior and execution.

**01 · [The system runs. Is it useful?](http://localhost:29418/vol1/lab_01_ml_intro/)**

Choose the next data, algorithm, or machine intervention using separate evidence of behavior and execution.

- A · Healthy runtime, changed behavior: What evidence would establish whether a changed population affects the task?
- B · Improve the right part of time: Which execution change is worth investigating first?
- C · Meet both requirements: Which candidate can you justify advancing under the fixed requirements?
- D · Name the efficiency lever: Which task factor changes under each host-study intervention?

**Next:** Where should the work live? — Defend a conditional placement or an explicit hold by screening locality, capacity, operation, and partition obligations.

**02 · [Where should the work live?](http://localhost:29418/vol1/lab_02_system_paradigms/)**

Defend a conditional placement or an explicit hold by screening locality, capacity, operation, and partition obligations.

- A · Locality: What part of a remote path can the selected change affect?
- B · Resources: Does the configuration fit, and what determines its time?
- C · Operation: Which operating choice meets the supplied service and resource requirements?
- D · Placement: Which placement survives the supplied gates, and what is still unknown?

**Next:** A late requirement makes earlier work conditional — Trace a requirement through the lifecycle and plan a justified repair, retest, and return path.

**03 · [A late requirement makes earlier work conditional](http://localhost:29418/vol1/lab_03_ml_lifecycle/)**

Trace a requirement through the lifecycle and plan a justified repair, retest, and return path.

- A · Requirements across artifacts: Which requirement changes the development plan?
- B · Checking time: When should the team pay to check the target?
- C · Iteration opportunities: Does a faster loop produce a releasable result?
- D · Release evidence: What evidence supports a release decision?
- E · Field feedback: What should production send backward?

**Next:** Make the evidence trustworthy and deliverable — Create a versioned data contract that supports independent evaluation and a functioning input path.

**04 · [Make the evidence trustworthy and deliverable](http://localhost:29418/vol1/lab_04_data_engr/)**

Create a versioned data contract that supports independent evaluation and a functioning input path.

- A · Define the label contract: What must a label distinguish?
- B · Independent evaluation: Can higher accuracy mean weaker evidence?
- C · Supplying execution: Why is execution waiting for data?
- D · Freshness policy: How fresh must the evidence be?
- E · Producer contracts: What happens when a producer changes?

**Next:** Open the model and inspect its state — Explain how topology, gradient dependencies, phase, and batching change work, movement, and live state.

**05 · [Open the model and inspect its state](http://localhost:29418/vol1/lab_05_nn_compute/)**

Explain how topology, gradient dependencies, phase, and batching change work, movement, and live state.

- A · Tensor dimensions: What becomes expensive when a dimension grows?
- B · Phase-specific state: Why can inference fit while training fails?
- C · Batching: Does batching make every request faster?
- D · Numerical representation: Can fewer bytes produce an invalid answer?
- E · Repairing the failed case: Which intervention fixes the observed failure?

**Next:** Choose structure for the task, not the label — Choose a task-supported architecture while explaining its information assumptions and dominant resource regime.

**06 · [Choose structure for the task, not the label](http://localhost:29418/vol1/lab_06_nn_arch/)**

Choose a task-supported architecture while explaining its information assumptions and dominant resource regime.

- A · Structural assumptions: When does a structural assumption help?
- B · Cost signatures: Does more compute guarantee proportional speedup?
- C · Input scaling: Which commitment breaks first as input grows?
- D · Architecture decision: Which architecture remains defensible?

**Next:** Make the graph executable — Choose an execution plan whose visibility, setup costs, backward state, and target support match the work.

**07 · [Make the graph executable](http://localhost:29418/vol1/lab_07_ml_frameworks/)**

Choose an execution plan whose visibility, setup costs, backward state, and target support match the work.

- A · Dispatch granularity: Why can small operations waste the machine?
- B · Compilation reuse: When does compilation repay setup?
- C · Graph fusion: What does fusion eliminate?
- D · Saved activation state: Which activations should remain stored?
- E · Runtime portability: Can the runtime execute the actual graph?

**Next:** Train to a target, not to a busy GPU — Choose or reject a training recipe using state fit, effective batch, numerical progress, and a cause-specific bottleneck diagnosis.

**08 · [Train to a target, not to a busy GPU](http://localhost:29418/vol1/lab_08_model_train/)**

Choose or reject a training recipe using state fit, effective batch, numerical progress, and a cause-specific bottleneck diagnosis.

- A · Training state: Which allocations make this training run fit?
- B · Physical and effective batch: Does a larger effective batch reach the same target sooner?
- C · Numerical progress: Does faster numerical execution produce accepted progress?
- D · Recomputation: Can checkpointing rescue a nonfitting run at an acceptable time cost?
- E · Busy versus useful work: Which tested intervention changes the time to useful progress?

**Next:** Make fewer examples earn their cost — Choose a selected pool or retain the full pool using learning value, coverage, complete preparation cost, and realized delivery.

**09 · [Make fewer examples earn their cost](http://localhost:29418/vol1/lab_09_data_selection/)**

Choose a selected pool or retain the full pool using learning value, coverage, complete preparation cost, and realized delivery.

- A · Learning allocation: Where should the next learning budget go?
- B · Coverage-constrained selection: Which examples can we remove while retaining required coverage?
- C · Complete selection cost: Does the selected pool repay preparation over its reuse horizon?
- D · Realized input-path savings: Do fewer selected examples actually reduce input work?
- E · Population shift: Does the selected pool remain useful under changed populations?

**Next:** The checkpoint is not the deployed artifact — Choose a supported deployment artifact on a quality/resource frontier, or hold conversion when evidence is missing.

**10 · [The checkpoint is not the deployed artifact](http://localhost:29418/vol1/lab_10_model_compress/)**

Choose a supported deployment artifact on a quality/resource frontier, or hold conversion when evidence is missing.

- A · Deployment allocations: Which resource must compression save?
- B · Precision and task evidence: How much precision can the task lose?
- C · Sparse execution: When do fewer weights execute faster?
- D · Distillation investment: Does the deployment lifetime repay student preparation?
- E · Ordered composition: Do individually useful transformations still compose?

**Next:** A faster kernel must survive the whole path — Choose a legal mapping and resource intervention with counted traffic and an explained application-level consequence.

**11 · [A faster kernel must survive the whole path](http://localhost:29418/vol1/lab_11_hw_accel/)**

Choose a legal mapping and resource intervention with counted traffic and an explained application-level consequence.

- A · Applicable ceilings: When does changing arithmetic or bandwidth help?
- B · Capacity and reuse: How much reuse fits locally?
- C · Executable contract: Can the specialized unit execute this workload?
- D · Complete path: How much local speedup reaches the application?
- E · Constrained choice: Which feasible candidate serves the chosen objective?

**Next:** Earn the right to make a performance claim — Decide which performance and behavioral claims are supported, contradicted, or still unresolved.

**12 · [Earn the right to make a performance claim](http://localhost:29418/vol1/lab_12_perf_bench/)**

Decide which performance and behavioral claims are supported, contradicted, or still unresolved.

- A · Measurement boundary: Does local speedup improve the user path?
- B · Measurement sufficiency: How much measurement supports the claim?
- C · Operating envelope: Does the winner survive sustained deployment?
- D · Behavioral evidence: Did efficiency preserve useful behavior?
- E · Protocol parity: Which comparison can we honestly report?

**Next:** Serve useful work when demand arrives — Choose a serving policy with an explicit demand envelope, waiting behavior, state fit, and lost-service accounting.

**13 · [Serve useful work when demand arrives](http://localhost:29418/vol1/lab_13_model_serving/)**

Choose a serving policy with an explicit demand envelope, waiting behavior, state fit, and lost-service accounting.

- A · Budget the whole service: Which request path delivers useful work?
- B · Keep headroom honestly: What happens when demand becomes bursty?
- C · Form work deliberately: When does waiting for a batch earn its cost?
- D · Fit the live workload: Do model weights fitting imply concurrent requests fit?
- E · Choose an operating policy: Which overload response preserves useful service?

**Next:** An update breaks the contract without crashing — Create an evidence-based incident response with replayable lineage, feasible rollout, and tested recovery.

**14 · [An update breaks the contract without crashing](http://localhost:29418/vol1/lab_14_ml_ops/)**

Create an evidence-based incident response with replayable lineage, feasible rollout, and tested recovery.

- A · Observe without assuming: Does an early proxy alert establish outcome failure?
- B · Replay what ran: Which requests and artifacts establish parity?
- C · Schedule retraining: When should the team spend resources on an update?
- D · Limit exposure while gathering evidence: Can the release fit while enough outcome evidence matures?
- E · Restore a compatible request contract: Does a fast routing revert establish recovery?

**Next:** Useful for whom, under which obligations? — Recommend release, restricted use, baseline retention, mitigation, or hold using justified obligations and observed consequences.

**15 · [Useful for whom, under which obligations?](http://localhost:29418/vol1/lab_15_responsible_engr/)**

Recommend release, restricted use, baseline retention, mitigation, or hold using justified obligations and observed consequences.

- A · Audit the intended benefit: Can more clicks mean less independently audited benefit?
- B · Compare mitigation consequences: Which errors change when the threshold changes?
- C · Follow the selected-label loop: How do labels censored now change the next policy?
- D · Account over the lifetime: Does lower inference energy justify the same-service lifecycle claim?
- E · Preserve an investigable system: What does the team lose when records are removed?

**Next:** Defend one coherent deployment decision — Integrate earlier concepts into a release, restriction, baseline, fallback, or hold decision without inventing a new score.

**16 · [Defend one coherent deployment decision](http://localhost:29418/vol1/lab_16_ml_conclusion/)**

Integrate earlier concepts into a release, restriction, baseline, fallback, or hold decision without inventing a new score.

- A · Audit the supporting story: Which claims does the actual ledger support?
- B · Carry an improvement through: Does the alternative help the complete design?
- C · Find the recommendation boundary: When does the same design stop meeting the requirement?
- D · Test claim transfer: Which physical and behavioral claims survive a new setting?

**Next:** Review the completed volume — Which decisions remain supported, which need new evidence, and what would reverse your recommendation?

### Volume 2

**00 · [Choose a fleet project](http://localhost:29418/vol2/lab_00_orientation/)**

Choose a fleet track and rehearse prediction, immediate feedback, saved comparisons, and explicit saving before technical investigations.

- A · Choose a fleet project: Which deployed service and preparation system will your team investigate?
- B · Try the investigation cycle: Which action changes the live briefing?
- C · Save and resume: Can you save your progress and return to it?
- D · Full route and next: How do the investigations build a supported project recommendation?

**Next:** Introduction — Defend when fleet coupling is necessary and how scaling, failures, and partitions change useful work.

**01 · [Introduction](http://localhost:29418/vol2/lab_01_introduction/)**

Defend when fleet coupling is necessary and how scaling, failures, and partitions change useful work.

- A · Scale moment: When must work leave one machine?
- B · Fleet Law: When does another worker stop helping?
- C · Reliability gap: Does reliable hardware make a reliable fleet?
- D · Partition consequences: What remains available during a partition?

**Next:** Compute Infrastructure — Select a workload-matched accelerator, node, and rack envelope that passes capacity, movement, facility, and ownership-cost constraints.

**02 · [Compute Infrastructure](http://localhost:29418/vol2/lab_02_compute_infra/)**

Select a workload-matched accelerator, node, and rack envelope that passes capacity, movement, facility, and ownership-cost constraints.

- A · Sustained compute: When does a faster accelerator fail to help?
- B · Memory capacity and bandwidth: Does the working set fit before timing it?
- C · Bandwidth hierarchy: Where should communicating work reside?
- D · Rack envelope: Can the facility power and cool the fleet?
- E · Infrastructure selection: Which feasible system earns its cost?

**Next:** Network Fabrics — Choose a fleet fabric by matching message size, shared cuts, isolation, and failure evidence to the traffic contract.

**03 · [Network Fabrics](http://localhost:29418/vol2/lab_03_communication/)**

Choose a fleet fabric by matching message size, shared cuts, isolation, and failure evidence to the traffic contract.

- A · Startup and transfer: When does bandwidth beat lower startup latency?
- B · Topology and traffic: Why do identical links perform differently?
- C · Contention isolation: When does sharing become unacceptable?
- D · Diagnosis and reliable delivery: Which counterfactual restores bounded delivery?
- E · Traffic-matched design: Which fabric plan survives a new mix?

**Next:** Data Storage — Place and package data so the complete input path feeds the fleet and checkpoint publication preserves restorable state.

**04 · [Data Storage](http://localhost:29418/vol2/lab_04_data_storage/)**

Place and package data so the complete input path feeds the fleet and checkpoint publication preserves restorable state.

- A · Input delivery budget: How many consumers can this path feed?
- B · Miss timing and finite buffering: Why can high hit rate still leave workers idle?
- C · Metadata-bound packaging: Buy bandwidth or change layout?
- D · Durable publication: When is a checkpoint recoverable?
- E · Lifecycle storage cost: Which placement earns its cost?

**Next:** Collective Communication — Choose a semantically correct collective and compare its topology, overlap, and compression costs at a fixed useful-work target.

**05 · [Collective Communication](http://localhost:29418/vol2/lab_05_collective_communication/)**

Choose a semantically correct collective and compare its topology, overlap, and compression costs at a fixed useful-work target.

- A · Collective algorithm crossover: When does ring versus tree reverse?
- B · Operation contracts and routed bounds: Reduce values or route records?
- C · Topology-aware hierarchy: When does topology change the schedule?
- D · Exposed overlap: How much communication can the schedule hide?
- E · Compression and quality: When is sending fewer bytes worthwhile?

**Next:** Distributed Training — Defend a parallel training plan against memory, communication, pipeline, and time-to-target limits.

**06 · [Distributed Training](http://localhost:29418/vol2/lab_06_dist_train/)**

Defend a parallel training plan against memory, communication, pipeline, and time-to-target limits.

- A · Update equivalence and scaling: When do additional workers stop helping?
- B · Sharded state: What does making the model fit cost?
- C · Parallel placement: Where should each dimension run?
- D · Pipeline schedule: Can a smaller bubble cause another failure?
- E · Time to quality: Is the fastest step the best policy?

**Next:** Preserve valid work when failures become routine — Preserve valid work when failures become routine

**07 · [Preserve valid work when failures become routine](http://localhost:29418/vol2/lab_07_fault_tolerance/)**

Preserve valid work when failures become routine

- A · Fleet interruption exposure: When does a rare fault become routine?
- B · Checkpoint interval and expected waste: How often should progress be saved?
- C · Recoverable checkpoint generation: Is the newest checkpoint recoverable?
- D · Replica placement across failure domains: Does redundancy survive a shared domain?
- E · Recovery pipeline resource allocation: Where should recovery bandwidth go?

**Next:** Allocate useful fleet progress under competing claims — Allocate useful fleet progress under competing claims

**08 · [Allocate useful fleet progress under competing claims](http://localhost:29418/vol2/lab_08_fleet_orch/)**

Allocate useful fleet progress under competing claims

- A · Atomic allocation and exclusive ownership: Why can jobs wait while resources are occupied?
- B · Topology-aware completion time: When should a job wait for better placement?
- C · Interruption economics and elastic resize contracts: Who pays for preemption?
- D · Feasible fairness and named utilization tiers: Is equal accelerator allocation fair?
- E · Ready serving reserve: How much serving capacity should stay ready?

**Next:** Recover useful progress from a busy fleet — Recover useful progress from a busy fleet

**09 · [Recover useful progress from a busy fleet](http://localhost:29418/vol2/lab_09_perf_engineering/)**

Recover useful progress from a busy fleet

- A · Nested useful-capacity boundaries: Where does offered fleet capacity go?
- B · Reproducible fleet measurement: Are these two results comparable?
- C · Term-matched program profiling: Which path term should the next change attack?
- D · Runtime cost of a program gain: Does the faster path repay its runtime cost?
- E · Joint fleet-efficiency decision: Which complete policy produces useful progress?

**Next:** Serve acceptable requests on time at a sustainable cost — Serve acceptable requests on time at a sustainable cost

**10 · [Serve acceptable requests on time at a sustainable cost](http://localhost:29418/vol2/lab_10_inference/)**

Serve acceptable requests on time at a sustainable cost

- A · Lifetime serving cost: When does serving outweigh training?
- B · Concurrent request-state admission: Why can a model fit while its service fails?
- C · Batch formation, iteration membership and prompt interference: When does batching save work but miss deadlines?
- D · Equal-budget serving placement: Replicate, shard, or split phases?
- E · Serving headroom and state-aware routing: What happens when a replicated pool loses capacity?

**Next:** Decide whether and how a heterogeneous fleet should learn — Decide whether and how a heterogeneous fleet should learn

**11 · [Decide whether and how a heterogeneous fleet should learn](http://localhost:29418/vol2/lab_11_edge_intelligence/)**

Decide whether and how a heterogeneous fleet should learn

- A · Placement of fleet adaptation: Where should adaptation occur?
- B · Background adaptation feasibility: Can adaptation fit without harming foreground work?
- C · Replay within a finite adaptation budget: How much past experience should remain?
- D · Local work to a common federated target: How much local work should precede communication?
- E · Participation, protocol survivors and adaptive release: Who contributes, and may the fleet release?

**Next:** Operate and change a coupled model portfolio — Operate and change a coupled model portfolio

**12 · [Operate and change a coupled model portfolio](http://localhost:29418/vol2/lab_12_ops_scale/)**

Operate and change a coupled model portfolio

- A · Coupling, adoption and useful utilization: Why does the same model count create different operating work?
- B · Dependency compatibility and feature-time consistency: Which consumers does one upstream change block?
- C · Canary exposure and observable evidence: When is the canary informative enough?
- D · Fleet alert routing and separate SLO budgets: Which signals deserve a page, and which budget blocks promotion?
- E · Version-bound release evidence: Which evidence must be regenerated?

**Next:** Security and Privacy — Choose controls for an explicit fleet adversary, compare privacy and runtime costs, and bound a release claim with lifecycle evidence.

**13 · [Security and Privacy](http://localhost:29418/vol2/lab_13_security_privacy/)**

Choose controls for an explicit fleet adversary, compare privacy and runtime costs, and bound a release claim with lifecycle evidence.

- A · Which boundary does the defense protect?: Which boundary does the defense protect?
- B · What utility survives stronger privacy?: What utility survives stronger privacy?
- C · Which repeated operations consume privacy budget?: Which repeated operations consume privacy budget?
- D · Which trust mechanism meets the deadline?: Which trust mechanism meets the deadline?
- E · Can the lifecycle policy be defended?: Can the lifecycle policy be defended?

**Next:** Robust AI — Diagnose silent failure, bound defense and fallback decisions, replay layered poisoning response, and distinguish label coverage from generated-claim evidence.

**14 · [Robust AI](http://localhost:29418/vol2/lab_14_robust_ai/)**

Diagnose silent failure, bound defense and fallback decisions, replay layered poisoning response, and distinguish label coverage from generated-claim evidence.

- A · Can a feature monitor see every silent failure?: Can a feature monitor see every silent failure?
- B · How much work does abstention move downstream?: How much work does abstention move downstream?
- C · Does the defense match the threat and envelope?: Does the defense match the threat and envelope?
- D · When does finite calibration force a full set?: When does finite calibration force a full set?

**Next:** Sustainable AI — Compare fleet interventions under explicit useful-work, facility, lifecycle, and demand boundaries.

**15 · [Sustainable AI](http://localhost:29418/vol2/lab_15_sustainable_ai/)**

Compare fleet interventions under explicit useful-work, facility, lifecycle, and demand boundaries.

- A · What limits the fleet before carbon does?: What limits the fleet before carbon does?
- B · When should hardware be replaced?: When should hardware be replaced?
- C · Is the cleanest place usable?: Is the cleanest place usable?
- D · Which mitigation survives its side effects?: Which mitigation survives its side effects?
- E · Can efficiency increase total emissions?: Can efficiency increase total emissions?

**Next:** Responsible AI — Make an accountable release decision from distinct fairness, observation, capacity, and data-governance evidence.

**16 · [Responsible AI](http://localhost:29418/vol2/lab_16_responsible_ai/)**

Make an accountable release decision from distinct fairness, observation, capacity, and data-governance evidence.

- A · Which fairness claim does the evidence support?: Which fairness claim does the evidence support?
- B · Can the fleet serve explanations and review?: Can the fleet serve explanations and review?
- C · What can delayed labels and sampling establish?: What can delayed labels and sampling establish?
- D · Which evidence profile meets the data obligation?: Which evidence profile meets the data obligation?
- E · What decision follows from the evidence?: What decision follows from the evidence?

**Next:** Fleet Synthesis — Replay inherited evidence, diagnose a cross-layer failure, compare equal-budget repairs, and defend a bounded fleet decision.

**17 · [Fleet Synthesis](http://localhost:29418/vol2/lab_17_fleet_synthesis/)**

Replay inherited evidence, diagnose a cross-layer failure, compare equal-budget repairs, and defend a bounded fleet decision.

- A · Which inherited evidence can we trust?: Which inherited evidence can we trust?
- B · When does scale stop improving useful work?: When does scale stop improving useful work?
- C · Can a fast fleet still violate its obligations?: Can a fast fleet still violate its obligations?
- D · Which equal-budget repair survives the whole fleet?: Which equal-budget repair survives the whole fleet?
- E · Does the repair survive a held-out condition?: Does the repair survive a held-out condition?

**Next:** Review the completed volume — Which decisions remain supported, which need new evidence, and what would reverse your recommendation?

### Volume 3

**00 · [Choose an agent project](http://localhost:29418/vol3/lab_00_orientation/)**

Choose an agent project and practice using comparisons to explain a systems decision.

- A · Choose your project: Which task and boundary will your agent serve?
- B · Learn the investigation cycle: What changes when a policy changes?
- C · Record a reasoned decision: What can the supplied comparison support?
- D · See the full route: How does one trajectory project connect the volume?

**Next:** Introduction — Define the repair contract and defend how much autonomy its evidence can support.

**01 · [Introduction](http://localhost:29418/vol3/lab_01_introduction/)**

Define the repair contract and defend how much autonomy its evidence can support.

- A · From proposal to a governed turn: Who may turn a proposal into an effect?
- B · Scope the delegated repair: What makes this repair request enforceable?
- C · Match closure to exposure: What evidence does this exposure require?
- D · Spend on accepted repairs: What did each accepted repair actually cost?

**Next:** The Foundation Model — Choose an invocation and output interface that fits the request, preserves status and supports scoped checks.

**02 · [The Foundation Model](http://localhost:29418/vol3/lab_02_foundation_model/)**

Choose an invocation and output interface that fits the request, preserves status and supports scoped checks.

- A · Admit a bounded request: What must a model call promise before dispatch?
- B · Separate completion from replay: Did generation finish, and can it be reproduced?
- C · Choose evidence for a candidate: What does this check actually establish?
- D · Price the interface across turns: Which interface cost grows across turns?

**Next:** Test-Time Compute — Choose a bounded deliberation policy, explain what evidence it adds, and defend when to abstain.

**03 · [Test-Time Compute](http://localhost:29418/vol3/lab_03_test_time_compute/)**

Choose a bounded deliberation policy, explain what evidence it adds, and defend when to abstain.

- A · Buy the next kind of work: Where should the next deliberation budget go?
- B · A good candidate is not a returned repair: Will the selector return a valid candidate?
- C · Stress the verifier: Can the verifier survive stronger selection pressure?
- D · Revise from a refuting observation: Which evidence must change after a refutation?

**Next:** Context Engineering — Assemble and audit a bounded, current working set that supports the next repair decision.

**04 · [Context Engineering](http://localhost:29418/vol3/lab_04_context_engineering/)**

Assemble and audit a bounded, current working set that supports the next repair decision.

- A · Pack the next decision: What belongs in the next working set?
- B · Stage stable content without hiding scope: Which prefix can remain stable?
- C · Compact and audit the continuation: What must survive compaction?
- D · Refresh before the next decision: Which context became stale?

**Next:** KV Cache Management — Choose exact state allocation and waiting policies under a real pool bound, then evaluate any semantic approximation separately.

**05 · [KV Cache Management](http://localhost:29418/vol3/lab_05_kv_cache/)**

Choose exact state allocation and waiting policies under a real pool bound, then evaluate any semantic approximation separately.

- A · Fit growing state: Will the next turn fit the cache pool?
- B · Count shared work once: Which tokens can safely share state?
- C · Keep exact reuse distinct from approximation: Is this reuse exact or approximate?
- D · Price the test wait: What should remain resident while a tool runs?

**Next:** Long-Term Memory — Choose a bounded memory policy that preserves provenance, honors writes/deletions and supports current repair decisions.

**06 · [Long-Term Memory](http://localhost:29418/vol3/lab_06_long_term_memory/)**

Choose a bounded memory policy that preserves provenance, honors writes/deletions and supports current repair decisions.

- A · Read evidence with an envelope: What does this retrieved record authorize?
- B · Bound the evidence path: Can useful evidence fit the retrieval budget?
- C · Write lessons without laundering guesses: Which observation may become a durable lesson?
- D · Serve current evidence across sessions: Does the stored evidence still apply?

**Next:** Tool Calling — Defend a tool-call policy that distinguishes structural validity, action-time permission, effect settlement, and bounded observation.

**07 · [Tool Calling](http://localhost:29418/vol3/lab_07_tool_calling/)**

Defend a tool-call policy that distinguishes structural validity, action-time permission, effect settlement, and bounded observation.

- A · A patch against a moving file: Is this patch valid at the moment it is applied?
- B · A reply disappears: What should happen after a missing reply?
- C · The test log is not the job: Did the job finish, or only the request?
- D · The catalog the model actually sees: Which tool descriptions should the model see?

**Next:** Agent Sandboxes — Choose a containment and reset policy whose permitted reach and carried state match the repair task.

**08 · [Agent Sandboxes](http://localhost:29418/vol3/lab_08_sandboxes/)**

Choose a containment and reset policy whose permitted reach and carried state match the repair task.

- A · Where the command can reach: Where can the admitted command reach?
- B · Enough authority for this repair: What authority does this repair need?
- C · An envelope for a persistent workspace: Which state may survive the next run?
- D · Release the workspace without releasing its secrets: What must be removed before workspace release?

**Next:** Agent Security — Defend how untrusted content may influence proposals without granting new authority or releasing protected information.

**09 · [Agent Security](http://localhost:29418/vol3/lab_09_security/)**

Defend how untrusted content may influence proposals without granting new authority or releasing protected information.

- A · A permitted tool can disclose the wrong data: Can an allowed tool still disclose protected data?
- B · Extracted fields retain their provenance: What trust survives extracting a field?
- C · Approval for this exact action: Does this approval cover the current action?
- D · Hold the boundary across a whole episode: Does the boundary hold across the full episode?

**Next:** The Agent Harness — Choose a continuation and admission policy that accounts for task budgets, pending liability, and approved authority.

**10 · [The Agent Harness](http://localhost:29418/vol3/lab_10_agent_harness/)**

Choose a continuation and admission policy that accounts for task budgets, pending liability, and approved authority.

- A · Who decides to continue: Who owns the decision to continue?
- B · A late result still has an owner: Who owns a result that arrives late?
- C · Pause is a control action; approval is a held proposal: What exactly is paused or approved?
- D · A budget that can actually stop work: Can this admission policy enforce the task budget?

**Next:** Durable Execution — Defend a restart protocol that reconciles durable intent, actual effects, and valid authority after interruption.

**11 · [Durable Execution](http://localhost:29418/vol3/lab_11_durable_execution/)**

Defend a restart protocol that reconciles durable intent, actual effects, and valid authority after interruption.

- A · What survived the crash: What evidence survived the interruption?
- B · The crash cut changes what absence proves: What does an absent record actually prove?
- C · A workspace and a record that belong together: Do the workspace and durable record agree?
- D · Replay evidence, do not resample history: What may be replayed after recovery?

**Next:** Failure Recovery — Choose a recovery route that contains partial effects and preserves a defensible task outcome.

**12 · [Failure Recovery](http://localhost:29418/vol3/lab_12_failure_recovery/)**

Choose a recovery route that contains partial effects and preserves a defensible task outcome.

- A · An external effect creates a duty: What duty remains after an external effect?
- B · Evidence before recovery preference: Which recovery route does the evidence permit?
- C · Repair before crossing the pivot: Can the repair remain reversible?
- D · Busy is not progress: Is continued activity making useful progress?

**Next:** Agent Evaluation — Defend what a matched evaluation establishes about task success, repeated reliability, and grader error.

**13 · [Agent Evaluation](http://localhost:29418/vol3/lab_13_evaluation/)**

Defend what a matched evaluation establishes about task success, repeated reliability, and grader error.

- A · What counts as an accepted repair: What counts as an accepted task?
- B · A fair trial needs protected evidence: Is the evaluation evidence protected and comparable?
- C · Calibrate the verdict, not the confidence of the report: Which verdict errors does this grader make?
- D · Does the candidate improve the system: Does the candidate improve the declared system?

**Next:** Operating Agents — Choose what evidence to retain and whether the candidate warrants a controlled release, restriction, or hold.

**14 · [Operating Agents](http://localhost:29418/vol3/lab_14_agent_operations/)**

Choose what evidence to retain and whether the candidate warrants a controlled release, restriction, or hold.

- A · Find the evidence boundary: Which record can explain this incident?
- B · Keep the rare failure: Which failures will the retention policy preserve?
- C · Test a cause: What does the intervention establish?
- D · Hold or expose: Does the evidence justify controlled exposure?

**Next:** Trajectory Curation — Defend which observed failures warrant intervention and which trace records may legitimately enter a learning dataset.

**15 · [Trajectory Curation](http://localhost:29418/vol3/lab_15_trajectory_curation/)**

Defend which observed failures warrant intervention and which trace records may legitimately enter a learning dataset.

- A · Earn a learning intervention: Is learning the right remedy for this failure?
- B · Admit without leakage: May this trajectory enter the training dataset?
- C · Teach recovery, preserve contrast: What contrast should the learning record preserve?
- D · Bound collection and lineage: Can withdrawal reach every derived record?

**Next:** Trajectory Fine-Tuning — Choose where a behavior belongs and construct a faithful, correctly weighted training record.

**16 · [Trajectory Fine-Tuning](http://localhost:29418/vol3/lab_16_fine_tuning/)**

Choose where a behavior belongs and construct a faithful, correctly weighted training record.

- A · Put behavior in its proper home: Does this behavior belong in context, a store, or weights?
- B · Choose what gets reinforced: Which tokens should contribute to the objective?
- C · Learn useful alternatives: Which alternative should receive greater weight?
- D · Release a scoped adaptation: What evidence would justify this adaptation?

**Next:** Reinforcement Learning from Verifiable Rewards — Defend a verifiable-reward learning proposal against proxy rewards, credit errors, invalid comparisons, and rollout limits.

**17 · [Reinforcement Learning from Verifiable Rewards](http://localhost:29418/vol3/lab_17_rlvr/)**

Defend a verifiable-reward learning proposal against proxy rewards, credit errors, invalid comparisons, and rollout limits.

- A · Question the reward: Does reward success mean task success?
- B · Inspect the update signal: What does the update signal attribute?
- C · Preserve informative groups: Which sampled groups contain useful contrast?
- D · Size the rollout loop: What limits the rollout learning loop?

**Next:** Multi-Agent Coordination — Choose whether delegation improves the task relative to a capable controller under the same permissions and budget.

**18 · [Multi-Agent Coordination](http://localhost:29418/vol3/lab_18_multi_agent/)**

Choose whether delegation improves the task relative to a capable controller under the same permissions and budget.

- A · Make delegation earn its keep: What does delegation add over a capable controller?
- B · Bound the child assignment: What belongs in the child contract?
- C · Validate the integration: Is a clean merge an accepted result?
- D · Choose the architecture: Which coordination design fits the task?

**Next:** Agent Economics — Defend an architecture using complete cohort accounting, useful throughput, conserved budget liabilities, and accepted outcomes.

**19 · [Agent Economics](http://localhost:29418/vol3/lab_19_agent_economics/)**

Defend an architecture using complete cohort accounting, useful throughput, conserved budget liabilities, and accepted outcomes.

- A · Reconcile the bill: Which initiated work belongs in the bill?
- B · Price the effective lever: Which intervention changes useful accepted work?
- C · Conserve capacity and money: What liabilities remain reserved?
- D · Choose a resolved-task system: Which architecture resolves the task at an acceptable cost?

**Next:** Conclusion — Defend a scoped agent-system dossier by combining prior evidence, unresolved obligations, and explicit reversal conditions.

**20 · [Conclusion](http://localhost:29418/vol3/lab_20_conclusion/)**

Defend a scoped agent-system dossier by combining prior evidence, unresolved obligations, and explicit reversal conditions.

- A · Declare the complete contract: What contract does the final system actually meet?
- B · Reconcile state and action: Do retained state and effect evidence agree?
- C · Choose the remedy: What remedy does the observed failure warrant?
- D · Defend the completed system: What can the final dossier support?

**Next:** Review the completed volume — Which decisions remain supported, which need new evidence, and what would reverse your recommendation?

### Volume 4

**00 · [Start the physical AI investigation](http://localhost:29418/vol4/lab_00_orientation/)**

Learn how to investigate tradeoffs without confusing a model of a machine with evidence from a physical machine.

- A · Your machine and evidence: Which evidence belongs to the running machine, and which belongs to your station?
- B · Your investigation route: What will you decide in this lab, and what will remain open?
- C · A changing experiment: What changes when a control moves, and what remains pinned?
- D · A bounded discussion: How would another person check your reasoning?

**Next:** The Causal Boundary — Decide what may remain a proposal, what requires independent permission, and which delay reduction preserves elementary stopping clearance.

**01 · [The Causal Boundary](http://localhost:29418/vol4/lab_01_boundary/)**

Decide what may remain a proposal, what requires independent permission, and which delay reduction preserves elementary stopping clearance.

- A · Locate the irreversible step: Which software recovery privilege disappears at onset, and what evidence would establish a feasible physical recovery?
- B · Spend stopping clearance: When would the preferred intervention reverse, and why does a faster inference not remove braking travel?
- C · Separate proposal from permission: Why does adding a veto preserve delegated authority, and which component must still act if the Brain stalls?
- D · Join the evidence at permission: Which record would make this decision reconstructable, and which fact would still need measurement?

**Next:** The Physical Body — Choose a usable operating condition by checking all physical budgets, not by trading one failed budget against another budget’s slack.

**02 · [The Physical Body](http://localhost:29418/vol4/lab_02_body/)**

Choose a usable operating condition by checking all physical budgets, not by trading one failed budget against another budget’s slack.

- A · Fresh enough and slow enough: Which budget becomes binding first and which measurement would justify the speed ceiling?
- B · More reduction is not free torque: Why can a higher reduction make one task easier while worsening acceleration or impact response?
- C · Burst now or repeat later: Would a slower cycle, staggered burst, or isolated permission supply address the observed failure, and what cost follows?
- D · Turn a specification into a limit: What can the permission path rely on now, and what bench campaign must precede a stronger claim?

**Next:** The Cognitive Brain — Choose a proposer split and chunk supply that fits computation and evidence validity while keeping model output checkable and nonauthoritative.

**03 · [The Cognitive Brain](http://localhost:29418/vol4/lab_03_brain/)**

Choose a proposer split and chunk supply that fits computation and evidence validity while keeping model output checkable and nonauthoritative.

- A · Find the cost of a fresh proposal: When does reducing weight traffic cease to address the dominant latency, and what must be reevaluated?
- B · Supply is not freshness: Which change repaired supply, which repaired freshness, and when should execution stop despite a nonempty buffer?
- C · Give each proposer the right job: What is gained by the split and which additional handoff or evaluation burden does it create?
- D · On time can still be wrong: Which checks can refuse this proposal, and which claim cannot be settled from the model’s output?

**Next:** The Nervous System — Choose transport and expiry rules whose bounded permission path remains able to stop while the proposer stalls or floods resources.

**04 · [The Nervous System](http://localhost:29418/vol4/lab_04_nervous/)**

Choose transport and expiry rules whose bounded permission path remains able to stop while the proposer stalls or floods resources.

- A · Match rates to the physics: Which loop must close inside the drive and which work can remain supervisory?
- B · Latest complete state, not every old event: Which timer fired first and why would extending a lease hide one failure while spending more stopping clearance?
- C · Demand a bound, then try to refute it: Which term lacks a justified bound and what adversarial workload would most directly challenge it?
- D · Find the writer that bypasses refusal: Can refusal still take effect after the host fails, and what wiring evidence is missing?

**Next:** Physical Data — Choose what to collect, label, retain and claim so that data covers the learner’s recoverable states without hiding physical cost or provenance.

**05 · [Physical Data](http://localhost:29418/vol4/lab_05_data/)**

Choose what to collect, label, retain and claim so that data covers the learner’s recoverable states without hiding physical cost or provenance.

- A · More of the center does not cover the edge: Where did the learner first leave support, and which collection change addresses that point?
- B · Which action did the label mean?: Which tap serves the intended learning objective and what transfer evidence is still required?
- C · Buy useful coverage, not just samples: Which cells justify the next collection effort, and what operating claim must remain narrower?
- D · Audit the missing episodes: Where does divergence begin, what does the record support, and what must not be imitated?

**Next:** Policy Training — Choose a training regime and decoder whose support and timing fit the latch task, then hand a versioned candidate to evaluation without granting it physical authority.

**06 · [Policy Training](http://localhost:29418/vol4/lab_06_training/)**

Choose a training regime and decoder whose support and timing fit the latch task, then hand a versioned candidate to evaluation without granting it physical authority.

- A · Two good actions can average into a bad one: Which representation fixes the ambiguity and which inference contract must it satisfy?
- B · A training score leaves out its dependency: What experience would justify the proposed recovery, rather than merely improve its training score?
- C · Spend trials where the simulator is wrong: Which probe changes the engineering decision and which simulation claim remains unsupported?
- D · An update creates a new candidate: What exactly can this new checkpoint claim, and what must the evaluation campaign test next?

**Next:** Closed-Loop Evaluation — Choose a bounded claim and evidence campaign whose population, complete denominator, outcome channels and statistical limits match that claim.

**07 · [Closed-Loop Evaluation](http://localhost:29418/vol4/lab_07_evaluation/)**

Choose a bounded claim and evidence campaign whose population, complete denominator, outcome channels and statistical limits match that claim.

- A · Define success before watching the run: Which exact population does the result describe, and what change would require a new protocol?
- B · Keep completion, autonomy and burden separate: Which attractive aggregate conceals the failure that matters to this claim?
- C · Choose the distortion you can tolerate: Which combination could refute the claim, and what would still require constrained physical testing?
- D · Clean runs have a finite ceiling: What claim can be defended now, and which gap becomes a runtime or architectural obligation?

**Next:** Sensor Perception — Choose sensing and a dated observation contract whose physical quantity, qualified error, age and failure behavior support the intended consumer.

**08 · [Sensor Perception](http://localhost:29418/vol4/lab_08_perception/)**

Choose sensing and a dated observation contract whose physical quantity, qualified error, age and failure behavior support the intended consumer.

- A · Measure the quantity the decision needs: Which sensor can veto this proposed motion and what has it actually measured?
- B · Date and place the measurement: Which calibration or clock premise must be renewed before the geometry can support clearance?
- C · Age and error spend the same clearance: Which intervention buys useful clearance without weakening the evidence behind its error bound?
- D · A complete frame can still be unusable: Which silent failure remains and what independent evidence would convert it into a detectable condition?

**Next:** Spatial Memory — Choose what a stored belief can still support, when it expires, and what requires a new observation rather than a more confident estimate.

**09 · [Spatial Memory](http://localhost:29418/vol4/lab_09_memory/)**

Choose what a stored belief can still support, when it expires, and what requires a new observation rather than a more confident estimate.

- A · Update where the measurement happened: Which estimator appears confident for the wrong reason, and what can innovations alone fail to reveal?
- B · Preserve occupancy before the name: Which representation serves this consumer and which information must reach it before identity is certain?
- C · Publishing does not renew evidence: Which action changes supporting evidence, which changes only publication, and what assumption must be checked before using this expiry horizon?
- D · Some ambiguity needs a different observation: What added observation would distinguish the alternatives, and what must remain unknown until it exists?

**Next:** Grounded Intent — Which target can the permission side grant, for how long, and which requests must be refused before trajectory planning?

**10 · [Grounded Intent](http://localhost:29418/vol4/lab_10_intent/)**

Which target can the permission side grant, for how long, and which requests must be refused before trajectory planning?

- A · Specify the grasp, then fit the evidence: Why did a better estimate make this same grasp admissible without changing the grasp’s physical requirements?
- B · A confident coordinate can name the wrong thing: Which failure could a torque or collision checker never discover, and what evidence would prevent it?
- C · A reachable target can expire before arrival: When did more accurate tracking stop helping, and why is a renewable target different from stretching one stale lease?
- D · Screen the corridor without freezing the mission: Which horizon rejects a useful corridor unnecessarily, and which leaves an unexamined conflict for the next stage?

**Next:** Trajectory Planning — How much executable motion should a record contain, and which seam and stop must remain valid when the next record never arrives?

**11 · [Trajectory Planning](http://localhost:29418/vol4/lab_11_planning/)**

How much executable motion should a record contain, and which seam and stop must remain valid when the next record never arrives?

- A · Replacement timing is not inference timing: Why can overlap sustain cadence without making any single replacement faster, and what happens if the overlap still fails?
- B · A clear tool path can exceed joint limits: Which cost did damping trade for bounded rates, and why did sampling only endpoints miss a violation?
- C · Join smoothly and reserve the real stop: Which seam property removed the torque step, and where did its longer transition spend stopping reserve?
- D · Convergence cannot overrule commitment: At what instant did continuing to wait become unjustifiable, even if the next candidate would converge?

**Next:** Safety Enforcement — Which commands can pass, which can be minimally corrected, and which state-matched fallback still has valid premises?

**12 · [Safety Enforcement](http://localhost:29418/vol4/lab_12_enforcement/)**

Which commands can pass, which can be minimally corrected, and which state-matched fallback still has valid premises?

- A · Price the entire stopping envelope: What did the distance-only rule admit that the kinetic rule refused, and which additional measurement could change that result?
- B · Correct the command only while the argument holds: When did correction retain task progress, and which failed premise forced a different response?
- C · Contact changes the margin being protected: Which margin replaced clearance, and what would a faster force detector still fail to guarantee?
- D · Choose the fallback from the measured state: Why does the same trigger choose different rungs, and which lower-layer claim remains unsupported on the classroom station?

**Next:** Workload Placement — Which coupling must be removed or bounded, and what loaded evidence is required before accepting a placement?

**13 · [Workload Placement](http://localhost:29418/vol4/lab_13_placement/)**

Which coupling must be removed or bounded, and what loaded evidence is required before accepting a placement?

- A · Independence is a resource-by-resource claim: Which resource invalidated your original independence claim, and why did a privilege boundary not bound its delay?
- B · A short burst can leave a long timing hangover: Why did stopping inference fail to restore timing immediately, and which claim did the observed overrun actually refute?
- C · Move the checker without blocking its reader: Which nonblocking failure preserved the tick, and where did relocation spend the latency budget it was meant to protect?
- D · Buy the remedy that addresses the coupling: What did your remedy actually isolate, what did it cost, and which surviving common cause still needs a test?

**Next:** Supervisory Intervention — When can a person take proposal authority on a named channel, and how does the machine protect itself before that person is ready?

**14 · [Supervisory Intervention](http://localhost:29418/vol4/lab_14_intervention/)**

When can a person take proposal authority on a named channel, and how does the machine protect itself before that person is ready?

- A · Ask for help without spending the last stopping margin: When did waiting for an expert cease to be defensible, and why must fallback be local?
- B · One channel, one proposer, one bounded lease: Which packet loss left the human and robot with different beliefs, and what prevented both from driving?
- C · Smooth transfer still has a physical price: Which blend window met both rate and distance requirements, and when was there no feasible compromise?
- D · Make authority visible and reconstructable: What could an assistant misunderstand, and which record field would prevent an investigator from making that same inference?

**Next:** Adversarial Verification — Which fault trials can refute the containment claim, what would count as valid evidence, and which conclusions remain out of reach?

**15 · [Adversarial Verification](http://localhost:29418/vol4/lab_15_verification/)**

Which fault trials can refute the containment claim, what would count as valid evidence, and which conclusions remain out of reach?

- A · Declare the oracle before seeing the stop: Which apparently passing trace did not test the claim at all, and which coupled corner remains unobserved?
- B · Derive attacks from both records and physics: Which attack passed the integrity checks, and what independent channel could test its physical consequence?
- C · Search for the smallest margin before moving hardware: Which counterexample was realizable, and what new evidence would be needed to carry a simulated margin to the bench?
- D · A campaign is not a fleet hazard rate: Which evidence licenses only existence, which licenses a distribution-bound estimate, and what still makes release premature?

**Next:** Deployment Release — Which configuration and envelope, if any, does the evidence justify, and what must withdraw that permission?

**16 · [Deployment Release](http://localhost:29418/vol4/lab_16_release/)**

Which configuration and envelope, if any, does the evidence justify, and what must withdraw that permission?

- A · Find the weakest premise rather than count green evidence: Which new measurement changed the argument while another pile of traces did not?
- B · Unknown is a release outcome: Why was a slower proposed limit still not enough without its own evidence, and which unknown caused refusal?
- C · Authorize this configuration, not an abstract capability: Which unchanged capability hid a changed configuration, and which linked evidence had to be reacquired?
- D · A better policy can reopen a containment claim: Which change demanded requalification outside the learning team, and why could shadow mode not supply it?

**Next:** Responsible Physical AI — Who can approve each premise, who bears the remaining exposure and work, and what evidence must prevent an unfair or unaccountable deployment?

**17 · [Responsible Physical AI](http://localhost:29418/vol4/lab_17_responsibility/)**

Who can approve each premise, who bears the remaining exposure and work, and what evidence must prevent an unfair or unaccountable deployment?

- A · An action log is not a decision owner: Which decision was invisible in telemetry, and what authority must its owner actually possess?
- B · Count oversight only when a person can perform it: Which failed condition made the role ceremonial, and what evidence could justify leaving the current stage?
- C · Detection gaps become different physical exposure: Who bears the unresolved risk, and which observation contract, fault coverage and release condition must change together?
- D · Retain evidence without turning people into scores: What must remain to reconstruct the event, what can be removed, and whose work increased under your selected restriction?

**Next:** Engineering Physical AI — Which claims transfer as reasoning, which need new evidence, and what is the next experiment worth performing?

**18 · [Engineering Physical AI](http://localhost:29418/vol4/lab_18_frontier/)**

Which claims transfer as reasoning, which need new evidence, and what is the next experiment worth performing?

- A · Transfer the questions, not the warehouse numbers: Which reasoning transferred immediately, and which apparently similar parameter needed a new measurement?
- B · More training cannot reveal an unobserved premise: Which proposed experiment could distinguish the states in time, and which only produced another confidence score?
- C · A written restriction needs an owner and enforcement: What prevents a later checkpoint from silently restoring normal speed, and which rate could the shield model legitimately act on?
- D · Choose the next capability by its missing evidence: Why is this the next evidence-generating action, what would reverse your recommendation, and what remains forbidden meanwhile?

**Next:** Review the completed volume — Which decisions remain supported, which need new evidence, and what would reverse your recommendation?
