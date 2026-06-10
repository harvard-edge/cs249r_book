# Float Exposition Worklist — `security_privacy.qmd` (vol2)

Graded against the Float Exposition Standard (type-level rubric).
Caption, fig-alt, in-figure labels, code comments, and callout interiors do not count toward the prose's job.
Only running body prose is judged.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| Figure | 🟠 High | 25 | 19 | 6 | 0 |
| Table | 🟠 High | 13 | 9 | 4 | 0 |
| **Total** | | **38** | **28** | **10** | **0** |

---

## Findings (⚠️ only — no 🛑)

---

### `fig-attack-surface` (figure 🟠) — def L81

**Ref sentence (L79):**
> Architectural complexity compounds these challenges. A contemporary ML deployment spans data ingestion pipelines, distributed training infrastructure, model serving systems, and continuous monitoring frameworks, each introducing distinct vulnerabilities as @fig-attack-surface maps across the ML lifecycle.

**Missing move:** The Interpret move. The prose names what the figure does (maps lifecycle vulnerabilities) but never states what the reader should conclude. The figure's organizing insight — that the four-layer taxonomy (Data, Model, Infrastructure, Supply Chain) represents increasing access depth and correspondingly different attacker profiles — lives only in the caption. The payoff paragraph (L85) is a chapter road-map sentence, not a figure lead-out.

**Takeaway currently lives in:** caption ("Defense requires a multi-layered approach: protecting data collection...hardening compute/network/orchestration...validating frameworks, firmware, and hardware provenance").

**Rule-compliant diff rewrite:**

```diff
- Architectural complexity compounds these challenges. A contemporary ML deployment spans data
- ingestion pipelines, distributed training infrastructure, model serving systems, and continuous
- monitoring frameworks, each introducing distinct vulnerabilities as @fig-attack-surface maps
- across the ML lifecycle.
+ Architectural complexity compounds these challenges. A contemporary ML deployment spans data
+ ingestion pipelines, distributed training infrastructure, model serving systems, and continuous
+ monitoring frameworks, each introducing distinct vulnerabilities. As @fig-attack-surface shows,
+ those vulnerabilities organize into four layers ordered by access depth: the Data layer (collection
+ and pipelines) is reachable with least privilege, while the Supply Chain layer (frameworks,
+ firmware, hardware provenance) requires the most sophisticated attacker. Defenses must therefore
+ be layered to match: a breach at any single layer does not automatically expose the layers above it.
```

---

### `fig-stuxnet` (figure 🟠) — def L300

**Ref sentence (L298):**
> @Fig-stuxnet maps these parallels between the Stuxnet attack chain and ML supply chain vulnerabilities.

**Missing move:** The Interpret move. The preceding paragraph describes four ML supply-chain vectors and their defenses, but the sentence introducing the figure is a pure pointer. The figure's load-bearing insight — that the structural parallel between the Stuxnet chain and ML supply-chain compromise is not just analogical but implies the same defensive principle (provenance verification before trust is extended) — is in the caption, not the prose. The payoff paragraph (L307) pivots entirely to the Jeep Cherokee incident.

**Takeaway currently lives in:** caption ("ML systems face analogous risks through compromised training data, backdoored dependencies, and tampered model weights").

**Rule-compliant diff rewrite:**

```diff
- @Fig-stuxnet maps these parallels between the Stuxnet attack chain and ML supply chain
- vulnerabilities.
+ @Fig-stuxnet maps these parallels between the Stuxnet attack chain and ML supply chain
+ vulnerabilities. The structural lesson is the same in both cases: trust that is implicitly
+ extended to an upstream component (a Windows update server, a PyPI package, a model checkpoint)
+ creates a path that bypasses all downstream controls. Cryptographic signing of model artifacts
+ and immutable provenance logs address the root cause rather than its symptoms, for the same
+ reason that Stuxnet required certificate forgery rather than direct network intrusion.
```

---

### `fig-data-poisoning-loop` (figure 🟠) — def L884

**Ref sentence (L882):**
> This data poisoning optimization loop (@fig-data-poisoning-loop) captures the iterative interplay between the attacker's objective and the model's training process.

**Missing move:** The Interpret move. The prose describes what the figure captures (attacker outer loop, model inner loop) but does not state the key insight the figure encodes: that the minimax structure makes defense computationally difficult because evaluating any defense requires re-running the full inner training loop for each candidate poisoning set. That insight is in the caption only. The payoff (L918) delivers a narrative example but does not circle back to the computational-hardness conclusion.

**Takeaway currently lives in:** caption ("This minimax dynamic makes defending against poisoning computationally difficult, as evaluating a defense requires simulating the full training process").

**Rule-compliant diff rewrite:**

```diff
- This data poisoning optimization loop (@fig-data-poisoning-loop) captures the iterative
- interplay between the attacker's objective and the model's training process.
+ This data poisoning optimization loop (@fig-data-poisoning-loop) captures the iterative
+ interplay between the attacker's objective and the model's training process. The minimax
+ structure is what makes defense computationally hard: to evaluate whether a proposed defense
+ rejects a poisoning set $\mathcal{S}_{\text{poison}}$, the defender must simulate the full
+ inner-loop training on $\mathcal{S} \cup \mathcal{S}_{\text{poison}}$ and measure the result.
+ No efficient shortcut exists, because the attacker optimizes precisely against whichever training
+ procedure the defender runs.
```

---

### `fig-threat-mitigation-flow` (figure 🟠) — def L1038

**Ref sentence (L1036):**
> The appropriate defense for a given threat depends on its type, attack vector, and where it occurs in the ML lifecycle. Matching threats to defenses becomes clearer through the decision flow in @fig-threat-mitigation-flow, which connects common threat categories, such as model theft, data poisoning, and adversarial examples, to corresponding defensive strategies. While real-world deployments may require more nuanced combinations of defenses as discussed in our layered defense framework, this flowchart serves as a conceptual guide for aligning threat models with practical mitigation techniques.

**Missing move:** The Interpret move. The prose frames the figure as a "conceptual guide" and names the threat categories, but never states the key conclusion the flow encodes. The payoff (L1042) says "This distinction between training-time and inference-time attacks is easiest to verify in a concrete deployment scenario" and pivots away. The figure's organizing insight — that the decision path branches first on attack-time (training vs. inference) and then on vector type, and that this branching corresponds to architectural decisions rather than add-on controls — lives only in the figure's cells and caption.

**Takeaway currently lives in:** caption and figure cells (threat rows with detection and mitigation columns).

**Rule-compliant diff rewrite:**

```diff
- Matching threats to defenses becomes clearer through the decision flow in
- @fig-threat-mitigation-flow, which connects common threat categories, such as model theft,
- data poisoning, and adversarial examples, to corresponding defensive strategies. While
- real-world deployments may require more nuanced combinations of defenses as discussed in our
- layered defense framework, this flowchart serves as a conceptual guide for aligning threat
- models with practical mitigation techniques.
+ Matching threats to defenses becomes clearer through the decision flow in
+ @fig-threat-mitigation-flow, which connects common threat categories to corresponding defensive
+ strategies. The flow branches first on attack-time: training-time attacks (data poisoning,
+ backdoor injection) require provenance controls and statistical validation applied before the
+ model is trained, while inference-time attacks (adversarial examples, membership inference)
+ require runtime defenses such as input preprocessing and output perturbation. Selecting the
+ wrong branch means applying a defense that is structurally too late in the lifecycle to address
+ the threat.
```

---

### `fig-side-channel-curves` (figure 🟠) — def L1302

**Ref sentence (L1300):**
> @Fig-side-channel-curves illustrates how cryptographic computations produce data-dependent power consumption signatures that reveal algorithmic state. These variations, while subtle, are measurable and reflect the internal state of the algorithm at specific points in time.

**Missing move:** The Interpret move. The prose names what the figure illustrates (data-dependent power variations) but does not state what the two-panel layout reveals. The right panel zooms into the region where the three traces diverge, showing that the amplitude difference at a specific window corresponds to three distinct data values (0000, 1111, 0101). That is the attack's concrete mechanism: those amplitude differences are the signal the neural network learns to classify. The payoff (L1373) arrives 70+ lines later and explains the general ML-SCA approach without referencing the figure again.

**Takeaway currently lives in:** fig-alt ("Right zooms into highlighted region, revealing amplitude differences between traces labeled with binary values 0000, 1111, and 0101").

**Rule-compliant diff rewrite:**

```diff
- @Fig-side-channel-curves illustrates how cryptographic computations produce data-dependent
- power consumption signatures that reveal algorithmic state. These variations, while subtle,
- are measurable and reflect the internal state of the algorithm at specific points in time.
+ @Fig-side-channel-curves illustrates how cryptographic computations produce data-dependent
+ power consumption signatures. The right panel is the attack signal: zooming into the region
+ of algorithmic interest reveals that the three power traces separate into distinct amplitude
+ bands corresponding to different intermediate data values (here labeled 0000, 1111, and 0101).
+ A neural network trained on labeled trace segments learns to associate that band structure with
+ the key-dependent S-box output, transforming key recovery from a statistical estimation problem
+ into a classification problem.
```

---

### `fig-secure-boot` (figure 🟠) — def L2013

**Ref sentence (L2011):**
> Secure Boot frequently works in tandem with hardware-based Trusted Execution Environments (TEEs) to create a more trusted execution stack. @Fig-secure-boot traces the layered verification sequence: platform firmware and boot components are verified before permitting execution of cryptographic operations or ML workloads [@nist2018sp800193]. In embedded systems, this architecture improves resilience against preruntime compromise.

**Missing move:** The Interpret move. The prose says the figure "traces the layered verification sequence" and notes it improves resilience, but the closing claim is vague. The key insight is what the two-column layout encodes: kernel verification and filesystem verification run as parallel chains and both must pass before any CRC check or ML runtime can proceed. A failure in either column halts boot, which means the security guarantee is conditional on both chains being valid. The payoff (L2068) pivots to Apple Face ID rather than delivering the figure's takeaway.

**Takeaway currently lives in:** caption ("ensures only authenticated code runs...safeguarding model data and preventing unauthorized model substitution").

**Rule-compliant diff rewrite:**

```diff
- @Fig-secure-boot traces the layered verification sequence: platform firmware and boot
- components are verified before permitting execution of cryptographic operations or ML
- workloads [@nist2018sp800193]. In embedded systems, this architecture improves resilience
- against preruntime compromise.
+ @Fig-secure-boot traces the layered verification sequence. The two-column layout is the
+ key structural point: kernel verification and filesystem verification run as parallel chains,
+ and both must pass before execution reaches the ML runtime. A failure in either column halts
+ boot entirely, which means model substitution or firmware tampering anywhere in the chain
+ is detected before the system can be used. For ML deployments, this guarantees that the
+ model-loading code itself has not been replaced, not just that the model file is signed.
```

---

### `tbl-defense-mapping` (table 🟠) — def L408

**Ref sentence (L399):**
> @Tbl-defense-mapping provides a concrete mapping from each attack surface layer to the defensive mechanisms and detection methods that protect it.

**Missing move:** The Interpret move. The cite sentence is a pure pointer. The payoff (L410) is a generic framing sentence ("The threat modeling framework provides the analytical foundation..."). No prose sentence names the load-bearing contrast the table encodes: that detection methods are architecturally separate from defensive mechanisms, and that the API/Interface layer is the only layer where both a privacy mechanism (differential privacy) and an access-control mechanism (rate limiting) are co-required because it is the only layer exposed to untrusted external query traffic.

**Takeaway currently lives in:** table cells (Defense Mechanisms and Detection Methods columns).

**Rule-compliant diff rewrite:**

```diff
- @Tbl-defense-mapping provides a concrete mapping from each attack surface layer to the
- defensive mechanisms and detection methods that protect it.
+ @Tbl-defense-mapping maps each attack surface layer to its defensive mechanisms and detection
+ methods. The API/Interface layer is the structurally distinctive row: it is the only layer
+ simultaneously exposed to untrusted external traffic and responsible for leaking model internals
+ through confidence scores and logits. Defending it therefore requires both output-limiting
+ mechanisms (differential privacy, output perturbation) and behavioral monitoring (query pattern
+ analysis, confidence distribution monitoring) deployed together. The other layers separate
+ prevention from detection more cleanly.
```

---

### `tbl-adversary-knowledge-spectrum` (table 🟠) — def L954

**Ref sentence (L944):**
> @Tbl-adversary-knowledge-spectrum categorizes this spectrum of knowledge levels, showing how access to model internals and training data determines both attack feasibility and defense complexity across different deployment environments.

**Missing move:** The Interpret move. The lead-in (L942) describes the three knowledge levels, but that is anticipation, not interpretation. The citation sentence is a pointer. The payoff (L956) pivots to a case study. No prose sentence states the key conclusion the table encodes: that adversarial transferability collapses the practical gap between white-box and black-box attacks (documented in the footnote, not the body prose), making the knowledge level a weaker predictor of attack feasibility than the table's structure implies.

**Takeaway currently lives in:** table cells (Typical Attack Methods column and the fn-adversarial-transferability footnote, which is not body prose).

**Rule-compliant diff rewrite:**

```diff
- @Tbl-adversary-knowledge-spectrum categorizes this spectrum of knowledge levels, showing how
- access to model internals and training data determines both attack feasibility and defense
- complexity across different deployment environments.
+ @Tbl-adversary-knowledge-spectrum categorizes this spectrum of knowledge levels. The table's
+ practical implication is that the white-box/black-box distinction is less protective than it
+ appears: adversarial transferability means perturbations crafted against a freely available
+ surrogate model fool the production target 60 to 80 percent of the time, so a black-box
+ attacker who cannot query the target directly can still mount an effective attack. Defenses
+ designed only for the white-box row therefore underestimate realistic black-box threat levels.
```

---

### `tbl-threats-models-summary` (table 🟠) — def L1034

**Ref sentence (L1026):**
> @Tbl-threats-models-summary categorizes these threats by lifecycle stage and attack vector, clarifying how vulnerabilities manifest and enabling targeted mitigation strategies.

**Missing move:** The Interpret move. The cite sentence is a pointer. The payoff (L1036) also names defenses as a pointer to @fig-threat-mitigation-flow. No prose delivers the key structural insight from the three-row table: that model theft, data poisoning, and adversarial attacks differ not just in stage but in the architectural boundary they violate. Model theft is a deployment-interface failure; data poisoning is a training-data-integrity failure; adversarial attacks are an inference-runtime failure. That structural reading of the rows is what guides defense selection.

**Takeaway currently lives in:** table cells (Lifecycle Stage, Attack Vector, and Example Impact columns).

**Rule-compliant diff rewrite:**

```diff
- @Tbl-threats-models-summary categorizes these threats by lifecycle stage and attack vector,
- clarifying how vulnerabilities manifest and enabling targeted mitigation strategies.
+ @Tbl-threats-models-summary categorizes these threats by lifecycle stage and attack vector.
+ The three rows differ structurally, not just temporally: model theft is a deployment-interface
+ failure (the attacker reaches the model through an API or artifact store), data poisoning is a
+ training-data-integrity failure (the attacker reaches the learning process through manipulated
+ inputs), and adversarial attacks are an inference-runtime failure (the attacker reaches the
+ decision boundary through crafted inputs at serving time). Each row therefore demands a defense
+ at a different architectural layer and cannot be addressed by controls designed for another row.
```

---

### `tbl-hw-security-comparison` (table 🟠) — def L2153

**Ref sentence (L2144):**
> @Tbl-hw-security-comparison compares their roles, use cases, and trade-offs for machine learning system design.

**Missing move:** The Interpret move. The cite sentence is a pure pointer. The payoff (L2155) gives a general "defense-in-depth" framing without extracting a specific row or trade-off. The table's load-bearing contrast — that TEEs protect runtime isolation but are memory-limited (constraining model size), while HSMs are the correct primitive for key custody but are expensive and low-I/O, making the choice depend on whether the threat is inference-time data exposure or key compromise — is not stated in body prose.

**Takeaway currently lives in:** table cells (Trade-offs column: "Added complexity, memory limits"; "High cost, integration overhead, limited I/O").

**Rule-compliant diff rewrite:**

```diff
- @Tbl-hw-security-comparison compares their roles, use cases, and trade-offs for machine
- learning system design.
+ @Tbl-hw-security-comparison compares their roles, use cases, and trade-offs for machine
+ learning system design. The trade-offs column drives selection: TEEs provide runtime isolation
+ but impose memory limits that constrain model size (making them impractical for large models
+ without quantization or offload), while HSMs provide tamper-resistant key management but at
+ high cost and limited I/O bandwidth. The choice is therefore not which primitive is strongest
+ in the abstract, but which trust boundary the deployment needs to protect, because each
+ mechanism defends a structurally different layer.
```

---

## Dangling reference

- `@fig-ondevice-gboard` (L1511) — referenced in body prose but no matching definition found in the chapter. This float is missing or mislabeled.
