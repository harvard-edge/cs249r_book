# Chapter 15: Trajectory Fine-Tuning & Distillation — Student Accessibility Audit

> **Auditor Persona:** Senior undergraduate, entry-level master's student, or first- to second-year PhD student in machine learning.
> **Prerequisites Assumed:** PyTorch, Transformer self-attention, prefill vs. decode, KV cache basics, GPU memory fundamentals, basic operating systems (processes, file descriptors, signals, cgroups), supervised fine-tuning (cross-entropy loss), preference optimization (DPO basics), LoRA fundamentals.
> **Pedagogical Guardrail:** No ungrounded mathematical formalism. Every equation must justify its existence with concrete physical units, hardware intuition, and engineering payoff.

---

## Executive Summary

Chapter 15 (*Trajectory Fine-Tuning & Distillation*) is the core algorithm and systems engine of **Part V: Learning from Trajectories**. It answers the primary modeling question of agent engineering: *Why can training an agent on its own best trajectories teach it to invent the tool outputs it was supposed to wait for, and how do we optimize neural policies to propose high-value actions without compromising runtime governance?*

The chapter deconstructs the profound differences between standard text fine-tuning and agent trajectory fine-tuning. A text document has a single author; an agent trajectory interleaves two distinct authors: the stochastic neural policy (proposing reasoning and tool calls) and the deterministic external environment (returning stdout, compiler diagnostics, and test results). Ingesting raw execution traces into standard language modeling objectives produces catastrophic operational failures: models learn to simulate environmental responses, memorize static tool schemas, and suffer quadratic compounding error drift. The chapter establishes an end-to-end post-training architecture:

1. **Where Agent Knowledge Lives (The Three Homes):**
   - **Weights = Procedure:** Cross-task invariant heuristics (which tool to reach for, how to read a stack trace, when to stop). Learns once, pays on every call.
   - **Context = Contracts:** Per-task mutable parameters (current tool schemas, task specifications, workspace paths, granted permissions). Must stay in context; memorization causes stale-argument crashes.
   - **Runtime = Enforcement:** Invariant security and safety gates (schema validation, authorization, sandboxing, verification). Must hold even when the model is wrong; cannot be moved into weights or relaxed after training.
2. **Trajectory Serialization & Causality:** Formulates the four-role message contract (`system`, `user`, `assistant`, `tool`). Enforces *Causal Monotonicity*: every decision step $k$ conditions strictly on states present before timestamp $T_k$, barring post-hoc hints or leakage. Preserves unvarnished evidence (nonzero exit codes, raw tracebacks) while using reserved tokens for call and turn boundaries.
3. **Observation Loss Masking:** Formulates the masked cross-entropy objective:
   $$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N_{\text{active}}} \sum_{t=1}^T m_t \cdot \log P_\theta(x_t \mid x_{<t})$$
   where $m_t = 1$ for model-written reasoning, tool calls, and end-of-turn tokens, while $m_t = 0$ for prompt and environment observation tokens. Eliminates the two cardinal hazards of unmasked training:
   - *Gradient Dilution:* Observations represent $\approx 77\%$ of trajectory tokens; unmasked loss wastes three-quarters of the gradient on high-entropy environmental noise.
   - *Role Confusion:* The model learns that tool output follows a tool call, causing the deployed agent to emit simulated test results instead of yielding control to the sandbox runtime.
4. **Sequence Packing & Block-Diagonal Attention:** Resolves the extreme Pareto distribution of agent trajectory lengths (256 to 32k tokens). Replaces naive zero-padding (which squanders $>88\%$ of accelerator FLOPs and VRAM on padding bubbles) with FlashAttention `varlen` sequence packing. Enforces *Block-Diagonal Attention* via `cu_seqlens` (setting cross-trajectory scores to $-\infty$) and resets Rotary Position Embeddings (RoPE) at boundaries ($\text{pos}[t] = t - c_k$), ensuring mathematical equivalence with isolated single-sequence execution while saving $4d \sum_{i<j} T_i T_j$ unnecessary FLOPs.
5. **Loss Normalization Across Variable Horizons:** Contrasts Per-Token Normalization ($\mathcal{L}_{\text{token}}$, which weights long, meandering, repetitive trajectories $32\times$ more heavily than concise fixes) with Per-Example Normalization ($\mathcal{L}_{\text{example}}$, which assigns equal weight $1/(B \cdot N_b)$ to each task), preventing models from learning exploratory bloat.
6. **Exposure Bias & The Quadratic Regret Bound:** Proves that behavior cloning on pristine demonstrations suffers from $O(\epsilon H^2)$ compounding error drift:
   $$J(\pi_\theta) - J(\pi^*) \le \epsilon \frac{H(H+1)}{2} = O(\epsilon H^2)$$
   Over a 30-turn task with a $2\%$ per-step error rate ($\epsilon = 0.02$), an unmonitored agent spends an expected $9.3$ turns in unmodeled error states. Solves this via on-policy dataset aggregation (DAgger) and fault injection, recovering the linear regret bound $O(\epsilon H)$ ($0.6$ turns off-path) while mandating that labeler corrections pass the verification cascade.
7. **Schema Robustness & Perturbation Operators:** Exposes how training on static schemas causes weight memorization (Parametric Recall Error): models bypass context and emit obsolete arguments. Counteracts this via four serialization perturbations: *Permutation* (shuffling argument order), *Renaming* (synthetic synonyms), *Distractor Injection* (unused tool declarations), and *Schema Dropout* (dropping optional arguments).
8. **Trajectory Direct Preference Optimization (TDPO):** Adapts DPO to multi-turn MDPs over winning ($\tau_w$) and losing ($\tau_l$) trajectories. Resolves negative gradient destruction (where depressing losing tokens destroys valid JSON/bash syntax) via joint SFT anchoring:
   $$\mathcal{L}_{\text{TDPO}} = \mathcal{L}_{\text{DPO}} + \gamma \mathcal{L}_{\text{SFT}}(\tau_w)$$
   Eliminates length bias through length-normalized implicit rewards $\hat{r}_{\text{norm}}(\tau)$, and aligns preference pairs at the branching decision pivot.
9. **Offline Reference Logit Caching:** Precomputes reference model log-probabilities $\sum \log \pi_{\text{ref}}(a_t \mid s_t)$ to local NVMe storage. Evicts the 140 GB reference model from GPU High Bandwidth Memory (HBM), halving VRAM pressure, doubling active micro-batch size $B$, and eliminating $50\%$ of forward FLOPs during DPO training.
10. **PEFT & Accelerator Memory Decomposition:** Breaks down accelerator memory into static state ($20\Phi$ bytes for full AdamW FP32: $1{,}400\text{ GB}$ for 70B) vs. Low-Rank Adaptation (LoRA $W = W_0 + \frac{\alpha}{r} B A$, adding $<1\%$ parameters and shrinking optimizer state by $99\%$). Analyzes dynamic activation memory $M_{\text{act}} = L \cdot b \cdot s \cdot (34d + 5as)$, SRAM tiling via FlashAttention, and multi-tenant adapter dispatch (S-LoRA/Punica) using on-chip Adapter Page Tables.
11. **Evaluating Fine-Tuned Agents (The Perplexity Inversion):** Proves why validation perplexity is negatively correlated with agent task completion. Establishes a 5-dimensional evaluation scorecard: Verified Task Success ($\text{pass}@k$ and $\text{pass}^k$), Function-Call Accuracy (malformed, schema violation, unknown, truncated), Schema Robustness (Schema-Following Accuracy vs. Parametric Recall Error), Cost per Accepted Task, and General Capability Retention.

---

## Pedagogical Friction Points by Severity

### Priority 1 (`[P1]`): Substantial Cognitive Friction or Missing Bridges

#### 1. The Action-Only Conditioning Distribution Shift Trap (Lines 210–239)
- **Location:** Section 2.2 (*Reasoning as target or context*, lines 210–239 and Table `@tbl-vol3-reasoning-mask-regimes`).
- **The Issue:**
  The text warns against training action tokens while retaining demonstrated reasoning tokens in the context window with $m_r = 0$:
  $$\mathcal{L}_{\text{action}}(\theta) = -\sum_{j=1}^K \log P_\theta(a_j \mid x_{\text{prompt}}, r_{1:M}, a_{<j})$$
- **Student Friction:**
  Students frequently ask: *"If chain-of-thought reasoning helps the model select the right action, why can't we provide reasoning in the prompt during training to guide the model, but mask its loss ($m_r = 0$) so the model isn't penalized for exact reasoning wording? Why does that cause the model to collapse at test time?"*
- **Actionable Fix:**
  Add a focused pedagogical bridge explaining teacher-forcing covariate shift:
  > *"**The Covariate Shift Trap of Masked Reasoning ($m_r = 0$)**:
  > Under teacher forcing during training, the transformer's attention heads compute query-key dot products across all preceding tokens in the context buffer:
  > $$a_j = \text{Softmax}\left(\frac{Q_j K_{[x, r, a_{<j}]}^T}{\sqrt{d}}\right) V$$
  > When reasoning tokens $r_{1:M}$ are present in the context buffer, the optimizer discovers that attending to $r_{1:M}$ (e.g., attending to the phrase 'the bug is in line 42 of auth.py') is the easiest way to predict the target tool call arguments. The feedforward weights rely entirely on those scratchpad representations.
  > If the deployed agent is configured to emit tool calls directly without generating reasoning at test time (to save latency), $r_{1:M}$ is completely absent from the KV cache. When the model computes $Q_j K_x^T$, the representations it relied upon do not exist. Conditioned on this truncated prefix, the policy suffers massive **covariate shift**: attention weights scatter, and the model either emits malformed tool syntax or hallucinates random file paths.
  > **Engineering Rule:**
  > - If the deployed model will generate reasoning before acting: train on reasoning ($m_r = 1, m_a = 1$).
  > - If the deployed model will act directly without reasoning: excise reasoning from the training text entirely ($[x, a]$ with $m_a = 1$).
  > - **Never** leave reasoning in the prompt with $m_r = 0$ unless an external upstream model reliably generates that reasoning at runtime."*

#### 2. The Formal Derivation of the $O(\epsilon H^2)$ Compounding Error Bound (Lines 378–425)
- **Location:** Section 3.1 (*Autoregressive error compounding over extended horizons*, lines 378–425 and Equations `@eq-vol3-cost-decomposition`–`@eq-vol3-linear-regret-bound`).
- **The Issue:**
  The chapter proves that behavior cloning cost grows as $O(\epsilon H^2)$, while on-policy DAgger training grows as $O(\epsilon H)$.
- **Student Friction:**
  Students without formal imitation learning theory often find the transition from the step error $\epsilon$ to the sum $\sum_{t=1}^H t\epsilon = \epsilon \frac{H(H+1)}{2}$ mysterious. Where does the factor $t$ come from?
- **Actionable Fix:**
  Break down the derivation step-by-step:
  > *"**Step-by-Step Derivation of Quadratic Exposure Bias**:
  > Let $\mathcal{E}_i$ be the event that the policy makes an error at step $i$. We assume that on demonstrated states, $P(\mathcal{E}_i) \le \epsilon$.
  > 1. **Probability of Past Mistakes:** At turn $t$, what is the probability that the agent has made *at least one* mistake in the preceding $t-1$ turns? By Boole's inequality (the union bound):
  >    $$P(\mathcal{E}_{<t}) = P\left(\bigcup_{i=1}^{t-1} \mathcal{E}_i\right) \le \sum_{i=1}^{t-1} P(\mathcal{E}_i) \le (t-1)\epsilon$$
  > 2. **Expected Cost at Step $t$:** Decompose expected step cost $C \in [0, 1]$ based on whether an error has already occurred:
  >    $$\mathbb{E}[C_t] = P(\neg \mathcal{E}_{<t}) \cdot \mathbb{E}[C_t \mid \neg \mathcal{E}_{<t}] + P(\mathcal{E}_{<t}) \cdot \mathbb{E}[C_t \mid \mathcal{E}_{<t}]$$
  >    - If no mistake occurred ($\neg \mathcal{E}_{<t}$), the policy is on-path, so it errs with probability at most $\epsilon$. Expected cost is $\le 1 \cdot \epsilon = \epsilon$.
  >    - If a mistake already occurred ($\mathcal{E}_{<t}$), the policy is off-path in an unmodeled state. In the worst case, every action is wrong, so $\mathbb{E}[C_t \mid \mathcal{E}_{<t}] \le 1$.
  >    $$\mathbb{E}[C_t] \le (1) \cdot \epsilon + (t-1)\epsilon \cdot (1) = t\epsilon$$
  > 3. **Summing Over the Entire Horizon $H$:**
  >    $$\text{Total Regret} = \sum_{t=1}^H \mathbb{E}[C_t] \le \sum_{t=1}^H t\epsilon = \epsilon \sum_{t=1}^H t = \epsilon \frac{H(H+1)}{2} = O(\epsilon H^2)$$
  > Under on-policy training (DAgger), the policy is trained on its own induced state distribution, so even when it makes a mistake, its error rate on subsequent recovery states remains bounded by $\epsilon$. Thus $\mathbb{E}[C_t] \le \epsilon$ at all times, yielding $\sum_{t=1}^H \epsilon = \epsilon H = O(\epsilon H)$."*

#### 3. Negative Gradient Destruction in Trajectory DPO and Joint TDPO Anchoring (Lines 521–536)
- **Location:** Section 5.1 (*Direct preference optimization on trajectory rollouts*, lines 521–536).
- **The Issue:**
  Equation `@eq-vol3-dpo-gradient` reveals that the DPO gradient applies a negative update:
  $$-\nabla_\theta \sum_{t=1}^{|\tau_l|} \log \pi_\theta(a_t^l \mid s_t^l)$$
  to all tokens in the losing trajectory $\tau_l$.
- **Student Friction:**
  Students ask: *"Why does pushing down the probability of a losing trajectory break tool calling? Isn't the whole point of preference optimization to teach the model not to take bad actions?"*
- **Actionable Fix:**
  Add a clear syntactic explanation:
  > *"**Why Unconstrained Negative Gradients Destroy Tool Syntax**:
  > Consider a losing trajectory where an agent attempted to fix a bug by running `grep -r "test" /src`, but provided an invalid regex syntax flag that failed.
  > The trajectory contains 500 tokens:
  > - 480 tokens consist of valid structural syntax: `{"name": "bash", "arguments": {"cmd": "..."}}`, common English reasoning, and valid shell commands.
  > - 20 tokens consist of the flawed flag or improper logic.
  > The standard DPO gradient applies a uniform negative multiplier across **all 500 tokens**. It pushes down the log-probability of `{`, `"name"`, `"bash"`, and spaces just as aggressively as it penalizes the flawed argument!
  > Over several training epochs, these cumulative negative updates erode the model's fundamental language modeling capabilities, causing **Tool Syntax Degeneration**: the model begins emitting unmatched curly brackets, missing quotes, or truncating JSON fields.
  > **The Solution (Joint TDPO):** Adding an auxiliary SFT loss on the winning trajectory:
  > $$\mathcal{L}_{\text{TDPO}} = \mathcal{L}_{\text{DPO}} + \gamma \mathcal{L}_{\text{SFT}}(\tau_w)$$
  > acts as an 'anchor'. The positive gradients from $\mathcal{L}_{\text{SFT}}$ replenish the probability mass of structural syntax tokens, allowing the contrastive DPO term to suppress specific decision errors without degrading grammatical competence."*

---

### Priority 2 (`[P2]`): Minor Tuning & Conceptual Polish

#### 1. Per-Token vs. Per-Example Loss Normalization Behavioral Consequence (Lines 287–306)
- **Location:** Section 3.1 (*Loss normalization across variable-length trajectories*, lines 287–306 and Table `@tbl-vol3-loss-normalization-comparison`).
- **The Issue:**
  The text contrasts per-token normalization $\mathcal{L}_{\text{token}}$ with per-example normalization $\mathcal{L}_{\text{example}}$.
- **Student Friction:**
  Students think of loss averaging as an arbitrary mathematical detail rather than a primary driver of agent behavior.
- **Actionable Fix:**
  Annotate the behavioral consequence with concrete numbers:
  > *"**The Behavioral Impact of Loss Normalization**:
  > Suppose a batch contains two trajectories:
  > - Trajectory 1: A concise 2-turn fix containing $N_1 = 500$ active tokens.
  > - Trajectory 2: A meandering 40-turn trace containing $N_2 = 16{,}000$ active tokens (where the agent wandered, ran unnecessary directory listings, and eventually stumbled upon the fix).
  > Under **Per-Token Normalization** ($\mathcal{L}_{\text{token}}$), the divisor is $500 + 16{,}000 = 16{,}500$. Trajectory 2 contributes $\frac{16{,}000}{16{,}500} \approx 97\%$ of the batch gradient! The policy update is dominated by imitating the rambling 40-turn trace.
  > Under **Per-Example Normalization** ($\mathcal{L}_{\text{example}}$), each trajectory's tokens are divided by their own length $N_b$, and the batch average is divided by $B=2$. Trajectory 1 contributes exactly $50\%$ of the gradient, and Trajectory 2 contributes $50\%$. Concise execution is valued equally with protracted execution."*

#### 2. Offline Reference Logit Caching HBM Economics (Lines 544–553)
- **Location:** Section 5.2 (*Offline reference logit caching and memory reduction*, lines 544–553).
- **The Issue:**
  The chapter states that reference logit caching halves GPU HBM memory pressure and saves 50% forward FLOPs.
- **Student Friction:**
  Why does an offline cache save 50% of FLOPs?
- **Actionable Fix:**
  Explicitly detail the training step compute balance:
  > *"**Compute and Memory Math for Reference Logit Caching**:
  > In standard online DPO, each training step must compute:
  > 1. $\pi_\theta(\tau_w)$ and $\pi_\theta(\tau_l)$ (Active policy forward pass: $2 \times \text{Forward FLOPs}$, requires backward graph).
  > 2. $\pi_{\text{ref}}(\tau_w)$ and $\pi_{\text{ref}}(\tau_l)$ (Reference policy forward pass: $2 \times \text{Forward FLOPs}$, frozen).
  > Total forward compute = $4 \times \text{Forward FLOPs}$. Furthermore, keeping $\pi_{\text{ref}}$ resident in GPU memory consumes $140\text{ GB}$ of VRAM (for a 70B model in 16-bit).
  > With **Offline Reference Logit Caching**:
  > - The reference forward pass is run once before training begins, and the scalar log-probabilities are saved to disk.
  > - During training, only $\pi_\theta$ is in GPU memory (saving $140\text{ GB}$ of HBM).
  > - Only the active forward passes are computed (cutting forward FLOPs from 4 to 2, a $50\%$ compute reduction).
  > - The scalar log-probabilities stream across PCIe at negligible bandwidth ($256\text{ KB/step}$ for a 64k token batch, consuming $<0.01\%$ of PCIe Gen5 bandwidth)."*

#### 3. FlashAttention SRAM Tiling vs. Activation Checkpointing (Lines 603–633)
- **Location:** Section 7.1 (*Training memory breakdown across regimes*, lines 603–633 and Equation `@eq-vol3-activation-memory`).
- **The Issue:**
  Equation `@eq-vol3-activation-memory` describes activation memory scaling as $L \cdot b \cdot s \cdot (34d + 5as)$, noting that FlashAttention and activation checkpointing reduce this.
- **Student Friction:**
  Students confuse the roles of FlashAttention and activation checkpointing, assuming they do the same thing.
- **Actionable Fix:**
  Add a clear operational distinction:
  > *"**Disentangling FlashAttention from Activation Checkpointing**:
  > - **FlashAttention (SRAM Tiling):** Solves the quadratic attention matrix memory $O(s^2)$. Standard attention writes the $s \times s$ attention score matrix ($QK^T$) and softmax probabilities to GPU HBM ($5as^2$ bytes per layer). FlashAttention tiles the computation directly in on-chip GPU SRAM ($228\text{ KB}$ per SM), computing the softmax online and writing *only* the final output to HBM. It eliminates the quadratic memory footprint without recomputing feedforward layers.
  > - **Activation Checkpointing (Layer Rematerialization):** Solves linear activation memory $O(s \cdot d)$. It drops the activations of intermediate linear projections, layer norms, and GeLU/SwiGLU activations during the forward pass, recomputing them on-the-fly during the backward pass.
  > **Combined Effect:** FlashAttention eliminates quadratic scaling with sequence length $s$; activation checkpointing collapses the linear multiplier from $34d$ to $2d$."*

---

## Chapter Structural & Mathematical Review

The chapter follows a tightly organized pedagogical flow across eight sections:

```
§ 1. Scope and Behavioral Boundaries of Supervised Fine-Tuning
     ├── Three Homes of Knowledge: Weights (Procedure), Context (Contracts), Runtime (Enforcement)
     └── H·S·A exposure boundaries remain strictly unchanged beneath the model
§ 2. Serializing Trajectories
     ├── Train on the serving template (4-role message contract, reserved boundary tokens)
     ├── Parallel tool calls and completion ordering
     └── Causal Monotonicity (prefix contains only pre-existing state; no post-hoc hints)
§ 3. Observation Loss Masking
     ├── Masked cross-entropy objective: active tokens m_t=1, observations/prompts m_t=0
     ├── Dangers of unmasked training: Gradient dilution (77% tokens) and Role confusion (simulated stdout)
     ├── Reasoning tokens dilemma: deliberation cloning vs action distillation vs covariate shift
     └── End-of-turn token handling and truncated turn quarantine
§ 4. Sequence Packing & Cross-Trajectory Contamination Prevention
     ├── Padding bubble roofline (extreme Pareto length distribution, 88% wasted FLOPs)
     ├── Loss normalization: Per-token (favors long runs) vs Per-example (equal task weight)
     └── Block-diagonal attention: FlashAttention varlen, cu_seqlens, and RoPE position resets
§ 5. Exposure Bias and On-Policy Data
     ├── Autoregressive error compounding: O(epsilon H^2) quadratic regret under behavior cloning
     ├── On-policy DAgger collection: O(epsilon H) linear regret
     └── Fault injection as offline recovery data (injected errors m=0, recovery actions m=1)
§ 6. Schema Robustness
     ├── Weight memorization vs in-context tool agility (Parametric Recall Error)
     └── 4 schema perturbation operators: Permutation, Renaming, Distractor injection, Schema dropout
§ 7. Trajectory Preference Optimization (DPO) & PEFT
     ├── Multi-turn trajectory DPO formulation
     ├── Negative gradient destruction & Joint TDPO (DPO + gamma SFT)
     ├── Length normalization and offline reference logit caching (halving HBM footprint)
     └── LoRA adaptation (W = W_0 + (alpha/r)BA), static memory breakdown, and multi-tenant dispatch
§ 8. Evaluating a Fine-Tuned Agent & Synthesis
     ├── Perplexity-Competence Inversion (lower validation loss != better agent)
     ├── 5-dimensional scorecard: Task Success (pass@k, pass^k), Call Accuracy, SFA/PRE, Cost, Retention
     └── 6-stage parameter optimization and release verification pipeline DAG
```

---

## Verification of Core Mathematical & Systems Claims

### 1. Where the Loss Goes in One Trajectory (Notebook `@nbk-15-gradient-budget-and-compute-allocation`)
- **Claim:** In a 16,384-token debugging trajectory (prompt: 1,384, reasoning: 1,400, tool calls: 960, observations: 12,640), observations represent $77.1\%$ of tokens. Masked training targets $N_{\text{active}} = 2{,}360$ tokens ($14.4\%$).
- **Verification:**
  - Total tokens: $1{,}200 + 184 + 1{,}400 + 960 + 12{,}640 = 16{,}384 \checkmark$.
  - Observation share: $12{,}640 / 16{,}384 = 0.77148 \approx 77.1\% \checkmark$.
  - Active tokens: $1{,}400 + 960 = 2{,}360 \checkmark$.
  - Active fraction: $2{,}360 / 16{,}384 = 0.14404 \approx 14.4\% \checkmark$.

### 2. Padded Batch Occupancy Bubble (Perspective `@psp-15-padding-bubble-roofline`)
- **Claim:** For $B=8$ sequences padded to $L_{\max} = 32{,}768$ with mean trajectory length $\bar{L} = 3{,}800$, batch occupancy is $11.6\%$, wasting over $88\%$ of accelerator FLOPs.
- **Verification:**
  - Total allocated elements: $8 \times 32{,}768 = 262{,}144$ tokens.
  - Active tokens: $8 \times 3{,}800 = 30{,}400$ tokens.
  - Occupancy: $30{,}400 / 262{,}144 = 0.115966 \approx 11.6\% \checkmark$.
  - Wasted compute: $1 - 0.116 = 88.4\% > 88\% \checkmark$.

### 3. Quadratic Error Compounding vs. Linear Regret (Notebook `@nbk-15-quantifying-trajectory-collapse-and-dagger`)
- **Claim:** Over $H=30$ turns with per-step error $\epsilon = 0.02$, probability of no mistakes is $54.5\%$. Behavior cloning bound yields $9.3$ turns off-path; on-policy bound yields $0.6$ turns (a $15.5\times$ gap).
- **Verification:**
  - $P(\text{no mistake}) = (1 - 0.02)^{30} = (0.98)^{30} \approx 0.54548 \approx 54.5\% \checkmark$.
  - Behavior cloning bound: $\epsilon \frac{H(H+1)}{2} = 0.02 \times \frac{30 \times 31}{2} = 0.02 \times 465 = 9.30\text{ turns} \checkmark$.
  - On-policy bound: $\epsilon H = 0.02 \times 30 = 0.60\text{ turns} \checkmark$.
  - Ratio: $9.3 / 0.6 = 15.5\times \checkmark$.

### 4. LoRA Parameter and Static Memory Scaling (Lines 588–602)
- **Claim:** Full fine-tuning with 16-bit mixed precision and FP32 AdamW requires $20\Phi$ bytes per parameter ($1{,}400\text{ GB}$ for 70B). LoRA rank $r=16$ on an 8B model ($L=32, d=4{,}096$) adds $16.78\text{M}$ parameters ($33.6\text{ MB}$ weights, $268.4\text{ MB}$ optimizer).
- **Verification:**
  - Full SFT per parameter: Weights (2 bytes) + Gradients (2 bytes) + FP32 Master Weights (4 bytes) + Momentum (4 bytes) + Variance (4 bytes) + Working Grad (4 bytes) = $20$ bytes.
  - For 70B: $70 \times 10^9 \times 20\text{ bytes} = 1{,}400\text{ GB} \checkmark$.
  - LoRA parameters for 4 projections ($W_q, W_k, W_v, W_o$): $4 \times (2 \times d \times r) \times L = 8 L d r$.
  - For 8B: $8 \times 32 \times 4{,}096 \times 16 = 16{,}777{,}216 \text{ parameters} \approx 16.78\text{M} \checkmark$.
  - Adapter weights (FP16): $16{,}777{,}216 \times 2 = 33{,}554{,}432\text{ bytes} = 32\text{ MiB} \approx 33.6\text{ MB} \checkmark$.
  - AdamW state for LoRA ($16 \times \Phi_{\text{LoRA}}$): $16 \times 16.78\text{M} \approx 268.4\text{ MB} \checkmark$.

---

## Illustrative Figure Opportunities

### Figure 15.1: The Three Homes of Agent Knowledge & The HSA Enforcement Boundary
- **Placement Suggestion:** Section 1 (*Scope and Behavioral Boundaries of Supervised Fine-Tuning*, near line 65).
- **Current State:** Table `@tbl-vol3-fine-tuning-scope` exists, but lacks a conceptual systems schematic.
- **Proposed Diagram:**
  A three-column systems architecture schematic:
  - **Left Column (Weights / Neural Policy):** "Procedure" — Decision heuristics, stack trace reading, recovery instincts, tool selection. Updated via SFT/LoRA.
  - **Center Column (Context Window):** "Contracts" — Active JSON tool schemas, task issue description, repo file tree, ephemeral security tokens. Passed dynamically per turn; NOT baked into weights.
  - **Right Column (Runtime Supervisor):** "Enforcement" — JSON Schema validator, seccomp/eBPF sandboxes, read-only test mount, token budget circuit breakers. Completely outside the model; immutable by fine-tuning.
  Annotated arrows show model actions proposing changes, which are checked by the Runtime before mutating the Context.
- **Student Pedagogical Payoff:** Permanently cements why fine-tuning cannot replace runtime security or allow removing tool schemas from prompts.

### Figure 15.2: FlashAttention Varlen Block-Diagonal Attention & RoPE Reset
- **Placement Suggestion:** Section 3.2 (*Block-diagonal attention and position isolation*, near line 325).
- **Current State:** SVG graphic exists, but lacks the hardware memory layout and CUDA thread block mapping.
- **Proposed Diagram:**
  A hardware execution diagram showing:
  - Top: Flattened 1D Jagged Tensor $[N_{\text{total}}, D]$ containing $\tau_1 (T_1=4), \tau_2 (T_2=3), \tau_3 (T_3=5)$ with cumulative sequence vector $\mathtt{cu\_seqlens} = [0, 4, 7, 12]$.
  - Middle: Attention Matrix partitioned into lower-triangular diagonal blocks. Off-diagonal blocks marked with red crosses ($-\infty$, completely skipped by CUDA thread blocks).
  - Bottom: RoPE Phase Angle timeline showing position indices resetting to $0$ at each trajectory boundary ($[0,1,2,3], [0,1,2], [0,1,2,3,4]$) preventing spurious rotational phase shifts.
- **Student Pedagogical Payoff:** Connects high-level sequence packing directly to GPU kernel thread scheduling and RoPE mathematical mechanics.

### Figure 15.3: Offline Reference Logit Caching Architecture in Trajectory DPO
- **Placement Suggestion:** Section 5.2 (*Offline reference logit caching and memory reduction*, near line 550).
- **Current State:** Textual explanation without a systems diagram.
- **Proposed Diagram:**
  A two-phase systems pipeline diagram:
  - **Phase 1 (Offline Preprocessing):** Frozen Reference Model $\pi_{\text{ref}}$ (140 GB) loaded on inference node; evaluates trajectories $(\tau_w, \tau_l)$ in forward-only mode; streams scalar log-probabilities $\sum \log \pi_{\text{ref}}$ to local PCIe Gen5 NVMe SSDs ($256\text{ KB/step}$).
  - **Phase 2 (Active Post-Training):** Reference model completely evicted from GPU HBM. Only Active Policy $\pi_\theta$ (140 GB) loaded into GPU memory (enabling $2\times$ micro-batch size). Background asynchronous DMA engine streams scalar logits from NVMe directly into the TDPO loss kernel over PCIe.
  Hardware callouts highlight: $-50\%$ HBM footprint, $-50\%$ forward FLOPs, zero pipeline bubble.
- **Student Pedagogical Payoff:** Shows students a real-world systems engineering optimization that allows training 70B parameter models with DPO on half the hardware typically required.

---

## Student Walkthrough & Classroom Simulation

### Classroom Setting
- **Course:** CS 8803 / EECS 598: *Systems for Autonomous and Agentic Machine Learning*.
- **Topic:** Lecture 15: *Trajectory Fine-Tuning, Loss Masking, and Policy Distillation*.
- **Scenario:** A graduate research team fine-tuned an open-source 8B model on a dataset of 5,000 successful coding trajectories. In their offline validation script, the loss dropped from $2.8$ to $0.4$, and token perplexity reached an impressive $1.2$. But when they plugged the model into their live agent harness, the agent failed $95\%$ of tasks: it started hallucinating bash outputs, never called external tools, and repeatedly sent commands with obsolete flags. They attend office hours for an architectural post-mortem.

### The Student Dialogues

**Professor / TA:** *"Let's look at your training loss implementation. Open your PyTorch training loop. What did your loss function look like?"*

**Student A:** *"Here it is: `loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=pad_token_id)`. We trained on all tokens in the sequence after padding was removed!"*

**Professor / TA:** *"Look at Section 2.1 of Chapter 15. In an agent trajectory, what percentage of tokens are observations returned by the environment?"*

**Student B:** *"According to Notebook 15.1, observations represent about 77% of the total tokens in a debugging trajectory!"*

**Student A:** *"Wait... so 77% of our loss was penalizing the model for not predicting file contents and pytest logs?"*

**Professor / TA:** *"Exactly! That is the classic **Role Confusion** failure mode. Because you trained on observation tokens, your model learned that tool output immediately follows a tool call. When deployed, instead of emitting the end-of-turn token `<|end_turn|>` and waiting for the sandbox to run pytest, your model generated its own fake pytest output:
```text
MODEL: pytest tests/auth_test.py
MODEL: tests/auth_test.py::test_login PASSED
MODEL: All tests pass! Task complete!
```
It never actually invoked the tool! What should you have done with observation tokens?"*

**Student C:** *"We should have used **Observation Loss Masking**! Set $m_t = 0$ on all prompts and environment observations, and set $m_t = 1$ strictly on the model's own reasoning, tool calls, and the end-of-turn token. That way, the model uses observations as context to condition on, but only receives gradients on its own actions."*

**Professor / TA:** *"Now, what about the obsolete CLI flags? Why did your model send `--old-format` when your system prompt explicitly declared `--json-format`?"*

**Student A:** *"Well, all 5,000 training trajectories in our dataset were collected two months ago, and they all used `--old-format`."*

**Professor / TA:** *"And because every single example in your training data used `--old-format`, gradient descent found the shortest path: it baked `--old-format` directly into the feedforward weights! The model stopped attending to the schema in the prompt because the schema was completely static. What does Section 4 teach us to do during training serialization?"*

**Student B:** *"We must apply **Schema Perturbations**: argument permutation, renaming with synthetic synonyms, distractor injection, and schema dropout. By making schemas vary across training examples, the model is forced to attend to the prompt's declared contract rather than memorizing parameter names as static procedures!"*

---

## Checklist & Actionable Recommendations

### Content Quality & Systems Rigor
- [x] **Three Homes of Knowledge:** Clear, non-negotiable sorting between Weights (Procedure), Context (Contracts), and Runtime (Enforcement).
- [x] **Trajectory Serialization Contract:** 4-role message schemas, causal monotonicity, and reserved boundary tokens rigorously detailed.
- [x] **Observation Loss Masking:** Formulates masked cross-entropy, active token budgets, and role confusion prevention.
- [x] **Sequence Packing & Attention Partitioning:** FlashAttention `varlen`, `cu_seqlens`, RoPE position resets, and FLOP elimination derived.
- [x] **Exposure Bias & Regret Bounds:** Step-by-step derivation of quadratic $O(\epsilon H^2)$ behavior cloning drift vs. linear $O(\epsilon H)$ on-policy DAgger bound.
- [x] **Schema Robustness:** Parametric Recall Error and four schema perturbation operators mathematically and operationally specified.
- [x] **Trajectory DPO & Syntax Preservation:** Joint TDPO ($\mathcal{L}_{\text{DPO}} + \gamma \mathcal{L}_{\text{SFT}}$) and length-normalized implicit rewards prevent tool syntax degeneration.
- [x] **Offline Reference Logit Caching:** Detailed hardware streaming economics (halving VRAM, doubling batch size, saving 50% forward FLOPs).
- [x] **PEFT & Accelerator Memory Decomposition:** $20\Phi$ byte full SFT vs. LoRA parameter breakdown and SRAM FlashAttention tiling.
- [x] **Perplexity-Competence Inversion & Scorecards:** Multi-metric evaluation framework superseding naive validation loss.

### Actionable Polish Items
1. **Explain the Covariate Shift Trap of Masked Reasoning (`[P1]`):** Explicitly warn that retaining reasoning in context with $m_r = 0$ while deploying without reasoning causes severe teacher-forcing covariate shift, mandating true reasoning excision.
2. **Detail Quadratic Regret Derivation Step-by-Step (`[P1]`):** Show explicitly how Boole's union bound yields $(t-1)\epsilon$ error probability, leading to $\sum t\epsilon = \epsilon H(H+1)/2$.
3. **Illustrate Negative Gradient Destruction in DPO (`[P1]`):** Explain that negative updates depress valid JSON/bash syntax tokens across losing trajectories, justifying the joint TDPO objective ($\mathcal{L}_{\text{DPO}} + \gamma \mathcal{L}_{\text{SFT}}$).
4. **Annotate Behavioral Consequence of Loss Normalization (`[P2]`):** Ground why per-token normalization allows a 16k-token wandering trace to contribute $32\times$ more gradient mass than a concise 500-token fix.
5. **Ground Offline Reference Logit Caching Economics (`[P2]`):** Provide concrete VRAM numbers (saving 140 GB for 70B models) and explain the 50% forward FLOP savings.
6. **Disentangle FlashAttention from Activation Checkpointing (`[P2]`):** Clarify that FlashAttention eliminates quadratic attention matrix memory in SRAM, while activation checkpointing collapses linear projection memory in HBM.

---

## Conclusion & Transition to Chapter 16

Chapter 15 delivers an authoritative, systems-first exposition of trajectory fine-tuning and policy distillation. It establishes how to serialize multi-turn agent interactions, isolate policy gradients via observation loss masking, eliminate batch padding bubbles via block-diagonal FlashAttention packing, overcome exposure bias via on-policy data and fault injection, and prevent tool syntax destruction in preference optimization.

With the fine-tuned and distilled policy now producing disciplined, schema-compliant, and self-healing action proposals:
- The policy operates at peak efficiency within the single-agent runtime harness.
- However, imitation learning remains strictly bounded by the ceiling of existing demonstrations.

The audit now advances directly to **Chapter 16: Verifiable RL (RLVR / GRPO)**, where we explore how autonomous agents break through the demonstration ceiling by optimizing policies directly against programmatic execution oracles and group-relative policy gradients.
