# Challenge 01: Breaking the Memory Wall 🧱

## The Prompt
You are the lead architect at a startup. You need to deploy a **Llama-3-70B** model for a high-priority customer.

**Constraints:**
- **Hardware:** You only have access to **NVIDIA A100 (80GB)** nodes.
- **Latency SLA:** The Time-to-First-Token (TTFT) must be under **50ms**.
- **Budget:** You must use the **minimum number of GPUs** possible.

## The Problem
A single A100 has 80GB of HBM. A 70B model at FP16 requires 140GB just for weights.

**Your Task:** Create a `solution.yaml` that defines the hardware configuration (nodes, precision, efficiency) to meet the SLA with minimum resources.

---

## How to Solve
1.  Study **Wall 2 (Memory)** and **Wall 4 (Serving)** in the [Textbook](https://mlsysbook.ai/vol1/ml_systems.html).
2.  Create your `solution.yaml` (see template below).
3.  Run the **Judge** to see if your architecture passes.

```bash
# Template for your solution
mlsysim eval Llama3_70B A100 --batch-size 1 --precision <YOUR_CHOICE> --nodes <YOUR_CHOICE>
```

## The "Judge" (Expected Output)
If your design fails, the judge will tell you which **Wall** you hit:
- `FAIL: Wall 2 (Memory) - Weight size exceeds total HBM capacity.`
- `FAIL: Wall 4 (Serving) - Latency exceeds 50ms SLA.`
- `PASS: Design meet all constraints. Efficiency: XX%.`

---

## 🏆 Leaderboard
Can you solve this with only 2 GPUs? How about 1 GPU using 4-bit quantization?

[Submit your solution logic to the Discussions!](https://github.com/harvard-edge/cs249r_book/discussions)
