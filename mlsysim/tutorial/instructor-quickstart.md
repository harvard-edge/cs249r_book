# Instructor Quick-Start Guide

> **Get mlsysim into your ML Systems course in 15 minutes.**

---

## The 15-Minute Adoption Path

### Minute 0–3: Install and Verify

```bash
pip install mlsysim
python3 -c "import mlsysim; print(mlsysim.__version__)"
# Should print: 0.1.0
```

### Minute 3–8: Your First Live Demo

Copy this into a Jupyter cell or Python script. This IS your first lecture demo:

```python
import mlsysim

# "Is Llama-3 8B compute-bound or memory-bound on an H100?"
profile = mlsysim.Engine.solve(
    mlsysim.Models.Language.Llama3_8B,
    mlsysim.Hardware.Cloud.H100,
    batch_size=1,
)

print(f"Bottleneck: {profile.bottleneck}")  # → Memory
print(f"MFU:        {profile.mfu:.3f}")     # → 0.003 (nearly idle compute!)
print(f"Latency:    {profile.latency:.2f}") # → 5.60 ms

# Now change batch_size to 256 and watch the bottleneck shift...
```

**The teaching moment:** Students predict "Compute-bound, because GPUs are fast."
They run it and see "Memory-bound, MFU=0.3%." Their intuition breaks. You rebuild
it with the Roofline model. That's the entire first lecture.

### Minute 8–12: Your First Homework Problem

> **Problem:** The H100 has 6.3× more FLOPS than the A100 (989 vs 156 TFLOPS FP16 dense).
> Use `Engine.solve()` to compare Llama-3-8B inference at batch=1 on both.
> How much faster is H100? Why isn't it 6.3×?

**Expected answer:** ~1.7× faster. Memory-bound workload scales with bandwidth
ratio (3.35/2.04 ≈ 1.64×), not FLOPS ratio. Students learn that advertising
FLOPS means nothing for memory-bound workloads.

### Minute 12–15: Plan Your Semester

| Week | Topic | mlsysim Exercise |
|------|-------|-----------------|
| 2 | Roofline Model | Batch-size sweep: find the compute↔memory crossover |
| 4 | LLM Serving | KV cache capacity: how many concurrent requests? |
| 6 | Quantization | Compress Llama-3 to INT4: what's the fleet impact? |
| 8 | Distributed Training | Scale from 8 to 256 GPUs: find the efficiency cliff |
| 10 | TCO & Carbon | Move training to Quebec: how much carbon saved? |
| 12 | Design Challenge | $5M budget, Llama-3 70B at 1000 QPS — design the fleet |

Each exercise takes 20–30 minutes of class time. Solutions are in
[exercises.md](exercises.md).

---

## What Students Need

- Python 3.10+
- `pip install mlsysim`
- No GPU required. No cloud account. Everything runs on a laptop CPU.
- See [prerequisites.md](prerequisites.md) for detailed setup.

## What You Get

| Material | File | Description |
|----------|------|-------------|
| Tutorial slides | `slides/tutorial_part1.tex` + `tutorial_part2.tex` | 102 Beamer slides with speaker notes |
| 8 exercises | `exercises.md` | Hands-on problems with expected answers |
| Cheat sheet | `cheatsheet.md` | Single-page reference (Iron Law + key equations) |
| Pre/post quiz | `assessment/quiz.md` | 10-question assessment with distractor analysis |
| Backward design | `DESIGN.md` | Learning goals, "aha moments," facilitation notes |
| 6 SVG figures | `slides/images/svg/` | Publication-quality diagrams (Roofline, AllReduce, etc.) |

## Auto-Grading Hint

mlsysim returns typed Pydantic objects. You can auto-grade by checking:

```python
# Student submits their analysis
result = mlsysim.Engine.solve(model, hardware, batch_size=student_batch)

# Auto-grade: check the bottleneck label
assert result.bottleneck == "Memory", "Expected Memory-bound at batch=1"

# Auto-grade: check MFU is in the right range
assert 0.001 < result.mfu < 0.01, f"MFU {result.mfu:.3f} outside expected range"

# Auto-grade: check feasibility
assert result.feasible, "Model should fit on this hardware"
```

This works with Gradescope, nbgrader, or any Python-based autograder.

## The Pedagogical Framework

mlsysim is organized around the **Iron Law of ML Systems**:

```
Time = FLOPs / (N × Peak × MFU × η_scaling × Goodput)
```

Every concept in your course maps to one term in this equation. Every homework
problem is about understanding which term is the bottleneck and how to improve it.
The 22-wall taxonomy provides the vocabulary; the Iron Law provides the structure.

See [laws-explained.md](../docs/laws-explained.md) for plain-English explanations
of all 22 constraints.

## Getting Help

- **Issues:** [github.com/harvard-edge/cs249r_book/issues](https://github.com/harvard-edge/cs249r_book/issues) (use the mlsysim template)
- **Documentation:** [mlsysbook.ai/mlsysim](https://mlsysbook.ai/mlsysim)
- **Citation:** See CITATION.cff in the package root
