# Math Audit Report: `book/quarto/contents/vol1/backmatter/appendix_assumptions.qmd`

## Checked scope

Audited system-assumption constants, napkin math, units, conversion factors, equations, numeric examples, and prose-equation consistency using direct reasoning only. No Gemini was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 65-79 and 112: the GPT-3 electricity example omits the 1024 accelerators behind the 25-day reference run.**  
  `GPT3_TRAINING_DAYS_REF` is `25 * day` in `mlsysim/core/constants.py`, with the source comment "Days on 1024 A100s." The appendix treats this as about 25 accelerator-days and computes `25 days x 24 h/day x 0.4 kW x $0.12/kWh = $28.80`. If the run used 1024 A100s for 25 days, the accelerator-day count is `25 x 1024 = 25,600`, and the electricity cost is `25 x 1024 x 24 x 0.4 x $0.12 = $29,491.20`, before PUE or other facility overhead.
  - Proposed correction: Either rename/restate the constant as wall-clock days on 1024 A100s and multiply by `1024` in `elec_cost`, or replace the prose with a true accelerator-day constant. The displayed formula should read `25 days x 1024 A100s x 24 h/day x 0.4 kW x $0.12/kWh ≈ $29,000`.

- **Line 112: total GPT-3 training work is labeled as FLOPS instead of FLOPs.**  
  `GPT3_TRAINING_OPS` has units of `flop`, i.e. total operations/work. FLOPS is a rate, `flop/second`. Saying GPT-3 training used `3.14e23 FLOPS` is dimensionally wrong and can confuse the later energy/time calculation.
  - Proposed correction: Change "roughly `{python} NapkinMath.gpt3_ops_str` FLOPS" to "roughly `{python} NapkinMath.gpt3_ops_str` FLOPs" or "floating-point operations."

### Medium Severity

- **Lines 63, 93-94, 110, and 445: accelerator capacity is converted from 80 GiB to about 86 decimal GB in the napkin example, while nearby prose says A100/H100 have 80 GB.**  
  The constants define `A100_MEM_CAPACITY = 80 * GiB` and `H100_MEM_CAPACITY = 80 * GiB`. The napkin example converts H100 capacity with `.m_as(GB)`, so the rendered value is about `85.9 GB`, rounded to `86 GB`. But the accelerator prose says "80 GB HBM2e" for A100, and public-facing accelerator capacity is normally quoted as 80 GB. The 7B training-state comparison remains true (`112 GB` exceeds both `80 GiB` and `85.9 GB`), but the appendix will appear to contradict itself.
  - Proposed correction: Keep capacity in GiB in the napkin example (`.m_as(GiB)` and label `GiB`), or change the prose to explicitly say "80 GiB (about 86 decimal GB)." Do not render the same capacity as both 80 GB and 86 GB without explaining the binary/decimal conversion.

- **Lines 557-558, 566, and 684: the claimed 200x register-to-DRAM energy gap does not follow from the listed constants.**  
  The constants are `ENERGY_REG_PJ = 0.01 pJ`, `ENERGY_DRAM_ACCESS_PJ = 640 pJ`, and `ENERGY_DRAM_PJ_PER_BYTE = 160 pJ/byte`. A whole DRAM access is `640 / 0.01 = 64,000x` a register access; even a per-byte DRAM estimate is `160 / 0.01 = 16,000x` per byte relative to the register constant. The stated `200x` gap is not supported by the table.
  - Proposed correction: Replace "200x gap between a register read and a DRAM access" with the ratio implied by the constants, e.g. "64,000x for a 640 pJ DRAM access versus a 0.01 pJ register access." If the intended teaching ratio is 200x, the table needs different reference energies and should state what access widths are being compared.

- **Lines 668 and 453-459: the H100 FP32 comparison cites a value that is not exposed in the H100 table.**  
  The pitfall says H100 has 990 TFLOPS in FP16 tensor operations and sixty TFLOPS in FP32 non-tensor operations, a roughly `989 / 60 ≈ 16.5x` ratio. The ratio is plausible as an approximation, but the H100 table exposes FP16, FP8, INT8, TF32, memory, capacity, and TDP only; there is no `H100_FLOPS_FP32` constant in the table or constants file. This weakens the instruction to "match the constant to the precision your workload actually uses."
  - Proposed correction: Add/expose an `H100_FLOPS_FP32` constant if this comparison should remain, or revise the prose to compare only exposed constants, such as FP16 tensor versus TF32 tensor (`989 / 494 ≈ 2.0x`). If retaining the FP32 text, say "about 60 TFLOPS" and include the corresponding constant/table row.

### Low Severity

- **Lines 11 and 25: the appendix points readers to `mlsys/constants.py`, but the imported source of truth is `mlsysim.core.constants`.**  
  This is not a numerical error, but it affects auditability: the code imports from `mlsysim.core.constants`, and the repository file is `mlsysim/mlsysim/core/constants.py`. A reader following the prose path may not find the constants being rendered.
  - Proposed correction: Replace `mlsys/constants.py` with `mlsysim/core/constants.py` or use the exact import path `mlsysim.core.constants` consistently.

- **Line 544: "four orders of magnitude" is ambiguous for a table mixing parameters, FLOPs, and accelerator-days.**  
  The table includes MobileNetV2 parameters/FLOPs, GPT-3 parameters/FLOPs/days, and GPT-4 accelerator-days. Comparing across these different dimensions is not meaningful. Even considering parameter counts alone, MobileNetV2 (`3.5e6`) to GPT-3 (`175e9`) spans about `5.0e4`, or nearly five orders of magnitude, not four.
  - Proposed correction: Either remove the exact order-count claim, or specify the dimension being compared. For parameter counts, "nearly five orders of magnitude" is closer for MobileNetV2 to GPT-3.

- **Line 537: the table label `GPT4_TRAINING_ACCELERATOR_DAYS` maps to `GPT4_TRAINING_GPU_DAYS`.**  
  The broader term "accelerator-days" may be intentional, but the Python identifier being rendered is `GPT4_TRAINING_GPU_DAYS`. This creates a traceability mismatch in an appendix whose purpose is to map table entries back to constants.
  - Proposed correction: Use the exact constant name in the table, or rename the Python constant and all references to the broader `GPT4_TRAINING_ACCELERATOR_DAYS`.

## Verified Correct

- Lines 75-76 and 108: the H100 ridge-point example is arithmetically consistent. `989 TFLOP/s / 3.35 TB/s = 295.2 FLOP/byte`, which rounds to 295 FLOP/byte.
- Lines 69, 76, and 108: the square GEMM arithmetic-intensity example is internally consistent under the usual simplified FP16 traffic model: `n / 3 = 4096 / 3 ≈ 1365 FLOP/byte`.
- Lines 70-77 and 110: the 7B mixed-precision Adam state estimate is internally consistent: `7e9 parameters x 16 bytes = 112e9 bytes = 112 GB` decimal.
- Lines 71 and 110: the 16 bytes/parameter decomposition is correct for BF16 weights (2), BF16 gradients (2), FP32 master weights (4), FP32 momentum (4), and FP32 variance (4).
- Lines 664 and 683: the qualitative MFU warning is mathematically consistent: using 30-50 percent of peak makes peak-only estimates roughly `1 / 0.5 = 2x` to `1 / 0.3 ≈ 3.3x` too optimistic.
