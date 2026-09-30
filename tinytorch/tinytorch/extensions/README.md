# TinyTorch Extensions Ecosystem

Modular, production-grade extensions that connect the 20 core TinyTorch modules to parameter-efficient fine-tuning, activation memory optimization, numerical precision dynamics, computational graph compilation, and bare-metal silicon acceleration.

Unlike the core modules (`tinytorch/core/` and `tinytorch/perf/`), which are exported from educational notebooks via `tito dev export`, extensions in `tinytorch/extensions/` are **hand-written, isolated, and tracked directly in git**.

Every extension adheres to a shared architectural contract: it falls back gracefully to standard NumPy when specialized compilers, C++ runtimes, or hardware accelerators are missing. As a result, importing `tinytorch.extensions` **never fails on any machine**.

---

## 1. The Four Canonical Extension Points

Every optimization in modern machine learning maps to one of four fundamental system boundaries. TinyTorch models each boundary with a clean, native Python abstraction:

| Boundary | Extension Class / Protocol | Physical System Bottleneck | Example Implementations |
|:---|:---|:---|:---|
| **1. Autograd Boundary** | Subclass `Function`<br>(`forward` / `backward`) | Graph node allocations & activation memory retention | `checkpoint.py`<br>`template.py` (`CustomScaledShift`) |
| **2. Architectural Boundary** | Subclass `Layer`<br>(`parameters`, weight freezing) | Parameter storage & Adam optimizer memory wall ($16\times$ weight bytes) | `lora.py` (`LoRALinear`)<br>`template.py` (`CustomResidualBlock`) |
| **3. Training Dynamics** | Parameter gradient hooks / `Optimizer` wrapper | Float16 numerical underflow & gradient stability | `loss_scaler.py` (`LossScaler`)<br>`template.py` (`CustomGradientTransform`) |
| **4. Silicon Boundary (FFI)** | `ctypes` C-ABI, Triton, Metal Performance Shaders | DRAM bandwidth memory wall & interpreter latency | `simd_ops.py`, `cpp_simd_gemm.cpp`<br>`mps_ops.py`, `triton_gelu.py`, `compile.py`<br>`template.py` (`custom_accelerated_op`) |

---

## 2. The Extension Contract

To ensure that extensions integrate seamlessly with the TinyTorch runtime, all extensions must satisfy the four-part **Extension Contract**:

1. **Systems Framing:** Clearly document the concrete physical resource bottleneck (e.g., DRAM bandwidth, cache line thrashing, float16 exponent underflow, or memory capacity) that the extension addresses.
2. **Tensor & NumPy Interoperability:** Accept either a TinyTorch `Tensor` or a NumPy `ndarray`. If passed a `Tensor`, return a `Tensor` (preserving the computational graph if differentiable). If passed a NumPy array, return an `ndarray`.
3. **Autograd Transparency:** Any operation participating in automatic differentiation must provide explicit forward and backward transformations that maintain exact gradient shape contracts.
4. **Graceful Defensive Fallback:** If native compilers (`clang++`, `g++`), multithreading runtimes (`libomp`), GPU libraries (Triton, CUDA), or Apple Silicon MPS are unavailable, fall back transparently to pure-Python or NumPy reference implementations without raising an exception on import.

---

## 3. How to Build the Next Extension in 5 Steps

TinyTorch includes a canonical boilerplate file at `tinytorch/extensions/template.py` providing working implementations for all four boundaries.

### Step 1: Identify Your Extension Boundary
Determine whether your optimization introduces:
- Custom autograd math or activation recomputation &rarr; **Subclass `Function`**
- Parameter freezing or new layer architectures &rarr; **Subclass `Layer`**
- Custom gradient scaling, clipping, or optimizer steps &rarr; **Wrap `Optimizer` / transform gradients**
- Bare-metal C++, CUDA, Triton, or Apple MPS acceleration &rarr; **Use C-ABI `ctypes` FFI or GPU runtime**

### Step 2: Copy the Starter Template
```bash
cp tinytorch/extensions/template.py tinytorch/extensions/my_extension.py
```

### Step 3: Implement the Logic and Defensive Fallback
Write your custom logic inside `my_extension.py`. Implement a feature-detection helper (`has_my_feature() -> bool`) and provide a clean fallback path so standard CPU environments run without errors:
```python
def my_accelerated_op(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    is_tensor = isinstance(x, Tensor)
    x_data = x.data if is_tensor else np.asarray(x)

    if has_custom_hardware():
        out_data = run_native_kernel(x_data)
    else:
        out_data = cpu_reference_kernel(x_data)

    return Tensor(out_data) if is_tensor else out_data
```

### Step 4: Write Unit Tests with Parity Verification
Create a matching test file in `tinytorch/tests/extensions/`:
```bash
cp tinytorch/tests/extensions/test_template.py tinytorch/tests/extensions/test_my_extension.py
```
Verify that:
1. Inputs as `Tensor` return `Tensor` with matching shapes and correct gradients.
2. Inputs as `np.ndarray` return `np.ndarray`.
3. Hardware/native outputs numerically match the pure NumPy reference within float32 tolerance (`np.allclose(out, ref, atol=1e-5)`).
4. When hardware is unavailable, fallback executes cleanly without throwing an exception.

Run your new test suite:
```bash
pytest tinytorch/tests/extensions/test_my_extension.py -v
```

### Step 5: Export Public Symbols
Register your public classes and functions in `tinytorch/extensions/__init__.py`:
```python
from .my_extension import MyExtensionClass, my_accelerated_op

__all__ = [
    ...,
    "MyExtensionClass",
    "my_accelerated_op",
]
```

---

## 4. Full Extension Catalog

### Systems & Modeling Extensions

- **`lora.py` (`LoRALinear`)**
  - *Bottleneck:* The Adam optimizer memory wall during full fine-tuning ($16\text{ bytes per parameter}$ for weights, gradients, momentum, and variance).
  - *Mechanism:* Freezes pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ (`requires_grad=False`) and injects low-rank trainable factors $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ ($r \ll d$).
  - *Impact:* Slashes optimizer memory footprint by over $98\%$ on transformer projection layers.

- **`checkpoint.py` (`checkpoint`)**
  - *Bottleneck:* Intermediate activation memory retention during the forward pass scaling linearly $O(L)$ with model depth.
  - *Mechanism:* Subclasses `Function` to discard intermediate activations during forward computation, storing only input boundary tensors and recomputing activations on-the-fly during the backward pass.
  - *Impact:* Converts $O(L)$ activation memory scaling into sublinear $O(\sqrt{L})$ with uniform segment partitioning, at the cost of a modest $33\%$ computational re-evaluation overhead.

- **`loss_scaler.py` (`LossScaler`)**
  - *Bottleneck:* IEEE 754 float16 exponent underflow during backpropagation (gradients smaller than $5.96 \times 10^{-8}$ underflow to zero).
  - *Mechanism:* Multiplies forward loss by dynamic scale factor $S$ ($2^{16}$ default) to shift small gradients into representable float16 range, checks for non-finite values (`inf`/`nan`), unscales gradients prior to optimizer stepping, and dynamically halves or doubles $S$.
  - *Impact:* Prevents gradient underflow across four orders of magnitude during mixed-precision training.

- **`compile.py` (`compile_graph`)**
  - *Bottleneck:* Intermediate tensor materialization in DRAM during eager execution of elementwise operator chains.
  - *Mechanism:* Traces eager AST operations, builds a symbolic compute graph, and fuses sequential elementwise operations (`add`, `mul`, `relu`, `gelu`) into a single unified Python or compiled loop.
  - *Impact:* Eliminates redundant DRAM round-trips, slashing memory bandwidth traffic by $40\%$.

### Hardware Acceleration Kernels

- **`cpp_simd_gemm.cpp` & `simd_ops.py` (`simd_matmul`, `simd_fused_bias_gelu`)**
  - *Bottleneck:* Python bytecode interpreter overhead ($0.12\text{ GFLOP/s}$) and CPU cache-line thrashing.
  - *Mechanism:* Bypasses the Python interpreter via raw `ctypes` pointer passing (`.data.ctypes.data_as`), organizes computation into $64 \times 64$ cache-blocked tiles, orders loops in $i, k, j$ sequence for contiguous memory access and compiler vectorization (AVX2 on x86, ARM NEON on Apple Silicon), and multithreads across cores using OpenMP.
  - *Impact:* Delivers up to a $170\times$ speedup over interpreted Python and an $8.6\times$ speedup over unfused activations.

- **`mps_ops.py` (`mps_matmul`)**
  - *Bottleneck:* CPU core saturation on large matrix products ($n \ge 1024$).
  - *Mechanism:* Offloads large matrix multiplications to Apple Silicon GPU cores via PyTorch Metal Performance Shaders (MPS).
  - *Impact:* Delivers up to $6\times$ higher throughput than CPU execution on $4096 \times 4096$ matrices, accounting for unified memory transfer latency.

- **`triton_gelu.py` (`triton_fused_gelu`)**
  - *Bottleneck:* GPU global memory bandwidth saturation from unfused bias addition and GELU activation.
  - *Mechanism:* Compiles an OpenAI Triton GPU kernel that loads inputs into on-chip SRAM/registers, computes bias addition and the Hendrycks GELU polynomial in registers, and writes back the final result in a single memory pass.
  - *Impact:* Eliminates intermediate DRAM writes on NVIDIA hardware accelerators.

---

## 5. Next-Level Extension Blueprints

Looking to contribute the next major capability to TinyTorch? Here are four high-impact architectural blueprints:

### Blueprint 1: FlashAttention (SRAM-Aware Online Softmax Tiling)
- *Systems Bottleneck:* Standard attention materializes an $S \times S$ score matrix in DRAM ($O(S^2)$ memory), causing catastrophic memory bandwidth stalls for long sequence lengths ($S \ge 2048$).
- *Implementation Strategy:*
  1. Subclass `Function` in `tinytorch/extensions/flash_attention.py`.
  2. Partition $Q, K, V$ into blocks ($B_r \times d$ and $B_c \times d$) that fit within CPU L1/L2 cache (or GPU shared SRAM).
  3. Implement online softmax: track running row-maximum scalars $m_i$ and running sum scalars $l_i$, updating the unnormalized output accumulator $O_i$ incrementally.
  4. Perform final normalization: $O_i^{\text{final}} = \text{diag}(l_i)^{-1} O_i$.
  5. In backward pass, recompute attention weights on-the-fly from input tiles.

### Blueprint 2: ScaleSim Systolic Array Tracing
- *Systems Bottleneck:* Profiling on physical hardware measures latency, but cannot answer architectural co-design questions (e.g., systolic array utilization, optimal SRAM buffer sizing, memory bandwidth stalls).
- *Implementation Strategy:*
  1. Author a `SimulatedLinear` layer subclassing `Layer` in `tinytorch/extensions/scalesim_tracer.py`.
  2. Intercept GEMM calls $(M, K, N)$ and emit cycle-accurate read/write address request traces.
  3. Export a ScaleSim-compatible workload topology file configured for Weight Stationary (`WS`) or Output Stationary (`OS`) dataflows.
  4. Invoke ScaleSim via Python `subprocess` and analyze PE array utilization and DRAM bandwidth stalls across TinyGPT model configurations.

### Blueprint 3: Muon / Matrix Orthogonalization Optimizer
- *Systems Bottleneck:* First- and second-order diagonal optimizers (AdamW) treat all matrix dimensions independently, failing to account for 2D spectral curvature during training.
- *Implementation Strategy:*
  1. Wrap `Optimizer` in `tinytorch/extensions/muon.py`.
  2. For 2D weight matrices, replace diagonal coordinate updates with polar decomposition / Newton-Schulz matrix iterations:
     $$X_{k+1} = \frac{1}{2} X_k (3 I - X_k^T X_k)$$
  3. Precondition the update matrix to enforce spectral normalization before applying the step.

### Blueprint 4: INT4 / FP4 Quantization Engine
- *Systems Bottleneck:* Linear layer weights consume $4\text{ bytes per parameter}$ in float32. Module 15 introduced uniform INT8 ($1\text{ byte}$); edge devices demand sub-byte precision ($0.5\text{ bytes}$).
- *Implementation Strategy:*
  1. Implement `Int4Linear` subclassing `Layer` in `tinytorch/extensions/int4.py`.
  2. Pack two 4-bit nibbles into a single `uint8` byte buffer.
  3. Implement per-channel affine scale and zero-point parameters.
  4. Author a vectorized C++ unpacking kernel in `cpp_simd_gemm.cpp` using bit-shifts (`>> 4` and `& 0x0F`) and FMA instructions to unpack and multiply directly inside CPU SIMD registers.

---

## 6. Verification and Testing

All extensions are thoroughly validated through automated test suites:

```bash
# Run all extension tests
pytest tinytorch/tests/extensions/ -v

# Run template tests
pytest tinytorch/tests/extensions/test_template.py -v

# Run hardware kernel tests (SIMD, MPS, Triton fallbacks)
pytest tinytorch/tests/extensions/test_hardware_ops.py -v

# Verify that code listings match documentation
python3 tinytorch/book/tools/listings.py --check
```
