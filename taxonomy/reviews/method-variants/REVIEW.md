# Review: methods, variants, and outputs

Status: proposed field corrections applied through `accepted-component-aliases.json` and `accepted-method-boundaries.json`. Deferred and provisional judgments below remain open. Sources inspected November 4, 2026. `decisions.json` preserves the selected records and all incident relationships, with proposed field changes. Stable identities and homes remain unchanged. This is targeted semantic review, not an independent admission study.

| Concept | Evidence and proposed disposition |
|---|---|
| Conformal prediction | [Angelopoulos, Barber, and Bates, §3.4](https://arxiv.org/html/2411.11824v1) identifies split conformal as a special case of the broader method. Broaden the definition and qualify the calibration-split requirement. The selective-prediction edge must not imply conditional accuracy on accepted predictions or universal robustness to shift. |
| Model export | [PyTorch export API, Overview](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export/api_reference.html) describes ahead-of-time capture and dynamic-shape constraints. Remove the assumption that shapes are fixed. Defer the tracing-JIT edge: graph capture alone does not establish JIT compilation. Either provide a specific supporting workflow or remove the edge. Other export edges remain provisional. |
| Mixed-precision training | [Croci et al., abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC8905452/) defines stochastic rounding as a rounding mode. Remove it as an equivalent name for mixed-precision training. The numerical-stability, loss-scaling, memory, matrix-unit, and operating-regime edges remain provisionally valid under their stated conditions. |
| Modality encoder | [LLaVA](https://arxiv.org/html/2304.08485v2) distinguishes the vision encoder from its visual features. Remove `visual tokens`, which names a representation rather than the network. Retain its incident edges provisionally; `vision encoder` still needs explicit narrower-entry-term treatment. The definition's cost generalization also needs broader evidence before acceptance. |
| Accelerator sharing and partitioning | [NVIDIA GPU pass-through introduction](https://docs.nvidia.com/vgpu/19.0/grid-vgpu-user-guide/using-gpu-pass-through.html) distinguishes assigning an entire GPU to one VM from sharing it among VMs. Remove the unqualified passthrough aliases. Retain the sharing edges provisionally; hardware isolation claims remain conditional on dedicated resources. Remaining implementation aliases are not newly attested by this review. |
| Partitioned and quantized vector index | [Faiss method table](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) distinguishes flat product quantization from IVF-PQ. Remove `product quantization` as an equivalent name for the combined index. Qualify the memory edge without asserting inevitable monotonic recall loss. Other index edges remain provisional. |

The proposed conditions are editorial synthesis from these distinctions. No performance measurement is claimed. Retaining an edge means this targeted reading found no additional necessary change; it is not exhaustive claim-by-claim source validation.

Run from the repository root:

```bash
python3 taxonomy/reviews/mechanism-boundaries/verify.py taxonomy/reviews/method-variants/decisions.json
```

The verifier checks the baseline and review inventory only. Application requires recorded review dispositions and provenance-preserving changes, including alias-removal rationales. An empty proposed alias list means remove the listed legacy alias, not admit a replacement concept. Mutable documentation must be preserved as evidence before acceptance.
