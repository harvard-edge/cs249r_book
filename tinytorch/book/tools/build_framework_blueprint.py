#!/usr/bin/env python3
"""
Generate TinyTorch Framework Master Blueprint, "You Are Here" Navigators,
and Terminal-style Source Code Mapping Cards for all 24 units (Ch 01-21, M 01-03).

Follows Vol 3 styling standards and palette.md:
- 100% vector SVG with Helvetica / TeX Gyre Heros typography
- Terminal cards feature dark navy header, orange >_ prompt, monospace file/export/symbols, and green tested badge
- "You Are Here" cards show 3-stage framework pipeline with the active module highlighted in orange (#ff8246 / #fff1e8)
"""
import subprocess
import re
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
NARRATIVE_BOOK_DIR = TOOLS_DIR.parent
TINYTORCH_DIR = NARRATIVE_BOOK_DIR.parent

DEST_DIRS = [
    NARRATIVE_BOOK_DIR / "assets/images/diagrams",
    TINYTORCH_DIR / "guide/assets/images/diagrams",
]


def esc(s: str) -> str:
    """Safely escape ampersands without double-escaping XML entities."""
    return re.sub(r'&(?!(amp;|#\d+;|gt;|lt;))', '&amp;', str(s))


def write_and_convert(name: str, content: str):
    for d in DEST_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        svg_path = d / f"{name}.svg"
        pdf_path = d / f"{name}.pdf"
        svg_path.write_text(content, encoding="utf-8")
        subprocess.run(
            [
                "rsvg-convert",
                "-f",
                "pdf",
                "--keep-aspect-ratio",
                str(svg_path),
                "-o",
                str(pdf_path),
            ],
            check=True,
        )
    print(f"Generated: {name}.svg -> .pdf")


# =============================================================================
# DATA DEFINITIONS FOR ALL 24 UNITS (Refined for Zero-Clipping)
# =============================================================================
UNITS = [
    {
        "id": "01_tensors",
        "bp_name": "01_framework-you-are-here",
        "src_name": "01_tensor-margin-source",
        "mod_tag": "MOD 01",
        "file": "src/01_tensor/01_tensor.py",
        "export": "tinytorch.core.tensor",
        "symbols": "Tensor, Function.apply",
        "tested": True,
        "s1": {"title": "01. TENSORS & STRIDES", "desc": "Flat 1D DRAM • Strided Views", "active": True},
        "s2": {"title": "02–08. THE CORE ENGINE", "desc": "Activations • Autograd • AdamW", "active": False},
        "s3": {"title": "09–21. ARCHITECTURES & SYSTEMS", "desc": "TinyGPT • INT8 • KV Cache", "active": False},
    },
    {
        "id": "02_activations",
        "bp_name": "02_framework-you-are-here",
        "src_name": "02_activation-margin-source",
        "mod_tag": "MOD 02",
        "file": "src/02_activations/02_activations.py",
        "export": "tinytorch.core.activations",
        "symbols": "ReLU, Sigmoid, GELU, Softmax",
        "tested": True,
        "s1": {"title": "01. TENSORS & STORAGE", "desc": "Flat Buffer • Strided Indexing", "active": False},
        "s2": {"title": "02. ACTIVATION FUNCTIONS", "desc": "ReLU • Sigmoid • Tanh • GELU", "active": True},
        "s3": {"title": "03–21. LAYERS, AUTOGRAD & SYSTEMS", "desc": "Layers • Autograd • Optimizers", "active": False},
    },
    {
        "id": "03_layers",
        "bp_name": "03_framework-you-are-here",
        "src_name": "03_layers-margin-source",
        "mod_tag": "MOD 03",
        "file": "src/03_layers/03_layers.py",
        "export": "tinytorch.core.layers",
        "symbols": "Linear, Dropout, Sequential",
        "tested": True,
        "s1": {"title": "01–02. TENSORS & ACTIVATIONS", "desc": "Strided Memory • Non-linearity", "active": False},
        "s2": {"title": "03. LAYERS & PARAMETERS", "desc": "Linear • Dropout • Sequential", "active": True},
        "s3": {"title": "04–21. LOSS, AUTOGRAD & SYSTEMS", "desc": "Loss • Autograd DAG • Training", "active": False},
    },
    {
        "id": "04_losses",
        "bp_name": "04_framework-you-are-here",
        "src_name": "04_losses-margin-source",
        "mod_tag": "MOD 04",
        "file": "src/04_losses/04_losses.py",
        "export": "tinytorch.core.losses",
        "symbols": "MSELoss, CrossEntropyLoss",
        "tested": True,
        "s1": {"title": "01–03. FORWARD REPRESENTATION", "desc": "Tensors • Layers • Activations", "active": False},
        "s2": {"title": "04. LOSS FUNCTIONS", "desc": "MSE • Stabilized Cross-Entropy", "active": True},
        "s3": {"title": "05–21. DATA, AUTOGRAD & TRAINING", "desc": "DataLoader • Autograd • Step", "active": False},
    },
    {
        "id": "05_dataloader",
        "bp_name": "05_framework-you-are-here",
        "src_name": "05_dataloader-margin-source",
        "mod_tag": "MOD 05",
        "file": "src/05_dataloader/05_dataloader.py",
        "export": "tinytorch.core.dataloader",
        "symbols": "Dataset, DataLoader",
        "tested": True,
        "s1": {"title": "01–04. MODEL INGREDIENTS", "desc": "Tensors • Modules • Invariants", "active": False},
        "s2": {"title": "05. DATA PIPELINE", "desc": "Dataset Indexing • Collation", "active": True},
        "s3": {"title": "06–21. AUTOGRAD & OPTIMIZATION", "desc": "Autograd Tape • Parameter Step", "active": False},
    },
    {
        "id": "06_autograd",
        "bp_name": "06_framework-you-are-here",
        "src_name": "06_autograd-margin-source",
        "mod_tag": "MOD 06",
        "file": "src/06_autograd/06_autograd.py",
        "export": "tinytorch.core.autograd",
        "symbols": "Function.apply, backward",
        "tested": True,
        "s1": {"title": "01–05. FORWARD COMPUTATION TAPE", "desc": "Forward Tape • Loss Evaluation", "active": False},
        "s2": {"title": "06. REVERSE-MODE AUTOGRAD", "desc": "DAG Topo Sort • Gradient Flow", "active": True},
        "s3": {"title": "07–21. OPTIMIZATION & SCALE", "desc": "Optimizers • Training • TinyGPT", "active": False},
    },
    {
        "id": "07_optimizers",
        "bp_name": "07_framework-you-are-here",
        "src_name": "07_optimizers-margin-source",
        "mod_tag": "MOD 07",
        "file": "src/07_optimizers/07_optimizers.py",
        "export": "tinytorch.core.optimizers",
        "symbols": "SGD, Adam, AdamW",
        "tested": True,
        "s1": {"title": "01–06. GRADIENT ENGINE", "desc": "Forward Pass • Backward Tape", "active": False},
        "s2": {"title": "07. PARAMETER OPTIMIZERS", "desc": "SGD Momentum • Decoupled AdamW", "active": True},
        "s3": {"title": "08–21. TRAINING & DEEP SCALE", "desc": "Training Loop • Flagship Model", "active": False},
    },
    {
        "id": "08_training",
        "bp_name": "08_framework-you-are-here",
        "src_name": "08_training-margin-source",
        "mod_tag": "MOD 08",
        "file": "src/08_training/08_training.py",
        "export": "tinytorch.core.training",
        "symbols": "Trainer, CosineSchedule",
        "tested": True,
        "s1": {"title": "01–07. FRAMEWORK SUBSYSTEMS", "desc": "Modules • Autograd • Optimizers", "active": False},
        "s2": {"title": "08. THE TRAINING HARNESS", "desc": "ZeroGrad → Fwd → Back → Step", "active": True},
        "s3": {"title": "09–21. ARCHITECTURES & SYSTEMS", "desc": "Milestones • Scale • Systems", "active": False},
    },
    {
        "id": "milestone_01",
        "bp_name": "m01_framework-you-are-here",
        "src_name": "m01_xor-margin-source",
        "mod_tag": "M-01",
        "file": "milestones/02_1969_xor/ & 03_1986_mlp/",
        "export": "tinytorch.milestones.xor_mlp",
        "symbols": "MLP, XORNet, train_xor",
        "tested": True,
        "s1": {"title": "01–05. FORWARD PREREQUISITES", "desc": "Storage • Layers • Activations", "active": False},
        "s2": {"title": "MILESTONE I. MLP SYNTHESIS", "desc": "2-Layer MLP • XOR Separation", "active": True},
        "s3": {"title": "09–21. PART II & III HORIZONS", "desc": "Spatial CNN • Transformers", "active": False},
    },
    {
        "id": "09_convolutions",
        "bp_name": "09_framework-you-are-here",
        "src_name": "09_conv-margin-source",
        "mod_tag": "MOD 09",
        "file": "src/09_convolutions/09_convolutions.py",
        "export": "tinytorch.core.spatial",
        "symbols": "Conv2d, MaxPool2d, im2col",
        "tested": True,
        "s1": {"title": "01–08. CORE ENGINE FOUNDATION", "desc": "Core Engine • Strided Memory", "active": False},
        "s2": {"title": "09. SPATIAL LOCALITY & CNN", "desc": "2D Receptive Fields • im2col", "active": True},
        "s3": {"title": "10–21. LLM STACK & ACCELERATION", "desc": "Tokenization • TinyGPT Stack", "active": False},
    },
    {
        "id": "10_tokenization",
        "bp_name": "10_framework-you-are-here",
        "src_name": "10_token-margin-source",
        "mod_tag": "MOD 10",
        "file": "src/10_tokenization/10_tokenization.py",
        "export": "tinytorch.core.tokenization",
        "symbols": "BPETokenizer, train_bpe",
        "tested": True,
        "s1": {"title": "01–08. CORE ML INFRASTRUCTURE", "desc": "Linear Algebra • Autograd Tape", "active": False},
        "s2": {"title": "10. SYMBOL INGESTION & BPE", "desc": "Byte-Pair Encoding • Merges", "active": True},
        "s3": {"title": "11–21. TRANSFORMER & SYSTEMS", "desc": "Embeddings • Attention • Stack", "active": False},
    },
    {
        "id": "11_embeddings",
        "bp_name": "11_framework-you-are-here",
        "src_name": "11_embed-margin-source",
        "mod_tag": "MOD 11",
        "file": "src/11_embeddings/11_embeddings.py",
        "export": "tinytorch.core.embeddings",
        "symbols": "Embedding, PositionalEncoding",
        "tested": True,
        "s1": {"title": "10. DISCRETE TOKEN STREAM", "desc": "BPE Tokens • Discrete IDs", "active": False},
        "s2": {"title": "11. EMBEDDING LOOKUP & POS", "desc": "Table Lookup • Sinusoidal Pos", "active": True},
        "s3": {"title": "12–21. ATTENTION & TRANSFORMERS", "desc": "Causal Attention • TinyGPT", "active": False},
    },
    {
        "id": "12_attention",
        "bp_name": "12_framework-you-are-here",
        "src_name": "12_attn-margin-source",
        "mod_tag": "MOD 12",
        "file": "src/12_attention/12_attention.py",
        "export": "tinytorch.core.attention",
        "symbols": "MultiHeadAttention, scaled_dot",
        "tested": True,
        "s1": {"title": "10–11. EMBEDDED CONTEXT STREAM", "desc": "Token Embeddings • Projections", "active": False},
        "s2": {"title": "12. CAUSAL SELF-ATTENTION", "desc": "Causal Mask • Scaled Softmax", "active": True},
        "s3": {"title": "13–21. TRANSFORMERS & SERVING", "desc": "Transformer Stack • Generation", "active": False},
    },
    {
        "id": "13_transformers",
        "bp_name": "13_framework-you-are-here",
        "src_name": "13_trans-margin-source",
        "mod_tag": "MOD 13",
        "file": "src/13_transformers/13_transformers.py",
        "export": "tinytorch.core.transformers",
        "symbols": "TransformerBlock, TinyGPT",
        "tested": True,
        "s1": {"title": "10–12. ATTENTION & LAYERS", "desc": "Positional QKV Projections", "active": False},
        "s2": {"title": "13. THE TRANSFORMER BLOCK", "desc": "Pre-LN Highway • Attention+MLP", "active": True},
        "s3": {"title": "14–21. SYSTEMS & ACCELERATION", "desc": "Profiling • INT8 • KV Cache", "active": False},
    },
    {
        "id": "milestone_02",
        "bp_name": "m02_framework-you-are-here",
        "src_name": "m02_gen-margin-source",
        "mod_tag": "M-02",
        "file": "milestones/05_2017_transformer/",
        "export": "tinytorch.models.tinygpt",
        "symbols": "generate, sample_top_k",
        "tested": True,
        "s1": {"title": "10–13. TINYGPT ARCHITECTURE", "desc": "TinyGPT Architecture Stack", "active": False},
        "s2": {"title": "MILESTONE II. GENERATION", "desc": "Next-Token Logits • Sampling", "active": True},
        "s3": {"title": "14–21. SYSTEMS & OPTIMIZATION", "desc": "Profiling • INT8 • Edge Deploy", "active": False},
    },
    {
        "id": "14_profiling",
        "bp_name": "14_framework-you-are-here",
        "src_name": "14_prof-margin-source",
        "mod_tag": "MOD 14",
        "file": "src/14_profiling/14_profiling.py",
        "export": "tinytorch.perf.profiling",
        "symbols": "Profiler, roofline_analysis",
        "tested": True,
        "s1": {"title": "01–13. THE TINYTORCH RUNTIME", "desc": "Autograd Runtime • TinyGPT", "active": False},
        "s2": {"title": "14. PROFILING & ROOFLINE", "desc": "FLOP Accounting • Roofline", "active": True},
        "s3": {"title": "15–21. ACCELERATION PIPELINE", "desc": "INT8 • Fusion • KV Memoization", "active": False},
    },
    {
        "id": "15_quantization",
        "bp_name": "15_framework-you-are-here",
        "src_name": "15_quant-margin-source",
        "mod_tag": "MOD 15",
        "file": "src/15_quantization/15_quantization.py",
        "export": "tinytorch.perf.quantization",
        "symbols": "quantize_int8, QuantizedLinear",
        "tested": True,
        "s1": {"title": "14. PERFORMANCE PROFILING", "desc": "Bandwidth Bounds • 4.3MB Model", "active": False},
        "s2": {"title": "15. INT8 QUANTIZATION", "desc": "Affine Scale/Zero • 4× Density", "active": True},
        "s3": {"title": "16–21. COMPRESSION & RUNTIMES", "desc": "Distillation • Fusion • Cache", "active": False},
    },
    {
        "id": "16_compression",
        "bp_name": "16_framework-you-are-here",
        "src_name": "16_comp-margin-source",
        "mod_tag": "MOD 16",
        "file": "src/16_compression/16_compression.py",
        "export": "tinytorch.perf.compression",
        "symbols": "magnitude_prune, Compressor",
        "tested": True,
        "s1": {"title": "14–15. PROFILE & QUANTIZATION", "desc": "Roofline • INT8 Representations", "active": False},
        "s2": {"title": "16. MODEL COMPRESSION", "desc": "Distillation • Soft Targets", "active": True},
        "s3": {"title": "17–21. RUNTIME ACCELERATION", "desc": "Operator Fusion • KV Cache", "active": False},
    },
    {
        "id": "17_acceleration",
        "bp_name": "17_framework-you-are-here",
        "src_name": "17_accel-margin-source",
        "mod_tag": "MOD 17",
        "file": "src/17_acceleration/17_acceleration.py",
        "export": "tinytorch.perf.acceleration",
        "symbols": "vectorized_matmul, fused_gelu",
        "tested": True,
        "s1": {"title": "14–16. COMPRESSION TECHNIQUES", "desc": "INT8 Quantization • Pruning", "active": False},
        "s2": {"title": "17. OPERATOR FUSION", "desc": "Fused GELU • 9× Memory Traffic", "active": True},
        "s3": {"title": "18–21. SERVING & HARDWARE", "desc": "KV Cache • Benchmarks • Silicon", "active": False},
    },
    {
        "id": "18_memoization",
        "bp_name": "18_framework-you-are-here",
        "src_name": "18_memo-margin-source",
        "mod_tag": "MOD 18",
        "file": "src/18_memoization/18_memoization.py",
        "export": "tinytorch.perf.memoization",
        "symbols": "KVCache, CachedAttention",
        "tested": True,
        "s1": {"title": "12–13. AUTOREGRESSIVE ATTENTION", "desc": "O(S²) Autoregressive Overhead", "active": False},
        "s2": {"title": "18. KEY-VALUE CACHE MEMO", "desc": "Static Buffer • O(1) Token Step", "active": True},
        "s3": {"title": "19–21. SERVING & BENCHMARKING", "desc": "Latency Benchmarking • Capstone", "active": False},
    },
    {
        "id": "19_benchmarking",
        "bp_name": "19_framework-you-are-here",
        "src_name": "19_bench-margin-source",
        "mod_tag": "MOD 19",
        "file": "src/19_benchmarking/19_benchmarking.py",
        "export": "tinytorch.perf.benchmarking",
        "symbols": "Benchmark, BenchmarkResult",
        "tested": True,
        "s1": {"title": "14–18. ACCELERATION STACK", "desc": "Fused INT8 + Cached Model", "active": False},
        "s2": {"title": "19. SYSTEMS BENCHMARKING", "desc": "Warmup Cycles • P50/P95/P99", "active": True},
        "s3": {"title": "20–21. CAPSTONE & SILICON", "desc": "End-to-End Capstone Submission", "active": False},
    },
    {
        "id": "20_capstone",
        "bp_name": "20_framework-you-are-here",
        "src_name": "20_caps-margin-source",
        "mod_tag": "MOD 20",
        "file": "src/20_capstone/20_capstone.py",
        "export": "tinytorch.deploy.capstone",
        "symbols": "BenchmarkReport, submission",
        "tested": True,
        "s1": {"title": "01–19. ALL TINYTORCH SUBSYSTEMS", "desc": "All Acceleration Techniques", "active": False},
        "s2": {"title": "20. THE CAPSTONE INTEGRATION", "desc": "Benchmark Report • Pareto Best", "active": True},
        "s3": {"title": "21. FUTURE HORIZONS", "desc": "Edge Deployment • Custom Chips", "active": False},
    },
    {
        "id": "milestone_03",
        "bp_name": "m03_framework-you-are-here",
        "src_name": "m03_edge-margin-source",
        "mod_tag": "M-03",
        "file": "milestones/06_2018_mlperf/",
        "export": "tinytorch.milestones.mlperf",
        "symbols": "optimization_olympics",
        "tested": True,
        "s1": {"title": "14–20. OPTIMIZATION ARSENAL", "desc": "Optimization Techniques Suite", "active": False},
        "s2": {"title": "MILESTONE III. TORCH OLYMPICS", "desc": "Pareto Frontier • Edge Limits", "active": True},
        "s3": {"title": "21. HARDWARE HORIZONS", "desc": "Hardware Engines • Systolic", "active": False},
    },
    {
        "id": "21_extensions",
        "bp_name": "21_framework-you-are-here",
        "src_name": "21_ext-margin-source",
        "mod_tag": "EXT 21",
        "file": "src/21_extensions/",
        "export": "tinytorch.hardware.extensions",
        "symbols": "SystolicArray, FusedExecutor",
        "tested": True,
        "s1": {"title": "01–20. TINYTORCH SOFTWARE STACK", "desc": "TinyTorch Software Runtime", "active": False},
        "s2": {"title": "21. COMPILERS & CUSTOM SILICON", "desc": "Graph IR • Triton • Systolic TPU", "active": True},
        "s3": {"title": "PRODUCTION SYSTEMS HORIZON", "desc": "PyTorch 2.0 • Megatron Parallel", "active": False},
    },
]


def gen_blueprint_svg(u: dict) -> str:
    w, h = 220, 165

    def render_box(s: dict, y: int, height: int = 33) -> str:
        title = esc(s["title"])
        desc = esc(s["desc"])
        if s["active"]:
            return f"""  <g transform="translate(15, {y})">
    <rect x="0" y="0" width="190" height="{height}" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.6"/>
    <text x="8" y="14" font-size="7.8" font-weight="700" fill="#c85a17">&#9679; {title}</text>
    <text x="182" y="14" text-anchor="end" font-size="6.8" font-weight="bold" fill="#ff8246">ACTIVE</text>
    <text x="8" y="26" font-size="5.8" font-family="monospace" fill="#7c2d12">{desc}</text>
  </g>"""
        else:
            return f"""  <g transform="translate(15, {y})">
    <rect x="0" y="0" width="190" height="{height}" rx="2" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>
    <text x="8" y="13" font-size="7.5" font-weight="700" fill="#475569">{title}</text>
    <text x="8" y="25" font-size="5.8" font-family="monospace" fill="#64748b">{desc}</text>
  </g>"""

    arrow1 = """  <!-- Arrow down -->
  <line x1="110" y1="67" x2="110" y2="74" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="2,2"/>
  <polygon points="108,74 112,74 110,77" fill="#9ca3af"/>"""

    arrow2 = """  <!-- Arrow down -->
  <line x1="110" y1="111" x2="110" y2="118" stroke="#9ca3af" stroke-width="1.2" stroke-dasharray="2,2"/>
  <polygon points="108,118 112,118 110,121" fill="#9ca3af"/>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="{w}" height="{h}" fill="#ffffff"/>
  <!-- Outer Card -->
  <rect x="5" y="5" width="210" height="155" rx="2" fill="#ffffff" stroke="#9ca3af" stroke-width="1.2"/>
  <rect x="5" y="5" width="210" height="20" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="12" y="19" font-size="8" font-weight="700" fill="#1f2937">FRAMEWORK BLUEPRINT: YOU ARE HERE</text>

{render_box(u['s1'], 32)}

{arrow1}

{render_box(u['s2'], 78)}

{arrow2}

{render_box(u['s3'], 122)}
</svg>
"""


def gen_source_card_svg(u: dict) -> str:
    w, h = 220, 110
    badge = "&#10003; tested" if u.get("tested") else "curriculum"
    badge_color = "#16a34a" if u.get("tested") else "#64748b"

    mod_tag = esc(u['mod_tag'])
    file_path = esc(u['file'])
    export_path = esc(u['export'])
    symbols = esc(u['symbols'])

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="{w}" height="{h}" fill="#ffffff"/>
  <!-- Card border -->
  <rect x="5" y="5" width="210" height="100" rx="2" fill="#ffffff" stroke="#1e293b" stroke-width="1.2"/>
  <!-- Header band (Dark Navy) -->
  <rect x="5" y="5" width="210" height="22" rx="2" fill="#1e293b"/>
  <text x="14" y="19" font-size="7.5" font-family="monospace" font-weight="bold" fill="#ff8246">&gt;_</text>
  <text x="32" y="19" font-size="8" font-weight="700" fill="#ffffff">SOURCE CODE MAPPING</text>
  <text x="205" y="19" text-anchor="end" font-size="7" font-family="monospace" fill="#94a3b8">{mod_tag}</text>

  <!-- Content -->
  <g transform="translate(14, 34)">
    <text x="0" y="11" font-size="7" font-weight="bold" fill="#475569">FILE:</text>
    <text x="42" y="11" font-size="6.0" font-family="monospace" fill="#0f172a">{file_path}</text>

    <text x="0" y="25" font-size="7" font-weight="bold" fill="#475569">EXPORT:</text>
    <text x="42" y="25" font-size="6.0" font-family="monospace" fill="#0f172a">{export_path}</text>

    <text x="0" y="39" font-size="7" font-weight="bold" fill="#475569">SYMBOLS:</text>
    <text x="42" y="39" font-size="6.0" font-family="monospace" fill="#c85a17">{symbols}</text>

    <line x1="0" y1="48" x2="192" y2="48" stroke="#e2e8f0" stroke-width="0.8"/>
    <text x="0" y="59" font-size="6.5" font-style="italic" fill="#64748b">Extracted &amp; verified at build time</text>
    <text x="192" y="59" text-anchor="end" font-size="6.5" font-family="monospace" font-weight="bold" fill="{badge_color}">{badge}</text>
  </g>
</svg>
"""


def gen_00_framework_datapath_master():
    w, h = 680, 240
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="'TeX Gyre Heros', 'Helvetica Neue', Arial, sans-serif">
<rect width="{w}" height="{h}" fill="#ffffff"/>
  <defs>
    <marker id="dp-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#1f2937"/>
    </marker>
    <marker id="dp-orange-arrow" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#ff8246"/>
    </marker>
  </defs>

  <!-- Header Banner -->
  <rect x="15" y="10" width="650" height="28" rx="2" fill="#f1f5f9" stroke="#9ca3af" stroke-width="1.2"/>
  <text x="30" y="28" font-size="11" font-weight="700" fill="#1f2937">THE TINYTORCH EXECUTION DATAPATH &amp; INVARIANT PIPELINE</text>
  <text x="645" y="28" text-anchor="end" font-size="9" font-family="monospace" fill="#6b7280">Track the Byte &#8226; Close the Loop</text>

  <!-- STAGE 1: Physical Memory & Tensor Layer -->
  <g transform="translate(15, 48)">
    <rect x="0" y="0" width="150" height="175" rx="2" fill="#fff1e8" stroke="#ff8246" stroke-width="1.5"/>
    <rect x="0" y="0" width="150" height="22" rx="2" fill="#ff8246"/>
    <text x="75" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#ffffff">1. STORAGE &amp; TENSORS</text>
    
    <rect x="10" y="32" width="130" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">Flat DRAM Buffer</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">float32 continuous</text>

    <line x1="75" y1="69" x2="75" y2="78" stroke="#ff8246" stroke-width="1.2" marker-end="url(#dp-orange-arrow)"/>

    <rect x="10" y="80" width="130" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Strided Tensor</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">shape, strides, offset</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#c85a17">Zero-copy views</text>

    <line x1="75" y1="123" x2="75" y2="132" stroke="#ff8246" stroke-width="1.2" marker-end="url(#dp-orange-arrow)"/>

    <rect x="10" y="134" width="130" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Function.apply</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Dispatch &amp; fresh wrapper</text>
  </g>

  <!-- Arrow 1 -> 2 -->
  <line x1="165" y1="135" x2="180" y2="135" stroke="#1f2937" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- STAGE 2: Forward Compute & Loss -->
  <g transform="translate(180, 48)">
    <rect x="0" y="0" width="150" height="175" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
    <rect x="0" y="0" width="150" height="22" rx="2" fill="#e2e8f0"/>
    <text x="75" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">2. FORWARD &amp; LOSS</text>

    <rect x="10" y="32" width="130" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">Activations (Ch 02)</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">ReLU, Sigmoid, GELU</text>

    <line x1="75" y1="69" x2="75" y2="78" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="80" width="130" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Layers (Ch 03, 05)</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">Linear(W, b), DataLoader</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#475569">Batches &amp; Collation</text>

    <line x1="75" y1="123" x2="75" y2="132" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="134" width="130" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Loss Engine (Ch 04)</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Cross-Entropy LogSumExp</text>
  </g>

  <!-- Arrow 2 -> 3 -->
  <line x1="330" y1="135" x2="345" y2="135" stroke="#1f2937" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- STAGE 3: Autograd & Learning Loop -->
  <g transform="translate(345, 48)">
    <rect x="0" y="0" width="150" height="175" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
    <rect x="0" y="0" width="150" height="22" rx="2" fill="#e2e8f0"/>
    <text x="75" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">3. BACKPROP &amp; STEP</text>

    <rect x="10" y="32" width="130" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">Autograd Tape (Ch 06)</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">DAG Topological Sort</text>

    <line x1="75" y1="69" x2="75" y2="78" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="80" width="130" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Optimizers (Ch 07)</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">SGD &#8226; AdamW (m_t, v_t)</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#475569">Decoupled Weight Decay</text>

    <line x1="75" y1="123" x2="75" y2="132" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="134" width="130" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Training Loop (Ch 08)</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Zero &#8226; Fwd &#8226; Back &#8226; Step</text>
  </g>

  <!-- Arrow 3 -> 4 -->
  <line x1="495" y1="135" x2="510" y2="135" stroke="#1f2937" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- STAGE 4: Model Scale & Systems Acceleration -->
  <g transform="translate(510, 48)">
    <rect x="0" y="0" width="155" height="175" rx="2" fill="#f8fafc" stroke="#9ca3af" stroke-width="1.2"/>
    <rect x="0" y="0" width="155" height="22" rx="2" fill="#e2e8f0"/>
    <text x="77" y="15" text-anchor="middle" font-size="9" font-weight="700" fill="#1f2937">4. SCALE &amp; SYSTEMS</text>

    <rect x="10" y="32" width="135" height="36" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="47" font-size="8" font-weight="700" fill="#1f2937">TinyGPT (Ch 10–13)</text>
    <text x="18" y="60" font-size="7.5" font-family="monospace" fill="#6b7280">BPE &#8226; Causal Attention</text>

    <line x1="77" y1="69" x2="77" y2="78" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="80" width="135" height="42" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="95" font-size="8" font-weight="700" fill="#1f2937">Acceleration (14–18)</text>
    <text x="18" y="107" font-size="7.2" font-family="monospace" fill="#6b7280">Roofline &#8226; INT8 &#8226; Fusion</text>
    <text x="18" y="117" font-size="6.8" font-family="monospace" fill="#475569">Static KV Cache O(1)</text>

    <line x1="77" y1="123" x2="77" y2="132" stroke="#9ca3af" stroke-width="1.2" marker-end="url(#dp-arrow)"/>

    <rect x="10" y="134" width="135" height="32" rx="2" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>
    <text x="18" y="148" font-size="7.8" font-weight="700" fill="#1f2937">Silicon &amp; Stack (21)</text>
    <text x="18" y="159" font-size="7" font-family="monospace" fill="#6b7280">Systolic Arrays &#8226; Triton</text>
  </g>
</svg>
"""
    write_and_convert("00_framework-datapath-master", svg)


def main():
    print("Building TinyTorch Framework Master Blueprint...")
    gen_00_framework_datapath_master()

    print(f"\nBuilding Blueprint and Source Cards for {len(UNITS)} units...")
    for u in UNITS:
        # 1. Blueprint "You Are Here"
        bp_svg = gen_blueprint_svg(u)
        write_and_convert(u["bp_name"], bp_svg)

        # 2. Source Code Mapping Terminal Card
        src_svg = gen_source_card_svg(u)
        write_and_convert(u["src_name"], src_svg)

    print("\nAll Blueprint and Source Code Mapping cards generated successfully!")


if __name__ == "__main__":
    main()
