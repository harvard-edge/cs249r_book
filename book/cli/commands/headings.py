"""
``binder headings`` — Heading-case enforcement for MIT Press style.

Enforces the H1/H2 headline-case + H3+ sentence-case policy documented in
`.claude/rules/book-prose.md` §10.3.1. Preserves ten documented exceptions:
acronyms, hyphenated-acronym compounds, digit-letter models (70B, 3D),
single-letter labels (Archetype A, N-models, S-curve), slash acronyms
(I/O, A/B, M/G/c/K), CamelCase product names (ResNet, FlashAttention),
lowercase-first API names (torch.compile, nn.Module), math spans
($B$, $\\alpha$-$\\beta$), named laws (Amdahl's Law, Young-Daly law),
and legislation (EU AI Act). Compound proper nouns (MLPerf Inference,
Oura Ring, Tensor Cores) are preserved phrase-level.

Subcommands:
    check    — Validate headings; exit 1 on any violation.
    dry-run  — Preview what apply would change across the whole book.
    apply    — Rewrite headings in place to sentence-case compliance.

This module owns the full implementation — no subprocess, no external
script. Invoked from:
    - ``./book/binder headings {check,dry-run,apply}`` (CLI)
    - ``validate.py:_run_heading_case`` (imports ``find_violations``
      and emits ``ValidationIssue`` objects for the rich check table)
    - ``.pre-commit-config.yaml`` → ``./book/binder headings check``
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ─── Rule data ───────────────────────────────────────────────────────────────

ACRONYMS = set("""
AI ML AIOps MLOps DevOps ClinAIOps MLSys
GPU GPUs CPU CPUs TPU TPUs NPU DSP ASIC FPGA SoC MCU HBM DRAM SRAM ROM EPROM NVRAM VRAM
NVMe PCIe PCI NVLink NVSwitch DMA MMIO RDMA CUDA ROCm OpenCL OpenMP MPI NCCL RCCL OpenACC
CNN CNNs RNN RNNs LSTM GRU MLP MLPs NN DNN BLAS cuBLAS cuDNN GEMM MAC MACs SIMD SIMT VLIW RISC CISC BatchNorm LayerNorm GroupNorm ReLU GELU SiLU PReLU LeakyReLU Sigmoid Softmax Tanh
FLOPs FLOP TFLOPs PFLOPs FLOPS TFLOPS PFLOPS TOPS IOPS MIPS FP32 FP16 FP8 FP4 BF16 INT8 INT4 TF32 FP64
GB TB KB MB PB EB GiB TiB KiB MiB Gbps Mbps Tbps ns μs ms kHz MHz GHz THz Hz Wh kWh MWh W kW MW
IO OS RTOS VM JVM JIT AOT RPC REST RESTful HTTP HTTPS TCP UDP IP DNS CDN TLS SSL SSH gRPC IPC UX UI
API APIs SDK CLI IDE GUI SLA SLAs SLI SLO SLOs RAID SSD HDD EEPROM SRE QA KPI
US UK EU USA FAA FDA IRB NHTSA DARPA NSF NIH IEEE ACM OSDI NSDI NeurIPS ICML ICLR CVPR ECCV ICCV ACL EMNLP NAACL KDD SIGMOD SIGCOMM SOSP FAST ATC HotOS MICRO ISCA HPCA SC
ImageNet BERT GPT PaLM T5 BART ViT CLIP ResNet AlexNet MobileNet MobileNets EfficientNet VGG Inception YOLO DLRM BLOOM Gemini Claude Mistral Qwen Kaplan Chinchilla LLaMA Llama
PyTorch TensorFlow JAX NumPy SciPy Pandas XGBoost ONNX TensorRT TVM MLIR LLVM OpenAI DeepMind Horovod DeepSpeed Megatron ZeRO FSDP DDP vLLM SGLang
AllReduce AllGather ReduceScatter AllToAll Broadcast Scatter Gather FlashAttention PagedAttention RoPE ALiBi MoE KV
MLPerf MLCommons LAPACK LINPACK D·A·M DAM C³ C^3 CCC C3
TinyML AutoML AutoAugment RandAugment NAS
NVIDIA AMD Intel ARM Apple Google Amazon Microsoft Meta Facebook IBM Qualcomm TSMC Samsung Uber Tesla DeepSeek Anthropic Cerebras Groq Graphcore SambaNova Tenstorrent Oura
MIT Stanford Berkeley CMU UIUC EPFL ETH Caltech Harvard Princeton Yale Columbia
Hennessy Patterson Amdahl Gustafson Turing Sutton Karpathy Kuhn Goodhart Horowitz Williams Waterman Knuth Tanenbaum Dean Chintala Huang LeCun Han Reddi Stoica Huyen Emer
HIPAA GDPR COPPA CCPA FERPA SOX HITRUST ISO SOC NAND NOR SLC MLC TLC QLC NaN Inf
HBM2 HBM3 HBM3e DDR4 DDR5 GDDR5 GDDR6 GDDR6X LPDDR4 LPDDR5
V100 A100 H100 H200 B100 B200 GB200 T4 L4 L40 RTX Xeon Epyc Ryzen M1 M2 M3 M4 Grace Hopper Blackwell Ampere Volta Pascal
TPUv1 TPUv2 TPUv3 TPUv4 TPUv5 TPUv5e TPUv5p
I II III IV V VI VII VIII IX X XI XII
Roofline Instinct Xavier Orin Jetson
Young-Daly Bayes
Adam AdamW SGD SGDM Adagrad RMSprop Lion Lamb Shampoo Muon
Clos Fat-Tree Dragonfly Torus Mesh Hypercube Butterfly
""".split())

COMPOUND_NAMES = {
    # Benchmark suites
    ("MLPerf", "Inference"), ("MLPerf", "Training"), ("MLPerf", "Tiny"),
    ("MLPerf", "Mobile"), ("MLPerf", "Client"), ("MLPerf", "HPC"),
    # Products
    ("Oura", "Ring"), ("Apple", "Watch"), ("Apple", "Silicon"), ("Google", "Glass"),
    ("Tensor", "Core"), ("Tensor", "Cores"),
    # Legislation shorthand
    ("EU", "Act"), ("AI", "Act"),
}

# Exact-match headings to skip entirely. For paper-title conventions that
# intentionally diverge from strict sentence case (e.g. "1-bit Adam" — "bit"
# stays lowercase per the paper, "Adam" stays capitalized as a proper noun).
SKIP_HEADINGS = {
    "1-bit Adam: Compression-aware optimization",
}

DAM_AXES = {"Data", "Algorithm", "Machine", "Computation", "Communication", "Coordination"}

# Concept terms from §10.3 that stay lowercase even at heading sentence start.
CONCEPT_TERMS_LOWER = {
    "iron law", "degradation equation", "verification gap", "bitter lesson",
    "ml node", "data wall", "compute wall", "memory wall", "power wall",
    "energy corollary", "machine learning operations", "transformer",
    "four pillars framework", "scaling laws", "information roofline",
    "long tail", "data gravity", "napkin math", "starving accelerator",
    "latency cliff", "roofline model",
}


# ─── Regexes ─────────────────────────────────────────────────────────────────

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s*\{[^}]*\})?\s*$")
TOKEN_RE = re.compile(
    r"""
    (?:[A-Za-z][A-Za-z0-9'·³²]*
       (?:[-/\.][A-Za-z0-9'·³²]+)*
    )
    | \d+[A-Za-z]*
    | :
    | [^\w\s]
    | \s+
    """,
    re.VERBOSE,
)


# ─── Public API: Violation + find_violations ─────────────────────────────────

@dataclass
class Violation:
    """A single heading-case violation found in a file."""
    path: str
    line: int
    current: str
    expected: str


def find_violations(paths: Iterable[str]) -> List[Violation]:
    """Return all heading-case violations across the given .qmd files.

    Used by ``binder check headers --scope case`` (native, no subprocess)
    and by ``binder headings check``.
    """
    result: List[Violation] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        if not path.endswith(".qmd"):
            continue
        for line_no, old_line, new_line in _process_file(path, dry_run=True):
            result.append(Violation(
                path=path, line=line_no, current=old_line, expected=new_line,
            ))
    return result


def apply_fixes(paths: Iterable[str]) -> Tuple[int, int]:
    """Apply heading-case fixes in place. Returns (files_changed, total_changes)."""
    files_changed = 0
    total_changes = 0
    for path in paths:
        if not os.path.isfile(path) or not path.endswith(".qmd"):
            continue
        changes = _process_file(path, dry_run=False)
        if changes:
            files_changed += 1
            total_changes += len(changes)
    return files_changed, total_changes


# ─── Internal: tokenization + case classification ────────────────────────────

def _is_single_letter_shape(w: str) -> bool:
    return bool(re.match(r"^[A-Z]-[a-z]", w))

def _is_lowercase_api_dotted(w: str) -> bool:
    return bool(re.match(r"^[a-z][a-z]*\.", w))

def _is_proper_whole(w: str) -> bool:
    """Strict whole-word proper-noun check (no hyphen-splitting heuristics)."""
    if w in ACRONYMS:
        return True
    # Plural: only for multi-letter acronyms (GPUs→GPU) — avoid "Is"→"I"
    if len(w) >= 3 and w.endswith("s") and w[:-1] in ACRONYMS:
        return True
    if len(w) >= 4 and w.endswith("'s") and w[:-2] in ACRONYMS:
        return True
    if _is_single_letter_shape(w):
        return True
    if _is_lowercase_api_dotted(w):
        return True
    if "/" in w and any(c.isupper() for c in w):
        return True
    # Single capital letter label (Archetype A, Case B, variable N)
    if len(w) == 1 and w.isupper() and w.isalpha():
        return True
    # Digit with letter suffix (70B, 3D, 8B)
    if re.match(r"^\d+[A-Z]+$", w):
        return True
    return False

def _is_proper_generic(w: str) -> bool:
    """CamelCase, all-caps, digit-model patterns."""
    if len(w) >= 2 and w.isupper() and w.isalpha():
        return True
    if re.match(r"^[A-Z][a-z]+[A-Z]", w):
        return True
    if re.match(r"^[A-Z][A-Za-z0-9\-]*\d", w):
        return True
    if any(c in w for c in "·") and any(c.isupper() for c in w):
        return True
    return False

def _is_wordlike(tok: str) -> bool:
    return bool(re.match(r"[A-Za-z]", tok)) and tok != ":"

def _parenthetical_axis(text: str) -> Optional[str]:
    m = re.search(r"\(([A-Z][a-z]+)\)\s*$", text)
    return m.group(1) if m and m.group(1) in DAM_AXES else None

def _case_hyphenated(w: str, is_start: bool) -> Optional[str]:
    """Apply sentence-case to each part of a hyphenated compound per §10.8."""
    parts = w.split("-")
    if len(parts) < 2:
        return None
    new_parts: List[str] = []
    for i, p in enumerate(parts):
        if not p:
            new_parts.append(p)
            continue
        preserve = False
        if p in ACRONYMS:
            preserve = True
        elif len(p) >= 3 and p.endswith("s") and p[:-1] in ACRONYMS:
            preserve = True
        elif len(p) >= 2 and p.isupper() and p.isalpha():
            preserve = True
        elif re.match(r"^[A-Z][a-z]+[A-Z]", p):
            preserve = True
        elif re.match(r"^[A-Z][A-Za-z0-9]*\d", p):
            preserve = True
        elif re.match(r"^\d+[A-Za-z]+$", p):
            preserve = True
        elif len(p) == 1 and p.isupper() and p.isalpha():
            preserve = True
        if preserve:
            new_parts.append(p)
            continue
        if i == 0 and is_start:
            new_parts.append(p[0].upper() + p[1:] if len(p) > 1 else p.upper())
        else:
            new_parts.append(p.lower())
    return "-".join(new_parts)

def _is_legislation_act(words: List[str], idx: int) -> bool:
    if idx == 0:
        return False
    prev = words[idx - 1]
    if prev in {"AI", "EU", "US", "UK", "HIPAA", "GDPR", "DMCA", "COPPA", "CCPA", "FERPA", "SOX"}:
        return True
    if idx >= 2 and words[idx - 2] in ACRONYMS and prev in ACRONYMS:
        return True
    return False

def _preprocess_math(text: str) -> str:
    # Normalize "C$^3$"-style LaTeX so the tokenizer treats it as a single word.
    text = re.sub(r"([A-Za-z])\$\^3\$", r"\1³", text)
    text = re.sub(r"([A-Za-z])\$\^2\$", r"\1²", text)
    return text

def _postprocess_math(text: str) -> str:
    text = text.replace("³", "$^3$").replace("²", "$^2$")
    return text

def _fix_sentence_case(text: str, paren_axis: Optional[str] = None) -> str:
    """Transform a heading text into sentence-case form (H3+ rule)."""
    # Stash math spans so tokenizer doesn't touch chars inside $...$
    math_spans: List[str] = []
    def stash(m):
        math_spans.append(m.group(0))
        return f"\x00MATHSPAN{len(math_spans) - 1}\x00"
    text = re.sub(r"\$[^$]+\$", stash, text)

    toks = [t for t in TOKEN_RE.findall(text) if t != ""]
    word_positions = [i for i, t in enumerate(toks) if _is_wordlike(t)]
    sentence_starts = set()
    if word_positions:
        sentence_starts.add(word_positions[0])
    for i, t in enumerate(toks):
        if t == ":":
            for j in range(i + 1, len(toks)):
                if _is_wordlike(toks[j]):
                    sentence_starts.add(j)
                    break
    words_only = [(i, t) for i, t in enumerate(toks) if _is_wordlike(t)]
    word_idx_map = {tok_idx: word_i for word_i, (tok_idx, _) in enumerate(words_only)}
    words_only_list = [t for _, t in words_only]

    out: List[str] = []
    prev_word: Optional[str] = None
    for i, t in enumerate(toks):
        if _is_wordlike(t):
            w = t
            is_start = i in sentence_starts
            wi = word_idx_map[i]
            # 1. Whole-word proper noun
            if _is_proper_whole(w):
                out.append(w); prev_word = w; continue
            # 2. Compound product/benchmark name
            if prev_word and (prev_word, w) in COMPOUND_NAMES:
                out.append(w); prev_word = w; continue
            # 3. Legislation Act/Law in proper-noun context
            if w in {"Act", "Law", "Rule", "Theorem", "Principle", "Axiom", "Directive", "Regulation"} and wi > 0:
                if prev_word and prev_word.endswith("'s") and prev_word[0].isupper():
                    out.append(w); prev_word = w; continue
                if _is_legislation_act(words_only_list, wi):
                    out.append(w); prev_word = w; continue
            # 4. Hyphenated compound §10.8
            if "-" in w and "/" not in w and "." not in w:
                hres = _case_hyphenated(w, is_start)
                if hres is not None:
                    out.append(hres); prev_word = w; continue
            # 5. Generic proper-noun patterns
            if _is_proper_generic(w):
                out.append(w); prev_word = w; continue
            # 6. Sentence start
            if is_start:
                out.append(w[0].upper() + w[1:] if len(w) > 1 else w.upper())
                prev_word = w; continue
            # 7. D·A·M axis
            if w in DAM_AXES and paren_axis == w:
                out.append(w); prev_word = w; continue
            # 8. Default lowercase
            out.append(w.lower()); prev_word = w
        else:
            out.append(t)
    result = "".join(out)
    # Restore math spans
    for i, span in enumerate(math_spans):
        result = result.replace(f"\x00MATHSPAN{i}\x00", span)
    return result

def _strip_ignore(src: str) -> set:
    """Return line numbers (1-indexed) to skip: code fences, HTML comments, YAML front matter."""
    src_no_html = re.sub(
        r"<!--.*?-->", lambda m: "\n" * (m.group(0).count("\n")), src, flags=re.DOTALL,
    )
    lines = src_no_html.split("\n")
    skip = set()
    in_fence = False
    in_yaml = False
    for i, line in enumerate(lines, 1):
        if i == 1 and line.strip() == "---":
            in_yaml = True; skip.add(i); continue
        if in_yaml:
            skip.add(i)
            if line.strip() == "---":
                in_yaml = False
            continue
        if re.match(r"^```", line):
            in_fence = not in_fence; skip.add(i); continue
        if in_fence:
            skip.add(i)
    return skip

def _process_file(path: str, dry_run: bool = True) -> List[Tuple[int, str, str]]:
    """Process a single .qmd file. Returns (line, old_line, new_line) tuples for H3+ violations."""
    with open(path) as fh:
        src = fh.read()
    skip = _strip_ignore(src)
    lines = src.split("\n")
    changes: List[Tuple[int, str, str]] = []
    for i, line in enumerate(lines, 1):
        if i in skip:
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        hashes, text, attrs = m.group(1), m.group(2).strip(), (m.group(3) or "")
        level = len(hashes)
        if level <= 2:
            continue
        if text == "Purpose":
            continue
        if text in SKIP_HEADINGS:
            continue
        axis = _parenthetical_axis(text)
        pre_text = _preprocess_math(text)
        new_text_raw = _fix_sentence_case(pre_text, axis)
        new_text = _postprocess_math(new_text_raw)
        if new_text != text:
            new_line = f"{hashes} {new_text}{attrs}"
            changes.append((i, line.rstrip(), new_line))
            if not dry_run:
                lines[i - 1] = new_line
    if changes and not dry_run:
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
    return changes


# ─── CLI: HeadingsCommand ────────────────────────────────────────────────────

class HeadingsCommand:
    """Native ``binder headings`` command group."""

    def __init__(self, config_manager, chapter_discovery):
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, args: List[str]) -> bool:
        parser = argparse.ArgumentParser(
            prog="binder headings",
            description="Heading-case enforcement (MIT Press §10.3.1)",
        )
        parser.add_argument(
            "subcommand",
            nargs="?",
            choices=["check", "dry-run", "apply"],
            help="Subcommand to run",
        )
        parser.add_argument("--path", default=None, help="File or directory")
        parser.add_argument("--vol1", action="store_true", help="Volume I only")
        parser.add_argument("--vol2", action="store_true", help="Volume II only")

        try:
            ns = parser.parse_args(args)
        except SystemExit:
            return ("-h" in args) or ("--help" in args)

        if not ns.subcommand:
            self._print_help()
            return True

        files = self._resolve_files(ns.path, ns.vol1, ns.vol2)

        if ns.subcommand == "check":
            return self._run_check(files)
        if ns.subcommand == "dry-run":
            return self._run_dry_run(files)
        if ns.subcommand == "apply":
            return self._run_apply(files)
        return False

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _print_help(self) -> None:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Subcommand", style="cyan", width=14)
        table.add_column("Description", style="white", width=60)
        table.add_row("check", "Validate headings; exit 1 on any violation")
        table.add_row("dry-run", "Preview changes across the whole book")
        table.add_row("apply", "Rewrite headings in place to compliance")
        console.print(Panel(table, title="binder headings <subcommand>", border_style="cyan"))
        console.print("[dim]Examples:[/dim]")
        console.print("  [cyan]./binder headings check[/cyan]")
        console.print("  [cyan]./binder headings dry-run[/cyan]")
        console.print("  [cyan]./binder headings apply[/cyan]")
        console.print()
        console.print("[dim]Rule reference: .claude/rules/book-prose.md §10.3.1[/dim]")
        console.print()

    # ------------------------------------------------------------------
    # File resolution
    # ------------------------------------------------------------------

    def _resolve_files(self, path: Optional[str], vol1: bool, vol2: bool) -> List[str]:
        if path:
            p = Path(path)
            if p.is_file():
                return [str(p)]
            if p.is_dir():
                return sorted(str(f) for f in p.rglob("*.qmd"))
        roots = []
        if vol1:
            roots.append("book/quarto/contents/vol1/**/*.qmd")
        if vol2:
            roots.append("book/quarto/contents/vol2/**/*.qmd")
        if not roots:
            roots = ["book/quarto/contents/**/*.qmd"]
        out: List[str] = []
        for pattern in roots:
            out.extend(glob.glob(pattern, recursive=True))
        return sorted(out)

    # ------------------------------------------------------------------
    # Subcommand runners
    # ------------------------------------------------------------------

    def _run_check(self, files: List[str]) -> bool:
        """Validate; return True on pass, False on fail. Prints diagnostic on fail."""
        violations = find_violations(files)
        if not violations:
            return True
        # Group by file
        by_file: dict = {}
        for v in violations:
            by_file.setdefault(v.path, []).append(v)
        console.print(
            f"\n[red]❌ Heading-case violations detected: {len(violations)} "
            f"heading(s) in {len(by_file)} file(s).[/red]"
        )
        console.print(
            "   The MIT Press style requires: H1/H2 headline case; H3+ sentence case."
        )
        console.print(
            "   See [cyan].claude/rules/book-prose.md §10.3.1[/cyan] for the rule + exceptions.\n"
        )
        for fpath, viols in by_file.items():
            short = fpath.replace("book/quarto/contents/", "")
            console.print(f"  [bold]{short}[/bold]")
            for v in viols:
                console.print(f"    L{v.line}")
                console.print(f"      [dim]current:[/dim]  {v.current[:140]}")
                console.print(f"      [dim]expected:[/dim] {v.expected[:140]}")
        console.print(f"\n   [cyan]Fix:[/cyan] ./book/binder headings apply")
        console.print(f"   Or manually edit the headings to match the expected form.\n")
        return False

    def _run_dry_run(self, files: List[str]) -> bool:
        violations = find_violations(files)
        if not violations:
            console.print("[green]✓ DRY RUN — no changes needed. 0 violations.[/green]")
            return True
        by_file: dict = {}
        for v in violations:
            by_file.setdefault(v.path, []).append(v)
        console.print(
            f"[yellow]DRY RUN — would change {len(violations)} heading(s) "
            f"across {len(by_file)} file(s).[/yellow]\n"
        )
        for fpath, viols in by_file.items():
            short = fpath.replace("book/quarto/contents/", "")
            console.print(f"\n[bold]== {short} ({len(viols)} changes) ==[/bold]")
            for v in viols:
                console.print(f"  L{v.line}")
                console.print(f"    [red]-[/red] {v.current[:140]}")
                console.print(f"    [green]+[/green] {v.expected[:140]}")
        console.print()
        return True  # dry-run is informational; always returns success

    def _run_apply(self, files: List[str]) -> bool:
        files_changed, total_changes = apply_fixes(files)
        console.print(
            f"[green]APPLIED — changed {total_changes} heading(s) "
            f"across {files_changed} file(s).[/green]"
        )
        return True
