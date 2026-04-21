#!/usr/bin/env python3
"""Apply sentence-case normalization to H3+ headings."""
import re, glob, os, sys

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
    ('MLPerf','Inference'), ('MLPerf','Training'), ('MLPerf','Tiny'),
    ('MLPerf','Mobile'), ('MLPerf','Client'), ('MLPerf','HPC'),
    ('Oura','Ring'), ('Apple','Watch'), ('Apple','Silicon'), ('Google','Glass'),
    ('Tensor','Core'), ('Tensor','Cores'),
    ('EU','Act'), ('AI','Act'),
}

# Exact-match headings to leave untouched. Use for domain-convention titles that
# don't follow strict sentence case (paper titles with domain-standard spellings).
SKIP_HEADINGS = {
    "1-bit Adam: Compression-aware optimization",  # paper title: "bit" stays lowercase, "Adam" stays capitalized
}
DAM_AXES = {"Data","Algorithm","Machine","Computation","Communication","Coordination"}
CONCEPT_TERMS_LOWER = {"iron law","degradation equation","verification gap","bitter lesson","ml node","data wall","compute wall","memory wall","power wall","energy corollary","machine learning operations","transformer","four pillars framework","scaling laws","information roofline","long tail","data gravity","napkin math","starving accelerator","latency cliff","roofline model"}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s*\{[^}]*\})?\s*$")
TOKEN_RE = re.compile(r"""
    (?:[A-Za-z][A-Za-z0-9'·³²]*
       (?:[-/\.][A-Za-z0-9'·³²]+)*
    )
    | \d+[A-Za-z]*
    | :
    | [^\w\s]
    | \s+
""", re.VERBOSE)

def is_single_letter_shape(w):
    return bool(re.match(r"^[A-Z]-[a-z]", w))

def is_lowercase_api_dotted(w):
    return bool(re.match(r"^[a-z][a-z]*\.", w))

def is_proper_whole(w):
    """Strict whole-word proper-noun check (no hyphen-splitting heuristics)."""
    if w in ACRONYMS: return True
    # Plural: only apply for multi-letter acronyms (GPUs→GPU) — avoid false match "Is"→"I"
    if len(w) >= 3 and w.endswith("s") and w[:-1] in ACRONYMS: return True
    if len(w) >= 4 and w.endswith("'s") and w[:-2] in ACRONYMS: return True
    if is_single_letter_shape(w): return True
    if is_lowercase_api_dotted(w): return True
    if "/" in w and any(c.isupper() for c in w): return True
    # Single capital letter label (Archetype A, Case B, variable N)
    if len(w)==1 and w.isupper() and w.isalpha(): return True
    # Digit with letter suffix (70B, 3D, 8B)
    if re.match(r"^\d+[A-Z]+$", w): return True
    return False

def is_proper_generic(w):
    """CamelCase, all-caps, digit-model patterns."""
    if len(w)>=2 and w.isupper() and w.isalpha(): return True
    if re.match(r"^[A-Z][a-z]+[A-Z]", w): return True
    if re.match(r"^[A-Z][A-Za-z0-9\-]*\d", w): return True
    if any(c in w for c in "·") and any(c.isupper() for c in w): return True
    return False

def is_wordlike(tok):
    return bool(re.match(r"[A-Za-z]", tok)) and tok != ":"

def parenthetical_axis(text):
    m = re.search(r"\(([A-Z][a-z]+)\)\s*$", text)
    return m.group(1) if m and m.group(1) in DAM_AXES else None

def case_hyphenated(w, is_start):
    """Apply sentence-case to each part of a hyphenated compound per §10.8.
    Preserves: acronyms, all-uppercase, CamelCase, digit-models, single capital letters."""
    parts = w.split('-')
    if len(parts) < 2:
        return None
    new_parts = []
    for i, p in enumerate(parts):
        if not p:
            new_parts.append(p); continue
        preserve = False
        if p in ACRONYMS: preserve = True
        elif len(p) >= 3 and p.endswith("s") and p[:-1] in ACRONYMS: preserve = True
        elif len(p)>=2 and p.isupper() and p.isalpha(): preserve = True
        elif re.match(r"^[A-Z][a-z]+[A-Z]", p): preserve = True
        elif re.match(r"^[A-Z][A-Za-z0-9]*\d", p): preserve = True
        elif re.match(r"^\d+[A-Za-z]+$", p): preserve = True
        elif len(p)==1 and p.isupper() and p.isalpha(): preserve = True
        if preserve:
            new_parts.append(p); continue
        if i == 0 and is_start:
            new_parts.append(p[0].upper() + p[1:] if len(p)>1 else p.upper())
        else:
            new_parts.append(p.lower())
    return "-".join(new_parts)

def is_legislation_act(words, idx):
    if idx == 0: return False
    prev = words[idx-1]
    if prev in {"AI","EU","US","UK","HIPAA","GDPR","DMCA","COPPA","CCPA","FERPA","SOX"}: return True
    if idx >= 2 and words[idx-2] in ACRONYMS and prev in ACRONYMS: return True
    return False

def preprocess_math(text):
    text = re.sub(r'([A-Za-z])\$\^3\$', r'\1³', text)
    text = re.sub(r'([A-Za-z])\$\^2\$', r'\1²', text)
    return text

def postprocess_math(text):
    text = text.replace('³', '$^3$').replace('²', '$^2$')
    return text

def fix_sentence_case(text, paren_axis=None):
    # Stash math spans so tokenizer doesn't touch chars inside $...$
    math_spans = []
    def stash(m):
        math_spans.append(m.group(0))
        return f"\x00MATHSPAN{len(math_spans)-1}\x00"
    text = re.sub(r'\$[^$]+\$', stash, text)
    toks = TOKEN_RE.findall(text)
    toks = [t for t in toks if t != ""]
    word_positions = [i for i,t in enumerate(toks) if is_wordlike(t)]
    sentence_starts = set()
    if word_positions:
        sentence_starts.add(word_positions[0])
    for i, t in enumerate(toks):
        if t == ":":
            for j in range(i+1, len(toks)):
                if is_wordlike(toks[j]):
                    sentence_starts.add(j); break
    words_only = [(i,t) for i,t in enumerate(toks) if is_wordlike(t)]
    word_idx_map = {tok_idx: word_i for word_i, (tok_idx, _) in enumerate(words_only)}
    words_only_list = [t for _,t in words_only]

    out = []
    prev_word = None
    for i, t in enumerate(toks):
        if is_wordlike(t):
            w = t
            is_start = i in sentence_starts
            wi = word_idx_map[i]
            # 1. Whole-word proper noun
            if is_proper_whole(w):
                out.append(w); prev_word = w; continue
            # 2. Compound product/benchmark name
            if prev_word and (prev_word, w) in COMPOUND_NAMES:
                out.append(w); prev_word = w; continue
            # 3. Legislation Act/Law in proper-noun context
            if w in {"Act","Law","Rule","Theorem","Principle","Axiom","Directive","Regulation"} and wi > 0:
                if prev_word and prev_word.endswith("'s") and prev_word[0].isupper():
                    out.append(w); prev_word = w; continue
                if is_legislation_act(words_only_list, wi):
                    out.append(w); prev_word = w; continue
            # 4. Hyphenated compound: §10.8
            if '-' in w and '/' not in w and '.' not in w:
                hres = case_hyphenated(w, is_start)
                if hres is not None:
                    out.append(hres); prev_word = w; continue
            # 5. Generic proper-noun patterns
            if is_proper_generic(w):
                out.append(w); prev_word = w; continue
            # 6. Sentence start
            if is_start:
                out.append(w[0].upper() + w[1:] if len(w)>1 else w.upper())
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

def strip_ignore(src):
    src_no_html = re.sub(r"<!--.*?-->", lambda m: "\n"*(m.group(0).count("\n")), src, flags=re.DOTALL)
    lines = src_no_html.split("\n")
    skip = set()
    in_fence = False; in_yaml = False
    for i, line in enumerate(lines, 1):
        if i==1 and line.strip()=="---":
            in_yaml=True; skip.add(i); continue
        if in_yaml:
            skip.add(i)
            if line.strip()=="---": in_yaml=False
            continue
        if re.match(r"^```", line):
            in_fence = not in_fence; skip.add(i); continue
        if in_fence: skip.add(i)
    return skip

def process_file(path, dry_run=True):
    src = open(path).read()
    skip = strip_ignore(src)
    lines = src.split("\n")
    changes = []
    for i, line in enumerate(lines, 1):
        if i in skip: continue
        m = HEADING_RE.match(line)
        if not m: continue
        hashes, text, attrs = m.group(1), m.group(2).strip(), (m.group(3) or "")
        level = len(hashes)
        if level <= 2: continue
        if text == "Purpose": continue
        if text in SKIP_HEADINGS: continue
        axis = parenthetical_axis(text)
        pre_text = preprocess_math(text)
        new_text_raw = fix_sentence_case(pre_text, axis)
        new_text = postprocess_math(new_text_raw)
        if new_text != text:
            new_line = f"{hashes} {new_text}{attrs}"
            changes.append((i, line, new_line))
            if not dry_run:
                lines[i-1] = new_line
    if changes and not dry_run:
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
    return changes

HELP = """Usage:
  python3 book/tools/audit/heading_fix.py [MODE] [FILES...]

Modes:
  dry-run         (default)  Preview changes across the whole book; exit 0.
  apply                       Apply changes in place; exit 0.
  check [FILES]               Pre-commit mode. If FILES given, only check those
                              files (typically staged .qmd files from pre-commit).
                              Exit 1 if any heading would be changed, 0 otherwise.
                              Prints the offending headings with suggested fixes.

Examples:
  python3 book/tools/audit/heading_fix.py dry-run
  python3 book/tools/audit/heading_fix.py apply
  python3 book/tools/audit/heading_fix.py check book/quarto/contents/vol1/introduction/introduction.qmd
"""

def main():
    args = sys.argv[1:]
    mode = args[0] if args else "dry-run"
    arg_files = args[1:] if len(args) > 1 else []

    if mode not in {"dry-run", "apply", "check", "-h", "--help"}:
        print(HELP, file=sys.stderr)
        sys.exit(2)
    if mode in {"-h", "--help"}:
        print(HELP); sys.exit(0)

    if mode == "check" and arg_files:
        # Only process files explicitly passed (pre-commit use case)
        files = [f for f in arg_files if f.endswith(".qmd") and "book/quarto/contents/" in f]
    else:
        # Cover every .qmd under book/quarto/contents/: vol1, vol2, frontmatter, backmatter
        files = sorted(glob.glob("book/quarto/contents/**/*.qmd", recursive=True))

    total = 0; per_file = []
    for f in files:
        if not os.path.isfile(f): continue
        # In check mode, never write; in apply mode, always write
        dry = (mode != "apply")
        changes = process_file(f, dry_run=dry)
        if changes:
            per_file.append((f, len(changes), changes))
            total += len(changes)

    if mode == "check":
        if total == 0:
            # Silent success for pre-commit
            sys.exit(0)
        # Report and fail
        print(f"\n❌ Heading-case violations detected: {total} heading(s) in {len(per_file)} file(s).")
        print("   The MIT Press style requires: H1/H2 headline case; H3+ sentence case.")
        print("   See .claude/rules/book-prose.md §10.3.1 for the full rule + exceptions.\n")
        for f, n, ch in per_file:
            short = f.replace('book/quarto/contents/','')
            print(f"  {short}")
            for i, old, new in ch:
                print(f"    L{i}")
                print(f"      current:  {old.strip()[:140]}")
                print(f"      expected: {new.strip()[:140]}")
        print(f"\n   Fix: python3 book/tools/audit/heading_fix.py apply")
        print(f"   Or manually edit the headings to match the expected form.\n")
        sys.exit(1)

    if mode == "dry-run":
        print(f"DRY RUN — would change {total} headings across {len(per_file)} files.\n")
        for f, n, ch in per_file:
            short = f.replace('book/quarto/contents/','')
            print(f"\n== {short} ({n} changes) ==")
            for i, old, new in ch:
                print(f"  L{i}")
                print(f"    - {old[:140]}")
                print(f"    + {new[:140]}")
    else:  # apply
        print(f"APPLIED — changed {total} headings across {len(per_file)} files.")

if __name__ == "__main__":
    main()
