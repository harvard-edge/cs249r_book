#!/usr/bin/env python3
"""QMD heading audit. Produces categorized report."""
import re, glob, json

SMALL_WORDS = {"a","an","and","as","at","but","by","en","for","from","if","in","nor","of","on","or","per","so","the","to","up","via","vs","vs.","with","yet"}

# Acronyms and canonical proper nouns from §10.3 / §10.9
ACRONYMS = set("""
AI ML AIOps MLOps DevOps ClinAIOps MLSys
GPU GPUs CPU CPUs TPU TPUs NPU DSP ASIC FPGA SoC MCU HBM DRAM SRAM ROM EPROM NVRAM VRAM
NVMe PCIe PCI NVLink NVSwitch DMA MMIO RDMA CUDA ROCm OpenCL OpenMP MPI NCCL RCCL OpenACC
CNN CNNs RNN RNNs LSTM GRU MLP MLPs NN DNN BLAS cuBLAS cuDNN GEMM MAC MACs SIMD SIMT VLIW RISC CISC
FLOPs FLOP TFLOPs PFLOPs FLOPS TFLOPS PFLOPS TOPS IOPS MIPS FP32 FP16 FP8 FP4 BF16 INT8 INT4 TF32 FP64
GB TB KB MB PB EB GiB TiB KiB MiB Gbps Mbps Tbps ns μs ms kHz MHz GHz THz Hz Wh kWh MWh W kW MW
I/O IO OS RTOS VM JVM JIT AOT RPC REST RESTful HTTP HTTPS TCP UDP IP DNS CDN TLS SSL SSH gRPC IPC UX UI
API APIs SDK CLI IDE GUI SLA SLAs SLI SLO SLOs RAID SSD HDD EEPROM CI/CD CI CD SRE QA KPI
US UK EU USA FAA FDA IRB NHTSA DARPA NSF NIH IEEE ACM OSDI NSDI NeurIPS ICML ICLR CVPR ECCV ICCV ACL EMNLP NAACL KDD SIGMOD SIGCOMM SOSP FAST ATC HotOS MICRO ISCA HPCA SC
ImageNet BERT GPT GPT-2 GPT-3 GPT-4 GPT-5 GPT-4o LLaMA Llama Llama-3 PaLM T5 BART ViT CLIP DALL-E ResNet AlexNet MobileNet MobileNets EfficientNet VGG Inception YOLO DLRM BLOOM Gemini Claude Mistral Qwen Kaplan Chinchilla
PyTorch TensorFlow JAX NumPy SciPy Pandas XGBoost ONNX TensorRT TVM MLIR LLVM OpenAI DeepMind Horovod DeepSpeed Megatron ZeRO FSDP DDP vLLM SGLang
AllReduce AllGather ReduceScatter AllToAll Broadcast Scatter Gather Ring-AllReduce FlashAttention PagedAttention RoPE ALiBi MoE KV
MLPerf MLCommons LAPACK LINPACK D·A·M D-A-M DAM C³ C^3 CCC C3
TinyML AutoML AutoAugment RandAugment NAS
NVIDIA AMD Intel ARM Apple Google Amazon Microsoft Meta Facebook IBM Qualcomm TSMC Samsung Uber Tesla DeepSeek Anthropic Cerebras Groq Graphcore SambaNova Tenstorrent
MIT Stanford Berkeley CMU UIUC EPFL ETH Caltech Harvard Princeton Yale Columbia
Hennessy Patterson Amdahl Gustafson Turing Sutton Karpathy Kuhn Goodhart Horowitz Williams Waterman Knuth Tanenbaum Dean Chintala Huang LeCun Han Reddi Stoica Huyen Emer
HIPAA GDPR COPPA CCPA FERPA SOX HITRUST PCI-DSS ISO SOC SOC2 NAND NOR SLC MLC TLC QLC NaN Inf
HBM2 HBM3 HBM3e DDR4 DDR5 GDDR5 GDDR6 GDDR6X LPDDR4 LPDDR5
V100 A100 H100 H200 B100 B200 GB200 T4 L4 L40 RTX Xeon Epyc Ryzen M1 M2 M3 M4 Grace Hopper Blackwell Ampere Volta Pascal
TPUv1 TPUv2 TPUv3 TPUv4 TPUv5 TPUv5e TPUv5p
I II III IV V VI VII VIII IX X XI XII
Roofline Instinct Xavier Orin Jetson
""".split())

# D·A·M and C³ axis labels that stay capitalized per §10.9
DAM_AXES = {"Data","Algorithm","Machine","Computation","Communication","Coordination"}

# Proper-noun-preserved "Law" phrases per §10.9
NAMED_LAWS = {"Amdahl's Law","Gustafson's Law","Little's Law","Moore's Law","Dennard's Law","Brooks's Law","Conway's Law","Metcalfe's Law","Dennard","Little's"}

CONCEPT_TERMS_LOWER = {"iron law","degradation equation","verification gap","bitter lesson","ml node","data wall","compute wall","memory wall","power wall","energy corollary","machine learning operations","transformer","four pillars framework","scaling laws","information roofline","long tail","data gravity","napkin math","starving accelerator","latency cliff","roofline model"}

def strip(src):
    src = re.sub(r"<!--.*?-->", "", src, flags=re.DOTALL)
    lines = src.split("\n"); out=[]; in_fence=False; in_yaml=False
    for i, line in enumerate(lines,1):
        if i==1 and line.strip()=="---":
            in_yaml=True; continue
        if in_yaml:
            if line.strip()=="---": in_yaml=False
            continue
        if re.match(r"^```", line):
            in_fence = not in_fence; continue
        if in_fence: continue
        out.append((i,line))
    return out

def parse(line):
    m = re.match(r"^(#{1,6})\s+(.*?)(?:\s*\{[^}]*\})?\s*$", line)
    if not m: return None, None
    return len(m.group(1)), m.group(2).strip()

def clean(t):
    t = re.sub(r"[*_`]", "", t)
    t = re.sub(r"\$[^$]+\$", "MATH", t)
    t = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", t)
    return t.strip()

def tokens(t): return re.findall(r"[A-Za-z][A-Za-z0-9.\-'·³²]*|\d+", t)

def proper(w):
    if w in ACRONYMS: return True
    if w.rstrip("s") in ACRONYMS: return True
    if w.endswith("'s") and w[:-2] in ACRONYMS: return True
    if len(w)>=2 and w.isupper() and w.isalpha(): return True
    if re.match(r"^[A-Z][a-z]+[A-Z]", w): return True
    if re.match(r"^[A-Z][A-Za-z0-9\-]*\d", w): return True
    if any(c in w for c in "·/") and any(c.isupper() for c in w): return True
    return False

def has_named_law_context(ws, idx):
    """Check if 'Law' at idx follows a possessive proper noun (e.g., Amdahl's Law)."""
    if idx == 0: return False
    prev = ws[idx-1]
    # Previous word ends with 's and starts with uppercase → possessive proper noun
    if prev.endswith("'s") and prev[0].isupper(): return True
    # Or is in named law set
    if prev in NAMED_LAWS: return True
    return False

def parenthetical_axis(t):
    """Detect '(Data)', '(Algorithm)', '(Machine)' etc at end of heading."""
    m = re.search(r"\(([A-Z][a-z]+)\)\s*$", t)
    return m.group(1) if m and m.group(1) in DAM_AXES else None

def audit_headline(t):
    c = clean(t); ws = tokens(c); bad=[]
    for i, w in enumerate(ws):
        if not re.match(r"[A-Za-z]", w): continue
        wl = w.lower()
        if wl in SMALL_WORDS and 0 < i < len(ws)-1: continue
        if w[0].islower() and not proper(w): bad.append(w)
    return bad

def audit_sentence(t):
    c = clean(t); ws = tokens(c)
    # Skip if heading contains a D·A·M axis in parens (axis words stay caps)
    axis = parenthetical_axis(c)
    after_colon={0}; ti=0; colon=False
    for m in re.finditer(r"([A-Za-z][A-Za-z0-9.\-'·³²]*|\d+|:)", c):
        tok=m.group(1)
        if tok==":": colon=True; continue
        if colon: after_colon.add(ti); colon=False
        ti+=1
    bad=[]
    for i, w in enumerate(ws):
        if not re.match(r"[A-Za-z]", w): continue
        if i in after_colon:
            if w[0].islower():
                ctx = " ".join(ws[i:i+3]).lower()
                if any(ctx.startswith(ct) for ct in CONCEPT_TERMS_LOWER): continue
                bad.append(("lower-start", w))
            continue
        # Not a first-position word — should be lowercase unless proper/acronym
        if w[0].isupper() and not proper(w):
            # Exception: D·A·M axis in parens
            if w in DAM_AXES and axis == w: continue
            # Exception: named-law pattern ("Amdahl's Law")
            if w in {"Law","Rule","Theorem","Principle","Axiom"} and has_named_law_context(ws, i): continue
            # Exception: second part of "X-and-Y" names stays uppercased
            bad.append(("upper-mid", w))
    return bad

def run():
    # Cover every .qmd under book/quarto/contents/: vol1, vol2, frontmatter, backmatter
    files = sorted(glob.glob("book/quarto/contents/**/*.qmd", recursive=True))
    all_f=[]
    for f in files:
        try: src = open(f).read()
        except: continue
        for i, line in strip(src):
            lvl, txt = parse(line)
            if not lvl or not txt: continue
            if txt in {"Purpose"}: continue
            if lvl <= 2: issues = audit_headline(txt)
            else: issues = audit_sentence(txt)
            if issues: all_f.append((lvl,f,i,line.rstrip(),issues))
    by = {l: [x for x in all_f if x[0]==l] for l in range(1,7)}
    summary = {f"H{l}": len(by[l]) for l in range(1,7)}
    return by, summary

by, summary = run()
print(json.dumps(summary, indent=2))
print()
# Show H3/H4 findings with file counts
from collections import Counter
for l in (3,4,5):
    if not by[l]: continue
    c = Counter(x[1] for x in by[l])
    print(f"\n## H{l}: {len(by[l])} findings across {len(c)} files. Top-10 files:")
    for f, n in c.most_common(10):
        print(f"  {n:3d}  {f.replace('book/quarto/contents/','')}")

# Save full report
with open("/tmp/heading_audit_full.txt","w") as fh:
    for l in range(1,7):
        if not by[l]: continue
        fh.write(f"\n\n# H{l} findings ({len(by[l])})\n")
        for lvl,f,i,ln,iss in by[l]:
            fh.write(f"{f.replace('book/quarto/contents/','')}:{i}\n")
            fh.write(f"  {ln[:160]}\n")
            fh.write(f"  -> {iss[:5]}\n")
print("\nFull report: /tmp/heading_audit_full.txt")
