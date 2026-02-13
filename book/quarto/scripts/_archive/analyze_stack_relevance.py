#!/usr/bin/env python3
"""
Analyze chapter content to recommend mlsysstack intensity values.

Reads each chapter's QMD file, scores it against curated keyword dictionaries
for each of the 8 stack layers, and produces recommended intensity values (0-100).

Approach:
  1. Curated keyword dictionaries with specificity weights
  2. Case-insensitive phrase matching with word-boundary awareness
  3. Weighted counts normalized by chapter length
  4. Cross-chapter normalization (TF-IDF-inspired)
  5. Nonlinear scaling to produce 0-100 intensity values

Usage:
  python3 analyze_stack_relevance.py [--verbose] [--apply]

Output:
  Table comparing current vs. recommended mlsysstack values per chapter.
"""

import re
import math
import sys
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

BOOK_ROOT = Path(__file__).resolve().parent.parent / "contents" / "vol1"

# Chapters with mlsysstack calls (order matches the book)
# Each entry: (directory_name, qmd_filename)
CHAPTERS = [
    ("introduction",       "introduction.qmd"),
    ("ml_systems",         "ml_systems.qmd"),
    ("workflow",           "workflow.qmd"),
    ("data_engineering",   "data_engineering.qmd"),
    ("nn_computation",          "nn_computation.qmd"),
    ("nn_architectures",  "nn_architectures.qmd"),
    ("frameworks",         "frameworks.qmd"),
    ("training",           "training.qmd"),
    ("data_selection",     "data_selection.qmd"),
    ("optimizations",      "model_compression.qmd"),
    ("hw_acceleration",    "hw_acceleration.qmd"),
    ("benchmarking",       "benchmarking.qmd"),
    ("serving",            "serving.qmd"),
    ("ops",                "ops.qmd"),
    ("responsible_engr",   "responsible_engr.qmd"),
    ("conclusion",         "conclusion.qmd"),
]

# Stack layer order (matches \mlsysstack argument order):
#   {hw}{fw}{models}{train}{serve}{ops}{apps}{data}
LAYER_ORDER = ["hardware", "frameworks", "models", "training", "serving", "operations", "applications", "data"]

# ─────────────────────────────────────────────────────────────────────────────
# Keyword Dictionaries
# ─────────────────────────────────────────────────────────────────────────────
# Each layer has a dict of { phrase: weight }.
#   - weight 3: Highly specific to this layer (e.g., "tensor core" → hardware)
#   - weight 2: Moderately specific (e.g., "GPU" → hardware)
#   - weight 1: Tangentially related (e.g., "compute" → hardware)
#
# Multi-word phrases are matched as-is (case-insensitive).
# Single words use word-boundary matching to avoid false positives.

KEYWORDS = {
    "hardware": {
        # Highly specific (weight 3) — refer to physical silicon/chips
        "tensor core": 3, "cuda core": 3, "systolic array": 3,
        "fpga": 3, "asic": 3, "tpu": 3, "npu": 3,
        "hardware accelerat": 3,  # catches acceleration, accelerator, accelerators
        "memory hierarchy": 3, "cache line": 3, "dram": 3, "sram": 3,
        "hbm": 3, "memory bandwidth": 3, "arithmetic intensity": 3,
        "roofline": 3, "flops": 3, "tops": 3,
        "silicon": 3, "die": 2, "wafer": 2,
        "clock frequency": 3,
        "alu": 3, "fpu": 3, "simd": 3, "simt": 3,
        "pcie": 3, "nvlink": 3,
        "von neumann": 3, "dataflow architecture": 3,
        "power consumption": 3, "tdp": 3,
        "coprocessor": 3,
        # Moderately specific (weight 2)
        "gpu": 2, "cpu": 2, "processor": 2, "compute unit": 2,
        "memory wall": 2, "hardware": 2, "accelerator": 2,
        "chip": 2,
    },

    "frameworks": {
        # Highly specific (weight 3) — named frameworks and their concepts
        "pytorch": 3, "tensorflow": 3, "jax": 3, "keras": 3,
        "onnx": 3, "tensorrt": 3,
        "autograd": 3, "automatic differentiation": 3,
        "computation graph": 3, "computational graph": 3,
        "graph execution": 3, "eager execution": 3, "eager mode": 3,
        "graph mode": 3, "static graph": 3, "dynamic graph": 3,
        "torch.compile": 3, "xla": 3, "mlir": 3,
        "intermediate representation": 3,
        "operator fusion": 3, "kernel fusion": 3,
        "operator dispatch": 3,
        "tensor library": 3, "tensor framework": 3,
        "torchscript": 3, "torch.fx": 3, "functorch": 3,
        "programming model": 2,
        # Moderately specific (weight 2)
        "framework": 2, "runtime": 2,
        "abstraction layer": 2, "compiler": 2,
    },

    "models": {
        # Highly specific (weight 3) — architecture names and structural concepts
        "neural network architecture": 3, "network architecture": 3,
        "convolutional neural network": 3, "cnn": 3,
        "recurrent neural network": 3, "rnn": 3, "lstm": 3, "gru": 3,
        "transformer": 3, "attention mechanism": 3, "self-attention": 3,
        "multi-head attention": 3, "cross-attention": 3,
        "mlp": 2, "perceptron": 3, "feedforward": 3,
        "encoder-decoder": 3,
        "resnet": 3, "vgg": 3, "efficientnet": 3,
        "bert": 3, "gpt": 3, "llm": 3, "large language model": 3,
        "diffusion model": 3, "generative model": 3, "gan": 3,
        "autoencoder": 3, "variational autoencoder": 3, "vae": 3,
        "embedding table": 3, "embedding layer": 3,
        "skip connection": 3, "residual connection": 3, "residual block": 3,
        "batch normalization": 3, "layer normalization": 3,
        "activation function": 3, "relu": 3, "gelu": 3, "sigmoid": 3, "softmax": 3,
        "dropout": 3,
        "inductive bias": 3, "receptive field": 3,
        "convolution": 2, "pooling": 2, "stride": 2, "kernel size": 2,
        "sequence model": 3, "language model": 3,
        "parameter count": 2, "model size": 2,
        # Moderately specific (weight 2)
        "neural network": 2, "deep neural": 2,
        "forward pass": 2, "architecture": 2,
    },

    "training": {
        # Highly specific (weight 3) — training process and optimization
        "backpropagation": 3, "backward pass": 3, "gradient descent": 3,
        "stochastic gradient descent": 3, "sgd": 3,
        "adam optimizer": 3, "optimizer": 2, "learning rate": 3,
        "learning rate schedule": 3, "cosine schedule": 3,
        "loss function": 3, "cross-entropy loss": 3,
        "training loop": 3, "training pipeline": 3, "training run": 3,
        "epoch": 3, "batch size": 3, "mini-batch": 3,
        "distributed training": 3, "data parallel": 3, "model parallel": 3,
        "pipeline parallel": 3, "tensor parallel": 3,
        "gradient accumulation": 3, "gradient checkpoint": 3,
        "mixed precision": 3, "fp16": 3, "bf16": 3,
        "overfitting": 3, "underfitting": 3,
        "hyperparameter": 3, "hyperparameter tuning": 3,
        "fine-tuning": 3, "transfer learning": 3, "pretraining": 3,
        "curriculum learning": 3,
        "allreduce": 3, "collective communication": 3,
        "nccl": 3, "parameter server": 3,
        "weight update": 2,
        "data augmentation": 3,
        # Moderately specific (weight 2)
        "training": 2, "gradient": 2,
        "convergence": 2, "checkpointing": 2,
    },

    "serving": {
        # Highly specific (weight 3) — inference serving and deployment runtime
        "model serving": 3, "inference server": 3, "serving system": 3,
        "inference latency": 3, "inference throughput": 3,
        "batch inference": 3, "online inference": 3, "real-time inference": 3,
        "model deployment": 3, "deployment pipeline": 2,
        "edge inference": 3, "edge deployment": 3, "on-device": 3,
        "triton inference": 3, "torchserve": 3, "tf serving": 3,
        "inference optimization": 3,
        "dynamic batching": 3, "request batching": 3,
        "model loading": 3, "model warm-up": 3,
        "sla": 3, "service level": 3,
        "load balancing": 3, "request routing": 3,
        "onnx runtime": 3, "tflite": 3, "coreml": 3,
        "mobile deployment": 3, "embedded deployment": 3,
        "kv cache": 3, "speculative decoding": 3,
        "continuous batching": 3, "vllm": 3,
        # Moderately specific (weight 2)
        "serving": 2, "inference": 2, "deployment": 2,
        "api endpoint": 2, "response time": 2,
    },

    "operations": {
        # Highly specific (weight 3) — MLOps, CI/CD, monitoring pipelines
        "mlops": 3, "devops": 3, "ml operations": 3,
        "ci/cd": 3, "continuous integration": 3, "continuous delivery": 3,
        "model registry": 3, "model versioning": 3,
        "experiment tracking": 3, "mlflow": 3, "wandb": 3,
        "observability": 3, "alerting": 3,
        "data drift": 3, "model drift": 3, "concept drift": 3,
        "feature store": 3,
        "pipeline orchestration": 3, "workflow orchestration": 3,
        "airflow": 3, "kubeflow": 3, "dagster": 3,
        "kubernetes": 3, "docker": 3,
        "infrastructure as code": 3, "terraform": 3,
        "reproducibility": 3,
        "a/b testing": 3, "canary deployment": 3, "blue-green": 3,
        "rollback": 3, "model rollback": 3,
        # Moderately specific (weight 2)
        "monitoring": 2, "logging": 2, "profiling": 2,
        "governance": 2, "compliance": 2, "audit": 2,
        "orchestration": 2, "automation": 2,
        "production system": 2, "production environment": 2,
    },

    "applications": {
        # Highly specific (weight 3) — ML application domains & responsible AI
        "computer vision": 3, "natural language processing": 3, "nlp": 3,
        "speech recognition": 3, "recommender system": 3, "recommendation system": 3,
        "autonomous driving": 3, "self-driving": 3,
        "object detection": 3, "image classification": 3,
        "text generation": 3, "machine translation": 3, "sentiment analysis": 3,
        "chatbot": 3, "virtual assistant": 3,
        "real-world application": 3,
        "user-facing": 3,
        "responsible ai": 3, "fairness": 3,
        "interpretability": 3, "explainability": 3,
        "accountability": 3, "societal impact": 3,
        "carbon footprint": 3, "sustainability": 3, "environmental impact": 3,
        # Moderately specific (weight 2)
        "ethical": 2, "ethics": 2, "privacy": 2,
        "transparency": 2, "use case": 2,
        "healthcare": 2, "medical": 2, "finance": 2,
    },

    "data": {
        # Highly specific (weight 3) — data pipelines, engineering, curation
        "data pipeline": 3, "data engineering": 3, "data ingestion": 3,
        "data preprocessing": 3, "data cleaning": 3, "data validation": 3,
        "data quality": 3, "data governance": 3,
        "data lake": 3, "data warehouse": 3, "data catalog": 3,
        "training data": 3, "test data": 2,
        "data labeling": 3, "annotation": 3,
        "data selection": 3, "data curation": 3, "data sampling": 3,
        "synthetic data": 3,
        "data loader": 3, "data loading": 3, "dataloader": 3,
        "etl": 3, "extract transform load": 3,
        "data distribution": 2, "class imbalance": 3,
        "data versioning": 3, "dvc": 3,
        "tokenization": 2, "tokenizer": 2,
        "coreset": 3, "coreset selection": 3,
        # Moderately specific (weight 2)
        "dataset": 2, "data augmentation": 2,
        "preprocessing": 2,
        "corpus": 2, "benchmark dataset": 2,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

def read_chapter_text(chapter_dir: str, qmd_file: str) -> str:
    """Read a chapter's QMD file and extract prose text (strip code, YAML, LaTeX)."""
    path = BOOK_ROOT / chapter_dir / qmd_file
    if not path.exists():
        print(f"  WARNING: {path} not found", file=sys.stderr)
        return ""

    raw = path.read_text(encoding="utf-8")

    # Remove YAML frontmatter
    raw = re.sub(r'^---\n.*?\n---\n', '', raw, count=1, flags=re.DOTALL)

    # Remove Python/code cells (```{python} ... ```)
    raw = re.sub(r'```\{python\}.*?```', '', raw, flags=re.DOTALL)
    raw = re.sub(r'```\{r\}.*?```', '', raw, flags=re.DOTALL)

    # Remove other fenced code blocks
    raw = re.sub(r'```\w*\n.*?```', '', raw, flags=re.DOTALL)

    # Remove LaTeX blocks (\begin{...} ... \end{...})
    raw = re.sub(r'\\begin\{(?:equation|align|gather|tikzpicture|marginfigure|figure)\*?\}.*?\\end\{(?:equation|align|gather|tikzpicture|marginfigure|figure)\*?\}', '', raw, flags=re.DOTALL)

    # Remove inline LaTeX commands but keep text arguments
    raw = re.sub(r'\\(?:text|textbf|textit|emph)\{([^}]*)\}', r'\1', raw)

    # Remove remaining LaTeX commands
    raw = re.sub(r'\\[a-zA-Z]+(?:\{[^}]*\})*', '', raw)

    # Remove HTML comments
    raw = re.sub(r'<!--.*?-->', '', raw, flags=re.DOTALL)

    # Remove Quarto div markers
    raw = re.sub(r'^:::\s*\{[^}]*\}\s*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^:::\s*$', '', raw, flags=re.MULTILINE)

    # Remove image references
    raw = re.sub(r'!\[.*?\]\(.*?\)(?:\{[^}]*\})?', '', raw)

    # Remove markdown headers (keep the text)
    raw = re.sub(r'^#+\s+', '', raw, flags=re.MULTILINE)

    return raw.lower()


def count_keyword_hits(text: str, keyword_dict: dict) -> dict:
    """Count occurrences of each keyword in text. Returns {phrase: count}."""
    hits = {}
    for phrase, weight in keyword_dict.items():
        phrase_lower = phrase.lower().strip()
        if ' ' in phrase_lower:
            count = text.count(phrase_lower)
        else:
            pattern = r'\b' + re.escape(phrase_lower) + r'\b'
            count = len(re.findall(pattern, text))
        if count > 0:
            hits[phrase] = count
    return hits


def score_layer(hits: dict, keyword_dict: dict, idf_weights: dict, word_count: int) -> float:
    """
    Score a single layer using TF-IDF-inspired weighting.

    TF = count / chapter_length (per 1000 words)
    IDF = log(N / doc_freq) for each keyword — words in every chapter count less
    Weight = keyword specificity weight (2-3)
    """
    if word_count == 0:
        return 0.0

    total_score = 0.0
    for phrase, count in hits.items():
        tf = count / (word_count / 1000.0)
        idf = idf_weights.get(phrase, 1.0)
        weight = keyword_dict[phrase]
        total_score += tf * idf * weight

    return total_score


def extract_current_values(chapter_dir: str, qmd_file: str) -> list:
    """Extract current mlsysstack values from a QMD file."""
    path = BOOK_ROOT / chapter_dir / qmd_file
    if not path.exists():
        return [0] * 8

    text = path.read_text(encoding="utf-8")
    match = re.search(r'\\mlsysstack\{(\d+)\}\{(\d+)\}\{(\d+)\}\{(\d+)\}\{(\d+)\}\{(\d+)\}\{(\d+)\}\{(\d+)\}', text)
    if match:
        return [int(match.group(i)) for i in range(1, 9)]
    return [0] * 8


def score_to_intensity(raw_scores: dict, chapter_name: str) -> dict:
    """
    Convert raw scores to 0-100 intensity values.

    Strategy:
      - Find the max score across all layers for this chapter
      - Scale the top layer to ~90
      - Apply sqrt scaling to compress the range (so secondary topics are visible)
      - Floor values below a threshold to 0
    """
    max_score = max(raw_scores.values()) if raw_scores else 1.0
    if max_score == 0:
        return {layer: 0 for layer in LAYER_ORDER}

    intensities = {}
    for layer in LAYER_ORDER:
        ratio = raw_scores[layer] / max_score

        # Square root scaling to expand the mid-range
        # This prevents secondary topics from being crushed to 0
        scaled = math.sqrt(ratio)

        # Map to 0-90 range (never quite 100 to leave room)
        intensity = int(scaled * 90)

        # Floor: anything below 10 goes to 0 (not worth showing)
        if intensity < 10:
            intensity = 0

        intensities[layer] = intensity

    return intensities


def analyze_chapters(verbose=False):
    """Run the full analysis across all chapters."""
    print("=" * 90)
    print("MLSYS STACK RELEVANCE ANALYSIS")
    print("=" * 90)
    print()

    # Phase 0: Read all chapter texts
    chapter_texts = {}
    chapter_word_counts = {}
    for chapter_dir, qmd_file in CHAPTERS:
        text = read_chapter_text(chapter_dir, qmd_file)
        chapter_texts[chapter_dir] = text
        chapter_word_counts[chapter_dir] = len(text.split())

    # Phase 1: Compute IDF weights
    # For each keyword in each layer, count how many chapters contain it.
    # IDF = log(N / doc_freq) — keywords in every chapter get downweighted.
    N = len(CHAPTERS)
    idf_weights = {}
    for layer in LAYER_ORDER:
        idf_weights[layer] = {}
        for phrase in KEYWORDS[layer]:
            phrase_lower = phrase.lower().strip()
            doc_count = 0
            for chapter_dir, _ in CHAPTERS:
                text = chapter_texts[chapter_dir]
                if ' ' in phrase_lower:
                    if phrase_lower in text:
                        doc_count += 1
                else:
                    if re.search(r'\b' + re.escape(phrase_lower) + r'\b', text):
                        doc_count += 1
            # IDF: log(N / doc_freq), minimum 0.3 to avoid zeroing out
            if doc_count > 0:
                idf_weights[layer][phrase] = max(0.3, math.log(N / doc_count))
            else:
                idf_weights[layer][phrase] = 1.0

    if verbose:
        print("IDF weights for ubiquitous terms:")
        for layer in LAYER_ORDER:
            for phrase, idf in sorted(idf_weights[layer].items(), key=lambda x: x[1]):
                if idf < 0.5:
                    print(f"  {layer:12s}  {phrase:<30s}  IDF={idf:.2f}")
        print()

    # Phase 2: Score each chapter against each layer with TF-IDF
    raw_scores = {}
    for chapter_dir, qmd_file in CHAPTERS:
        text = chapter_texts[chapter_dir]
        word_count = chapter_word_counts[chapter_dir]

        scores = {}
        for layer in LAYER_ORDER:
            hits = count_keyword_hits(text, KEYWORDS[layer])
            scores[layer] = score_layer(hits, KEYWORDS[layer], idf_weights[layer], word_count)

        raw_scores[chapter_dir] = scores

        if verbose:
            print(f"\n{chapter_dir} ({word_count:,} words):")
            max_s = max(scores.values()) if scores.values() else 1
            for layer in LAYER_ORDER:
                bar_len = int(scores[layer] / max(max_s, 1) * 40)
                bar = "█" * bar_len
                print(f"  {layer:15s}: {scores[layer]:7.1f}  {bar}")

    # Phase 3: Cross-chapter normalization
    # For each layer, the highest-scoring chapter gets ~90
    layer_maxes = {}
    for layer in LAYER_ORDER:
        max_val = max(raw_scores[ch][layer] for ch, _ in CHAPTERS)
        layer_maxes[layer] = max_val if max_val > 0 else 1.0

    # Phase 4: Compute intensities using cross-chapter normalization
    all_intensities = {}
    for chapter_dir, _ in CHAPTERS:
        intensities = {}
        for layer in LAYER_ORDER:
            # Ratio relative to the highest-scoring chapter for this layer
            cross_ratio = raw_scores[chapter_dir][layer] / layer_maxes[layer]

            # Apply power scaling to shape the distribution
            # x^0.6 expands mid-range while keeping primary dominant
            scaled = cross_ratio ** 0.6

            # Map to 0-90 range
            intensity = int(scaled * 90)

            # Floor: below 10 → 0 (avoid faint distracting shading)
            if intensity < 10:
                intensity = 0

            intensities[layer] = intensity

        all_intensities[chapter_dir] = intensities

    # Phase 5: Report
    print("\n")
    print("=" * 100)
    print("COMPARISON: CURRENT vs. RECOMMENDED")
    print("=" * 100)
    print()

    # Header
    abbrs = ["HW", "FW", "MOD", "TRN", "SRV", "OPS", "APP", "DAT"]
    print(f"{'Chapter':<22s}  {'':8s}  ", end="")
    for a in abbrs:
        print(f"{a:>5s}", end="")
    print("   Visual")
    print("-" * 100)

    def bar_chart(vals, max_val=90, width=30):
        """Render a compact inline bar chart."""
        chars = "▁▂▃▄▅▆▇█"
        result = []
        for v in vals:
            idx = min(int(v / max_val * (len(chars) - 1)), len(chars) - 1)
            result.append(chars[idx] if v > 0 else " ")
        return "".join(result)

    for chapter_dir, qmd_file in CHAPTERS:
        current = extract_current_values(chapter_dir, qmd_file)
        recommended = all_intensities[chapter_dir]
        rec_vals = [recommended[layer] for layer in LAYER_ORDER]

        # Current
        print(f"{chapter_dir:<22s}  {'current':8s}  ", end="")
        for v in current:
            print(f"{v:5d}", end="")
        print(f"   {bar_chart(current)}")

        # Recommended
        print(f"{'':22s}  {'recomm':8s}  ", end="")
        for v in rec_vals:
            print(f"{v:5d}", end="")
        print(f"   {bar_chart(rec_vals)}")

        # Delta
        print(f"{'':22s}  {'delta':8s}  ", end="")
        for i, layer in enumerate(LAYER_ORDER):
            delta = recommended[layer] - current[i]
            if delta == 0:
                print(f"{'·':>5s}", end="")
            elif delta > 0:
                print(f"{'+' + str(delta):>5s}", end="")
            else:
                print(f"{delta:5d}", end="")
        print()
        print()

    # Phase 5: Output LaTeX replacement lines
    print("=" * 90)
    print("RECOMMENDED LaTeX (copy-paste ready)")
    print("=" * 90)
    print()
    for chapter_dir, qmd_file in CHAPTERS:
        recommended = all_intensities[chapter_dir]
        vals = [str(recommended[layer]) for layer in LAYER_ORDER]
        latex = "\\mlsysstack{" + "}{".join(vals) + "}"
        print(f"  {chapter_dir:<22s}  {latex}")

    print()
    return all_intensities


def apply_changes(all_intensities: dict):
    """Apply recommended values to QMD files."""
    for chapter_dir, qmd_file in CHAPTERS:
        path = BOOK_ROOT / chapter_dir / qmd_file
        if not path.exists():
            continue

        text = path.read_text(encoding="utf-8")
        recommended = all_intensities[chapter_dir]
        vals = [str(recommended[layer]) for layer in LAYER_ORDER]
        new_latex = "\\mlsysstack{" + "}{".join(vals) + "}"

        # Replace existing mlsysstack call
        pattern = r'\\mlsysstack\{\d+\}\{\d+\}\{\d+\}\{\d+\}\{\d+\}\{\d+\}\{\d+\}\{\d+\}'
        match = re.search(pattern, text)
        if match:
            new_text = text[:match.start()] + new_latex + text[match.end():]
        else:
            new_text = text

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"  ✓ Updated {chapter_dir}/{qmd_file}")
        else:
            print(f"  · No change needed for {chapter_dir}/{qmd_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Keyword detail report (for debugging)
# ─────────────────────────────────────────────────────────────────────────────

def keyword_detail_report(chapter_dir: str, qmd_file: str):
    """Show which keywords are hitting for a specific chapter."""
    text = read_chapter_text(chapter_dir, qmd_file)
    word_count = len(text.split())

    print(f"\nKEYWORD DETAIL: {chapter_dir} ({word_count:,} words)")
    print("=" * 70)

    for layer in LAYER_ORDER:
        print(f"\n  {layer.upper()}:")
        raw_hits = count_keyword_hits(text, KEYWORDS[layer])
        display_hits = []
        for phrase, count in raw_hits.items():
            weight = KEYWORDS[layer][phrase]
            display_hits.append((phrase, count, weight, count * weight))

        display_hits.sort(key=lambda x: -x[3])
        for phrase, count, weight, score in display_hits[:15]:
            print(f"    {phrase:<35s}  count={count:3d}  weight={weight}  score={score:5.0f}")

        if not display_hits:
            print(f"    (no matches)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    apply = "--apply" in sys.argv
    detail = "--detail" in sys.argv

    if detail:
        # Show keyword hits for a specific chapter
        chapter = sys.argv[sys.argv.index("--detail") + 1] if "--detail" in sys.argv else "nn_architectures"
        for ch_dir, ch_file in CHAPTERS:
            if ch_dir == chapter:
                keyword_detail_report(ch_dir, ch_file)
                break
        sys.exit(0)

    intensities = analyze_chapters(verbose=verbose)

    if apply:
        print("\nAPPLYING CHANGES...")
        apply_changes(intensities)
        print("\nDone! Re-render to see the updated diagrams.")
    else:
        print("\nTo apply these values, run with --apply flag.")
        print("To see keyword detail for a chapter, run with --detail <chapter_name>.")
