# Cross-Reference Generation & Integration Recipe

## Overview
This recipe documents the complete process for generating AI-powered cross-references with explanations and integrating them into the ML Systems textbook.

## Prerequisites

### Software Requirements
```bash
# Python dependencies
pip install sentence-transformers scikit-learn numpy torch pyyaml pypandoc requests

# Ollama for AI explanations
brew install ollama  # macOS
# or: curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Download recommended model (best quality from experiments)
ollama run llama3.1:8b
```

### Hardware
- **GPU recommended** for domain-adapted model training
- **16GB+ RAM** for processing 93 sections across 19 chapters
- **SSD storage** for faster model loading

## Step 1: Generate Cross-References with Explanations

### Quick Command (Recommended)
```bash
# Generate cross-references with explanations using optimal settings
python3 ./scripts/cross_refs/cross_refs.py \
    -g \
    -m ./scripts/cross_refs/t5-mlsys-domain-adapted/ \
    -o data/cross_refs.json \
    -d ./contents/core/ \
    -t 0.5 \
    --explain \
    --ollama-model llama3.1:8b
```

### Parameters Explained
- **`-t 0.5`**: Similarity threshold (0.5 = 230 refs, good balance of quality/quantity)
- **`--ollama-model llama3.1:8b`**: Best quality model from systematic experiments
- **Domain-adapted model**: `t5-mlsys-domain-adapted/` provides better results than base models

### Alternative Thresholds
```bash
# Higher quality, fewer references (92 refs)
python3 ./scripts/cross_refs/cross_refs.py ... -t 0.6

# More references, lower quality (294 refs)  
python3 ./scripts/cross_refs/cross_refs.py ... -t 0.4

# Very high quality, very few (36 refs)
python3 ./scripts/cross_refs/cross_refs.py ... -t 0.65
```

### Expected Output
```
✅ Generated 230 cross-references across 18 files.
📊 Average similarity: 0.591
📄 Results saved to: data/cross_refs.json
```

## Step 2: Quality Evaluation (Optional)

### Evaluate with LLM Judges
```bash
# Evaluate sample with Student, TA, Instructor judges
python3 ./scripts/cross_refs/evaluate_explanations.py \
    data/cross_refs.json \
    --sample 20 \
    --output evaluation_results.json
```

### Expected Quality Metrics
- **Target Score**: 3.5+ out of 5.0
- **Student Judge**: Most accepting (focuses on clarity)
- **TA Judge**: Most critical (focuses on pedagogy)
- **Instructor Judge**: Balanced (focuses on academic rigor)

## Step 3: Integration into Book

### Configure Quarto
Ensure `_quarto.yml` has cross-reference configuration:
```yaml
cross-references:
  file: "data/cross_refs.json"
  enabled: true

filters:
  - lua/inject_crossrefs.lua    # Must come before custom-numbered-blocks
  - custom-numbered-blocks
  - lua/margin-connections.lua  # Must come after custom-numbered-blocks
```

### Test with Single Chapter
```bash
# Test with introduction only
quarto render contents/core/introduction/introduction.qmd --to pdf
```

### Build Full Book
```bash
# Render complete book
quarto render --to pdf
```

## Step 4: Handle Common Issues

### Float Issues ("Too many unprocessed floats")
If you get float overflow errors, add to `tex/header-includes.tex`:
```latex
\usepackage{placeins}
\newcommand{\sectionfloatclear}{\FloatBarrier}
\newcommand{\chapterfloatclear}{\clearpage}

% Flush floats at sections and chapters
\let\oldsection\section
\renewcommand{\section}{\sectionfloatclear\oldsection}

\let\oldchapter\chapter  
\renewcommand{\chapter}{\chapterfloatclear\oldchapter}
```

### Missing References
If some cross-references don't resolve:
```bash
# Check section IDs are correct
grep -r "sec-" contents/core/ | head -10

# Regenerate with verbose logging
python3 ./scripts/cross_refs/cross_refs.py ... --verbose
```

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# List available models
ollama list
```

## Step 5: Optimization Settings

### Model Selection Priority
1. **llama3.1:8b** - Best quality (8.0/10 from experiments) ⭐
2. **qwen2.5:7b** - Fast alternative (7.8/10 quality)
3. **gemma2:9b** - Good balance
4. **phi3:3.8b** - High quality but verbose

### Threshold Guidelines
| Use Case | Threshold | Expected Count | Quality |
|----------|-----------|----------------|---------|
| **Recommended** | 0.5 | 230 refs | Good balance |
| High quality | 0.6 | 92 refs | Excellent |
| Comprehensive | 0.4 | 294 refs | Acceptable |
| Elite only | 0.65 | 36 refs | Premium |

## Troubleshooting

### Performance Issues
- **Slow generation**: Use `qwen2.5:7b` instead of `llama3.1:8b`
- **Memory issues**: Reduce `--max-suggestions` from 5 to 3
- **Large output**: Use higher threshold (0.6+)

### Quality Issues
- **Poor explanations**: Check Ollama model is correct version
- **Generic text**: Regenerate with different `--seed` value
- **Wrong direction**: Verify file ordering in `_quarto.yml`

### Build Issues
- **LaTeX errors**: Check `tex/header-includes.tex` for conflicts
- **Missing sections**: Verify all `.qmd` files have proper section IDs
- **Slow builds**: Use `quarto render --cache` for faster rebuilds

## File Structure
```
scripts/cross_refs/
├── cross_refs.py              # Main generation script
├── evaluate_explanations.py   # LLM judge evaluation
├── filters.yml               # Content filtering rules
├── t5-mlsys-domain-adapted/  # Domain-adapted model
└── RECIPE.md                 # This documentation

data/
└── cross_refs.json          # Generated cross-references

lua/
├── inject_crossrefs.lua     # Injection filter
└── margin-connections.lua   # PDF margin rendering
```

## Success Metrics
- ✅ **230 cross-references** generated with threshold 0.5
- ✅ **3.6+ average quality** from LLM judge evaluation
- ✅ **Clean PDF build** without float or reference errors
- ✅ **Margin notes** render correctly in PDF output
- ✅ **Connection callouts** display properly in HTML

## Maintenance

### Updating Cross-References
When content changes significantly:
```bash
# Regenerate cross-references
python3 ./scripts/cross_refs/cross_refs.py -g ... 

# Re-evaluate quality
python3 ./scripts/cross_refs/evaluate_explanations.py ...

# Test build
quarto render --to pdf
```

### Model Updates
When new Ollama models become available:
```bash
# Download new model
ollama run new-model:version

# Test with sample
python3 ./scripts/cross_refs/cross_refs.py ... --ollama-model new-model:version --sample 10

# Evaluate quality difference
python3 ./scripts/cross_refs/evaluate_explanations.py ...
```

---

**Last Updated**: July 2025  
**Tested With**: Quarto 1.5+, Ollama 0.3+, Python 3.8+ 