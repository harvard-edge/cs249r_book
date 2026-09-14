# Figure Caption Improvement Script

## Overview
This script improves figure and table captions in the ML Systems textbook using local Ollama LLM models. It provides automated caption enhancement with strong, educational language while maintaining proper formatting.

## Prerequisites

### Software Requirements
```bash
# Python dependencies (included in main requirements.txt)
pip install pypandoc pyyaml requests pillow

# Ollama for LLM caption improvement
brew install ollama  # macOS
# or: curl -fsSL https://ollama.ai/install.sh | sh  # Linux

# Download recommended models
ollama pull qwen2.5:7b      # Default model (good balance)
ollama pull gemma2:9b       # High quality alternative
ollama pull llama3.2:3b     # Fast lightweight option
```

### Hardware Requirements
- **8GB+ RAM** for LLM processing
- **SSD storage** for faster model loading
- **GPU optional** but improves performance

## Quick Start

### Improve All Captions (Recommended)
```bash
# Process all core chapters with default model
python3 scripts/improve_figure_captions.py -d contents/core/

# Use specific model
python3 scripts/improve_figure_captions.py -d contents/core/ -m gemma2:9b

# Process specific files
python3 scripts/improve_figure_captions.py -f contents/core/introduction/introduction.qmd
```

## Command Line Options

### Main Modes
All main options have both short and long forms:

| Option | Short | Purpose |
|--------|-------|---------|
| `--improve` | `-i` | **LLM caption improvement (default mode)** |
| `--build-map` | `-b` | Build content map and save to JSON |
| `--analyze` | `-a` | Quality analysis + file validation |
| `--repair` | `-r` | Fix formatting issues only |

### Additional Options
| Option | Short | Purpose |
|--------|-------|---------|
| `--model` | `-m` | Specify Ollama model (default: qwen2.5:7b) |
| `--files` | `-f` | Process specific QMD files |
| `--directories` | `-d` | Process directories (follows _quarto-html.yml order) |
| `--save-json` |  | Save detailed content map to JSON |
| `--list-models` |  | List available Ollama models |

## Usage Examples

### Complete Caption Improvement
```bash
# Default workflow - improve all captions
python3 scripts/improve_figure_captions.py -d contents/core/

# Equivalent explicit command
python3 scripts/improve_figure_captions.py --improve -d contents/core/

# With different model
python3 scripts/improve_figure_captions.py -i -d contents/core/ -m gemma2:9b

# Multiple directories
python3 scripts/improve_figure_captions.py -d contents/core/ -d contents/frontmatter/
```

### Analysis and Utilities
```bash
# Build content map only
python3 scripts/improve_figure_captions.py --build-map -d contents/core/
python3 scripts/improve_figure_captions.py -b -d contents/core/

# Analyze caption quality and validate structure
python3 scripts/improve_figure_captions.py --analyze -d contents/core/
python3 scripts/improve_figure_captions.py -a -d contents/core/

# Fix formatting issues only (no LLM)
python3 scripts/improve_figure_captions.py --repair -d contents/core/
python3 scripts/improve_figure_captions.py -r -d contents/core/
```

### Development and Debugging
```bash
# Save detailed JSON output for inspection
python3 scripts/improve_figure_captions.py -d contents/core/ --save-json

# List available Ollama models
python3 scripts/improve_figure_captions.py --list-models

# Process single file for testing
python3 scripts/improve_figure_captions.py -f contents/core/introduction/introduction.qmd -m gemma2:9b
```

## Model Selection Guide

### Recommended Models
| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **qwen2.5:7b** | ⭐⭐⭐ | ⭐⭐⭐⭐ | **Default - best balance** |
| **gemma2:9b** | ⭐⭐ | ⭐⭐⭐⭐⭐ | High quality output |
| **llama3.2:3b** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Fast processing |
| **mistral:7b** | ⭐⭐⭐ | ⭐⭐⭐⭐ | Alternative option |

### Model Installation
```bash
# Install specific models
ollama pull qwen2.5:7b
ollama pull gemma2:9b
ollama pull llama3.2:3b

# Check installed models
ollama list
```

## Caption Quality Standards

### Formatting Rules
- **Figures**: `**Bold Title**: Sentence case explanation.`
- **Tables**: `: **Bold Title**: Sentence case explanation.` (note colon prefix)
- **Word limit**: Maximum 100 words per caption
- **Language**: Strong, direct educational language

### Language Improvements
The script automatically:
- ✅ **Removes weak starters**: "Illustrates", "Shows", "Demonstrates"
- ✅ **Uses direct language**: "Neural networks process..." instead of "This shows how..."
- ✅ **Fixes capitalization**: Proper sentence case after periods
- ✅ **Normalizes spacing**: Single spaces, clean formatting
- ✅ **Educational focus**: Clear, learning-oriented explanations

### Before/After Examples

**Before (weak):**
```
Illustrates how machine learning models can serve as amplifiers.
```

**After (strong):**
```
**Amplification Effects**: Machine learning models enable threat actors to scale attacks by automating target identification and payload generation.
```

## Processing Workflow

### What the Script Does
1. **Extract**: Finds all figures and tables in QMD files (follows _quarto-html.yml order)
2. **Analyze**: Builds content map with context extraction
3. **Improve**: Uses LLM to generate better captions with quality validation
4. **Update**: Applies improvements directly to QMD files
5. **Validate**: Ensures proper formatting and structure

### Content Map Structure
The script builds a comprehensive map including:
- **270 figures** across core chapters (Markdown, TikZ, Code blocks)
- **92 tables** with proper caption detection
- **Context extraction** using paragraph-level analysis
- **100% success rate** with robust extraction patterns

## Troubleshooting

### Common Issues

#### Ollama Connection Problems
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Check available models
ollama list
```

#### Extraction Failures
```bash
# Analyze extraction issues
python3 scripts/improve_figure_captions.py --analyze -d contents/core/

# Build content map to see details
python3 scripts/improve_figure_captions.py --build-map -d contents/core/
```

#### Quality Issues
```bash
# Try different model
python3 scripts/improve_figure_captions.py -d contents/core/ -m gemma2:9b

# Check specific file
python3 scripts/improve_figure_captions.py -f problematic_file.qmd --save-json
```

### Performance Optimization
- **Use qwen2.5:7b** for best speed/quality balance
- **Process single files** for testing: `-f filename.qmd`
- **Use llama3.2:3b** for fastest processing
- **Enable JSON output** only when debugging: `--save-json`

## Output Files

### Generated Files
```
content_map.json           # Detailed content structure (if --save-json)
improvements_YYYYMMDD_HHMMSS.json  # Summary of changes made
```

### Content Map Structure
```json
{
  "figures": {
    "fig-ai-timeline": {
      "qmd_file": "contents/core/introduction/introduction.qmd",
      "type": "tikz",
      "original_caption": "...",
      "new_caption": "...",
      "improved": true
    }
  },
  "tables": { ... },
  "metadata": {
    "extraction_stats": {
      "figures_found": 270,
      "tables_found": 92,
      "extraction_failures": 0,
      "success_rate": 100.0
    }
  }
}
```

## Integration with Book Build

### Quarto Compatibility
The script works seamlessly with Quarto's build process:
- **Preserves**: All Quarto attributes (`{#fig-id .class}`)
- **Maintains**: Reference links and cross-references
- **Follows**: _quarto-html.yml chapter ordering
- **Supports**: TikZ, Markdown, and code block figures

### Build Process
```bash
# 1. Improve captions
python3 scripts/improve_figure_captions.py -d contents/core/

# 2. Build book normally
quarto render

# 3. Check results
open build/html/index.html
```

## Best Practices

### Development Workflow
1. **Test on single file** first: `-f filename.qmd`
2. **Use analyze mode** to check structure: `--analyze`
3. **Try different models** for quality comparison
4. **Save JSON output** for debugging: `--save-json`
5. **Commit script changes** but review QMD changes carefully

### Production Workflow
1. **Use default settings** for consistent results
2. **Process all core chapters**: `-d contents/core/`
3. **Verify improvements** before committing QMD files
4. **Test Quarto build** after caption updates

### Quality Assurance
- **Automatic validation**: 100-word limit, proper formatting
- **Language improvements**: Strong, educational tone
- **Context preservation**: Maintains technical accuracy
- **Format consistency**: Proper table/figure formatting

## Success Metrics

### Extraction Quality
- ✅ **100% success rate** (270 figures, 92 tables found)
- ✅ **Perfect format detection** (TikZ, Markdown, Code blocks)
- ✅ **Robust table parsing** (handles `: **bold**: format`)
- ✅ **Context-aware processing** (paragraph-level analysis)

### Caption Quality
- ✅ **Strong language** (eliminates weak starters)
- ✅ **Educational focus** (clear learning objectives)
- ✅ **Proper formatting** (consistent spacing, capitalization)
- ✅ **Technical accuracy** (preserves domain knowledge)

---

**Last Updated**: December 2024
**Tested With**: Quarto 1.5+, Ollama 0.3+, Python 3.8+
**Script Version**: 2.0 (streamlined options)
