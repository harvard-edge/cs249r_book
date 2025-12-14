# How to Regenerate Cross-References

This guide explains how to regenerate all cross-references for the MLSysBook using Gemma 2 27B model.

## Prerequisites

1. **Ollama**: Ensure Ollama is installed and running
   ```bash
   # Check if Ollama is running
   ollama list

   # If gemma2:27b is not available, pull it
   ollama pull gemma2:27b
   ```

2. **Python Dependencies** (optional for enhanced generation):
   ```bash
   pip install sentence-transformers scikit-learn numpy torch pyyaml pypandoc requests
   ```

## Method 1: Simple Regeneration Script (Recommended)

Run the provided regeneration script:

```bash
cd /Users/VJ/GitHub/MLSysBook
python3 tools/scripts/cross_refs/regenerate_xrefs.py
```

This script will:
1. Check if Ollama is running with gemma2:27b
2. Clean up all existing _xrefs.json files
3. Generate base cross-references using production_xref_generator.py
4. Enhance explanations with Gemma 2 if available
5. Save files to `quarto/contents/core/[chapter]/[chapter]_xrefs.json`

## Method 2: Using Production Generator Directly

```bash
cd tools/scripts/cross_refs
python3 production_xref_generator.py
```

This generates cross-references without LLM enhancement but is faster.

## Method 3: Using manage_cross_references.py (Advanced)

For more control with specific models:

```bash
cd tools/scripts/cross_refs
python3 manage_cross_references.py \
  --generate \
  --model sentence-t5-base \
  --output cross_refs.json \
  --dirs ../../../quarto/contents/core/ \
  --explain \
  --ollama-model gemma2:27b
```

## Files Generated

The scripts generate individual `_xrefs.json` files for each chapter:
- `introduction/introduction_xrefs.json`
- `ml_systems/ml_systems_xrefs.json`
- `dl_primer/dl_primer_xrefs.json`
- ... (one for each chapter)

## Structure of _xrefs.json Files

Each file contains:
```json
{
  "cross_references": {
    "sec-[section-id]": [
      {
        "target_chapter": "chapter_name",
        "target_section": "sec-target-id",
        "connection_type": "prerequisite|foundation|extends|complements",
        "concepts": ["concept1", "concept2"],
        "strength": 0.0-1.0,
        "quality": 0.0-1.0,
        "explanation": "Brief explanation",
        "placement": "chapter_start|section_start|section_end|sidebar",
        "priority": 1-3
      }
    ]
  }
}
```

## Using the Cross-References

The `inject-xrefs.lua` filter automatically uses these files during PDF/HTML generation:

```bash
# Build with cross-references
./binder pdf intro  # Builds introduction chapter with cross-refs
./binder pdf all    # Builds entire book with cross-refs
```

## Configuration

### Default Model
The default model is set to `gemma2:27b` in:
- `tools/scripts/cross_refs/manage_cross_references.py`
- `tools/scripts/cross_refs/llm_enhanced_xrefs.py`

To change the default, modify the `selected_model` variable or use `--ollama-model` flag.

### Hybrid Mode Settings
The inject-xrefs.lua filter uses hybrid mode by default, configured with:
- `MAX_CHAPTER_REFS = 8` - Maximum references in chapter-level box
- `MAX_SECTION_REFS = 3` - Maximum references per section
- `PRIORITY_THRESHOLD = 2` - Show priority 1-2 in sections
- `STRENGTH_THRESHOLD = 0.25` - Show connections >25% strength

## Troubleshooting

1. **Ollama not running**: Start Ollama first
   ```bash
   ollama serve
   ```

2. **Model not available**: Pull the model
   ```bash
   ollama pull gemma2:27b
   ```

3. **Python dependencies missing**: The production generator works without ML libraries
   ```bash
   python3 production_xref_generator.py
   ```

4. **Files not in correct location**: Ensure you're in the MLSysBook root directory

## Notes

- Generation takes ~2-5 minutes depending on whether LLM enhancement is enabled
- The hybrid placement mode reduces visual clutter by ~60% while maintaining pedagogical value
- Cross-references point to specific sections, not just chapters
- Each reference includes both chapter name and section number (e.g., "**ML Systems**: (§2.1)")
