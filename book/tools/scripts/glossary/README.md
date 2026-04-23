# Glossary Management Scripts

Scripts for managing the ML Systems textbook glossary system.

## Quick Commands

### Full Rebuild (when chapters change)
```bash
cd /Users/VJ/GitHub/MLSysBook
python3 book/tools/scripts/glossary/build_global_glossary.py
python3 book/tools/scripts/glossary/generate_glossary.py
```

### Generate Specific Volume
```bash
python3 book/tools/scripts/glossary/generate_glossary.py --volume vol1
python3 book/tools/scripts/glossary/generate_glossary.py --volume vol2
```

## Data Flow

```
Chapter QMDs → Agent → Individual JSONs → build_global_glossary.py → Volume JSONs → generate_glossary.py → glossary.qmd
```

## Scripts

- **`build_global_glossary.py`** - Main aggregation script (chapter JSONs → volume JSONs)
- **`generate_glossary.py`** - Page generator (volume JSONs → volume glossary.qmd files)
- **`clean_master_glossary.py`** - Legacy cleanup script
- **`smart_consolidation.py`** - Advanced term consolidation
- **`rule_based_consolidation.py`** - Rule-based term consolidation

## Source Files

- **Vol1 chapter glossaries**: `quarto/contents/vol1/*/<chapter>_glossary.json`
- **Vol2 chapter glossaries**: `quarto/contents/vol2/*/<chapter>_glossary.json`

Individual chapter glossaries are the source of truth. Edit those, then rebuild.

## Output Files

- **Volume 1 JSON**: `quarto/contents/vol1/backmatter/glossary/vol1_glossary.json`
- **Volume 1 page**: `quarto/contents/vol1/backmatter/glossary/glossary.qmd`
- **Volume 2 JSON**: `quarto/contents/vol2/backmatter/glossary/vol2_glossary.json`
- **Volume 2 page**: `quarto/contents/vol2/backmatter/glossary/glossary.qmd`

Each volume has its own self-contained glossary.
