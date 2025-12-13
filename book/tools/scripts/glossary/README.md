# Glossary Management Scripts

Scripts for managing the ML Systems textbook glossary system.

## Quick Commands

### Full Rebuild (when chapters change)
```bash
cd /Users/VJ/GitHub/MLSysBook
python3 tools/scripts/glossary/build_master_glossary.py
python3 tools/scripts/glossary/generate_glossary.py
```

### Individual Chapter Update
```bash
# 1. Use glossary-builder agent to update specific chapter
# 2. Then rebuild:
python3 tools/scripts/glossary/build_master_glossary.py
python3 tools/scripts/glossary/generate_glossary.py
```

## Data Flow

```
Chapter QMDs → Agent → Individual JSONs → build_master_glossary.py → Master JSON → generate_glossary.py → glossary.qmd
```

## Scripts

- **`build_master_glossary.py`** - Main aggregation script (chapter JSONs → master JSON)
- **`generate_glossary.py`** - Page generator (master JSON → glossary.qmd)
- **`clean_master_glossary.py`** - Legacy cleanup script (use build_master instead)

## Source Files

- **Individual glossaries**: `quarto/contents/core/*/\*_glossary.json` (22 files)
- **Master glossary**: `quarto/contents/data/master_glossary.json`
- **Published page**: `quarto/contents/backmatter/glossary/glossary.qmd`

Individual chapter glossaries are the source of truth. Edit those, then rebuild.
