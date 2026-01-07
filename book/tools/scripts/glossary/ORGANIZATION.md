# Glossary System File Organization

This document outlines the complete file organization for the glossary system.

## Directory Structure

```
MLSysBook/book/
├── quarto/contents/
│   ├── vol1/                                 # Volume 1 chapters
│   │   ├── introduction/
│   │   │   └── introduction_glossary.json    # Chapter-specific glossary
│   │   ├── ml_systems/
│   │   │   └── ml_systems_glossary.json      # Chapter-specific glossary
│   │   ├── [... more vol1 chapters ...]
│   │   ├── frontmatter/                      # Vol1 frontmatter
│   │   │   ├── foreword.qmd
│   │   │   └── acknowledgements/
│   │   └── backmatter/glossary/              # Vol1 glossary
│   │       ├── vol1_glossary.json            # Aggregated Vol1 terms
│   │       └── glossary.qmd                  # Vol1 glossary page
│   │
│   ├── vol2/                                 # Volume 2 chapters
│   │   ├── infrastructure/
│   │   │   └── infrastructure_glossary.json
│   │   ├── [... more vol2 chapters ...]
│   │   ├── frontmatter/                      # Vol2 frontmatter (if needed)
│   │   └── backmatter/glossary/              # Vol2 glossary
│   │       ├── vol2_glossary.json            # Aggregated Vol2 terms
│   │       └── glossary.qmd                  # Vol2 glossary page
│   │
│   └── frontmatter/                          # Shared frontmatter
│       ├── about/
│       └── socratiq/
│
└── tools/scripts/glossary/                   # Processing scripts
    ├── build_global_glossary.py              # Aggregates chapters → volume JSONs
    ├── generate_glossary.py                  # Generates JSONs → QMD pages
    ├── consolidate_similar_terms.py          # Manual consolidation rules
    ├── smart_consolidation.py                # LLM-based similarity detection
    ├── rule_based_consolidation.py           # Academic best practices
    ├── clean_master_glossary.py              # Utility for cleanup
    ├── README.md                             # Quick start documentation
    └── ORGANIZATION.md                       # This file
```

## Data Flow

```
Chapter Glossaries → Volume Glossaries → Published Glossary Pages
   (vol1: 16 files)   (vol1_glossary.json)   (vol1/glossary.qmd)
   (vol2:  7 files)   (vol2_glossary.json)   (vol2/glossary.qmd)
         ↓                    ↓                     ↓
     Source of           Aggregated             Volume-specific
       truth             & deduplicated         user-facing pages
```

Each volume has its own self-contained glossary with no cross-volume dependencies.

## File Status Summary

### Volume-Specific Glossaries

**Volume 1 Glossary:**
- Source: `quarto/contents/vol1/*/<chapter>_glossary.json` (16 files)
- Aggregated: `quarto/contents/vol1/backmatter/glossary/vol1_glossary.json`
- Published: `quarto/contents/vol1/backmatter/glossary/glossary.qmd`

**Volume 2 Glossary:**
- Source: `quarto/contents/vol2/*/<chapter>_glossary.json` (7 files)
- Aggregated: `quarto/contents/vol2/backmatter/glossary/vol2_glossary.json`
- Published: `quarto/contents/vol2/backmatter/glossary/glossary.qmd`

## Current Statistics

- **Vol1 chapter glossaries**: 16 files (~593 raw terms, ~462 unique)
- **Vol2 chapter glossaries**: 7 files (~287 raw terms, ~250 unique)
- **Processing scripts**: 7 Python files
- **Published glossaries**: 2 QMD files (Vol1, Vol2)

## Usage Workflow

1. **Edit terms**: Modify individual chapter glossary JSON files
2. **Rebuild all**: Run `python3 book/tools/scripts/glossary/build_global_glossary.py`
3. **Generate pages**: Run `python3 book/tools/scripts/glossary/generate_glossary.py`
4. **Optional**: Generate specific volume only:
   ```bash
   python3 book/tools/scripts/glossary/generate_glossary.py --volume vol1
   ```

All files are now properly organized with volume-specific glossaries.
