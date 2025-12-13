# Glossary System File Organization

This document outlines the complete file organization for the glossary system.

## Directory Structure

```
MLSysBook/
├── quarto/contents/
│   ├── core/                              # Chapter directories (source of truth)
│   │   ├── introduction/
│   │   │   └── introduction_glossary.json # Chapter-specific glossary
│   │   ├── ml_systems/
│   │   │   └── ml_systems_glossary.json   # Chapter-specific glossary
│   │   ├── [... 20 more chapters ...]
│   │   └── generative_ai/
│   │       └── generative_ai_glossary.json
│   │
│   ├── data/                              # Aggregated data
│   │   ├── global_glossary.json           # Global aggregated glossary
│   │   └── global_glossary.backup.json   # Backup of previous version
│   │
│   └── backmatter/glossary/               # Published glossary
│       └── glossary.qmd                   # Final glossary page
│
└── tools/scripts/glossary/                # Processing scripts
    ├── build_global_glossary.py           # Aggregates chapters → global
    ├── generate_glossary.py               # Generates master → QMD page
    ├── consolidate_similar_terms.py       # Manual consolidation rules
    ├── smart_consolidation.py             # LLM-based similarity detection
    ├── rule_based_consolidation.py        # Academic best practices
    ├── clean_master_glossary.py           # Utility for cleanup
    ├── README.md                          # Documentation
    └── ORGANIZATION.md                    # This file
```

## Data Flow

```
Chapter Glossaries → Global Glossary → Published Glossary
      (22 files)         (1 file)         (1 QMD file)
         ↓                   ↓                  ↓
     Source of           Aggregated         Final user-
       truth             & deduplicated     facing page
```

## File Status Summary

### ✅ Properly Located Files

**Chapter Glossaries (22 files):**
- `quarto/contents/core/*/chapter_glossary.json`
- Status: ✅ All in proper chapter directories
- Purpose: Source of truth for individual chapter terms

**Global Glossary:**
- `quarto/contents/data/global_glossary.json`
- Status: ✅ In proper data directory
- Purpose: Aggregated and deduplicated terms from all chapters

**Published Glossary:**
- `quarto/contents/backmatter/glossary/glossary.qmd`
- Status: ✅ In proper backmatter location
- Purpose: Final user-facing glossary page

**Processing Scripts (7 files):**
- `tools/scripts/glossary/*.py`
- Status: ✅ All in proper tools directory
- Purpose: Data processing and generation pipeline

### 🧹 Cleaned Up

**Removed Files:**
- `quarto/contents/backmatter/._glossary_xref.json` (macOS hidden file)
- `quarto/contents/backmatter/glossary/._glossary_xref.json` (macOS hidden file)

## Current Statistics

- **Chapter glossaries**: 22 files (810 raw terms)
- **Master glossary**: 611 unique terms after deduplication
- **Processing scripts**: 7 Python files
- **Published glossary**: 1 QMD file with 611 terms in 26 alphabetical sections

## Usage Workflow

1. **Edit terms**: Modify individual chapter glossary JSON files
2. **Rebuild master**: Run `python3 tools/scripts/glossary/build_global_glossary.py`
3. **Generate page**: Run `python3 tools/scripts/glossary/generate_glossary.py`
4. **Optional cleanup**: Run consolidation scripts for quality improvement

All files are now properly organized and in their correct locations.
