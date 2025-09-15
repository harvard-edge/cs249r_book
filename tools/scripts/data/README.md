# Glossary System Scripts

This directory contains the scripts for managing the ML Systems textbook glossary.

## ✅ Proper Data Flow

```
Individual Chapter Glossaries (Source of Truth)
    ↓
build_master_glossary.py (Smart Aggregation)
    ↓
Master Glossary (Clean Output) 
    ↓
generate_glossary.py (Page Generation)
    ↓
glossary.qmd (Published Page)
```

## Scripts Overview

### 1. `build_master_glossary.py` ⭐ **Main Script**
- **Purpose**: Aggregates individual chapter glossaries into clean master glossary
- **Input**: `quarto/contents/core/**/*_glossary.json` (22 chapter files)
- **Output**: `quarto/contents/data/master_glossary.json`
- **Features**:
  - Smart deduplication (merged 199 duplicates from 816 → 617 terms)
  - Cross-chapter term detection (102 multi-chapter terms)
  - Intelligent definition selection (prefers core chapters, longest definitions)
  - Proper chapter attribution

**Usage:**
```bash
python3 build_master_glossary.py
```

### 2. `generate_glossary.py` 
- **Purpose**: Generates the glossary page from clean master data
- **Input**: `quarto/contents/data/master_glossary.json`
- **Output**: `quarto/contents/backmatter/glossary/glossary.qmd`
- **Features**:
  - Alphabetical organization (A-Z sections)
  - Quarto cross-reference links (@sec-chapter format)
  - Academic formatting standards
  - Multi-chapter attribution

**Usage:**
```bash
python3 generate_glossary.py
```

### 3. `clean_master_glossary.py` (Legacy)
- **Purpose**: One-time cleanup script (now replaced by proper aggregation)
- **Status**: Kept for reference, but `build_master_glossary.py` is preferred

## Complete Workflow

### Initial Setup / Major Updates
```bash
# 1. Rebuild master glossary from chapter sources
python3 build_master_glossary.py

# 2. Generate glossary page
python3 generate_glossary.py

# 3. Commit changes
git add . && git commit -m "update: rebuild glossary system"
```

### Regular Updates
When chapter glossaries are updated:
```bash
# Just rebuild and regenerate
python3 build_master_glossary.py && python3 generate_glossary.py
```

## Data Sources (Source of Truth)

Individual chapter glossaries are the authoritative source:
```
quarto/contents/core/introduction/introduction_glossary.json
quarto/contents/core/training/training_glossary.json
quarto/contents/core/workflow/workflow_glossary.json
... (22 chapters total)
```

## Outputs

### Master Glossary
- **File**: `quarto/contents/data/master_glossary.json`
- **Format**: Structured JSON with metadata
- **Terms**: 617 unique terms (after deduplication)
- **Cross-references**: 102 terms appear in multiple chapters

### Published Glossary Page
- **File**: `quarto/contents/backmatter/glossary/glossary.qmd`
- **Format**: Quarto markdown with cross-reference links
- **Integration**: Available in HTML, PDF, and EPUB navigation
- **Features**: Interactive tooltips throughout the book

## Statistics

**Current Glossary (as of latest build):**
- **Total unique terms**: 617
- **Raw terms across chapters**: 816
- **Duplicates merged**: 199
- **Multi-chapter terms**: 102
- **Single-chapter terms**: 515
- **Coverage**: All 22 chapters

**Example Multi-Chapter Terms:**
- `federated learning`: 10 chapters
- `transfer learning`: 9 chapters  
- `quantization`: 8 chapters
- `artificial intelligence`: 7 chapters
- `edge computing`: 6 chapters

## Quality Features

1. **Smart Deduplication**: Merges terms like "large language models" + "large_language_model"
2. **Definition Selection**: Prefers definitions from core chapters (dl_primer, training, etc.)
3. **Chapter Attribution**: Properly tracks where terms appear vs. where they're primary
4. **Consistent Formatting**: Standardizes term names and removes formatting artifacts
5. **Cross-References**: Uses proper Quarto @sec- links for navigation

## Maintenance

- **Source Updates**: Edit individual chapter glossaries (they're the source of truth)
- **Regeneration**: Run build → generate scripts after changes
- **Backup**: Scripts automatically create backups before overwriting
- **Validation**: Check for proper @sec- link resolution during full book builds