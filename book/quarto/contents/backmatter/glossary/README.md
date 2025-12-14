# ML Systems Textbook Glossary System {#sec-ml-systems-textbook-glossary-system}

This directory contains the comprehensive glossary system for the ML Systems textbook. The glossary provides definitions for 611+ technical terms used throughout the book, with automatic cross-references and interactive tooltips.

## 📚 What This System Provides {#sec-ml-systems-textbook-glossary-system-system-provides-e08f}

### For Readers {#sec-ml-systems-textbook-glossary-system-readers-e678}
- **Interactive tooltips** when hovering over terms throughout the book
- **Comprehensive alphabetical glossary** with 611+ terms across 26 letter sections
- **Cross-chapter references** showing where terms appear or are discussed
- **Academic-quality definitions** suitable for both undergraduate and graduate students

### For Authors/Maintainers   {#sec-ml-systems-textbook-glossary-system-authorsmaintainers-27b8}
- **Automated term aggregation** from individual chapter glossaries
- **Intelligent deduplication** and similarity detection
- **Cross-reference validation** ensuring all links work properly
- **Quality assurance** through rule-based and LLM-based consolidation

## 🏗️ System Architecture {#sec-ml-systems-textbook-glossary-system-system-architecture-a055}

### Data Flow {#sec-ml-systems-textbook-glossary-system-data-flow-b3a7}
```
Individual Chapter Glossaries → Global Glossary → Published Glossary Page
        (22 JSON files)           (1 JSON file)        (1 QMD file)
            ↓                          ↓                     ↓
      Source of Truth              Aggregated           User-Facing
     (810 raw terms)            (611 unique terms)       (Web Page)
```

### File Structure {#sec-ml-systems-textbook-glossary-system-file-structure-a801}
```
quarto/contents/
├── core/                                    # Chapter directories
│   ├── introduction/
│   │   └── introduction_glossary.json      # Chapter-specific terms
│   ├── ml_systems/
│   │   └── ml_systems_glossary.json        # Chapter-specific terms
│   └── [... 20 more chapters ...]
│
├── data/
│   ├── global_glossary.json                # Aggregated & deduplicated terms
│   └── global_glossary.backup.json         # Backup of previous version
│
└── backmatter/glossary/
    ├── glossary.qmd                        # Published glossary page
    └── README.md                           # This documentation
```

## 🔄 How It Works {#sec-ml-systems-textbook-glossary-system-works-b18b}

### 1. Chapter Glossaries (Source of Truth) {#sec-ml-systems-textbook-glossary-system-1-chapter-glossaries-source-truth-312f}
Each chapter has its own JSON glossary file containing terms specific to that chapter:

```json
{
  "metadata": {
    "chapter": "introduction",
    "total_terms": 27,
    "generated_date": "2024-09-15"
  },
  "terms": [
    {
      "term": "artificial intelligence",
      "definition": "A field of computer science focused on creating systems that can perform tasks typically requiring human intelligence...",
      "chapter_source": "introduction",
      "aliases": [],
      "see_also": []
    }
  ]
}
```

### 2. Global Glossary (Aggregated Data) {#sec-ml-systems-textbook-glossary-system-2-global-glossary-aggregated-data-8694}
The global glossary aggregates all chapter terms, handles deduplication, and manages cross-chapter references:

```json
{
  "metadata": {
    "type": "global_glossary",
    "version": "1.0",
    "total_terms": 611,
    "source": "aggregated_from_chapter_glossaries"
  },
  "terms": [
    {
      "term": "artificial intelligence",
      "definition": "A field of computer science focused on creating systems...",
      "appears_in": ["introduction", "ml_systems", "responsible_ai"],
      "chapter_source": "introduction",
      "aliases": [],
      "see_also": []
    }
  ]
}
```

### 3. Published Glossary (User-Facing) {#sec-ml-systems-textbook-glossary-system-3-published-glossary-userfacing-1e82}
The final QMD file provides the user-facing glossary page with proper Quarto formatting and cross-references.

## 🛠️ Processing Scripts {#sec-ml-systems-textbook-glossary-system-processing-scripts-6943}

All processing scripts are located in `tools/scripts/glossary/`:

### Core Scripts {#sec-ml-systems-textbook-glossary-system-core-scripts-e183}

1. **`build_global_glossary.py`** - Main aggregation script
   - Reads all 22 chapter glossary files
   - Deduplicates and merges similar terms
   - Handles multi-chapter attribution
   - Generates the global glossary JSON

2. **`generate_glossary.py`** - Page generation script
   - Reads the global glossary JSON
   - Generates the final QMD page with proper formatting
   - Automatically discovers chapter section IDs for cross-references
   - Creates alphabetical organization with term counts

### Quality Assurance Scripts {#sec-ml-systems-textbook-glossary-system-quality-assurance-scripts-d080}

3. **`smart_consolidation.py`** - Intelligent similarity detection
   - Detects 115+ groups of potentially similar terms
   - Uses multiple similarity metrics (sequence matching, word overlap, subset detection)
   - Provides framework for LLM-based consolidation decisions
   - Logs all decisions for review

4. **`rule_based_consolidation.py`** - Academic best practices
   - Applies standard consolidation rules (singular/plural, acronyms, formatting)
   - Follows academic publishing guidelines
   - Handles common formatting inconsistencies
   - Prioritizes foundational chapters for definitions

5. **`consolidate_similar_terms.py`** - Manual consolidation rules
   - Implements specific consolidation rules for known issues
   - Handles edge cases not caught by automated systems
   - Provides fine-grained control over term merging

### Utility Scripts {#sec-ml-systems-textbook-glossary-system-utility-scripts-2f7f}

6. **`clean_global_glossary.py`** - Cleanup and validation
   - Validates JSON structure and required fields
   - Checks for orphaned references
   - Provides data quality reports

## 📝 Usage Instructions {#sec-ml-systems-textbook-glossary-system-usage-instructions-7699}

### For Content Authors {#sec-ml-systems-textbook-glossary-system-content-authors-7406}

#### Adding New Terms {#sec-ml-systems-textbook-glossary-system-adding-new-terms-b437}
1. Edit the appropriate chapter glossary file: `quarto/contents/core/[chapter]/[chapter]_glossary.json`
2. Add your term following the JSON schema
3. Run the rebuild process (see below)

#### Editing Existing Terms   {#sec-ml-systems-textbook-glossary-system-editing-existing-terms-3624}
1. Find the term in its source chapter glossary
2. Update the definition or metadata
3. Run the rebuild process to propagate changes

### For Maintainers {#sec-ml-systems-textbook-glossary-system-maintainers-975c}

#### Complete Rebuild Process {#sec-ml-systems-textbook-glossary-system-complete-rebuild-process-78a5}
```bash
# 1. Aggregate chapter glossaries into global glossary
python3 tools/scripts/glossary/build_global_glossary.py

# 2. Generate the published glossary page
python3 tools/scripts/glossary/generate_glossary.py

# 3. Optional: Run quality assurance
python3 tools/scripts/glossary/smart_consolidation.py  # Analysis only
python3 tools/scripts/glossary/rule_based_consolidation.py  # Apply fixes
```

#### Quality Assurance Workflow {#sec-ml-systems-textbook-glossary-system-quality-assurance-workflow-2e81}
```bash
# Check for similar terms that need consolidation
python3 tools/scripts/glossary/smart_consolidation.py

# Apply academic best practices
python3 tools/scripts/glossary/rule_based_consolidation.py

# Rebuild after fixes
python3 tools/scripts/glossary/build_global_glossary.py
python3 tools/scripts/glossary/generate_glossary.py
```

## 📊 Current Statistics {#sec-ml-systems-textbook-glossary-system-current-statistics-bb2b}

- **Total Terms**: 611 unique terms (down from 810 raw terms after deduplication)
- **Chapter Coverage**: 22 chapters with individual glossaries
- **Multi-Chapter Terms**: 104 terms that appear in multiple chapters
- **Single-Chapter Terms**: 507 terms specific to one chapter
- **Alphabetical Sections**: 26 letter sections (A-Z)
- **Similar Term Groups**: 115 groups identified for potential consolidation

## 🔧 Technical Features {#sec-ml-systems-textbook-glossary-system-technical-features-4036}

### Automatic Cross-Reference Resolution {#sec-ml-systems-textbook-glossary-system-automatic-crossreference-resolution-f11f}
The system automatically discovers actual section IDs from chapter files rather than relying on hardcoded mappings. This ensures cross-references always work correctly.

### Intelligent Deduplication {#sec-ml-systems-textbook-glossary-system-intelligent-deduplication-0f55}
- **Singular/Plural Merging**: "adversarial example" + "adversarial examples" → "adversarial example"
- **Acronym Standardization**: "GPU" + "graphics processing unit" → "graphics processing unit (GPU)"
- **Formatting Consistency**: "Moore's law" vs "moores law" → "Moore's law"

### Multi-Chapter Attribution {#sec-ml-systems-textbook-glossary-system-multichapter-attribution-410a}
Terms appearing in multiple chapters are properly attributed:
- **Single Chapter**: "Chapter: @sec-introduction"
- **Multiple Chapters**: "Appears in: @sec-dl-primer, @sec-dnn-architectures, @sec-frameworks"

### Interactive Integration {#sec-ml-systems-textbook-glossary-system-interactive-integration-0b2a}
The glossary integrates with the book through:
- **Lua Filters**: Automatic term detection and tooltip injection
- **CSS Styling**: Responsive tooltips that don't get cut off
- **Cross-References**: Clickable links between glossary and chapters

## 🚀 Future Enhancements {#sec-ml-systems-textbook-glossary-system-future-enhancements-890b}

### Planned Features {#sec-ml-systems-textbook-glossary-system-planned-features-8032}
- **LLM Integration**: Automatic term consolidation using Claude/GPT APIs
- **Term Validation**: Automatic checking for undefined terms used in chapters
- **Synonym Detection**: Advanced similarity detection for related concepts
- **Export Formats**: PDF, EPUB, and standalone HTML versions

### Quality Improvements {#sec-ml-systems-textbook-glossary-system-quality-improvements-fccd}
- **Definition Quality Scoring**: Automatic assessment of definition clarity
- **Coverage Analysis**: Detection of missing key terms
- **Consistency Checking**: Validation of term usage across chapters

## 📋 Maintenance Notes {#sec-ml-systems-textbook-glossary-system-maintenance-notes-d4ad}

### Best Practices {#sec-ml-systems-textbook-glossary-system-best-practices-8327}
1. **Chapter glossaries are the source of truth** - always edit there first
2. **Run full rebuild after any changes** to ensure consistency
3. **Use quality assurance scripts** regularly to catch issues
4. **Test cross-references** in full website builds, not isolated files
5. **Review consolidation logs** when running automated tools

### Common Issues {#sec-ml-systems-textbook-glossary-system-common-issues-afab}
- **Cross-references not working**: Usually means viewing single file vs full website
- **Duplicate terms**: Use smart consolidation to identify and fix
- **Missing definitions**: Check individual chapter files for completeness
- **Broken acronyms**: Update rule-based consolidation for new patterns

## 🆘 Troubleshooting {#sec-ml-systems-textbook-glossary-system-troubleshooting-8d50}

### Cross-References Show as "?" {#sec-ml-systems-textbook-glossary-system-crossreferences-show-f00a}
This usually means you're viewing the glossary in isolation. Cross-references only work in the full website build:
```bash
quarto render  # Full website
# OR
quarto preview # Development server
```

### Terms Not Appearing {#sec-ml-systems-textbook-glossary-system-terms-appearing-54ce}
1. Check if term exists in chapter glossary
2. Verify JSON syntax is valid
3. Run build script to regenerate global glossary
4. Check for case sensitivity issues

### Consolidation Issues {#sec-ml-systems-textbook-glossary-system-consolidation-issues-d66b}
1. Review similarity detection results: `smart_consolidation.py`
2. Check consolidation logs in `quarto/contents/data/`
3. Manually edit problem terms in chapter files
4. Re-run the full rebuild process

---

**Last Updated**: September 2024
**System Version**: 1.0
**Total Terms**: 611
**Coverage**: Complete for all 22 chapters
