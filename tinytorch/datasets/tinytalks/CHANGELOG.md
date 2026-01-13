# Changelog

All notable changes to the TinyTalks dataset will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-28

### Added
- Initial release of TinyTalks dataset
- 301 Q&A pairs across 5 difficulty levels:
  - Level 1: Greetings & Identity (47 pairs)
  - Level 2: Simple Facts (82 pairs)
  - Level 3: Basic Math (45 pairs)
  - Level 4: Common Sense Reasoning (87 pairs)
  - Level 5: Multi-turn Context (40 pairs)
- Train/validation/test splits (70/15/15)
- Comprehensive README with usage examples
- DATASHEET.md following "Datasheets for Datasets" best practices
- CC BY 4.0 license
- Generation script (`scripts/generate_tinytalks.py`)
- Validation script (`scripts/validate_dataset.py`)
- Statistics script (`scripts/stats.py`)
- Example usage script (`examples/demo_usage.py`)

### Dataset Statistics
- Total size: ~17.5 KB
- Character vocabulary: 65 unique characters
- Word vocabulary: 865 unique words
- Average question length: 4.8 words (21.6 characters)
- Average answer length: 6.1 words (29.0 characters)

### Validation
- ✅ UTF-8 encoding
- ✅ Unix line endings (LF)
- ✅ No duplicate questions
- ✅ No empty questions or answers
- ✅ Proper punctuation
- ✅ Balanced splits with no overlap

---

## [Unreleased]

### Planned for v1.1.0
- Add 50 more Level 4-5 pairs for better reasoning
- Expand math questions to include simple multiplication tables
- Add more conversational context pairs

### Planned for v2.0.0
- Multi-language support (Spanish, French)
- Expanded to 500+ pairs
- Add difficulty scores for each Q&A pair

### Planned for v3.0.0
- Expand to 1,000+ pairs
- Add more complex reasoning tasks
- Include multi-hop questions
- Add entity recognition annotations

---

## Version History

- **1.0.0** (2025-01-28) - Initial release

