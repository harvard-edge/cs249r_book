# Chapter Glossaries

This directory contains chapter-specific glossary term lists that track which terms are used in each chapter.

## Structure

Each file follows the naming pattern: `{chapter_name}_terms.yml`

## Format

```yaml
chapter: introduction
chapter_title: "Introduction to ML Systems"
terms_used:
  - artificial_intelligence
  - machine_learning  
  - deep_learning
  - neural_networks
  - ml_systems
terms_introduced:  # First appearance in book
  - ml_systems
  - systems_engineering
key_terms:  # Most important for this chapter
  - artificial_intelligence
  - machine_learning
  - ml_systems
```

## Purpose

1. **Progressive Learning**: Track which terms students encounter when
2. **Chapter Summaries**: Generate "Key Terms" boxes for each chapter
3. **Selective Marking**: Only mark terms relevant to current chapter
4. **Study Guides**: Create chapter-specific study materials
5. **Dependency Tracking**: Ensure prerequisites are introduced first

## Generation

These files are generated automatically by the glossary-builder agent when processing chapters.

## Usage

The auto-glossary filter can optionally use these files to:
- Only mark terms introduced up to current chapter
- Highlight new terms in each chapter
- Create progressive glossaries that grow through the book