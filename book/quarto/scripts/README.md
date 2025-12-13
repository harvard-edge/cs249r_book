# Post-Render Scripts

This directory contains scripts that run after Quarto builds to fix various issues.

## fix_cross_references.py

### Problem It Solves

When using **selective rendering** to speed up builds (only building index + introduction instead of all 20+ chapters), Quarto cannot resolve cross-references to chapters that weren't built. This results in broken references appearing as `?@sec-chapter-name` in the HTML output.

This particularly affects:
- **Glossary**: Contains 800+ cross-references to all chapters (hence the original script name)
- **Introduction**: References many other chapters for context
- **Any chapter** that references other chapters

### How It Works

1. **Runs automatically** as a post-render hook after Quarto finishes building
2. **Scans ALL HTML files** in the `_build/html/` directory
3. **Finds unresolved references** that appear as `<strong>?@sec-xxx</strong>`
4. **Converts them to proper links**: `<strong><a href="../path/to/chapter.html#sec-xxx">Chapter Title</a></strong>`

### Configuration

The script is configured as a post-render hook in the Quarto configuration files:

```yaml
# In config/_quarto-html.yml
project:
  post-render:
    - scripts/clean_svgs.py
    - scripts/fix_cross_references.py  # Fixes cross-references
```

### Maintenance

When adding new chapters to the book:

1. Add the chapter's section ID to `CHAPTER_MAPPING` dictionary
2. Add the chapter's display title to `CHAPTER_TITLES` dictionary
3. Ensure the section ID matches what's in the `.qmd` file (e.g., `{#sec-new-chapter}`)

Example:
```python
CHAPTER_MAPPING = {
    # ... existing chapters ...
    "sec-new-chapter": "../core/new_chapter/new_chapter.html#sec-new-chapter",
}

CHAPTER_TITLES = {
    # ... existing chapters ...
    "sec-new-chapter": "New Chapter Title",
}
```

### Manual Testing

```bash
# Test on all HTML files
python3 scripts/fix_cross_references.py

# Test on specific file
python3 scripts/fix_cross_references.py _build/html/contents/backmatter/glossary/glossary.html
```

### Output

The script provides clear output showing what it fixed:

```
🔗 [Cross-Reference Fix] Scanning 3 HTML files...
✅ Fixed 850 cross-references in 2 files:
   📄 contents/backmatter/glossary/glossary.html: 810 refs
   📄 contents/core/introduction/introduction.html: 40 refs
```

## clean_svgs.py

Cleans up SVG files generated during the build process.

---

## Why These Scripts Exist

These post-render scripts enable **fast iterative development** by allowing selective chapter builds while maintaining a fully functional website with working cross-references. Without them, developers would need to build all 20+ chapters every time (taking minutes) just to test changes to a single chapter.
