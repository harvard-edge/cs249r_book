# Glossary Implementation Notes

## Analysis of debruine/quarto-glossary Extension

### How It Works:
1. **Shortcode-based**: Uses `{{< glossary "term" >}}` syntax
2. **Manual marking**: Authors must mark each term occurrence
3. **HTML-only**: Returns `pandoc.Null()` for non-HTML formats
4. **Simple YAML**: Reads flat key-value pairs from YAML file
5. **CSS styling**: Purple underlined text with hover tooltips

### Limitations for Our Use Case:
- **No automatic detection**: Requires manual markup (violates clean source principle)
- **Single format**: Only works for HTML, not PDF/EPUB
- **No smart matching**: No plural handling, acronym expansion, etc.
- **No occurrence control**: Can't limit to first occurrence only
- **Simple structure**: Doesn't support our hierarchical glossary

## Our Custom Implementation

### Design Principles:
1. **Clean source files**: No manual markup required
2. **Multi-format support**: Different rendering for HTML/PDF/EPUB
3. **Smart detection**: Handle plurals, acronyms, case variations
4. **Configurable marking**: First occurrence only, skip code blocks
5. **Integration**: Works with existing filter pipeline

### Implementation Strategy:

#### Phase 1: Basic Auto-Detection
- `auto-glossary.lua`: Simple term detection and marking
- Supports basic term matching
- Format-specific rendering

#### Phase 2: Advanced Features  
- `auto-glossary-advanced.lua`: Full implementation
- Hierarchical glossary support
- Smart term matching (word boundaries, variants)
- Integration with existing sidenote system for PDF
- Configurable behavior per format

### Rendering by Format:

#### HTML:
- Bootstrap tooltips on hover
- Optional click popups
- Links to glossary page

#### PDF/LaTeX:
- Sidenotes (integrate with existing system)
- Footnotes (alternative)
- Hyperlinks to glossary section

#### EPUB:
- Links to glossary chapter
- Inline definitions (optional)

### Configuration:
```yaml
filter-metadata:
  auto-glossary:
    glossary-file: "data/master_glossary.yml"
    mark-first-only: true
    skip-code: true
    skip-headings: true
    formats:
      html: tooltip
      pdf: sidenote
      epub: link
```

### Integration Points:
1. Add to filter pipeline in `_quarto.yml`
2. Ensure it runs before other text-processing filters
3. Coordinate with cross-reference injection filter
4. Share glossary data with other filters if needed

### Testing Strategy:
1. Test basic term detection
2. Verify format-specific output
3. Check first-occurrence-only logic
4. Ensure code blocks are skipped
5. Test with full book build

### Future Enhancements:
- Glossary term analytics (which terms are used where)
- Context-aware definitions (different definitions based on chapter)
- Automatic acronym expansion on first use
- Machine learning for term importance ranking
- Interactive glossary navigation in HTML