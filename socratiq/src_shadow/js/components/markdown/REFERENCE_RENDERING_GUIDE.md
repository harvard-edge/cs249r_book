# Reference Rendering System Guide

## Overview

The Reference Rendering System enhances the STREAMDOWN markdown renderer with footnote reference functionality. It converts markdown footnotes `[^ref-1]` into interactive pill-style links with tooltips and scroll-to-element functionality.

## Features

### ✅ **Pill-Style Reference Links**
- Converts `[^ref-1]` to clickable pill-style links `(1)`
- Hover effects with scaling and color changes
- Accessible keyboard navigation

### ✅ **Interactive Tooltips**
- Hover tooltips showing paragraph content and ID
- Smart positioning to stay within viewport
- Includes "Scroll to paragraph" link

### ✅ **Scroll-to-Element Functionality**
- Smooth scrolling to referenced paragraphs
- Temporary highlighting of target paragraphs
- Works with `data-fuzzy-id` attributes

### ✅ **Full Markdown Compatibility**
- Works with all existing STREAMDOWN features
- Compatible with Mermaid diagrams, LaTeX, tables
- Preserves all custom containers and styling

## Usage

### Basic Markdown Syntax

```markdown
This is a paragraph with a reference[^ref-1] to important information.

[^ref-1]: (id=p-1757047226191-5rhhu0559) This is the definition text that will appear in the tooltip.
```

### Multiple References

```markdown
Data protection involves multiple strategies[^ref-1] and techniques[^ref-2] to ensure data quality.

[^ref-1]: (id=p-1757047226191-5rhhu0559) First definition with paragraph ID.

[^ref-2]: (id=p-1757047226193-lv53eb2n8) Second definition with different paragraph ID.
```

### Reference Format

The reference definition format is:
```
[^ref-id]: (id=paragraph-id) Definition text
```

Where:
- `ref-id`: Unique reference identifier (e.g., `ref-1`, `ref-2`)
- `paragraph-id`: The `data-fuzzy-id` of the target paragraph
- `Definition text`: The content shown in the tooltip

## Integration

### Automatic Integration

The reference rendering system is automatically integrated with the STREAMDOWN markdown renderer. No additional setup is required.

### Manual Integration

If you need to use the reference renderer independently:

```javascript
import { processReferences, initializeReferenceRenderer } from './reference_renderer.js';

// Initialize the renderer
initializeReferenceRenderer(shadowElement);

// Process markdown with references
const processedHTML = processReferences(markdownText, htmlContent);
```

## Styling

### Default Styles

The system includes comprehensive CSS styling:

```css
.reference-pill {
  background: #e5e7eb;
  color: #374151;
  padding: 0.125rem 0.375rem;
  border-radius: 0.75rem;
  font-size: 0.75rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.reference-pill:hover {
  background: #d1d5db;
  transform: scale(1.05);
}
```

### Dark Mode Support

The system automatically adapts to dark mode:

```css
@media (prefers-color-scheme: dark) {
  .reference-pill {
    background: #374151;
    color: #e5e7eb;
  }
}
```

### Custom Styling

You can override the default styles by targeting the CSS classes:

```css
.reference-pill {
  background: your-custom-color;
  /* Your custom styles */
}
```

## User Experience

### Reference Link Interaction

1. **Hover**: Shows tooltip with paragraph content and ID
2. **Click**: Smoothly scrolls to the referenced paragraph
3. **Highlight**: Target paragraph is temporarily highlighted
4. **Accessibility**: Full keyboard navigation support

### Tooltip Content

The tooltip displays:
- Reference number
- Paragraph ID
- Definition text
- "Scroll to paragraph" link

### Scroll Behavior

- Smooth scrolling with 100px offset from top
- Temporary highlight with blue border and background
- 3-second highlight duration
- Works with any element having `data-fuzzy-id`

## Technical Details

### Parsing Algorithm

1. **Extract References**: Find all `[^ref-id]` patterns
2. **Extract Definitions**: Find all `[^ref-id]: definition` patterns
3. **Generate Links**: Convert references to interactive HTML
4. **Position Tooltips**: Calculate optimal tooltip positioning

### Performance Optimizations

- **Lazy Processing**: Only processes content with references
- **Efficient DOM Queries**: Caches paragraph elements
- **Minimal Re-renders**: Updates only changed references
- **Event Delegation**: Efficient event handling

### Browser Compatibility

- **Modern Browsers**: Full support (Chrome, Firefox, Safari, Edge)
- **ES6 Features**: Uses modern JavaScript features
- **CSS Grid/Flexbox**: For tooltip positioning
- **Smooth Scrolling**: Uses native `scrollTo` with `behavior: 'smooth'`

## Error Handling

### Graceful Degradation

- **Missing Definitions**: Keeps original `[^ref-1]` text
- **Invalid Paragraph IDs**: Logs warning, no scroll action
- **Tooltip Errors**: Falls back to simple text display
- **Build Errors**: Returns original content unchanged

### Debug Information

Enable debug logging:

```javascript
// The system logs warnings for missing paragraphs
console.warn(`Paragraph with ID ${paragraphId} not found`);
```

## Testing

### Test File

Use the included test file to verify functionality:

```javascript
import { runReferenceRendererTests } from './test_reference_renderer.js';
runReferenceRendererTests();
```

### Demo Page

Open `demo_reference_rendering.html` in a browser to see the system in action.

### Manual Testing

1. Create markdown with references
2. Process through STREAMDOWN renderer
3. Verify pill links appear
4. Test hover tooltips
5. Test scroll functionality

## Troubleshooting

### Common Issues

**References not appearing:**
- Check markdown syntax: `[^ref-1]` and `[^ref-1]: definition`
- Verify paragraph IDs match `data-fuzzy-id` attributes
- Ensure reference renderer is initialized

**Tooltips not showing:**
- Check CSS z-index conflicts
- Verify tooltip positioning calculations
- Test in different browsers

**Scroll not working:**
- Verify paragraph has `data-fuzzy-id` attribute
- Check for JavaScript errors in console
- Test with different paragraph IDs

### Debug Steps

1. **Check Console**: Look for error messages
2. **Inspect HTML**: Verify reference links are generated
3. **Test CSS**: Ensure styles are applied correctly
4. **Verify IDs**: Check paragraph IDs match references

## Future Enhancements

### Planned Features

- **Reference Counter**: Show total references in document
- **Reference Index**: Generate reference list at end
- **Cross-References**: Link between related references
- **Export Support**: Include references in exported content

### Extension Points

The system is designed for easy extension:

```javascript
// Custom reference processing
function customReferenceProcessor(markdownText, htmlContent) {
  // Your custom logic
  return processedHTML;
}
```

## API Reference

### Functions

#### `parseReferences(markdownText)`
Parses markdown text to extract references and definitions.

**Parameters:**
- `markdownText` (string): The markdown content to parse

**Returns:**
- `Object`: `{ references: string[], definitions: Object }`

#### `processReferences(markdownText, htmlContent)`
Processes HTML content to add reference rendering.

**Parameters:**
- `markdownText` (string): Original markdown text
- `htmlContent` (string): Processed HTML content

**Returns:**
- `string`: HTML with reference rendering

#### `initializeReferenceRenderer(shadowElement)`
Initializes the reference rendering system.

**Parameters:**
- `shadowElement` (HTMLElement): The shadow DOM element

#### `scrollToParagraph(paragraphId)`
Scrolls to a paragraph by its ID.

**Parameters:**
- `paragraphId` (string): The paragraph ID to scroll to

## Examples

### Complete Example

```markdown
# Machine Learning Systems

Data protection is crucial for ML systems[^ref-1]. It involves multiple strategies[^ref-2] to ensure data quality and prevent issues like data poisoning[^ref-3].

## Key Concepts

Understanding distribution shifts[^ref-1] helps practitioners design better systems. Data cleaning techniques[^ref-2] are essential for maintaining data quality.

[^ref-1]: (id=p-1757047226191-5rhhu0559) Distribution shifts arise from a variety of underlying mechanisms—both natural and system-driven. Understanding these mechanisms helps practitioners detect, diagnose, and design mitigation strategies.

[^ref-2]: (id=p-1757047226193-lv53eb2n8) Data poisoning can be avoided by cleaning data, which involves identifying and removing or correcting noisy, incomplete, or inconsistent data points. Techniques such as data deduplication, missing value imputation, and outlier removal can be applied to improve the quality of the training data.

[^ref-3]: (id=p-1757047226199-nzlrt9lvd) A model trained on daytime traffic data failing at night exemplifies distribution shifts, where the training and deployment environments differ, challenging the model's ability to generalize.
```

This will render as:
- Clickable pill links `(1)`, `(2)`, `(3)` in the text
- Hover tooltips showing the definitions
- Click-to-scroll functionality to the referenced paragraphs
- Temporary highlighting of target paragraphs

## Support

For issues or questions about the reference rendering system:

1. Check this documentation
2. Review the test files
3. Examine the demo page
4. Check browser console for errors
5. Verify markdown syntax and paragraph IDs

The system is designed to be robust and provide helpful error messages when issues occur.
