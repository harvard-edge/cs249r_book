# Migration Guide: Old markdown.js → streamdown_markdown.js

## Overview
This guide helps you migrate from the old `markdown.js` to the new `streamdown_markdown.js` that integrates **actual STREAMDOWN functionality** for powerful streaming markdown while preserving all custom features.

## What's Changed

### ✅ What STREAMDOWN Now Handles (Using Real STREAMDOWN Libraries):
- **Mermaid diagrams** - Using STREAMDOWN's mermaid library with proper rendering
- **Code blocks** - Using STREAMDOWN's shiki highlighter for syntax highlighting  
- **Math expressions** - Using STREAMDOWN's katex integration for LaTeX rendering
- **Incomplete markdown parsing** - Using STREAMDOWN's parseIncompleteMarkdown algorithm
- **Markdown parsing** - Using STREAMDOWN's marked library with GFM support
- **Security features** - Built-in protection against malicious content
- **Streaming optimization** - Designed for AI-powered streaming

### 🔧 What We Preserved (Custom Features):
- **Custom Containers**: `spoiler`, `info`, `network-warning`, `loader`
- **Editable Input Components**: `.editable-text` with `.enter-button`
- **Question Processing**: `%%%` marker and follow-up question generation
- **Custom Event System**: `aiActionCompleted` events for editable text
- **Content Normalization**: `normalizeMarkdownText` function

### 🚫 What We Removed:
- **Wikipedia links** - Double backslash pattern removed
- **Redundant code** - Old markdown-it configurations that STREAMDOWN handles better
- **Manual highlighting** - Replaced with STREAMDOWN's shiki highlighter
- **Manual mermaid** - Replaced with STREAMDOWN's mermaid integration

## Migration Steps

### Step 1: Update Imports in index.js

**Before:**
```javascript
import { updateMarkdownPreview } from "./components/markdown/markdown";
import { 
  initiateMarkdown, 
  initializeSpoilerEditing,
  reinitializeEditableInputs,
} from "./components/markdown/markdown";
```

**After:**
```javascript
import { updateMarkdownPreview } from "./components/markdown/streamdown_markdown";
import { 
  initiateMarkdown, 
  initializeSpoilerEditing,
  reinitializeEditableInputs,
  StreamdownMarkdownRenderer
} from "./components/markdown/streamdown_markdown";
```

### Step 2: Update Any Other Import References

Search for any other files that import from the old markdown.js:
```bash
grep -r "from.*markdown/markdown" ../js/
```

Update them to use `streamdown_markdown` instead.

### Step 3: Test the STREAMDOWN Integration

Use the enhanced test file to verify STREAMDOWN features work:
```javascript
import { 
  testStreamdownIntegration, 
  testStreamingRenderer, 
  testStreamdownFeatures 
} from './test_streamdown_integration.js';

// Test basic STREAMDOWN functionality
testStreamdownIntegration();

// Test individual STREAMDOWN features
await testStreamdownFeatures();

// Test streaming (if you have a shadow element)
await testStreamingRenderer(shadowElement);
```

### Step 4: Use STREAMDOWN Streaming Features

The new implementation includes powerful streaming capabilities:

```javascript
// Create a STREAMDOWN streaming renderer
const renderer = new StreamdownMarkdownRenderer(shadowElement);

// Start streaming with STREAMDOWN's incomplete markdown parsing
await renderer.startStream('markdown-preview');

// Add chunks as they arrive from AI (with STREAMDOWN processing)
chunks.forEach(async (chunk, index) => {
  setTimeout(async () => {
    await renderer.addChunk(chunk, 'markdown-preview');
  }, index * 100);
});

// Complete streaming
setTimeout(async () => {
  await renderer.completeStream('markdown-preview');
}, chunks.length * 100);
```

## STREAMDOWN Features Now Available

### 1. **Incomplete Markdown Parsing**
STREAMDOWN automatically handles incomplete markdown during streaming:
- Incomplete bold: `**text` → `**text**`
- Incomplete italic: `*text` → `*text*`
- Incomplete code: `` `code `` → `` `code` ``
- Incomplete links: `[text` → `[text](streamdown:incomplete-link)`
- Incomplete math: `$math` → `$math$`

### 2. **Advanced Syntax Highlighting**
Using STREAMDOWN's shiki highlighter:
```javascript
// Supports 50+ languages with proper theming
const highlighted = await highlightCode(code, 'javascript', 'github-light');
```

### 3. **Mermaid Diagram Rendering**
Using STREAMDOWN's mermaid integration:
```javascript
// Automatic diagram rendering with error handling
const svg = await renderMermaid(mermaidCode, uniqueId);
```

### 4. **Math Expression Rendering**
Using STREAMDOWN's katex integration:
```javascript
// Inline math: $E = mc^2$
// Block math: $$\sum_{i=1}^{n} x_i$$
```

### 5. **Security Features**
Built-in protection against:
- XSS attacks
- Malicious links
- Unsafe content

## API Compatibility

### Functions That Work Exactly the Same:
- `updateMarkdownPreview(text, clone, isResearch, markdownPreviewId)`
- `initiateMarkdown(shadowElement)`
- `initializeSpoilerEditing(preview)`
- `reinitializeEditableInputs(shadowRoot)`

### New STREAMDOWN Functions Available:
- `StreamdownMarkdownRenderer` - Class for STREAMDOWN-powered streaming
- `parseIncompleteMarkdown(text)` - STREAMDOWN's incomplete markdown parsing
- `highlightCode(code, language, theme)` - STREAMDOWN's syntax highlighting
- `renderMermaid(chart, containerId)` - STREAMDOWN's mermaid rendering
- `renderMath(text)` - STREAMDOWN's math rendering

## Custom Containers Still Work

All your existing custom containers continue to work exactly the same:

```markdown
:::spoiler Editable Content
This content is editable with a submit button.
:::

:::info
This is an information box.
:::

:::network-warning
This is a network warning.
:::

:::loader
This shows a loading animation.
:::
```

## Question Processing Still Works

The `%%%` marker and question processing works exactly the same:

```markdown
# Some content here

%%%

What is the main topic?
How does this work?
```

## STREAMDOWN Benefits

1. **🚀 Better Performance**: STREAMDOWN is optimized for streaming
2. **🎨 Better Rendering**: Professional syntax highlighting and math rendering
3. **🔒 Enhanced Security**: Built-in protection against malicious content
4. **📊 Better Diagrams**: Improved Mermaid rendering with error handling
5. **🧮 Math Support**: Full LaTeX/Katex integration
6. **🔄 Streaming Optimized**: Handles incomplete markdown gracefully
7. **🧹 Cleaner Code**: Removed redundant markdown parsing code
8. **📦 Smaller Bundle**: Less code to maintain
9. **🔧 Same Features**: All custom functionality preserved

## Testing STREAMDOWN Features

### Test Incomplete Markdown Parsing:
```javascript
const incompleteText = "This is **incomplete bold text";
const fixed = parseIncompleteMarkdown(incompleteText);
// Result: "This is **incomplete bold text**"
```

### Test Syntax Highlighting:
```javascript
const code = 'console.log("Hello World");';
const highlighted = await highlightCode(code, 'javascript');
// Returns properly highlighted HTML
```

### Test Mermaid Rendering:
```javascript
const mermaidCode = 'graph TD\n    A[Start] --> B[End]';
const svg = await renderMermaid(mermaidCode, 'test-id');
// Returns rendered SVG
```

### Test Math Rendering:
```javascript
const mathText = 'The formula is $E = mc^2$ and $$\\sum_{i=1}^{n} x_i$$';
const rendered = renderMath(mathText);
// Returns HTML with rendered math
```

## Troubleshooting

### If Something Breaks:

1. **Check imports**: Make sure all imports point to `streamdown_markdown`
2. **Test STREAMDOWN features**: Use the test functions to verify functionality
3. **Check console**: Look for any error messages
4. **Verify dependencies**: Ensure all STREAMDOWN libraries are installed

### Common Issues:

- **Import errors**: Update all import paths
- **Missing dependencies**: Run `npm install marked mermaid shiki katex`
- **Container not rendering**: Check container syntax
- **Editable inputs not working**: Verify event listeners are attached
- **STREAMDOWN features not working**: Check if libraries are properly initialized

## Performance Improvements

With STREAMDOWN integration, you'll see:
- **Faster rendering** of code blocks and diagrams
- **Better streaming performance** with incomplete markdown handling
- **Reduced bundle size** by removing redundant code
- **Professional appearance** with STREAMDOWN's styling

## Rollback Plan

If you need to rollback:
1. Revert import changes in `index.js`
2. The old `markdown.js` file is still there
3. No data loss - all functionality preserved

## Next Steps

1. ✅ Update imports
2. ✅ Test STREAMDOWN functionality  
3. ✅ Test custom containers
4. ✅ Test editable inputs
5. ✅ Test question processing
6. ✅ Test streaming features
7. ✅ Remove old `markdown.js` file (after confirming everything works)

## Support

If you encounter any issues:
1. Check the test file for examples
2. Verify all imports are updated
3. Test each STREAMDOWN feature individually
4. Check browser console for errors
5. Use the test functions to debug specific features

The new implementation gives you the full power of STREAMDOWN while preserving all your unique custom features!