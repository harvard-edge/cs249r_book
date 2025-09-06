# Footnote Placement Rules

## CRITICAL: Footnote Definitions Must Follow Their Usage Immediately

### The Rule
**Footnote definitions MUST be placed immediately after the paragraph containing their reference(s).**

### Why This Matters
1. **Easier editing** - You can see the term and its definition together
2. **Better review** - Context is clear when term and definition are adjacent
3. **Logical flow** - Readers (and editors) don't have to jump around
4. **Maintainability** - Changes to paragraphs and their footnotes stay together

## Correct Placement Examples

### ✅ CORRECT - Definition immediately after paragraph
```markdown
The evolution of AI began at the Dartmouth Conference[^fn-dartmouth] in 1956, 
where pioneers like John McCarthy and Claude Shannon[^fn-shannon] first coined 
the term "artificial intelligence."

[^fn-dartmouth]: **Dartmouth Conference**: The 1956 summer workshop where AI was born...
[^fn-shannon]: **Claude Shannon**: Father of information theory who laid foundations...

The next breakthrough came with the perceptron[^fn-perceptron] in 1957, which 
showed how machines could learn from examples.

[^fn-perceptron]: **Perceptron**: First artificial neural network that could learn...
```

### ❌ INCORRECT - Bulk definitions at section end
```markdown
The evolution of AI began at the Dartmouth Conference[^fn-dartmouth] in 1956, 
where pioneers like John McCarthy and Claude Shannon[^fn-shannon] first coined 
the term "artificial intelligence."

The next breakthrough came with the perceptron[^fn-perceptron] in 1957, which 
showed how machines could learn from examples.

[Many paragraphs later...]

[^fn-dartmouth]: **Dartmouth Conference**: The 1956 summer workshop where AI was born...
[^fn-shannon]: **Claude Shannon**: Father of information theory who laid foundations...
[^fn-perceptron]: **Perceptron**: First artificial neural network that could learn...
```

## Implementation Strategy for Footnote Agent

### Phase 1: Planning
1. Read through the entire chapter
2. Identify terms needing footnotes
3. Create a plan listing:
   - Which paragraph contains each term
   - What footnote will be added
   - Where definition will be placed (right after that paragraph)

### Phase 2: Execution
Process the chapter paragraph by paragraph:
1. Add inline references `[^fn-term]` where needed in the paragraph
2. Immediately after that paragraph, add ALL definitions for terms in that paragraph
3. Move to next paragraph
4. Repeat

### Example Workflow
```python
# Pseudocode for footnote placement
for paragraph in chapter:
    terms_to_footnote = identify_terms(paragraph)
    
    if terms_to_footnote:
        # Add inline references
        paragraph = add_inline_refs(paragraph, terms_to_footnote)
        
        # Add definitions immediately after
        definitions = create_definitions(terms_to_footnote)
        insert_after_paragraph(definitions)
```

## Special Cases

### Multiple Terms in One Sentence
If a sentence has multiple footnoted terms, all definitions go after that paragraph:
```markdown
Modern ML relies on GPUs[^fn-gpu], TPUs[^fn-tpu], and specialized ASICs[^fn-asic].

[^fn-gpu]: **Graphics Processing Unit**: Originally for gaming, now essential for ML...
[^fn-tpu]: **Tensor Processing Unit**: Google's custom AI accelerator...
[^fn-asic]: **Application-Specific Integrated Circuit**: Custom chips designed for specific tasks...
```

### Long Paragraphs
Even if a paragraph is very long, its footnotes still go immediately after it, not broken up.

### Lists and Code Blocks
Footnotes in lists or before code blocks should have definitions after the list/code block ends:
```markdown
Key concepts include:
- Supervised learning[^fn-supervised]
- Unsupervised learning[^fn-unsupervised]
- Reinforcement learning[^fn-reinforcement]

[^fn-supervised]: **Supervised Learning**: Learning from labeled examples...
[^fn-unsupervised]: **Unsupervised Learning**: Finding patterns without labels...
[^fn-reinforcement]: **Reinforcement Learning**: Learning through trial and reward...
```

## Verification Checklist

After adding footnotes, verify:
- [ ] Every footnote definition appears right after its paragraph
- [ ] No "orphaned" definitions far from their references
- [ ] No bulk footnote sections at end of document
- [ ] Related footnotes (same paragraph) are grouped together
- [ ] Reading flow is natural - term → definition → next content

## Benefits of This Approach

1. **For Editors**: Easy to review term and definition together
2. **For Readers**: Natural flow, immediate clarification
3. **For Maintenance**: Changes to content and footnotes stay synchronized
4. **For Quarto**: No issues with rendering or cross-references
5. **For Agents**: Clear structure for future updates