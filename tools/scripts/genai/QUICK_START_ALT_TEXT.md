# Quick Start: Testing Alt Text Generation

## Immediate Next Steps

### 1. Install Dependencies (if not already installed)
```bash
cd /Users/VJ/GitHub/MLSysBook/tools/scripts/genai
pip install -r requirements.txt
```

### 2. Setup Ollama for Local Testing
```bash
# Install Ollama from https://ollama.ai (if not installed)
# Then pull the vision model
ollama pull llava
```

### 3. Test with Introduction Chapter (Dry Run)
```bash
cd /Users/VJ/GitHub/MLSysBook
python tools/scripts/genai/generate_alt_text.py --chapter intro --provider ollama --dry-run
```

This will:
1. Build the introduction chapter using `./binder html intro`
2. Extract all figures from the built HTML
3. Generate alt text using Ollama's llava model
4. Show what changes would be made (but not actually modify files)

### 4. Review the Output
Check the console output and `generate_alt_text.log` to see:
- How many figures were found
- What alt text was generated
- Which files would be updated

### 5. If Happy, Run for Real
```bash
python tools/scripts/genai/generate_alt_text.py --chapter intro --provider ollama
```

This will actually modify your `.qmd` files to add `fig-alt` attributes.

## What This Script Does Differently

**Key Innovation**: We use the **rendered output** as the source of truth because:
1. Your TikZ code is unreadable without rendering
2. The built HTML has all figures with clean IDs
3. We can see exactly what readers see
4. Figure IDs provide perfect matching back to source

**Workflow**:
```
Source .qmd → Build → Rendered HTML → Extract Figures → Vision AI → Alt Text → Update Source .qmd
     ↑                                                                               |
     └───────────────────────────────────────────────────────────────────────────────┘
```

## Example Output

For a figure like this in your HTML:
```html
<figure class="quarto-float quarto-float-fig figure">
  <figcaption>Figure 2: AI Development Timeline</figcaption>
  <img src="introduction_files/mediabag/25cf57367...svg">
</figure>
```

The script will:
1. Extract figure ID: `fig-ai-timeline`
2. Download/locate the image
3. Send to vision model with caption context
4. Get alt text: "Timeline showing evolution from symbolic AI in 1950s to modern LLMs"
5. Find in source: `![Timeline](image.svg){#fig-ai-timeline}`
6. Update to: `![Timeline](image.svg){#fig-ai-timeline fig-alt="Timeline showing..."}`

## Testing Tips

1. **Start small**: Test with just the intro chapter first
2. **Use dry-run**: Always do a dry run first to preview changes
3. **Check the log**: `generate_alt_text.log` has detailed progress
4. **Review quality**: Read the generated alt text to ensure it's useful
5. **Iterate**: If quality isn't great, you can adjust the prompt in the script

## Troubleshooting First Run

### If binder build fails:
```bash
# Test binder directly
cd /Users/VJ/GitHub/MLSysBook
./binder html intro
```

### If Ollama connection fails:
```bash
# Make sure Ollama is running
ollama serve

# Test it works
ollama run llava "Describe this image" < some_test_image.png
```

### If figure matching fails:
The script logs which figures it finds and which .qmd files it searches.
Check the log to see if:
- Figures were extracted from HTML
- The figure IDs match what's in your .qmd files
- The .qmd files were found in the right directory

## Next Steps After Testing

Once you verify this works on the introduction chapter:

1. **Process other chapters**: Run for each chapter individually
2. **Review all changes**: Use git diff to review the added alt text
3. **Iterate on quality**: Adjust the prompt if needed for better descriptions
4. **Scale up**: Process the entire book chapter by chapter

## Scaling to Full Book

```bash
# Process each chapter
for chapter in intro ml_systems dl_primer ai_workflow data_engineering; do
  echo "Processing $chapter..."
  python tools/scripts/genai/generate_alt_text.py --chapter $chapter --provider ollama
  git add .
  git commit -m "feat(accessibility): Add alt text to $chapter chapter"
done
```

## Cost Considerations

**Ollama (Local)**:
- ✅ Free
- ✅ Unlimited usage
- ✅ Fast for testing
- ⚠️ May be less accurate than GPT-4 Vision

**OpenAI**:
- ⚠️ Costs per image (roughly $0.01-0.03 per image)
- ✅ Higher quality descriptions
- ✅ Better understanding of technical content
- 💡 Use for final production after testing workflow with Ollama

## Questions to Consider

1. **Quality bar**: What level of detail do you want in alt text?
2. **Technical terms**: Should alt text use technical ML terminology?
3. **Length**: Prefer shorter (accessible) or longer (detailed)?
4. **Review process**: Who should review generated alt text?

You can adjust the `ALT_TEXT_PROMPT` in the script to guide the model's behavior.

## Ready to Test?

Run this command to start:
```bash
cd /Users/VJ/GitHub/MLSysBook
python tools/scripts/genai/generate_alt_text.py --chapter intro --provider ollama --dry-run
```

Then check the output and let me know how it goes!


