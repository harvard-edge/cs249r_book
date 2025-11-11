# Alt Text Generation Tool

This tool automatically generates accessible alt text for images in the Machine Learning Systems book by combining local builds with AI vision models.

## Overview

The alt text generator works by:

1. **Building locally**: Uses `./binder html <chapter>` to build the specified chapter
2. **Parsing HTML**: Extracts all `<figure>` elements with their IDs, captions, and context
3. **Image analysis**: Sends images to a vision model (OpenAI GPT-4 Vision or Ollama's llava)
4. **Source update**: Finds corresponding figures in `.qmd` files and adds `fig-alt` attributes

## Why This Approach?

We scrape from the **built HTML** rather than directly from source because:
- Many figures use TikZ code, which is unreadable without rendering
- We get the actual visual output that readers see
- Context (captions, sections) is cleanly extracted from rendered HTML
- Figure IDs provide perfect matching back to source

## Prerequisites

### For OpenAI (Cloud)
```bash
export OPENAI_API_KEY=your_api_key_here
```

### For Ollama (Local)
1. Install Ollama from https://ollama.ai
2. Pull the llava vision model:
```bash
ollama pull llava
```

### Python Dependencies
```bash
cd /Users/VJ/GitHub/MLSysBook/tools/scripts/genai
pip install -r requirements.txt
```

## Usage

### Basic Usage (Ollama, recommended for testing)
```bash
# Process the introduction chapter
python generate_alt_text.py --chapter intro --provider ollama

# Dry run to see what would happen without modifying files
python generate_alt_text.py --chapter intro --provider ollama --dry-run
```

### Using OpenAI
```bash
# Process the introduction chapter with OpenAI
python generate_alt_text.py --chapter intro --provider openai
```

### Advanced Options
```bash
# Use a different Ollama model
python generate_alt_text.py --chapter intro --provider ollama --model llava:13b

# Use a different OpenAI model
python generate_alt_text.py --chapter intro --provider openai --model gpt-4-vision-preview

# Specify Ollama URL (if not localhost)
python generate_alt_text.py --chapter intro --provider ollama --ollama-url http://192.168.1.100:11434
```

## Chapter Names

Common chapter identifiers:
- `intro` or `introduction` - Introduction chapter
- `ml_systems` - ML Systems chapter
- `dl_primer` - Deep Learning Primer
- `ai_workflow` - AI Workflow
- etc.

The script will search for matching files in the `quarto/contents/core/` directory.

## How It Works

### 1. Building the Chapter
```bash
./binder html intro
```
This builds just the introduction chapter to `quarto/_build/html/`.

### 2. Extracting Figures
The script parses the HTML looking for:
```html
<figure class="quarto-float quarto-float-fig figure">
  <div aria-describedby="fig-ai-timeline-caption-...">
    <img src="introduction_files/mediabag/25cf57367...svg" class="img-fluid figure-img">
  </div>
  <figcaption id="fig-ai-timeline-caption-...">
    Figure 2: AI Development Timeline: ...
  </figcaption>
</figure>
```

It extracts:
- Figure ID: `fig-ai-timeline` (from the caption ID)
- Image path: `introduction_files/mediabag/25cf57367...svg`
- Caption text: "Figure 2: AI Development Timeline: ..."
- Section heading: From the nearest preceding `<h1>`, `<h2>`, or `<h3>`

### 3. Generating Alt Text
For each image, the script:
- Encodes the image as base64
- Sends it to the vision model with context (caption, section)
- Gets back descriptive alt text following accessibility guidelines

**Prompt guidelines:**
- Be concise (1-2 sentences, ideally under 125 characters)
- Describe what's visually important, not what's obvious from caption
- Focus on key information the image conveys
- For diagrams: structure, flow, relationships
- For graphs: trends, comparisons, insights
- Don't start with "Image of" or "Figure showing"

### 4. Updating Source Files
The script searches `.qmd` files for figure references like:
- `![caption](image.png){#fig-ai-timeline}`
- `#| label: fig-ai-timeline`
- `id="fig-ai-timeline"`

It adds or updates the `fig-alt` attribute:

**Before:**
```markdown
![AI Development Timeline](images/timeline.svg){#fig-ai-timeline}
```

**After:**
```markdown
![AI Development Timeline](images/timeline.svg){#fig-ai-timeline fig-alt="Timeline showing evolution from symbolic AI in 1950s through neural networks to modern large language models"}
```

Or for block figures:
```markdown
#| label: fig-ai-timeline
#| fig-cap: "AI Development Timeline"
#| fig-alt: "Timeline showing evolution from symbolic AI in 1950s through neural networks to modern large language models"
```

## Output

The script provides:
- Progress logging to console and `generate_alt_text.log`
- Summary statistics at the end:
  - Total figures found
  - Alt text generated
  - Files updated
  - Errors encountered

## Troubleshooting

### "Could not find HTML file"
- Make sure the chapter name is correct
- Check that the build succeeded
- Look in `quarto/_build/html/contents/core/` for the HTML file

### "Could not find image file"
- The script tries multiple locations for images
- Check the build output directory structure
- Image might be in a subdirectory or mediabag

### "Could not find figure in .qmd file"
- The figure ID in HTML might not match the source
- Check the figure ID format in your `.qmd` files
- Try searching manually: `grep -r "fig-your-id" quarto/contents/core/`

### Ollama connection error
```bash
# Make sure Ollama is running
ollama serve

# Test it's working
curl http://localhost:11434/api/tags
```

### OpenAI API errors
- Check your API key is set: `echo $OPENAI_API_KEY`
- Verify you have credits: https://platform.openai.com/usage
- Check rate limits if processing many images

## Best Practices

1. **Start with dry run**: Always test with `--dry-run` first
2. **One chapter at a time**: Process chapters individually to catch issues early
3. **Review generated text**: Alt text quality matters for accessibility
4. **Use Ollama for testing**: Free and fast for iterating on the workflow
5. **Use OpenAI for production**: Generally produces higher quality descriptions
6. **Commit incrementally**: Commit each chapter separately for easier review

## Future Enhancements

Potential improvements:
- [ ] Batch processing of multiple chapters
- [ ] Alt text quality validation
- [ ] Support for updating existing alt text selectively
- [ ] Integration with the `binder` CLI
- [ ] Cache generated alt text to avoid regenerating
- [ ] Support for other image formats (PDF figures, etc.)

## Examples

### Successful Run
```bash
$ python generate_alt_text.py --chapter intro --provider ollama

2025-10-24 10:30:00 - INFO - Starting alt text generation for chapter: intro
2025-10-24 10:30:00 - INFO - Provider: ollama, Model: llava
2025-10-24 10:30:05 - INFO - Successfully built chapter: intro
2025-10-24 10:30:06 - INFO - Extracting figures from: .../introduction.html
2025-10-24 10:30:06 - INFO - Found figure: fig-ai-timeline
2025-10-24 10:30:06 - INFO - Found figure: fig-ml-workflow
2025-10-24 10:30:06 - INFO - Extracted 2 figures
2025-10-24 10:30:10 - INFO - Generating alt text for fig-ai-timeline using Ollama
2025-10-24 10:30:15 - INFO - Generated alt text: Timeline showing evolution...
2025-10-24 10:30:20 - INFO - Adding alt text to .../introduction.qmd for fig-ai-timeline
2025-10-24 10:30:20 - INFO - Successfully updated .../introduction.qmd

============================================================
ALT TEXT GENERATION SUMMARY
============================================================
Total figures found: 2
Alt text generated: 2
Files updated: 2
Errors: 0
============================================================

✅ Successfully processed chapter: intro
📝 Check the log file for details: generate_alt_text.log
```

### Dry Run
```bash
$ python generate_alt_text.py --chapter intro --provider ollama --dry-run

...
[DRY RUN] Would update .../introduction.qmd
[DRY RUN] Line 45: ![AI Timeline](images/timeline.svg){#fig-ai-timeline}
[DRY RUN] New line: ![AI Timeline](images/timeline.svg){#fig-ai-timeline fig-alt="Timeline..."}
...

⚠️  DRY RUN MODE - No files were actually modified
```

## Contributing

If you improve the alt text generation quality or workflow:
1. Test thoroughly with multiple chapters
2. Update this README with your changes
3. Add examples of improvements
4. Document any new dependencies or requirements

## Related Tools

Other genai tools in this directory:
- `header_update.py` - Update section headers
- `quizzes.py` - Generate quizzes
- `footnote_assistant.py` - Add scholarly footnotes
- `fix_dashes.py` - Fix dash usage in text


