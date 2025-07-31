# Pre-commit Hook: External Image Validation

## Overview

This repository includes a pre-commit hook that validates all Quarto markdown files (`.qmd`) to ensure they don't contain external image references. This helps maintain build reliability and ensures the book can be built offline.

## How It Works

The `validate-external-images` hook runs automatically before each commit and:

1. **Scans** all `.qmd` files in `book/contents/` 
2. **Detects** images with `#fig-` references that use external URLs (http/https)
3. **Fails** the commit if external images are found
4. **Provides** clear instructions on how to fix the issue

## Example Output

When external images are detected:

```
❌ External image found in book/contents/core/chapter/chapter.qmd: fig-example → https://example.com/image.png

💡 To fix, run:
   python3 tools/scripts/download_external_images.py -f book/contents/core/chapter/chapter.qmd
```

## Fixing External Images

When the hook fails, you have two options:

### Option 1: Download Images Locally (Recommended)

**Single file:**
```bash
python3 tools/scripts/download_external_images.py -f path/to/file.qmd
```

**All files in directory:**
```bash
python3 tools/scripts/download_external_images.py -d book/contents/
```

**Preview what would be downloaded:**
```bash
python3 tools/scripts/download_external_images.py -d book/contents/ --dry-run
```

### Option 2: Bypass Hook (Not Recommended)

Only use this for special cases where external images are intentionally required:

```bash
git commit --no-verify -m "your commit message"
```

## Benefits

- **Build Reliability**: Local images prevent broken builds due to external URL changes
- **Offline Capability**: The book can be built without internet access
- **Performance**: Faster builds with local images
- **Consistency**: Ensures all contributors follow the same image management practices

## Configuration

The hook is configured in `.pre-commit-config.yaml`:

```yaml
- id: validate-external-images
  name: "Check for external images in Quarto files"
  entry: python3 tools/scripts/download_external_images.py --validate book/contents/
  language: system
  pass_filenames: false
  files: ^book/contents/.*\.qmd$
```

## Running Manually

You can run the validation manually at any time:

```bash
# Check all files
pre-commit run validate-external-images --all-files

# Check only staged files
pre-commit run validate-external-images

# Run the validation script directly
python3 tools/scripts/download_external_images.py --validate book/contents/
```