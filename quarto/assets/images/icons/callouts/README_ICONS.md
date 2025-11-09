# Callout Icons

This directory contains icons for all custom callout blocks used in MLSysBook.

## Icon Specifications

- **Size**: 43x43 pixels
- **Formats**: PNG (for HTML/EPUB) and PDF (for PDF output)
- **Style**: Circular background with border, centered icon
- **Naming**: `icon_callout-{type}.{png|pdf}`

## Available Icons

### Interactive Content
- **callout-colab** (🔬 Orange) - Hands-on Colab notebooks
  - Background: `#FFE8D5` (light peach)
  - Border: `#FF6B35` (vibrant orange)
  - Icon: Stylized microscope shape

### Learning & Assessment
- **callout-quiz-question** (Purple) - Self-check questions
- **callout-quiz-answer** (Green) - Self-check answers

### Resources
- **callout-resource-slides** (Teal) - Slide decks
- **callout-resource-videos** (Teal) - Video content
- **callout-resource-exercises** (Teal) - Practice exercises

### Content Organization
- **callout-chapter-connection** (Crimson) - Cross-chapter connections
- **callout-code** (Blue-gray) - Code listings
- **callout-definition** (Navy) - Key definitions
- **callout-example** (Teal) - Examples and demonstrations

## Creating New Icons

To create a new callout icon matching the existing style:

### Using ImageMagick

```bash
cd quarto/assets/images/icons/callouts

# Create PNG
magick -size 43x43 xc:none \
  -fill "#BACKGROUND_COLOR" -draw "circle 21.5,21.5 21.5,2" \
  -fill none -stroke "#BORDER_COLOR" -strokewidth 2.5 -draw "circle 21.5,21.5 21.5,2" \
  [additional drawing commands for icon] \
  icon_callout-NAME.png

# Create PDF
magick -size 43x43 xc:none \
  -fill "#BACKGROUND_COLOR" -draw "circle 21.5,21.5 21.5,2" \
  -fill none -stroke "#BORDER_COLOR" -strokewidth 2.5 -draw "circle 21.5,21.5 21.5,2" \
  [additional drawing commands for icon] \
  icon_callout-NAME.pdf
```

### Design Guidelines

1. **Consistency**: Match the circular style and 43x43 size
2. **Colors**: Use colors from the callout group definition in `_quarto.yml`
3. **Simplicity**: Keep icons simple and recognizable at small sizes
4. **Contrast**: Ensure icon is visible against the background
5. **Formats**: Always create both PNG and PDF versions

## Configuration

Icons are automatically loaded by the custom-numbered-blocks extension:

```yaml
filter-metadata:
  mlsysbook-ext/custom-numbered-blocks:
    icon-path: "assets/images/icons/callouts"
    icon-format: "png"  # Format for HTML/EPUB builds
```

For PDF builds, the extension automatically uses `.pdf` versions when available.

## Troubleshooting

**Icon not showing?**
- Verify both PNG and PDF versions exist
- Check filename follows the pattern: `icon_callout-{type}.{png|pdf}`
- Ensure the callout class is defined in `_quarto.yml` under `classes:`
- Clear Quarto cache: `quarto clean` or `rm -rf _build .quarto`

**Wrong colors?**
- Check the `colors:` array in the group definition in `_quarto.yml`
- Format: `["BACKGROUND_HEX", "BORDER_HEX"]` (without `#` prefix)

**Icon too large/small?**
- Icons should be exactly 43x43 pixels
- Verify with: `file icon_callout-NAME.png`

