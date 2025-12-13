# Margin Video Extension

A Quarto extension for embedding YouTube videos as margin notes with automatic numbering and format-specific rendering.

## Usage

```markdown
{{< margin-video "YOUTUBE_URL" "VIDEO_TITLE" "AUTHOR" >}}
```

### Basic Examples

```markdown
{{< margin-video "https://youtu.be/aircAruvnKk" "Neural Networks" "3Blue1Brown" >}}
{{< margin-video "https://www.youtube.com/watch?v=FwFduRA_L6Q" "CNN Demo" "Yann LeCun" >}}
```

### Advanced Usage with Options

```markdown
<!-- Custom aspect ratio -->
{{< margin-video "https://youtu.be/aircAruvnKk" "Neural Networks" "3Blue1Brown" aspect-ratio="4/3" >}}

<!-- Start at specific time (in seconds) -->
{{< margin-video "https://youtu.be/aircAruvnKk" "Neural Networks" "3Blue1Brown" start="120" >}}

<!-- Enable autoplay (use sparingly) -->
{{< margin-video "https://youtu.be/aircAruvnKk" "Neural Networks" "3Blue1Brown" autoplay="true" >}}
```

## Supported Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `aspect-ratio` | Video aspect ratio | `"16/9"` | `aspect-ratio="4/3"` |
| `start` | Start time in seconds | none | `start="120"` |
| `autoplay` | Enable autoplay | `false` | `autoplay="true"` |

## Features

- **HTML**: Renders as margin video with iframe embed and auto-numbering
- **PDF**: Generates QR code with margin note for mobile scanning
- **YouTube validation**: Only accepts YouTube URLs with clear error messages
- **Responsive design**: Videos adapt to available space

## Requirements

- Quarto >= 1.2.0
- YouTube URLs only (youtu.be or youtube.com)

## Output

### HTML
- Places video in `.column-margin` with auto-numbered caption
- Uses CSS counters for automatic "Video 1:", "Video 2:" numbering
- Responsive iframe with 16:9 aspect ratio

### PDF
- QR code linking to video
- Formatted margin note with title and author
- FontAwesome TV icon with clickable link

## Installation

This extension is bundled with the MLSysBook project. For standalone use:

```bash
quarto add path/to/margin-video
```

## License

Part of the MLSysBook project.
