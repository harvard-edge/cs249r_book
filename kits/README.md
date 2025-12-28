# Hardware Kits

*Hands-on embedded ML labs for the MLSysBook*

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-publish-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-publish-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai/kits-blue)](https://mlsysbook.ai/kits)

This directory contains hands-on embedded ML labs using Arduino, Raspberry Pi, and other microcontroller platforms.

**[Read Online](https://mlsysbook.ai/kits)** | **[PDF](https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf)**

---

## Platforms

| Platform | Description |
|----------|-------------|
| **Arduino Nicla Vision** | Compact AI camera board with STM32H7 |
| **Seeed XIAO ESP32S3** | Tiny ESP32-S3 with camera support |
| **Grove Vision AI V2** | No-code AI vision module |
| **Raspberry Pi** | Full Linux SBC for edge AI |

---

## Quick Start

```bash
# Build HTML site
cd kits
ln -sf config/_quarto-html.yml _quarto.yml
quarto render

# Build PDF
ln -sf config/_quarto-pdf.yml _quarto.yml
quarto render --to titlepage-pdf

# Preview with live reload
quarto preview
```

---

## Directory Structure

```
kits/
├── contents/                # Lab content
│   ├── arduino/             # Arduino Nicla Vision labs
│   ├── seeed/               # Seeed XIAO & Grove Vision labs
│   ├── raspi/               # Raspberry Pi labs
│   └── shared/              # Shared resources (DSP, features)
├── assets/                  # Images, styles, scripts
├── config/                  # Quarto configurations
│   ├── _quarto-html.yml     # Website config
│   └── _quarto-pdf.yml      # PDF config
├── tex/                     # LaTeX includes for PDF
├── filters/                 # Lua filters
└── index.qmd                # Landing page
```

---

## Labs Overview

Each platform includes labs covering:

- **Setup** - Hardware setup and environment configuration
- **Image Classification** - CNN-based image recognition
- **Object Detection** - Real-time object detection
- **Keyword Spotting** - Audio wake word detection
- **Motion Classification** - IMU-based gesture recognition

---

## Related

- **[MLSysBook](../README.md)** - Main textbook
- **[TinyTorch](../tinytorch/)** - Build ML frameworks from scratch
- **[Website](https://mlsysbook.ai/kits)** - Read labs online

---

## Authors

- **Marcelo Rovai** - Primary author
- **Vijay Janapa Reddi** - Harvard University

---

## License

Content is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
