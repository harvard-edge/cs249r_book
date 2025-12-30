# Hardware Kits

*Hands-on Embedded ML Labs for Real Devices*

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-publish-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-publish-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai/kits-blue)](https://mlsysbook.ai/kits)
[![PDF](https://img.shields.io/badge/Download-PDF-red)](https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf)

**[Read Online](https://mlsysbook.ai/kits)** | **[PDF](https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf)**

---

## What This Is

The Hardware Kits teach you how to deploy ML models to real embedded devices. You will face actual hardware constraints: limited memory, power budgets, and latency requirements that do not exist in cloud environments.

This is where AI systems meet the physical world.

---

## What You Will Learn

| Concept | What You Do |
|---------|-------------|
| **Image Classification** | Deploy CNN models to classify images in real-time on microcontrollers |
| **Object Detection** | Run YOLO-style detection on camera-equipped boards |
| **Keyword Spotting** | Build always-on wake word detection with audio DSP |
| **Motion Classification** | Use IMU sensors for gesture and activity recognition |
| **Model Optimization** | Quantize and compress models to fit in KB of RAM |
| **Power Management** | Balance accuracy vs battery life for edge deployment |

### Hardware Platforms

| Platform | Description | Best For |
|----------|-------------|----------|
| **Arduino Nicla Vision** | Compact AI camera board with STM32H7 | Vision projects, ultra-low power |
| **Seeed XIAO ESP32S3** | Tiny ESP32-S3 with camera support | WiFi-connected vision |
| **Grove Vision AI V2** | No-code AI vision module | Rapid prototyping |
| **Raspberry Pi** | Full Linux SBC for edge AI | Complex pipelines, prototyping |

---

## Quick Start

### For Learners

1. Pick a platform from the [labs](https://mlsysbook.ai/kits)
2. Follow the setup guide for your hardware
3. Complete the labs in order: Setup → Image Classification → Object Detection → Keyword Spotting

### For Contributors

```bash
cd kits

# Build HTML site
ln -sf config/_quarto-html.yml _quarto.yml
quarto render

# Build PDF
ln -sf config/_quarto-pdf.yml _quarto.yml
quarto render --to titlepage-pdf

# Preview with live reload
quarto preview
```

---

## Labs Overview

Each platform includes progressive labs:

| Lab | What You Build | Skills |
|-----|----------------|--------|
| **Setup** | Hardware setup and environment configuration | Toolchain, flashing, debugging |
| **Image Classification** | CNN-based image recognition | Model deployment, inference |
| **Object Detection** | Real-time object detection | YOLO, bounding boxes |
| **Keyword Spotting** | Audio wake word detection | DSP, MFCC features |
| **Motion Classification** | IMU-based gesture recognition | Sensor fusion, time series |

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

## Documentation

| Audience | Resources |
|----------|-----------|
| **Learners** | [Online Labs](https://mlsysbook.ai/kits) ・ [PDF](https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf) |
| **Contributors** | See build instructions above |

---

## Contributing

We welcome contributions to the hardware labs! To contribute:

1. Fork and clone the repository
2. Add or improve lab content in `contents/`
3. Test your changes with `quarto preview`
4. Submit a PR with a clear description

---

## Related

| Component | Description |
|-----------|-------------|
| **[Main README](../README.md)** | Project overview and ecosystem |
| **[Textbook](../book/)** | ML Systems concepts and theory |
| **[TinyTorch](../tinytorch/)** | Build ML frameworks from scratch |
| **[Website](https://mlsysbook.ai/kits)** | Read labs online |

---

## Authors

- **Marcelo Rovai** - Primary author
- **Vijay Janapa Reddi** - Harvard University

---

## License

Content is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
