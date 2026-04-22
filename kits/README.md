# Hardware Kits

*Hands-on Embedded ML Labs for Real Devices*

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/kits-validate-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/kits-validate-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai/kits-blue)](https://mlsysbook.ai/kits)
[![PDF](https://img.shields.io/badge/Download-PDF-red)](https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf)

**[Read Online](https://mlsysbook.ai/kits)** | **[PDF](https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf)**

---

<!-- EARLY-RELEASE-CALLOUT:START -->
> [!NOTE]
> **📌 Early release (2026)**
>
> Hardware Kits shipped with the **2026** MLSysBook refresh. Labs, build recipes, board notes, and PDF exports are **actively iterated** as hardware and SDKs evolve.
>
> **Feedback** — [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or pull requests.
<!-- EARLY-RELEASE-CALLOUT:END -->

## What This Is

The Hardware Kits teach you how to deploy ML models to real embedded devices. You will face actual hardware constraints: limited memory, power budgets, and latency requirements that do not exist in cloud environments.

This is where AI systems meet the physical world.

---

## 🎓 What You Will Learn

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="25%">Concept</th>
      <th align="left" width="75%">What You Do</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>🖼️ Image Classification</b></td>
      <td>Deploy CNN models to classify images in real-time on microcontrollers</td>
    </tr>
    <tr>
      <td align="center"><b>🎯 Object Detection</b></td>
      <td>Run YOLO-style detection on camera-equipped boards</td>
    </tr>
    <tr>
      <td align="center"><b>🗣️ Keyword Spotting</b></td>
      <td>Build always-on wake word detection with audio DSP</td>
    </tr>
    <tr>
      <td align="center"><b>👋 Motion Classification</b></td>
      <td>Use IMU sensors for gesture and activity recognition</td>
    </tr>
    <tr>
      <td align="center"><b>🗜️ Model Optimization</b></td>
      <td>Quantize and compress models to fit in KB of RAM</td>
    </tr>
    <tr>
      <td align="center"><b>🔋 Power Management</b></td>
      <td>Balance accuracy vs battery life for edge deployment</td>
    </tr>
  </tbody>
</table>

### 🛠️ Hardware Platforms

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="25%">Platform</th>
      <th align="left" width="50%">Description</th>
      <th align="left" width="25%">Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Arduino Nicla Vision</b></td>
      <td>Compact AI camera board with STM32H7</td>
      <td>Vision projects, ultra-low power</td>
    </tr>
    <tr>
      <td align="center"><b>Seeed XIAO ESP32S3</b></td>
      <td>Tiny ESP32-S3 with camera support</td>
      <td>WiFi-connected vision</td>
    </tr>
    <tr>
      <td align="center"><b>Grove Vision AI V2</b></td>
      <td>No-code AI vision module</td>
      <td>Rapid prototyping</td>
    </tr>
    <tr>
      <td align="center"><b>Raspberry Pi</b></td>
      <td>Full Linux SBC for edge AI</td>
      <td>Complex pipelines, prototyping</td>
    </tr>
  </tbody>
</table>

---

## 🚀 Quick Start

### For Learners

1. Pick a platform from the [labs](https://mlsysbook.ai/kits)
2. Follow the setup guide for your hardware
3. Complete the labs in order: Setup → Image Classification → Object Detection → Keyword Spotting

### For Contributors

<kbd>cd kits</kbd>

**Build HTML site**
<kbd>ln -sf config/_quarto-html.yml _quarto.yml</kbd>
<kbd>quarto render</kbd>

**Build PDF**
<kbd>ln -sf config/_quarto-pdf.yml _quarto.yml</kbd>
<kbd>quarto render --to titlepage-pdf</kbd>

**Preview with live reload**
<kbd>quarto preview</kbd>

---

## 🔬 Labs Overview

Each platform includes progressive labs:

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="25%">Lab</th>
      <th align="left" width="50%">What You Build</th>
      <th align="left" width="25%">Skills</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><b>Setup</b></td>
      <td>Hardware setup and environment configuration</td>
      <td>Toolchain, flashing, debugging</td>
    </tr>
    <tr>
      <td align="center"><b>Image Classification</b></td>
      <td>CNN-based image recognition</td>
      <td>Model deployment, inference</td>
    </tr>
    <tr>
      <td align="center"><b>Object Detection</b></td>
      <td>Real-time object detection</td>
      <td>YOLO, bounding boxes</td>
    </tr>
    <tr>
      <td align="center"><b>Keyword Spotting</b></td>
      <td>Audio wake word detection</td>
      <td>DSP, MFCC features</td>
    </tr>
    <tr>
      <td align="center"><b>Motion Classification</b></td>
      <td>IMU-based gesture recognition</td>
      <td>Sensor fusion, time series</td>
    </tr>
  </tbody>
</table>

---

## 📂 Directory Structure

```text
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

## 📚 Documentation

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="20%">Who</th>
      <th align="left" width="80%">Resources</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="20%" align="center"><b>Learners</b></td>
      <td><a href="https://mlsysbook.ai/kits">Online Labs</a> ・ <a href="https://mlsysbook.ai/kits/assets/downloads/Hardware-Kits.pdf">PDF</a></td>
    </tr>
    <tr>
      <td width="20%" align="center"><b>Contributors</b></td>
      <td>See build instructions above</td>
    </tr>
  </tbody>
</table>

---

## 🤝 Contributing

We welcome contributions to the hardware labs! To contribute:

1. Fork and clone the repository
2. Add or improve lab content in `contents/`
3. Test your changes with <kbd>quarto preview</kbd>
4. Submit a PR with a clear description

---

## 🔗 Related

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="25%">Component</th>
      <th align="left" width="75%">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="25%" align="center"><b><a href="../README.md">Main README</a></b></td>
      <td>Project overview and ecosystem</td>
    </tr>
    <tr>
      <td width="25%" align="center"><b><a href="../book/">Textbook</a></b></td>
      <td>ML Systems concepts and theory</td>
    </tr>
    <tr>
      <td width="25%" align="center"><b><a href="../tinytorch/">TinyTorch</a></b></td>
      <td>Build ML frameworks from scratch</td>
    </tr>
    <tr>
      <td width="25%" align="center"><b><a href="https://mlsysbook.ai/kits">Website</a></b></td>
      <td>Read labs online</td>
    </tr>
  </tbody>
</table>

---

## Contributors

Thanks to these wonderful people who helped improve the hardware kits!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪 🛠️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=80" width="80px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />✍️ 🧑‍💻 🎨 tutorial</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/salmanmkc"><img src="https://avatars.githubusercontent.com/u/32169182?v=4?v=4?s=80" width="80px;" alt="Salman Chishti"/><br /><sub><b>Salman Chishti</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Pratham-ja"><img src="https://avatars.githubusercontent.com/u/114498234?v=4?v=4?s=80" width="80px;" alt="Pratham Chaudhary"/><br /><sub><b>Pratham Chaudhary</b></sub></a><br />🧑‍💻</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for tool, test, video, or doc
```

---

## License

Content is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
