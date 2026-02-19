# Machine Learning Systems

*Principles and Practices of Engineering Artificially Intelligent Systems*

[![Build](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Build&logo=githubactions)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
[![Website](https://img.shields.io/badge/Read-mlsysbook.ai-blue)](https://mlsysbook.ai)
[![PDF](https://img.shields.io/badge/Download-PDF-red)](https://mlsysbook.ai/pdf)
[![EPUB](https://img.shields.io/badge/Download-EPUB-green)](https://mlsysbook.ai/epub)

**[Read Online](https://mlsysbook.ai)** | **[PDF](https://mlsysbook.ai/pdf)** | **[EPUB](https://mlsysbook.ai/epub)**

---

## What This Is

The ML Systems textbook teaches you how to engineer AI systems that work in the real world. It bridges machine learning theory with systems engineering practice, covering everything from neural network fundamentals to production deployment.

This directory contains the textbook source and build system for contributors.

---

## What You Will Learn

| ML Concepts | Systems Engineering |
|-------------|---------------------|
| Neural networks and deep learning | Memory hierarchies and caching |
| Model architectures (CNNs, Transformers) | Hardware accelerators (GPUs, TPUs, NPUs) |
| Training and optimization | Distributed systems and parallelism |
| Inference and deployment | Power and thermal management |
| Compression and quantization | Latency, throughput, and efficiency |

### The ML ↔ Systems Bridge

| You know... | You will learn... |
|-------------|-------------------|
| How to train a model | How training scales across GPU clusters |
| That quantization shrinks models | How INT8 math maps to silicon |
| What a transformer is | Why KV-cache dominates memory |
| Models run on GPUs | How schedulers balance latency vs throughput |
| Edge devices have limits | How to co-design models and hardware |

### Book Structure

| Part | Focus | Chapters |
|------|-------|----------|
| **Foundations** | ML and systems basics | Introduction, ML Primer, DL Primer, AI Acceleration |
| **Workflow** | Production pipeline | Workflows, Data Engineering, Frameworks |
| **Training** | Learning at scale | Training, Distributed Training, Efficient AI |
| **Deployment** | Real-world systems | Inference, On-Device AI, Hardware Benchmarking, Ops |
| **Advanced** | Frontier topics | Privacy, Security, Responsible AI, Sustainable AI, Genertic AI, Frontiers |

---

## What Makes This Book Different

**Systems first**: Start with hardware constraints and work up to algorithms, not the other way around.

**Production focus**: Every concept connects to real deployment scenarios, not just research benchmarks.

**Open and evolving**: Community-driven updates keep content current with a fast-moving field.

**Hands-on companion**: Pair with [TinyTorch](../tinytorch/) to build what you learn from scratch.

---

## Quick Start

### For Readers

```bash
# Read online
open https://mlsysbook.ai

# Download formats
curl -O https://mlsysbook.ai/pdf
curl -O https://mlsysbook.ai/epub
```

### For Contributors

```bash
cd book

# First time setup
./binder setup
./binder doctor

# Daily workflow
./binder clean              # Clean build artifacts
./binder build              # Build HTML book
./binder preview intro      # Preview chapter with live reload

# Build all formats
./binder pdf                # Build PDF
./binder epub               # Build EPUB

# Utilities
./binder help               # Show all commands
./binder list               # List chapters
```

---

## Directory Structure

```
book/
├── quarto/              # Book source (Quarto markdown)
│   ├── contents/        # Chapter content
│   │   ├── core/        # Core chapters
│   │   ├── labs/        # Hands-on labs
│   │   ├── frontmatter/ # Preface, about, changelog
│   │   └── backmatter/  # References, glossary
│   ├── assets/          # Images, downloads
│   └── _quarto.yml      # Quarto configuration
├── cli/                 # Binder CLI tool
├── docker/              # Development containers
├── docs/                # Documentation
├── tools/               # Build scripts
└── binder               # CLI entry point
```

---

## Documentation

| Audience | Resources |
|----------|-----------|
| **Readers** | [Online Book](https://mlsysbook.ai) ・ [PDF](https://mlsysbook.ai/pdf) ・ [EPUB](https://mlsysbook.ai/epub) |
| **Contributors** | [CONTRIBUTING.md](docs/CONTRIBUTING.md) ・ [BUILD.md](docs/BUILD.md) |
| **Developers** | [DEVELOPMENT.md](docs/DEVELOPMENT.md) ・ [BINDER.md](docs/BINDER.md) |

---

## Contributing

We welcome contributions! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

1. **Fork and clone** the repository
2. **Set up** your environment: `./binder setup`
3. **Find an issue** or propose a change
4. **Make your changes** in the `quarto/contents/` directory
5. **Preview** your changes: `./binder preview <chapter>`
6. **Submit a PR** with a clear description

---

## Related

| Component | Description |
|-----------|-------------|
| **[Main README](../README.md)** | Project overview and ecosystem |
| **[TinyTorch](../tinytorch/)** | Build ML frameworks from scratch |
| **[Hardware Kits](../kits/)** | Deploy to Arduino, Raspberry Pi, edge devices |
| **[Website](https://mlsysbook.ai)** | Read the book online |

---

## Contributors

Thanks to these wonderful people who helped improve the book!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧠 🔎 🧪 🛠️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Mjrovai"><img src="https://avatars.githubusercontent.com/Mjrovai?v=4?s=80" width="80px;" alt="Marcelo Rovai"/><br /><sub><b>Marcelo Rovai</b></sub></a><br />🧑‍💻 🎨 🧪</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/GabrielAmazonas"><img src="https://avatars.githubusercontent.com/GabrielAmazonas?v=4?s=80" width="80px;" alt="Gabriel Amazonas"/><br /><sub><b>Gabriel Amazonas</b></sub></a><br />🪲 ✍️ 🧠</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kai4avaya"><img src="https://avatars.githubusercontent.com/kai4avaya?v=4?s=80" width="80px;" alt="Kai Kleinbard"/><br /><sub><b>Kai Kleinbard</b></sub></a><br />🧑‍💻 🛠️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/didier-durand"><img src="https://avatars.githubusercontent.com/didier-durand?v=4?s=80" width="80px;" alt="Didier Durand"/><br /><sub><b>Didier Durand</b></sub></a><br />✍️ 🪲</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hzeljko"><img src="https://avatars.githubusercontent.com/hzeljko?v=4?s=80" width="80px;" alt="Zeljko Hrcek"/><br /><sub><b>Zeljko Hrcek</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jasonjabbour"><img src="https://avatars.githubusercontent.com/jasonjabbour?v=4?s=80" width="80px;" alt="Jason Jabbour"/><br /><sub><b>Jason Jabbour</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/uchendui"><img src="https://avatars.githubusercontent.com/uchendui?v=4?s=80" width="80px;" alt="Ikechukwu Uchendu"/><br /><sub><b>Ikechukwu Uchendu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Naeemkh"><img src="https://avatars.githubusercontent.com/Naeemkh?v=4?s=80" width="80px;" alt="Naeem Khoshnevis"/><br /><sub><b>Naeem Khoshnevis</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sara-Khosravi"><img src="https://avatars.githubusercontent.com/Sara-Khosravi?v=4?s=80" width="80px;" alt="Sara Khosravi"/><br /><sub><b>Sara Khosravi</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/V0XNIHILI"><img src="https://avatars.githubusercontent.com/V0XNIHILI?v=4?s=80" width="80px;" alt="Douwe den Blanken"/><br /><sub><b>Douwe den Blanken</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/18jeffreyma"><img src="https://avatars.githubusercontent.com/18jeffreyma?v=4?s=80" width="80px;" alt="Jeffrey Ma"/><br /><sub><b>Jeffrey Ma</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shanzehbatool"><img src="https://avatars.githubusercontent.com/shanzehbatool?v=4?s=80" width="80px;" alt="shanzehbatool"/><br /><sub><b>shanzehbatool</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eliasab16"><img src="https://avatars.githubusercontent.com/eliasab16?v=4?s=80" width="80px;" alt="Elias"/><br /><sub><b>Elias</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/JaredP94"><img src="https://avatars.githubusercontent.com/JaredP94?v=4?s=80" width="80px;" alt="Jared Ping"/><br /><sub><b>Jared Ping</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ishapira1"><img src="https://avatars.githubusercontent.com/ishapira1?v=4?s=80" width="80px;" alt="Itai Shapira"/><br /><sub><b>Itai Shapira</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8863743b4f26c1a20e730fcf7ebc3bc0?d=identicon&s=100?v=4?s=80" width="80px;" alt="Maximilian Lam"/><br /><sub><b>Maximilian Lam</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jaysonzlin"><img src="https://avatars.githubusercontent.com/jaysonzlin?v=4?s=80" width="80px;" alt="Jayson Lin"/><br /><sub><b>Jayson Lin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sophiacho1"><img src="https://avatars.githubusercontent.com/sophiacho1?v=4?s=80" width="80px;" alt="Sophia Cho"/><br /><sub><b>Sophia Cho</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/andreamurillomtz"><img src="https://avatars.githubusercontent.com/andreamurillomtz?v=4?s=80" width="80px;" alt="Andrea"/><br /><sub><b>Andrea</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alxrod"><img src="https://avatars.githubusercontent.com/alxrod?v=4?s=80" width="80px;" alt="Alex Rodriguez"/><br /><sub><b>Alex Rodriguez</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/korneelf1"><img src="https://avatars.githubusercontent.com/korneelf1?v=4?s=80" width="80px;" alt="Korneel Van den Berghe"/><br /><sub><b>Korneel Van den Berghe</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/foundingnimo"><img src="https://avatars.githubusercontent.com/foundingnimo?v=4?s=80" width="80px;" alt="Nimo"/><br /><sub><b>Nimo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/colbybanbury"><img src="https://avatars.githubusercontent.com/colbybanbury?v=4?s=80" width="80px;" alt="Colby Banbury"/><br /><sub><b>Colby Banbury</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zishenwan"><img src="https://avatars.githubusercontent.com/zishenwan?v=4?s=80" width="80px;" alt="Zishen Wan"/><br /><sub><b>Zishen Wan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/mmaz"><img src="https://avatars.githubusercontent.com/mmaz?v=4?s=80" width="80px;" alt="Mark Mazumder"/><br /><sub><b>Mark Mazumder</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ma3mool"><img src="https://avatars.githubusercontent.com/ma3mool?v=4?s=80" width="80px;" alt="Abdulrahman Mahmoud"/><br /><sub><b>Abdulrahman Mahmoud</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DivyaAmirtharaj"><img src="https://avatars.githubusercontent.com/DivyaAmirtharaj?v=4?s=80" width="80px;" alt="Divya Amirtharaj"/><br /><sub><b>Divya Amirtharaj</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/srivatsankrishnan"><img src="https://avatars.githubusercontent.com/srivatsankrishnan?v=4?s=80" width="80px;" alt="Srivatsan Krishnan"/><br /><sub><b>Srivatsan Krishnan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arnaumarin"><img src="https://avatars.githubusercontent.com/arnaumarin?v=4?s=80" width="80px;" alt="marin-llobet"/><br /><sub><b>marin-llobet</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aptl26"><img src="https://avatars.githubusercontent.com/aptl26?v=4?s=80" width="80px;" alt="Aghyad Deeb"/><br /><sub><b>Aghyad Deeb</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/James-QiuHaoran"><img src="https://avatars.githubusercontent.com/James-QiuHaoran?v=4?s=80" width="80px;" alt="Haoran Qiu"/><br /><sub><b>Haoran Qiu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Ekhao"><img src="https://avatars.githubusercontent.com/Ekhao?v=4?s=80" width="80px;" alt="Emil Njor"/><br /><sub><b>Emil Njor</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ELSuitorHarvard"><img src="https://avatars.githubusercontent.com/ELSuitorHarvard?v=4?s=80" width="80px;" alt="ELSuitorHarvard"/><br /><sub><b>ELSuitorHarvard</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kaiM0ves"><img src="https://avatars.githubusercontent.com/kaiM0ves?v=4?s=80" width="80px;" alt="kaiM0ves"/><br /><sub><b>kaiM0ves</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/oishib"><img src="https://avatars.githubusercontent.com/oishib?v=4?s=80" width="80px;" alt="oishib"/><br /><sub><b>oishib</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jared-ni"><img src="https://avatars.githubusercontent.com/jared-ni?v=4?s=80" width="80px;" alt="Jared Ni"/><br /><sub><b>Jared Ni</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AditiR-42"><img src="https://avatars.githubusercontent.com/AditiR-42?v=4?s=80" width="80px;" alt="Aditi Raju"/><br /><sub><b>Aditi Raju</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MichaelSchnebly"><img src="https://avatars.githubusercontent.com/MichaelSchnebly?v=4?s=80" width="80px;" alt="Michael Schnebly"/><br /><sub><b>Michael Schnebly</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/VThuong99"><img src="https://avatars.githubusercontent.com/VThuong99?v=4?s=80" width="80px;" alt="Thuong Duong"/><br /><sub><b>Thuong Duong</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/leo47007"><img src="https://avatars.githubusercontent.com/leo47007?v=4?s=80" width="80px;" alt="Yu-Shun Hsiao"/><br /><sub><b>Yu-Shun Hsiao</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BaeHenryS"><img src="https://avatars.githubusercontent.com/BaeHenryS?v=4?s=80" width="80px;" alt="Henry Bae"/><br /><sub><b>Henry Bae</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/eimlav"><img src="https://avatars.githubusercontent.com/eimlav?v=4?s=80" width="80px;" alt="Eimhin Laverty"/><br /><sub><b>Eimhin Laverty</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jaywonchung"><img src="https://avatars.githubusercontent.com/jaywonchung?v=4?s=80" width="80px;" alt="Jae-Won Chung"/><br /><sub><b>Jae-Won Chung</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ShvetankPrakash"><img src="https://avatars.githubusercontent.com/ShvetankPrakash?v=4?s=80" width="80px;" alt="Shvetank Prakash"/><br /><sub><b>Shvetank Prakash</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/marcozennaro"><img src="https://avatars.githubusercontent.com/marcozennaro?v=4?s=80" width="80px;" alt="Marco Zennaro"/><br /><sub><b>Marco Zennaro</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aryatschand"><img src="https://avatars.githubusercontent.com/aryatschand?v=4?s=80" width="80px;" alt="Arya Tschand"/><br /><sub><b>Arya Tschand</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arbass22"><img src="https://avatars.githubusercontent.com/arbass22?v=4?s=80" width="80px;" alt="Andrew Bass"/><br /><sub><b>Andrew Bass</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pongtr"><img src="https://avatars.githubusercontent.com/pongtr?v=4?s=80" width="80px;" alt="Pong Trairatvorakul"/><br /><sub><b>Pong Trairatvorakul</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/euranofshin"><img src="https://avatars.githubusercontent.com/euranofshin?v=4?s=80" width="80px;" alt="Eura Nofshin"/><br /><sub><b>Eura Nofshin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0c931fcfd03cd548d44c90602dd773ba?d=identicon&s=100?v=4?s=80" width="80px;" alt="Matthew Stewart"/><br /><sub><b>Matthew Stewart</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/af39c27c6090c50a1921a9b6366e81cc?d=identicon&s=100?v=4?s=80" width="80px;" alt="Emeka Ezike"/><br /><sub><b>Emeka Ezike</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jianqingdu"><img src="https://avatars.githubusercontent.com/jianqingdu?v=4?s=80" width="80px;" alt="jianqingdu"/><br /><sub><b>jianqingdu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jzhou1318"><img src="https://avatars.githubusercontent.com/jzhou1318?v=4?s=80" width="80px;" alt="Jennifer Zhou"/><br /><sub><b>Jennifer Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vitasam"><img src="https://avatars.githubusercontent.com/vitasam?v=4?s=80" width="80px;" alt="The Random DIY"/><br /><sub><b>The Random DIY</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/468ef35acc69f3266efd700992daa369?d=identicon&s=100?v=4?s=80" width="80px;" alt="Fatima Shah"/><br /><sub><b>Fatima Shah</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BrunoScaglione"><img src="https://avatars.githubusercontent.com/BrunoScaglione?v=4?s=80" width="80px;" alt="Bruno Scaglione"/><br /><sub><b>Bruno Scaglione</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Allen-Kuang"><img src="https://avatars.githubusercontent.com/Allen-Kuang?v=4?s=80" width="80px;" alt="Allen-Kuang"/><br /><sub><b>Allen-Kuang</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/4ad8cdf19eb3b666ace97d3eedb19278?d=identicon&s=100?v=4?s=80" width="80px;" alt="Tess314"/><br /><sub><b>Tess314</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/taunoe"><img src="https://avatars.githubusercontent.com/taunoe?v=4?s=80" width="80px;" alt="Tauno Erik"/><br /><sub><b>Tauno Erik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/gnodipac886"><img src="https://avatars.githubusercontent.com/gnodipac886?v=4?s=80" width="80px;" alt="gnodipac886"/><br /><sub><b>gnodipac886</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/serco425"><img src="https://avatars.githubusercontent.com/serco425?v=4?s=80" width="80px;" alt="Sercan Aygün"/><br /><sub><b>Sercan Aygün</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TheHiddenLayer"><img src="https://avatars.githubusercontent.com/TheHiddenLayer?v=4?s=80" width="80px;" alt="TheHiddenLayer"/><br /><sub><b>TheHiddenLayer</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Gjain234"><img src="https://avatars.githubusercontent.com/Gjain234?v=4?s=80" width="80px;" alt="Gauri Jain"/><br /><sub><b>Gauri Jain</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/FinAminToastCrunch"><img src="https://avatars.githubusercontent.com/FinAminToastCrunch?v=4?s=80" width="80px;" alt="Fin Amin"/><br /><sub><b>Fin Amin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alex-oesterling"><img src="https://avatars.githubusercontent.com/alex-oesterling?v=4?s=80" width="80px;" alt="Alex Oesterling"/><br /><sub><b>Alex Oesterling</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AbenezerKb"><img src="https://avatars.githubusercontent.com/AbenezerKb?v=4?s=80" width="80px;" alt="Abenezer Angamo"/><br /><sub><b>Abenezer Angamo</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BravoBaldo"><img src="https://avatars.githubusercontent.com/BravoBaldo?v=4?s=80" width="80px;" alt="Baldassarre Cesarano"/><br /><sub><b>Baldassarre Cesarano</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Jahnic-kb"><img src="https://avatars.githubusercontent.com/Jahnic-kb?v=4?s=80" width="80px;" alt="Jahnic Beck"/><br /><sub><b>Jahnic Beck</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aethernavshulkraven-allain"><img src="https://avatars.githubusercontent.com/aethernavshulkraven-allain?v=4?s=80" width="80px;" alt="अरनव शुक्ला | Arnav Shukla"/><br /><sub><b>अरनव शुक्ला | Arnav Shukla</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/RinZ27"><img src="https://avatars.githubusercontent.com/RinZ27?v=4?s=80" width="80px;" alt="Rin"/><br /><sub><b>Rin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bilgeacun"><img src="https://avatars.githubusercontent.com/bilgeacun?v=4?s=80" width="80px;" alt="Bilge Acun"/><br /><sub><b>Bilge Acun</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/atcheng2"><img src="https://avatars.githubusercontent.com/atcheng2?v=4?s=80" width="80px;" alt="Andy Cheng"/><br /><sub><b>Andy Cheng</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/arighosh05"><img src="https://avatars.githubusercontent.com/arighosh05?v=4?s=80" width="80px;" alt="Aritra Ghosh"/><br /><sub><b>Aritra Ghosh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/abigailswallow"><img src="https://avatars.githubusercontent.com/abigailswallow?v=4?s=80" width="80px;" alt="abigailswallow"/><br /><sub><b>abigailswallow</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/YangZhou1997"><img src="https://avatars.githubusercontent.com/YangZhou1997?v=4?s=80" width="80px;" alt="Yang Zhou"/><br /><sub><b>Yang Zhou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/XaicuL"><img src="https://avatars.githubusercontent.com/XaicuL?v=4?s=80" width="80px;" alt="JEON HYUNJUN(Luciano)"/><br /><sub><b>JEON HYUNJUN(Luciano)</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/emmanuel2406"><img src="https://avatars.githubusercontent.com/emmanuel2406?v=4?s=80" width="80px;" alt="Emmanuel Rassou"/><br /><sub><b>Emmanuel Rassou</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jasonlyik"><img src="https://avatars.githubusercontent.com/jasonlyik?v=4?s=80" width="80px;" alt="Jason Yik"/><br /><sub><b>Jason Yik</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/jessicaquaye"><img src="https://avatars.githubusercontent.com/jessicaquaye?v=4?s=80" width="80px;" alt="Jessica Quaye"/><br /><sub><b>Jessica Quaye</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cursoragent"><img src="https://avatars.githubusercontent.com/cursoragent?v=4?s=80" width="80px;" alt="Cursor Agent"/><br /><sub><b>Cursor Agent</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/happyappledog"><img src="https://avatars.githubusercontent.com/happyappledog?v=4?s=80" width="80px;" alt="happyappledog"/><br /><sub><b>happyappledog</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/snuggs"><img src="https://avatars.githubusercontent.com/snuggs?v=4?s=80" width="80px;" alt="Snuggs"/><br /><sub><b>Snuggs</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/swilcock0"><img src="https://avatars.githubusercontent.com/swilcock0?v=4?s=80" width="80px;" alt="Sam Wilcock"/><br /><sub><b>Sam Wilcock</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sjohri20"><img src="https://avatars.githubusercontent.com/sjohri20?v=4?s=80" width="80px;" alt="Shreya Johri"/><br /><sub><b>Shreya Johri</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/skmur"><img src="https://avatars.githubusercontent.com/skmur?v=4?s=80" width="80px;" alt="Sonia Murthy"/><br /><sub><b>Sonia Murthy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/fc4f3460cdfb9365ab59bdeafb06413e?d=identicon&s=100?v=4?s=80" width="80px;" alt="Costin-Andrei Oncescu"/><br /><sub><b>Costin-Andrei Oncescu</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/0d6b8616427d8b19d425c9808692e347?d=identicon&s=100?v=4?s=80" width="80px;" alt="formlsysbookissue"/><br /><sub><b>formlsysbookissue</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/7cd8d5dfd83071f23979019d97655dc5?d=identicon&s=100?v=4?s=80" width="80px;" alt="Annie Laurie Cook"/><br /><sub><b>Annie Laurie Cook</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/5aa037840c0ca11ee42784ed4843c655?d=identicon&s=100?v=4?s=80" width="80px;" alt="Parampreet Singh"/><br /><sub><b>Parampreet Singh</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/b15b6e0e9adf58099905c1a0fd474cb9?d=identicon&s=100?v=4?s=80" width="80px;" alt="Vijay Edupuganti"/><br /><sub><b>Vijay Edupuganti</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f88052cca4f401d9b0f43aed0a53434a?d=identicon&s=100?v=4?s=80" width="80px;" alt="Jothi Ramaswamy"/><br /><sub><b>Jothi Ramaswamy</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/35a8d9ffd03f05e79a2c6ce6206a56f2?d=identicon&s=100?v=4?s=80" width="80px;" alt="Batur Arslan"/><br /><sub><b>Batur Arslan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/bd53d146aa888548c8db4da02bf81e7a?d=identicon&s=100?v=4?s=80" width="80px;" alt="Curren Iyer"/><br /><sub><b>Curren Iyer</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/8d8410338458e08bd5e4b96f58e1c217?d=identicon&s=100?v=4?s=80" width="80px;" alt="Edward Jin"/><br /><sub><b>Edward Jin</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/28c6123d2c9f75578d3ccdedb0df3d11?d=identicon&s=100?v=4?s=80" width="80px;" alt="Tess Watt"/><br /><sub><b>Tess Watt</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/ef139181fe00190f21730f6912532e9e?d=identicon&s=100?v=4?s=80" width="80px;" alt="bluebaer7"/><br /><sub><b>bluebaer7</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/f5d58ba6aa9b00189d4c018d370e8f43?d=identicon&s=100?v=4?s=80" width="80px;" alt="yanjingl"/><br /><sub><b>yanjingl</b></sub></a><br />✍️</td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/a5a47df988ab1720dd706062e523ca32?d=identicon&s=100?v=4?s=80" width="80px;" alt="a-saraf"/><br /><sub><b>a-saraf</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/c2dc311aa8122d5f5f061e1db14682b1?d=identicon&s=100?v=4?s=80" width="80px;" alt="songhan"/><br /><sub><b>songhan</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/4814aad67982ab07a69006a1ce9d2a72?d=identicon&s=100?v=4?s=80" width="80px;" alt="jvijay"/><br /><sub><b>jvijay</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/harvard-edge/cs249r_book/graphs/contributors"><img src="https://www.gravatar.com/avatar/43b1feff77c8a95fd581774fb8ec891f?d=identicon&s=100?v=4?s=80" width="80px;" alt="Zishen"/><br /><sub><b>Zishen</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/BunningsWarehouseOfficial"><img src="https://avatars.githubusercontent.com/u/49220945?v=4?v=4?s=80" width="80px;" alt="Kristian Radoš"/><br /><sub><b>Kristian Radoš</b></sub></a><br />✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/minhdang26403"><img src="https://avatars.githubusercontent.com/u/86156224?v=4?v=4?s=80" width="80px;" alt="Dang Truong"/><br /><sub><b>Dang Truong</b></sub></a><br />🧑‍💻</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/pipme"><img src="https://avatars.githubusercontent.com/pipme?v=4?s=80" width="80px;" alt="pipme"/><br /><sub><b>pipme</b></sub></a><br />✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for doc, review, translation, or design
```

---

## License

Book content is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0).

See [LICENSE.md](../LICENSE.md) for details.
