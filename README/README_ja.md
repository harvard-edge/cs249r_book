# 機械学習システム
*人工知能システム工学の原則と実践*

<p align="center">
  <a href="../README.md">English</a> •
  <a href="README_zh.md">中文</a> •
  <a href="README_ja.md">日本語</a> •
  <a href="README_ko.md">한국어</a>
</p>

<div align="center">

<p align="center">

  [![Book](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/book-validate-dev.yml?branch=dev&label=Book&logo=githubactions&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/book-validate-dev.yml)
  [![TinyTorch](https://img.shields.io/github/actions/workflow/status/harvard-edge/cs249r_book/tinytorch-validate-dev.yml?branch=dev&label=TinyTorch&logo=python&cacheSeconds=300)](https://github.com/harvard-edge/cs249r_book/actions/workflows/tinytorch-validate-dev.yml)
  ![Updated](https://img.shields.io/github/last-commit/harvard-edge/cs249r_book/dev?label=Updated&logo=git&cacheSeconds=300)
  [![License](https://img.shields.io/badge/License-CC--BY--NC--ND%204.0-blue.svg)](https://github.com/harvard-edge/cs249r_book/blob/dev/LICENSE.md)
  [![Cite](https://img.shields.io/badge/Cite-IEEE%202024-blue?logo=ieee)](#-citation--license)
  [![Fund Us](https://img.shields.io/badge/Fund%20Us-Open%20Collective-blue.svg?logo=open-collective)](https://opencollective.com/mlsysbook)

</p>

<p align="center">

  <!-- Reader Navigation -->
  **[📖 オンラインで読む](https://mlsysbook.ai)** •
  **[Tiny🔥Torch](https://mlsysbook.ai/tinytorch)** •
  **[📄 PDF ダウンロード](https://mlsysbook.ai/assets/downloads/Machine-Learning-Systems.pdf)** •
  **[📓 EPUB ダウンロード](https://mlsysbook.ai/epub)** •
  **[🌐 エコシステムを探検](https://mlsysbook.org)**

</p>

📚 **2026年にMIT Pressからハードカバー版が出版予定**

</div>

---

## ミッション

**世界はAIシステムを急いで作っていますが、エンジニアリングが足りません。**

それが私たちが言うAIエンジニアリングです。

**AIエンジニアリングは、実世界で効率的で信頼性があり、安全で頑丈な知能システムを構築する学問です。単にモデルを作るだけではありません。**

**私たちのミッション:** ソフトウェアエンジニアリングとコンピュータエンジニアリングに加えて、AIエンジニアリングを基礎学問として位置付け、エンドツーエンドの知能システムを設計・構築・評価する方法を教えることです。AIの長期的な影響は、アイデアを実際に動く信頼できるシステムに変えるエンジニアによって形作られます。

---

## このリポジトリに含まれるもの

このリポジトリは、AIシステム工学のためのオープンラーニングスタックです。

テキストブックのソース、TinyTorch、ハードウェアキット、そして原則と実行可能なコード・実デバイスを結びつける共同ラボ（co‑labs）を含みます。

---

## 始め方

目的に合わせてパスを選んでください。

**READ** [テキストブック](https://mlsysbook.ai)から始めます。まずは[Chapter 1](https://www.mlsysbook.ai/contents/core/introduction/introduction.html)と[Benchmarking chapter](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)を見てください。

**BUILD** [Getting Started guide](https://mlsysbook.ai/tinytorch/getting-started.html)に従ってTinyTorchを始めます。Module 01から始めてCNN、Transformer、MLPerfベンチマークへ進みます。

**DEPLOY** [ハードウェアキット](https://mlsysbook.ai/kits)を選び、ArduinoやRaspberry Piなどのエッジデバイスで実験します。

**CONNECT** [Discussions](https://github.com/harvard-edge/cs249r_book/discussions)で挨拶してください。できるだけ早く返信します。

---

## 学習スタック

以下の図は、テキストブックがハンズオンやデプロイとどのように結びつくかを示しています。テキストブックを読んで、好きなパスを選んでください:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                           MACHINE LEARNING SYSTEMS                            │
│                              Read the Textbook                                │
│                                                                               │
│                    Theory • Concepts • Best Practices                         │
│                                                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                          ┌─────────────┼─────────────┐
                          │             │             │
                          ▼             ▼             ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            HANDS‑ON ACTIVITIES                                │
│                           (pick one or all)                                   │
│                                                                               │
│     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐     │
│     │                 │      │                 │      │                 │     │
│     │    SOFTWARE     │      │    TINYTORCH    │      │    HARDWARE     │     │
│     │    CO‑LABS      │      │    FRAMEWORK    │      │      LABS       │     │
│     │                 │      │                 │      │                 │     │
│     │ EXPLORE         │      │ BUILD           │      │ DEPLOY          │     │
│     │                 │      │                 │      │                 │     │
│     │ Run controlled  │      │ Understand      │      │ Engineer under  │     │
│     │ experiments on  │      │ frameworks by   │      │ real constraints│     │
│     │ latency, memory,│      │ implementing    │      │ memory, power,  │     │
│     │ energy, cost    │      │ them            │      │ timing, safety  │     │
│     │                 │      │                 │      │                 │
│     │ (coming 2026)   │      │                 │      │ Arduino, Pi     │
│     └─────────────────┘      └─────────────────┘      └─────────────────┘     │
│                                                                               │
│           EXPLORE                  BUILD                   DEPLOY             │
│                                                                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                                                               │
│                                  AI OLYMPICS                                  │
│                                 Prove Mastery                                 │
│                                                                               │
│       Compete across all tracks • University teams • Public leaderboards      │
│                                                                               │
│                                (coming 2026)                                  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

|   | Component | What You Do | Link |
|---|-----------|-------------|------|
| **READ** | [📖 テキストブック](https://mlsysbook.ai) | MLシステムの概念を理解 | [book/](book/README.md) |
| **EXPLORE** | 🔮 Software Co‑Labs | レイテンシ・メモリ・エネルギー・コストの実験 | *Coming 2026* |
| **BUILD** | [🔥 TinyTorch](https://mlsysbook.ai/tinytorch) | フレームワーク実装を体験 | [tinytorch/](tinytorch/README.md) |
| **DEPLOY** | [🔧 Hardware Kits](https://mlsysbook.ai/kits) | メモリ・電力・時間・安全性の制約下でハードウェアをエンジニアリング | [kits/](kits/README.md) |
| **PROVE** | 🏆 AI Olympics | すべてのトラックで競争・ベンチマーク | *Coming 2026* |

**各パスが教えること:**
- **EXPLORE**は *なぜ* — トレードオフを理解。バッチサイズ・精度・モデル構造を変えるとレイテンシ・メモリ・精度がどう変わるかを確認。
- **BUILD**は *どうやって* — 内部構造を理解。autograd、optimizer、attention を自分で実装して TensorFlow・PyTorch が実際にどう動くか体感。
- **DEPLOY**は *どこで* — 制約条件を理解。実際のメモリ上限・電力予算・レイテンシ要件を持つハードウェアで実験。

---

## 学べること

この教科書は、機械学習とシステム工学の交差点を考える方法を教えます。各章はアルゴリズムの概念とそれを実際に動かすインフラを結びつけます。

### ML ↔ Systems Bridge

| ML Concept | Systems Concept | What You Learn |
|------------|-----------------|----------------|
| Model parameters | Memory constraints | 限られたリソースデバイスに大規模モデルをどう合わせるか |
| Inference latency | Hardware acceleration | GPU、TPU、アクセラレータがニューラルネットをどう実行するか |
| Training convergence | Compute efficiency | 混合精度・最適化手法でコストを削減する方法 |
| Model accuracy | Quantization and pruning | 性能を保ちつつモデルを圧縮する方法 |
| Data requirements | Pipeline infrastructure | 効率的なデータロード・前処理パイプラインの構築方法 |
| Model deployment | MLOps practices | プロダクションでモデルを監視・バージョン管理・更新する方法 |
| Privacy constraints | On‑device learning | データをクラウドに送らずに学習・適応する方法 |

### 本の構成

| Part | Focus | Chapters |
|------|-------|----------|
| **I. Foundations** | 基礎概念 | Introduction, ML Systems, DL Primer, Architectures |
| **II. Design** | ビルディングブロック | Workflow, Data Engineering, Frameworks, Training |
| **III. Performance** | 高速化 | Efficient AI, Optimizations, HW Acceleration, Benchmarking |
| **IV. Deployment** | 実装 | MLOps, On‑device Learning, Privacy, Robustness |
| **V. Trust** | 正しく作る | Responsible AI, Sustainable AI, AI for Good |
| **VI. Frontiers** | 次のステップ | Emerging trends and future directions |

---

## 特徴

この本は「生きている」教科書です。分野が成長すれば継続的に更新し、コミュニティの意見を取り入れます。

AIは稲妻のように速く変わりますが、それを動かすエンジニアリングブロックは見出しほど速くは変わりません。このプロジェクトはその安定した基盤の上に築かれています。

レゴを思い出してください。新しいセットが次々出ますが、ブロック自体は変わりません。ブロックの組み合わせ方を学べば何でも作れます。ここでの "AIブロック" は、AIを動かす堅固なシステム原則です。

読んだり、実験したり、フィードバックしたりすることで、次の学習者へのアクセス性を高める手助けをしてください。

### Research to Teaching Loop

研究と教育を同じループで使います: システム課題定義 → 参考実装構築 → ベンチマーク → カリキュラム・ツール化 → 他者が再現・拡張できるように。

| Loop Step | Research Artifacts | Teaching Artifacts |
|-----------|-------------------|-------------------|
| **Measure** | Benchmarks, suites, metrics | Benchmarking chapter, assignments |
| **Build** | Reference systems, compilers, runtimes | TinyTorch modules, co‑labs |
| **Deploy** | Hardware targets, constraints, reliability | Hardware labs, kits |

---

## 支援のお願い

私たちは **2030年までに100万人の学習者** を目指しています。AIエンジニアリングを孤立した慣例ではなく、共有できる学問にするためです。スター、シェア、貢献はすべてこの動きを加速させます。

### なぜGitHub Starsが重要か?

<div align="center">

*測定されたものは改善される。*

各スターは、AIシステムを厳密かつ実世界の制約を考慮してエンジニアリングすべきだと信じる学習者・教育者・支援者です。

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date)](https://star-history.com/#harvard-edge/cs249r_book&Date)

1 学習者 → 10 学習者 → 100 学習者 → 1,000 学習者 → **10,000 学習者** → 100,000 学習者 → **1M 学習者**

</div>

Starsは目標ではなくシグナルです。

可視的なコミュニティは、大学・財団・産業パートナーがこの資料を採用・ハードウェアを寄付・ワークショップを支援しやすくし、その結果は次世代の教室・コホート・学習者へのハードルを下げます。

支援金は [Open Collective](https://opencollective.com/mlsysbook) に流れ、TinyML4D ワークショップ・恵まれない教室向けハードウェアキット・無料・オープンリソースの維持に使われます。

ワンクリックで次の教室・次の貢献者・次のAIエンジニア世代を開くことができます。

### ミッションへの寄付

<div align="center">

All contributions go to [Open Collective](https://opencollective.com/mlsysbook), a transparent fund that supports educational outreach.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

</div>

---

## コミュニティとリソース

| Resource | Description |
|---|---|
| [📖 **テキストブック**](https://mlsysbook.ai) | インタラクティブなオンライン教科書 |
| [🔥 **TinyTorch**](https://mlsysbook.ai/tinytorch) | MLフレームワークを最初から実装 |
| [🔧 **Hardware Kits**](https://mlsysbook.ai/kits) | Arduino、Raspberry Pi、エッジデバイスへデプロイ |
| [🌐 **Ecosystem**](https://mlsysbook.org) | リソース・ワークショップ・コミュニティ |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | 質問・アイデア |

---

## コントリビューション

私たちは教科書・TinyTorch・ハードウェアキットへの貢献を歓迎します！

| I want to… | Go here |
|--------------|---------|
| 誤字修正・章改善 | [book/docs/CONTRIBUTING.md](book/docs/CONTRIBUTING.md) |
| TinyTorch モジュール追加・バグ修正 | [tinytorch/CONTRIBUTING.md](tinytorch/CONTRIBUTING.md) |
| ハードウェア実験改善 | [kits/README.md](kits/README.md) |
| Issue 報告 | [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues) |
| 質問 | [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) |

---

## 引用とライセンス

### 引用
```bibtex
@inproceedings{reddi2024mlsysbook,
  title        = {MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering},
  author       = {Reddi, Vijay Janapa},
  booktitle    = {2024 International Conference on Hardware/Software Codesign and System Synthesis (CODES+ ISSS)},
  pages        = {41--42},
  year         = {2024},
  organization = {IEEE},
  url          = {https://mlsysbook.org}
}
```

### ライセンス

このプロジェクトは二重ライセンス構造を使用します:

| Component | License | What It Means |
|-----------|---------|---------------|
| **Book content** | [CC BY‑NC‑ND 4.0](LICENSE.md) | 出典表示・非営利・改変禁止の条件で自由配布 |
| **TinyTorch code** | [Apache 2.0](tinytorch/LICENSE) | 自由使用・修正・配布・特許保護含む |

テキストブックの内容（章・図・解説）は教育資料であり、出典表示と非営利利用を前提に自由に共有できます。ソフトウェアフレームワークは誰でも使用・修正・統合できるよう設計されたツールです。

---

## 貢献者

以下の素晴らしい方々がこのリソースをより良くするために貢献してくださいました:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ... (contributors omitted for brevity) -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<div align="center">

**[⭐ GitHubでスターを付ける](https://github.com/harvard-edge/cs249r_book#support-this-work) • [✉️ 購読する](https://buttondown.email/mlsysbook) • [💬 ディスカッションに参加](https://github.com/harvard-edge/cs249r_book/discussions) • [🌐 mlsysbook.ai を訪問](https://mlsysbook.ai)**

*MLSysBook コミュニティの献身によって作られました。*

</div>
