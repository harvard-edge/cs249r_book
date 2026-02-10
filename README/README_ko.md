# 머신러닝 시스템
*인공지능 시스템 엔지니어링의 원리와 실천*

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
  **[📖 온라인 읽기](https://mlsysbook.ai)** •
  **[Tiny🔥Torch](https://mlsysbook.ai/tinytorch)** •
  **[📄 PDF 다운로드](https://mlsysbook.ai/assets/downloads/Machine-Learning-Systems.pdf)** •
  **[📓 EPUB 다운로드](https://mlsysbook.ai/epub)** •
  **[🌐 생태계 탐험](https://mlsysbook.org)**

</p>

📚 **2026년 MIT Press에서 하드커버 출판 예정**

</div>

---

## 미션

**세상은 AI 시스템을 급히 만들고 있습니다. 하지만 엔지니어링은 부족합니다.**

그 격차가 바로 우리가 말하는 AI 엔지니어링입니다.

**AI 엔지니어링은 실제 세계에서 효율적이고, 신뢰할 수 있으며, 안전하고, 견고한 지능형 시스템을 구축하는 학문입니다. 단순히 모델만 만드는 것이 아니라요.**

**우리의 미션:** 소프트웨어 엔지니어링과 컴퓨터 엔지니어링에 이어 AI 엔지니어링을 기본 학문으로 자리매김하도록, 엔드‑투‑엔드 지능형 시스템을 설계·구축·평가하는 방법을 가르치는 것입니다. AI의 장기적 영향은 아이디어를 실제 작동하고 신뢰할 수 있는 시스템으로 바꿀 수 있는 엔지니어에 의해 형성될 것입니다.

---

## 이 저장소에 포함된 내용

이 저장소는 AI 시스템 엔지니어링을 위한 오픈 학습 스택입니다.

텍스트북 소스, TinyTorch, 하드웨어 키트, 그리고 원리와 실행 가능한 코드·실제 장치를 연결하는 향후 공동 실험(co‑labs) 등을 포함합니다.

---

## 시작하기

목표에 따라 경로를 선택하세요.

**READ** [텍스트북](https://mlsysbook.ai)부터 시작하세요. [Chapter 1](https://www.mlsysbook.ai/contents/core/introduction/introduction.html)과 [Benchmarking chapter](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)을 살펴보세요.

**BUILD** [Getting Started guide](https://mlsysbook.ai/tinytorch/getting-started.html)를 따라 TinyTorch를 시작하세요. Module 01부터 시작해 CNN, Transformer, MLPerf 벤치마크까지 진행합니다.

**DEPLOY** [하드웨어 키트](https://mlsysbook.ai/kits)를 선택해 Arduino, Raspberry Pi 등 엣지 디바이스에서 실험을 진행하세요.

**CONNECT** [Discussions](https://github.com/harvard-edge/cs249r_book/discussions)에서 인사해 주세요. 가능한 한 빠르게 답변드리겠습니다.

---

## 학습 스택

아래 그림은 텍스트북이 실습 및 배포와 어떻게 연결되는지를 보여줍니다. 텍스트북을 읽고, 원하는 경로를 선택하세요:

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
| **READ** | [📖 텍스트북](https://mlsysbook.ai) | ML 시스템 개념 이해 | [book/](book/README.md) |
| **EXPLORE** | 🔮 Software Co‑Labs | 레이턴시·메모리·에너지·비용 실험 | *Coming 2026* |
| **BUILD** | [🔥 TinyTorch](https://mlsysbook.ai/tinytorch) | 프레임워크 구현을 직접 경험 | [tinytorch/](tinytorch/README.md) |
| **DEPLOY** | [🔧 Hardware Kits](https://mlsysbook.ai/kits) | 메모리·전력·시간·안전 제약 하드웨어 엔지니어링 | [kits/](kits/README.md) |
| **PROVE** | 🏆 AI Olympics | 모든 트랙에서 경쟁·벤치마크 | *Coming 2026* |

**각 경로가 가르치는 내용:**
- **EXPLORE**는 *왜* — 트레이드오프 이해. 배치 크기·정밀도·모델 구조를 바꾸면 레이턴시·메모리·정확도가 어떻게 변하는지 확인.
- **BUILD**는 *어떻게* — 내부 구조 이해. autograd, optimizer, attention을 직접 구현해 TensorFlow·PyTorch가 실제로 어떻게 동작하는지 체험.
- **DEPLOY**는 *어디서* — 제약 조건 이해. 실제 메모리 한계·전력 예산·레이턴시 요구사항을 갖는 하드웨어에서 실험.

---

## 배울 내용

이 교재는 머신러닝과 시스템 엔지니어링의 교차점을 생각하도록 가르칩니다. 각 장은 알고리즘 개념과 이를 실제로 동작하게 하는 인프라를 연결합니다.

### ML ↔ Systems Bridge

| ML Concept | Systems Concept | What You Learn |
|------------|-----------------|----------------|
| Model parameters | Memory constraints | 제한된 자원 디바이스에 큰 모델을 어떻게 맞출지 |
| Inference latency | Hardware acceleration | GPU, TPU, 가속기가 신경망을 어떻게 실행하는지 |
| Training convergence | Compute efficiency | 혼합 정밀도·최적화 기법으로 비용을 어떻게 줄이는지 |
| Model accuracy | Quantization and pruning | 성능을 유지하면서 모델을 압축하는 방법 |
| Data requirements | Pipeline infrastructure | 효율적인 데이터 로딩·전처리 파이프라인 구축 방법 |
| Model deployment | MLOps practices | 프로덕션에서 모델을 모니터링·버전 관리·업데이트하는 방법 |
| Privacy constraints | On‑device learning | 데이터를 클라우드에 보내지 않고 학습·적응하는 방법 |

### 책 구조

| Part | Focus | Chapters |
|------|-------|----------|
| **I. Foundations** | 핵심 개념 | Introduction, ML Systems, DL Primer, Architectures |
| **II. Design** | 빌딩 블록 | Workflow, Data Engineering, Frameworks, Training |
| **III. Performance** | 빠르게 만들기 | Efficient AI, Optimizations, HW Acceleration, Benchmarking |
| **IV. Deployment** | 실제 적용 | MLOps, On‑device Learning, Privacy, Robustness |
| **V. Trust** | 올바르게 만들기 | Responsible AI, Sustainable AI, AI for Good |
| **VI. Frontiers** | 다음 단계 | Emerging trends and future directions |

---

## 차별점

이 책은 살아있는 교재입니다. 분야가 성장함에 따라 지속적으로 업데이트하고, 커뮤니티 입력을 반영합니다.

AI는 번개처럼 빠르게 변하지만, 이를 작동하게 하는 엔지니어링 블록은 헤드라인만큼 빠르게 변하지 않습니다. 이 프로젝트는 그 안정적인 기반 위에 세워졌습니다.

레고를 떠올려 보세요. 새로운 세트가 계속 나오지만, 블록 자체는 변하지 않죠. 블록 맞추는 법을 배우면 무엇이든 만들 수 있습니다. 여기서 "AI 블록"은 AI가 작동하도록 하는 견고한 시스템 원칙입니다.

읽기, 실험, 피드백을 통해 다음 학습자를 위한 접근성을 높이는 데 함께해 주세요.

### Research to Teaching Loop

연구와 교육을 같은 루프로 사용합니다: 시스템 문제 정의 → 레퍼런스 구현 구축 → 벤치마크 → 커리큘럼·툴링으로 전환 → 다른 사람들이 재현·확장 가능하게.

| Loop Step | Research Artifacts | Teaching Artifacts |
|-----------|-------------------|-------------------|
| **Measure** | Benchmarks, suites, metrics | Benchmarking chapter, assignments |
| **Build** | Reference systems, compilers, runtimes | TinyTorch modules, co‑labs |
| **Deploy** | Hardware targets, constraints, reliability | Hardware labs, kits |

---

## 이 작업을 지원해 주세요

우리는 **2030년까지 1백만 명의 학습자**를 목표로 합니다. AI 엔지니어링을 고립된 관행이 아닌 공유 가능한 학문으로 만들기 위해요. 별, 공유, 기여는 모두 이 움직임을 촉진합니다.

### 왜 GitHub Stars가 중요한가?

<div align="center">

*측정된 것이 개선됩니다.*

각 스타는 AI 시스템을 엄격하고 실제 제약을 고려해 엔지니어링해야 한다고 믿는 학습자·교육자·지원자입니다.

[![Stars](https://img.shields.io/github/stars/harvard-edge/cs249r_book?style=for-the-badge&logo=github&color=gold)](https://github.com/harvard-edge/cs249r_book/stargazers)

[![Star History Chart](https://api.star-history.com/svg?repos=harvard-edge/cs249r_book&type=Date)](https://star-history.com/#harvard-edge/cs249r_book&Date)

1 학습자 → 10 학습자 → 100 학습자 → 1,000 학습자 → **10,000 학습자** → 100,000 학습자 → **1M 학습자**

</div>

Stars는 목표가 아니라 신호입니다.

가시적인 커뮤니티는 대학·재단·산업 파트너가 이 자료를 채택·하드웨어를 기부·워크숍을 지원하기 쉽게 만들고, 그 결과는 차세대 교실·코호트·학습자를 위한 장벽을 낮춥니다.

지원금은 [Open Collective](https://opencollective.com/mlsysbook)로 흐르고, TinyML4D 워크숍·소외된 교실용 하드웨어 키트·무료·오픈 리소스 유지에 쓰입니다.

한 번 클릭으로 다음 교실·다음 기여자·다음 AI 엔지니어 세대를 열 수 있습니다.

### 사명을 위한 기부

<div align="center">

All contributions go to [Open Collective](https://opencollective.com/mlsysbook), a transparent fund that supports educational outreach.

[![Open Collective](https://img.shields.io/badge/💝%20Support%20AI%20Education-Open%20Collective-blue.svg?style=for-the-badge)](https://opencollective.com/mlsysbook)

</div>

---

## 커뮤니티와 리소스

| Resource | Description |
|---|---|
| [📖 **텍스트북**](https://mlsysbook.ai) | 인터랙티브 온라인 교재 |
| [🔥 **TinyTorch**](https://mlsysbook.ai/tinytorch) | ML 프레임워크를 처음부터 구현 |
| [🔧 **Hardware Kits**](https://mlsysbook.ai/kits) | Arduino, Raspberry Pi, 엣지 디바이스에 배포 |
| [🌐 **Ecosystem**](https://mlsysbook.org) | 리소스·워크숍·커뮤니티 |
| [💬 **Discussions**](https://github.com/harvard-edge/cs249r_book/discussions) | 질문·아이디어 |

---

## 기여하기

우리는 교재·TinyTorch·하드웨어 키트에 대한 기여를 환영합니다!

| I want to… | Go here |
|--------------|---------|
| 오타 수정·챕터 개선 | [book/docs/CONTRIBUTING.md](book/docs/CONTRIBUTING.md) |
| TinyTorch 모듈 추가·버그 수정 | [tinytorch/CONTRIBUTING.md](tinytorch/CONTRIBUTING.md) |
| 하드웨어 실험 개선 | [kits/README.md](kits/README.md) |
| 이슈 보고 | [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues) |
| 질문하기 | [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions) |

---

## 인용 및 라이선스

### 인용
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

### 라이선스

이 프로젝트는 이중 라이선스 구조를 사용합니다:

| Component | License | What It Means |
|-----------|---------|---------------|
| **Book content** | [CC BY‑NC‑ND 4.0](LICENSE.md) | 출처 표시·비상업·변경 금지 조건으로 자유 배포 |
| **TinyTorch code** | [Apache 2.0](tinytorch/LICENSE) | 자유 사용·수정·배포·특허 보호 포함 |

텍스트북 내용(챕터·그림·설명)은 교육 자료이며, 출처 표시와 비상업적 사용을 전제로 자유롭게 공유됩니다. 소프트웨어 프레임워크는 누구나 사용·수정·통합할 수 있도록 설계된 도구입니다.

---

## 기여자들

다음 훌륭한 분들이 이 리소스를 더 나은 것으로 만들기 위해 기여해 주셨습니다:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- ... (contributors table omitted for brevity) -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<div align="center">

**[⭐ GitHub에 별 달기](https://github.com/harvard-edge/cs249r_book#support-this-work) • [✉️ 구독하기](https://buttondown.email/mlsysbook) • [💬 토론 참여](https://github.com/harvard-edge/cs249r_book/discussions) • [🌐 mlsysbook.ai 방문](https://mlsysbook.ai)**

*MLSysBook 커뮤니티의 헌신으로 만들어졌습니다.*

</div>
