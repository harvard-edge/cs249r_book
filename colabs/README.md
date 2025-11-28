# Lens: Interactive ML Systems Analysis Toolkit

**Tagline**: See ML systems trade-offs through a new lens

This directory contains the design documentation and planning for **Lens**, the pedagogical framework for interactive ML systems exploration used throughout the textbook.

## What is Lens?

Lens is a lightweight analytical modeling toolkit that:
- Provides simple analytical models for hardware/network/deployment trade-offs
- Wraps existing battle-tested tools (Flower, Rooflini, electricityMap)
- Offers unified API across all chapters and deployment paradigms
- Builds progressively from simple (Ch02) to complex (Ch18)
- Focuses on systems thinking and trade-off exploration

**Not a simulator**: Lens uses analytical models (fast, pedagogically focused) rather than cycle-accurate simulation.

## Key Documentation

### Current Planning
- **[NAMING_DISCUSSION.md](NAMING_DISCUSSION.md)** - Why "Lens" instead of "MLS Simulator"
- **[MLS_SIMULATOR_BUILD_VS_LEVERAGE.md](MLS_SIMULATOR_BUILD_VS_LEVERAGE.md)** - Build vs leverage analysis, hybrid approach recommendation
- **[PYTHON_TO_NOTEBOOK_WORKFLOW.md](PYTHON_TO_NOTEBOOK_WORKFLOW.md)** - Python source to Jupyter notebook conversion workflow
- **[CHAPTER_MAPPING.md](CHAPTER_MAPPING.md)** - Progressive Lens module availability across chapters

### Archive
- **[archive/](archive/)** - Previous planning iterations (v1-v3 master plans, original vision)

## Framework Architecture

### Hybrid Approach
**Build analytical models for**:
- Hardware performance (cloud/edge/mobile/TinyML)
- Network tier simulation (latency, bandwidth, cost)
- Drift models and lifecycle management
- Reliability (SDC, fault injection)

**Wrap existing tools for**:
- Roofline analysis → Rooflini
- Federated learning → Flower
- Carbon modeling → electricityMap API
- Adversarial attacks → CleverHans/ART

**Inspired by academic tools**:
- MAESTRO (analytical dataflow cost model)
- Timeloop (accelerator modeling)
- SCALE-Sim (systolic array concepts)

## Lens Colabs Structure

Each Lens colab follows the **OERC framework**:
1. **Observe**: Present a systems scenario/trade-off
2. **Explore**: Interactive exploration with Lens toolkit
3. **Reason**: Guided analysis and critical thinking
4. **Connect**: Link to production systems and research

**Duration**: 20-30 minutes per colab

## Progressive Complexity

- **Ch02**: Simple deployment paradigm comparison (cloud/edge/mobile/TinyML)
- **Ch11**: Roofline model introduction (Lens becomes central analysis tool)
- **Ch14**: Federated learning with Flower integration
- **Ch18**: Multi-dimensional carbon-aware scheduling

## Development Roadmap

### Phase 1: Core Analytical Models (Weeks 1-6)
- Hardware performance models
- Network tier simulation
- Basic workload characterization

### Phase 2: Tool Integration (Weeks 7-11)
- Roofline wrapper (Rooflini)
- Federated wrapper (Flower)
- Carbon API integration

### Phase 3: Pilot Colabs (Weeks 12-14)
- Ch02: Deployment paradigms
- Ch11: Roofline analysis
- Ch14: Federated learning

### Phase 4: Validation (Weeks 15-16)
- Student testing
- Accuracy validation (±20% target)
- Documentation

## Directory Structure

```
colabs/
├── src/                          # Python source files (version controlled)
│   ├── ch01_ai_triangle.py       # Chapter 1: AI Triangle colab
│   ├── ch02_deployment.py        # Chapter 2: Deployment paradigms
│   └── utils/                    # Reusable utilities
│       ├── ai_triangle_sim.py    # AI Triangle simulator
│       └── visualization.py      # Common plotting functions
│
├── notebooks/                    # Generated Jupyter notebooks (for students)
│   ├── ch01_ai_triangle.ipynb    # Ready for Google Colab
│   ├── ch02_deployment.ipynb
│   └── ...
│
├── docs/                         # Planning and design documentation
│   ├── NAMING_DISCUSSION.md
│   ├── PYTHON_TO_NOTEBOOK_WORKFLOW.md
│   └── ...
│
└── README.md                     # This file
```

## Workflow for Authors

### Creating New Colabs
1. Write colab as Python file in `colabs/src/chXX_topic.py` using percent format
2. Use `# %% [markdown]` for text cells, `# %%` for code cells
3. Extract reusable code to `colabs/src/utils/`
4. Convert to notebook: `jupytext --to notebook src/chXX_topic.py --output notebooks/chXX_topic.ipynb`
5. Test in Google Colab
6. Commit both `.py` (source) and `.ipynb` (distribution)

See [PYTHON_TO_NOTEBOOK_WORKFLOW.md](PYTHON_TO_NOTEBOOK_WORKFLOW.md) for details.

## Next Steps

1. ✅ **Created lens-colab-designer agent** - Expert at designing OERC-structured pedagogical notebooks
2. ✅ **Created colab-writer agent** - Transforms designs into executable notebooks
3. ✅ **First colab implemented** - Ch01 AI Triangle (Python source + notebook)
4. ⏳ **Test Ch01 colab** - Upload to Google Colab and validate student experience
5. ⏳ **Redesign remaining chapters** - Using Lens framework and hybrid tool approach
6. ⏳ **Prototype Phase 1** - Core analytical models for Ch02 proof-of-concept
7. ⏳ **Implement Lens package** - `pip install lens-mlsys` or embedded in Colab

---

**Total Estimated Colabs**: 45-50 across 20 chapters (consolidated from original 98)

**Package name**: `lens` (or `lens-mlsys` if PyPI conflict)

**Import style**: `from lens import hardware, roofline, federated, carbon`
