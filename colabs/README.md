# MLSysBook Interactive Colabs

This directory contains interactive Google Colab notebooks that complement the MLSysBook textbook, providing hands-on demonstrations of key concepts at strategic learning junctions.

## Overview

Each Colab is designed as a **"Concept Bridge"** that:
- Illuminates a specific concept with minimal, runnable code
- Shows immediate results connecting theory to observable behavior
- Complements (not duplicates) TinyTorch hands-on implementation
- Completes in 5-10 minutes to maintain reading flow

## Directory Structure

```
colabs/
├── README.md                          # This file
├── docs/                              # Documentation and specifications
│   ├── COLAB_INTEGRATION_PLAN.md      # Complete specifications
│   ├── COLAB_PLACEMENT_MATRIX.md      # Quick reference table
│   ├── COLAB_CHAPTER_OUTLINE.md       # Chapter-by-chapter placement
│   ├── COLAB_TEMPLATE_SPECIFICATION.md # Template standards
│   ├── COLAB_TEMPLATE_EXAMPLE.md      # Working example
│   └── COLAB_STANDARDS_SUMMARY.md     # Quick checklist
├── ch03_dl_primer/                    # Chapter 3 Colabs
├── ch06_data_engineering/             # Chapter 6 Colabs
├── ch08_training/                     # Chapter 8 Colabs
├── ch10_optimizations/                # Chapter 10 Colabs
│   └── quantization_demo.ipynb        # ✓ Quantization demonstration
└── [other chapters...]
```

## Available Colabs

### Phase 1 (v0.5.0 MVP) - 5 Colabs

| Chapter | Colab | Status | Link |
|---------|-------|--------|------|
| Ch 3: DL Primer | Gradient Descent Visualization | 🚧 Planned | - |
| Ch 6: Data Engineering | Data Quality Impact | 🚧 Planned | - |
| Ch 8: Training | Training Dynamics Explorer | 🚧 Planned | - |
| Ch 10: Optimizations | Quantization Demo | ✅ Complete | [Open in Colab](link) |
| Ch 11: Hardware Acceleration | CPU vs GPU vs TPU | 🚧 Planned | - |

### Phase 2 (v0.5.1) - 13 Colabs
Status: Planned

### Phase 3 (v0.5.2) - 10 Colabs
Status: Planned

**Total Planned**: 28 Colabs across 18 chapters

## Using These Colabs

### For Readers

1. **Read the textbook section first** - Colabs complement, not replace, textbook content
2. **Click "Open in Colab"** - Launches notebook in Google Colab (free account sufficient)
3. **Follow the notebook** - Execute cells sequentially
4. **Experiment** - Modify parameters and explore
5. **Connect back to theory** - Review textbook with new insights

### For Contributors

1. **Read the documentation** - Start with `docs/COLAB_STANDARDS_SUMMARY.md`
2. **Follow the template** - Use `docs/COLAB_TEMPLATE_SPECIFICATION.md`
3. **Review examples** - See `docs/COLAB_TEMPLATE_EXAMPLE.md`
4. **Test thoroughly** - Must run in < 10 minutes on Colab Free Tier
5. **Submit PR** - Follow contribution guidelines

## Development Standards

Every MLSysBook Colab must:

- ✅ Have ONE clear learning objective
- ✅ Complete in < 10 minutes on Colab Free Tier
- ✅ Connect explicitly to textbook section
- ✅ Include quantitative results
- ✅ Follow MLSysBook visual standards
- ✅ Be reproducible (seeds set)
- ✅ Include MLSysBook branding

See `docs/COLAB_STANDARDS_SUMMARY.md` for complete checklist.

## Quick Start for Development

```bash
# 1. Review template
cat colabs/docs/COLAB_TEMPLATE_SPECIFICATION.md

# 2. Copy and rename template (when available)
cp colabs/TEMPLATE.ipynb colabs/ch##_chapter/notebook_name.ipynb

# 3. Follow the 10-section structure
# 4. Test on Colab Free Tier
# 5. Run pre-publication checklist
# 6. Submit for review
```

## Integration with Textbook

Colabs are referenced in the textbook using special callout blocks:

```markdown
::: {.callout-colab}
## Interactive Exercise: Quantization in Action

Experience INT8 quantization reducing model size and latency.

**Learning Objective**: Understand quantization trade-offs

**Estimated Time**: 6-8 minutes

[![Open In Colab](badge)](link-to-colab)
:::
```

## Support and Feedback

- **Documentation Issues**: [GitHub Issues](https://github.com/harvard-edge/cs249r_book/issues)
- **Questions**: [GitHub Discussions](https://github.com/harvard-edge/cs249r_book/discussions)
- **Book Website**: https://mlsysbook.ai

## License

All Colabs are licensed under **CC BY-NC-SA 4.0** (same as MLSysBook).

---

**Status**: Phase 1 Development (1/5 Colabs complete)  
**Last Updated**: November 5, 2025  
**Maintainer**: MLSysBook Team

