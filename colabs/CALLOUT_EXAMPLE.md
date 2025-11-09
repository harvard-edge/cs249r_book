# How to Add Colab Callouts to Textbook

This document shows how to integrate Colab notebooks into textbook chapters using the `callout-colab` block.

## Basic Usage

Place the callout block at strategic points in your `.qmd` file after introducing the theoretical concept:

```markdown
::: {.callout-colab}
## 🔬 Hands-On: Quantization in Action

Experience INT8 quantization reducing model size and inference latency through hands-on experimentation.

**Learning Objective**: Understand quantization trade-offs by measuring actual size reduction and speedup

**Estimated Time**: 6-8 minutes

**What You'll Do**:
- Apply post-training quantization to MobileNetV2
- Measure size reduction (~75%) and speedup (2-4x)  
- Visualize weight distributions before and after

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/colabs/ch10_optimizations/quantization_demo.ipynb)

:::
```

## Visual Result

The callout will render with:
- **Light peach background** (FFF5E6) - Warm and inviting
- **Vibrant orange border** (FF6B35) - Signals action and interactivity
- **"Interactive Colab" label** - Clear purpose
- **Colab badge** - One-click launch

## Placement Strategy

### ✅ DO: Place After Theory Introduction

```markdown
## Quantization and Precision Optimization

[Theory explanation of how quantization works...]

[Mathematical formulation...]

[Benefits and trade-offs discussion...]

::: {.callout-colab}
[Colab callout here - lets students experience what they just learned]
:::

[Continue with advanced topics...]
```

### ❌ DON'T: Place Before Theory

```markdown
## Quantization

::: {.callout-colab}
[Too early - students haven't learned the concepts yet]
:::

[Theory should come first...]
```

### ❌ DON'T: Overuse

```markdown
## Section 1

[Content...]

::: {.callout-colab}
[Colab 1]
:::

## Section 2

[Content...]

::: {.callout-colab}
[Colab 2 - Too many interrupts reading flow]
:::
```

## Components of a Good Callout

### 1. Title with Emoji
```markdown
## 🔬 Hands-On: [Descriptive Title]
```

**Emoji Suggestions**:
- 🔬 Hands-On / Experiment
- 🎯 Practice / Exercise
- 💻 Code / Implementation
- 🚀 Action / Launch

### 2. Brief Description (1-2 sentences)
```markdown
Experience INT8 quantization reducing model size and latency.
```

### 3. Learning Objective (Specific, Measurable)
```markdown
**Learning Objective**: Understand quantization trade-offs by measuring actual metrics
```

### 4. Estimated Time
```markdown
**Estimated Time**: 6-8 minutes
```

### 5. What You'll Do (3-4 bullets)
```markdown
**What You'll Do**:
- Apply technique to real model
- Measure concrete metrics
- Visualize results
```

### 6. Colab Badge Link
```markdown
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebook-url)
```

## URL Format for Colab Badge

```
https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/colabs/[chapter_dir]/[notebook_name].ipynb
```

**Example**:
```
https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/colabs/ch10_optimizations/quantization_demo.ipynb
```

## Complete Examples by Chapter

### Chapter 3: Gradient Descent Visualization

```markdown
::: {.callout-colab}
## 🎯 Hands-On: Gradient Descent Visualization

Visualize how gradient descent navigates loss landscapes and see how learning rate affects convergence.

**Learning Objective**: Understand gradient descent dynamics through interactive visualization

**Estimated Time**: 5-7 minutes

**What You'll Do**:
- Visualize 2D/3D loss surfaces
- Adjust learning rate and see convergence paths
- Compare SGD vs momentum vs Adam optimizers

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/colabs/ch03_dl_primer/gradient_descent_visualization.ipynb)

:::
```

### Chapter 6: Data Quality Impact

```markdown
::: {.callout-colab}
## 🔬 Hands-On: Data Quality Impact

Quantify how data quality affects model performance by training on clean vs. noisy data.

**Learning Objective**: Measure the relationship between data quality and model accuracy

**Estimated Time**: 5-7 minutes

**What You'll Do**:
- Train identical models on clean and noisy datasets
- Introduce different types of noise (label, feature, missing values)
- Measure and visualize accuracy degradation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/colabs/ch06_data_engineering/data_quality_impact.ipynb)

:::
```

### Chapter 11: Hardware Comparison

```markdown
::: {.callout-colab}
## 💻 Hands-On: CPU vs GPU vs TPU Performance

Compare inference performance across different hardware accelerators available in Colab.

**Learning Objective**: Understand hardware-specific optimization benefits

**Estimated Time**: 5-7 minutes

**What You'll Do**:
- Run identical operations on CPU, GPU, and TPU
- Measure throughput and latency differences
- Visualize scaling behavior with batch size

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harvard-edge/cs249r_book/blob/main/colabs/ch11_hw_acceleration/cpu_gpu_tpu_comparison.ipynb)

:::
```

## Format-Specific Content

### Show Different Content for HTML vs PDF

```markdown
::: {.content-visible when-format="html"}
::: {.callout-colab}
## 🔬 Hands-On: Quantization in Action

[Full callout with badge]

[![Open in Colab](badge)](url)

:::
:::

::: {.content-visible when-format="pdf"}
> **Interactive Exercise Available Online**
>
> An interactive Colab notebook demonstrating quantization is available at:
> https://mlsysbook.ai/colabs/ch10/quantization
>
> Scan QR code to access: [QR code image]
:::
```

## Testing Your Callout

### Local Preview

1. Build the book locally:
```bash
cd quarto
quarto preview
```

2. Navigate to your chapter and verify:
   - [ ] Callout renders with orange styling
   - [ ] Badge displays correctly
   - [ ] Link works when clicked
   - [ ] Placement flows naturally with text

### Visual Checklist

- [ ] Background is light peach (not too strong)
- [ ] Border is vibrant orange (stands out but not overwhelming)
- [ ] "Interactive Colab" label is visible
- [ ] Badge is properly sized
- [ ] Emoji renders correctly
- [ ] Bullets are formatted
- [ ] Link is clickable

## Style Guide

### Writing Style
- **Active voice**: "Visualize..." not "You will visualize..."
- **Action verbs**: Apply, Measure, Compare, Visualize, Explore
- **Concrete outcomes**: "75% smaller" not "much smaller"
- **Friendly tone**: Inviting, not intimidating

### Length
- **Title**: 3-6 words
- **Description**: 1-2 sentences
- **What You'll Do**: 3-4 bullets, each 5-10 words

### Tone
- Exciting but not hyperbolic
- Educational but not pedantic
- Technical but accessible

## Common Mistakes to Avoid

### ❌ Too Vague
```markdown
**Learning Objective**: Learn about quantization
```

### ✅ Specific and Measurable
```markdown
**Learning Objective**: Measure 4x size reduction from INT8 quantization
```

---

### ❌ Too Long
```markdown
**What You'll Do**:
- First we will load a pre-trained MobileNetV2 model from PyTorch...
- Then measure the baseline FP32 model size in megabytes...
- [5 more long bullets]
```

### ✅ Concise
```markdown
**What You'll Do**:
- Load pre-trained MobileNetV2
- Measure baseline FP32 size
- Apply INT8 quantization
- Compare metrics
```

---

### ❌ No Connection to Text
```markdown
[Random Colab unrelated to current section]
```

### ✅ Reinforces Just-Learned Concepts
```markdown
[Colab placed right after explaining quantization theory]
```

---

## Maintenance

When updating callouts:

1. **Check links** - Ensure notebook path is correct
2. **Update time estimates** - If notebook execution time changes
3. **Refresh content** - Keep "What You'll Do" aligned with notebook
4. **Test rendering** - Verify styling after Quarto updates

---

**Remember**: Callouts should enhance, not interrupt, the reading experience. Place them strategically where hands-on reinforcement adds maximum value.

