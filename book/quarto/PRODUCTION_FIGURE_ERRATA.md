# Production Figure Errata & Reconciliation

**Date**: February 3, 2026  
**Document**: Machine Learning Systems - Volume I  
**Purpose**: Address figure numbering discrepancies and missing captions

---

## CRITICAL: Chapter Numbering Clarification

The source files use the following chapter structure:

| Chapter # | Chapter Name | Figure Range | Figure Count |
|-----------|--------------|--------------|--------------|
| 1 | Introduction | 1.1 - 1.8 | 8 figures |
| 2 | ML Systems | 2.1 - 2.12 | 12 figures |
| 3 | ML Workflow | 3.1 - 3.6 | 6 figures |
| 4 | Data Engineering | 4.1 - 4.17 | 17 figures |
| 5 | Neural Computation | 5.1 - 5.20 | 20 figures |
| 6 | Network Architectures | 6.1 - 6.10 | 10 figures |
| 7 | ML Frameworks | 7.1 - 7.14 | 14 figures |
| 8 | Model Training | 8.1 - 8.19 | 19 figures |
| 9 | Data Selection | 9.1 - 9.13 | 13 figures |
| 10 | Model Compression | 10.1 - 10.32 | 32 figures |
| 11 | Hardware Acceleration | 11.1 - 11.14 | 14 figures |
| 12 | Benchmarking | 12.1 - 12.13 | 13 figures |
| 13 | Model Serving | 13.1 - 13.7 | 7 figures |
| 14 | ML Operations | 14.1 - 14.12 | 12 figures |
| 15 | Responsible Engineering | 15.1 - 15.6 | 6 figures |
| 16 | Conclusion | 16.1 | 1 figure |

**TOTAL: 204 figures**

### Note on Production Feedback

If production's "alt text doc" shows figures starting at 2.1 because "chapter one has no figures":
- This may indicate the **Introduction chapter is numbered differently** in the PDF manuscript
- OR the PDF was generated with a different configuration
- Please verify the chapter numbering in the actual PDF manuscript

---

## Errata Response: Chapter 2 Issues

Production reported these issues for "Chapter 2":

| Production Says | Source File Shows | Resolution |
|-----------------|-------------------|------------|
| "Figure 2.5 - ML System Lifecycle appears as Fig. 7" | Figure 1.5 "ML System Lifecycle" is in Introduction (Ch. 1) | **Check if Introduction is Chapter 1 or unnumbered in PDF** |
| "Figure 2.6 - Era of Scale missing alt text" | Figure 1.8 "The Era of Scale" in Introduction HAS alt text | Alt text exists in source |
| "Figure 2.7 - Algorithmic Efficiency appears as Fig. 5" | Figure 1.7 "Algorithmic Efficiency Trajectory" in Introduction | Numbering mismatch |
| "Figure 2.8 - Title & caption missing" | Not clear which figure this refers to | Need clarification |

---

## FIGURES MISSING TITLE & CAPTION (28 total)

These figures have alt text but NO title/caption. They need editorial review:

### Chapter 1: Introduction
- **Figure 1.6** - Alt: "Five pillars diagram: Data Engineering, Training Systems..."
  - *This is the Five-Pillar Framework diagram*

### Chapter 2: ML Systems  
- **Figure 2.3** - Alt: "Aerial view of Google Cloud TPU data center..."
  - *This is the Cloud Data Center Scale image*
- **Figure 2.5** - Alt: "Collection of IoT devices arranged on a surface..."
  - *This is the Edge Device Deployment image*
- **Figure 2.7** - Alt: "Small development boards including Arduino Nano BLE Sense..."
  - *This is the TinyML System Scale image*

### Chapter 3: ML Workflow
- **Figure 3.4** - Alt: "Two side-by-side retinal fundus images..."
  - *This is the Retinal Hemorrhages (Diabetic Retinopathy) image*

### Chapter 4: Data Engineering
- **Figure 4.3** - Alt: "Diagram showing voice-activated device with microphone..."
  - *This is the Keyword Spotting System diagram*
- **Figure 4.5** - Alt: "Historical black-and-white photograph from 1914..."
  - *This is the Data Source Noise (1914 traffic) image*
- **Figure 4.9** - Alt: "Two-panel visualization showing raw audio waveform..."
  - *This is the Audio Feature Transformation image*
- **Figure 4.11** - Alt: "Three versions of same street scene showing annotation..."
  - *This is the Data Annotation Granularity image*
- **Figure 4.12** - Alt: "Grid of example images showing labeling challenges..."
  - *This is the Labeling Ambiguity image*
- **Figure 4.14** - Alt: "Pipeline showing audio waveform and text transcript..."
  - *This is the Multilingual Data Preparation image*

### Chapter 5: Neural Computation
- **Figure 5.1** - Alt: "Nested circles diagram showing AI as outermost circle..."
  - *This is the AI Hierarchy diagram*
- **Figure 5.4** - Alt: "Decision tree flowchart for activity classification..."
  - *This is the Activity Classification Decision Tree*
- **Figure 5.5** - Alt: "Three-panel image showing HOG feature extraction..."
  - *This is the HOG Method image*
- **Figure 5.7** - Alt: "Side-by-side comparison of biological neuron and artificial neuron..."
  - *This is the Biological-to-Artificial Neuron Mapping*
- **Figure 5.8** - Alt: "Log-scale scatter plot showing training compute in FLOPS..."
  - *This is the Computational Growth chart*
- **Figure 5.18** - Alt: "Grid of handwritten digit samples from USPS dataset..."
  - *This is the Handwritten Digit Variability image*

### Chapter 7: ML Frameworks
- **Figure 7.12** - NEEDS INVESTIGATION

### Chapter 8: Model Training
- **Figure 8.6** - NEEDS INVESTIGATION

### Chapter 9: Data Selection
- **Figure 9.1** - NEEDS INVESTIGATION

### Chapter 10: Model Compression
- **Figure 10.15** - NEEDS INVESTIGATION
- **Figure 10.19** - NEEDS INVESTIGATION
- **Figure 10.30** - NEEDS INVESTIGATION
- **Figure 10.31** - NEEDS INVESTIGATION

### Chapter 11: Hardware Acceleration
- **Figure 11.3** - NEEDS INVESTIGATION

### Chapter 15: Responsible Engineering
- **Figure 15.1** - NEEDS INVESTIGATION
- **Figure 15.2** - NEEDS INVESTIGATION
- **Figure 15.3** - NEEDS INVESTIGATION

---

## DALL-E Cover Images

Each chapter begins with a DALL-E generated cover image. These typically:
- Have alt text describing the image
- Do NOT have fig-cap (caption) because they are decorative
- Should be documented separately if needed for accessibility

Chapters with DALL-E cover images that need entries in alt text doc:
- Chapter 1: Introduction cover
- Chapter 2: ML Systems cover
- Chapter 3: ML Workflow cover  
- Chapter 4: Data Engineering cover
- Chapter 5: Neural Computation cover
- (and all subsequent chapters)

---

## Next Steps

1. **Verify PDF chapter numbering** - Check if Introduction is Chapter 1 or Chapter 0 in the rendered PDF
2. **Add missing titles/captions** - The 28 figures listed above need editorial review
3. **Reconcile alt text doc** - The external alt text document needs to match source file numbering
4. **Review DALL-E images** - Confirm if cover images need entries in alt text doc

---

## Full Figure List

See `FIGURE_LIST_VOL1.md` for complete figure inventory with:
- All figure numbers
- Titles
- Full captions
- Alt text

Generated: February 3, 2026
