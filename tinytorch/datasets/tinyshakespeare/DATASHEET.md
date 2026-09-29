# Datasheet for TinyShakespeare

Datasheet format based on *Datasheets for Datasets* (Gebru et al., 2018/2021).

## Motivation

### For what purpose was the dataset created?
TinyShakespeare was compiled to provide an accessible, public-domain text corpus for teaching autoregressive sequence modeling without the massive compute overhead of industrial web corpora.

### Who created the dataset?
Originally extracted from Project Gutenberg by Andrej Karpathy (2015). Bundled and curated for TinyTorch by the Harvard Edge Lab.

---

## Composition

### What do the instances that comprise the dataset represent?
The dataset consists of continuous English dramatic dialogue and poetry from William Shakespeare plays (including *Coriolanus*, *Julius Caesar*, and sonnets).

### How many instances are there in total?
* **Bundled Sample (`tinyshakespeare_sample.txt`):** 21,096 characters (801 lines).
* **Full Corpus:** 1,115,394 characters (~40,000 lines).

### What data does each instance consist of?
Plain UTF-8 text with speaker turns labeled in capital letters followed by colons (e.g. `First Citizen:`, `All:`).

---

## Collection Process

### How was the data acquired?
Extracted from public domain electronic texts of William Shakespeare works provided by Project Gutenberg.

---

## Preprocessing, Cleaning, and Labeling

### Was any preprocessing done?
Character normalization to standard UTF-8. No synthetic alterations or modernizations were applied to preserve historical meter and poetic vocabulary.

---

## Uses

### What tasks has the dataset been used for?
* Character-level autoregressive language modeling.
* Vocabulary construction and embedding projection.
* Temperature and top-k sampling demonstration.

### What tasks should the dataset NOT be used for?
TinyShakespeare should not be used for modern factual question answering, grammar instruction, or safety evaluations. It reflects 16th-century Early Modern English dialogue.

---

## Distribution & Maintenance

### How will the dataset be distributed?
The offline sample is bundled directly in the TinyTorch repository under `tinytorch/datasets/tinyshakespeare/`. The full corpus is hosted publicly on GitHub and Hugging Face.

### What is the dataset license?
Public Domain (CC0 / Project Gutenberg License).
