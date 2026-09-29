# Datasheet for TinyDigits

Datasheet format based on *Datasheets for Datasets* (Gebru et al., 2018/2021).

## Motivation

### For what purpose was the dataset created?
TinyDigits was created to provide a lightweight, deterministic, 100% offline vision dataset for educational machine learning frameworks. Standard vision datasets like full MNIST (55 MB uncompressed) or CIFAR-10 (170 MB) introduce download latency, network failures, and lengthy training loops in classroom environments. TinyDigits provides genuine 2D pixel inputs that train in under 10 seconds on a single CPU core.

### Who created the dataset?
Curated by the TinyTorch development team at the Harvard Edge Lab.

### Who funded the creation of the dataset?
Harvard Edge Lab educational infrastructure research.

---

## Composition

### What do the instances that comprise the dataset represent?
Each instance represents an 8x8 normalized grayscale image of a handwritten numerical digit from 0 to 9, paired with its integer class label.

### How many instances are there in total?
* **Total Instances:** 1,200 samples.
* **Training Set (`train.pkl`):** 1,000 samples (strictly balanced: exactly 100 samples per digit 0 through 9).
* **Test Set (`test.pkl`):** 200 samples (strictly balanced: exactly 20 samples per digit 0 through 9).

### Does the dataset contain all possible instances or is it a sample?
It is a curated, stratified subsample of the optical recognition of handwritten digits dataset distributed via scikit-learn (originally created by E. Alpaydin and C. Kaynak, Bogazici University).

### What data does each instance consist of?
* `images`: NumPy float32 array of shape `(8, 8)` with pixel intensities scaled to the range `[0.0, 1.0]`.
* `labels`: NumPy int64 scalar in the range `[0, 9]`.

### Is any information missing from individual instances?
No. All 1,200 instances have complete pixel grids and verified ground-truth labels.

---

## Collection Process

### How was the data associated with each instance acquired?
Raw 8x8 normalized digit arrays were extracted from the canonical NIST downsampled distribution using a deterministic random seed (`seed=42`).

### What mechanisms were used to collect the data?
Scripted automated extraction using `create_tinydigits.py`.

---

## Preprocessing, Cleaning, and Labeling

### Was any preprocessing done?
1. Pixel values were divided by 16.0 to normalize the original 4-bit integer values into floating-point range `[0.0, 1.0]`.
2. Exact class balance was enforced to eliminate majority-class bias.
3. Serialized into standard Python pickle dictionaries without external dependencies.

---

## Uses

### What tasks has the dataset been used for?
* Multi-Layer Perceptron (MLP) classification in Milestone 03.
* 2D Convolutional Neural Network (CNN) feature extraction in Milestone 04.
* INT8 Quantization and SVD compression verification in Part III.

### What tasks should the dataset NOT be used for?
TinyDigits should not be used as a high-capacity production computer vision benchmark. The 8x8 resolution is intentionally downsampled for rapid educational verification on CPUs.

---

## Distribution & Maintenance

### How will the dataset be distributed?
Shipped directly inside the TinyTorch git repository under `tinytorch/datasets/tinydigits/` and hosted via the Harvard Edge organization on Hugging Face.

### What is the dataset license?
BSD 3-Clause License (inherited from scikit-learn and the original Bogazici University digit distribution).
