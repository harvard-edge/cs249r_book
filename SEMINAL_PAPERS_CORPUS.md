# Volume 1 Seminal Papers Corpus

This document defines the **core corpus of papers** that should be cited in each chapter, with justification for why each is seminal.

Generated: January 29, 2026

---

## How to Use This Document

For each chapter:
1. Check if the paper is already cited
2. If not cited but topic is discussed → ADD the citation
3. If topic is not discussed → SKIP (don't force citations)

---

## Chapter 1: Introduction

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Computing Machinery and Intelligence | Turing | 1950 | Introduced Turing Test, framed machine intelligence |
| A Proposal for the Dartmouth Summer Research Project | McCarthy et al. | 1955 | Coined "artificial intelligence", launched AI as field |
| The Perceptron | Rosenblatt | 1957 | First learning algorithm that adjusts weights from data |
| Perceptrons: An Introduction to Computational Geometry | Minsky & Papert | 1969 | Proved perceptron limitations, caused first AI winter |
| Learning Representations by Back-Propagating Errors | Rumelhart, Hinton, Williams | 1986 | Popularized backpropagation, enabled deep learning |
| ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky et al. | 2012 | Sparked deep learning revolution |
| Software 2.0 | Karpathy | 2017 | Framed shift from code to learned models |
| The Bitter Lesson | Sutton | 2019 | Showed computation beats encoded expertise |
| Hidden Technical Debt in ML Systems | Sculley et al. | 2015 | Established ML systems engineering as discipline |
| AI and Compute | Amodei & Hernandez | 2018 | Quantified exponential growth in AI compute |

---

## Chapter 2: ML Systems

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| In-Datacenter Performance Analysis of a TPU | Jouppi et al. | 2017 | First TPU disclosure, established domain-specific accelerators |
| Hitting the Memory Wall | Wulf & McKee | 1995 | Coined "memory wall", identified fundamental bottleneck |
| MobileNets | Howard et al. | 2017 | Enabled efficient mobile deployment |
| Communication-Efficient Learning (FedAvg) | McMahan et al. | 2017 | Established federated learning |
| Widening Access to Applied ML with TinyML | Reddi et al. | 2022 | Democratized ML on resource-constrained devices |
| MLPerf Tiny Benchmark | Banbury, Reddi et al. | 2021 | First benchmark for microcontroller ML |
| Deep Learning Recommendation Model (DLRM) | Naumov et al. | 2019 | Industry-standard recommendation architecture |
| Roofline Model | Williams et al. | 2009 | Framework for compute vs memory-bound analysis |

---

## Chapter 3: Deep Learning Primer

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Learning Representations by Back-Propagating Errors | Rumelhart et al. | 1986 | Standard training algorithm |
| Rectified Linear Units Improve RBMs | Nair & Hinton | 2010 | Established ReLU as default activation |
| Adam: A Method for Stochastic Optimization | Kingma & Ba | 2014 | Default optimizer for most applications |
| Dropout: Preventing Overfitting | Srivastava et al. | 2014 | Standard regularization technique |
| Batch Normalization | Ioffe & Szegedy | 2015 | Enables faster, stable training |
| Understanding Difficulty of Training Deep Networks | Glorot & Bengio | 2010 | Xavier/Glorot initialization |
| Deep Learning (Nature) | LeCun, Bengio, Hinton | 2015 | Landmark review marking mainstream acceptance |
| Approximation by Superpositions of Sigmoidal Function | Cybenko | 1989 | Universal approximation theorem |
| Delving Deep into Rectifiers (He Init) | He et al. | 2015 | Initialization for ReLU networks |

---

## Chapter 4: DNN Architectures

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Gradient-based Learning (LeNet) | LeCun et al. | 1998 | First successful CNN |
| ImageNet Classification (AlexNet) | Krizhevsky et al. | 2012 | Deep learning breakthrough |
| Very Deep CNNs (VGGNet) | Simonyan & Zisserman | 2014 | Showed depth improves performance |
| Going Deeper with Convolutions (GoogLeNet) | Szegedy et al. | 2015 | Multi-scale inception modules |
| Deep Residual Learning (ResNet) | He et al. | 2016 | Skip connections enabled 100+ layers |
| Densely Connected CNNs (DenseNet) | Huang et al. | 2017 | Feature reuse through dense connectivity |
| Long Short-Term Memory | Hochreiter & Schmidhuber | 1997 | Gating for long-term dependencies |
| GRU | Cho et al. | 2014 | Simpler alternative to LSTM |
| Neural Machine Translation (Attention) | Bahdanau et al. | 2014 | Introduced attention mechanism |
| Attention Is All You Need | Vaswani et al. | 2017 | Transformer architecture |
| BERT | Devlin et al. | 2019 | Bidirectional pre-training paradigm |
| GPT | Radford et al. | 2018 | Autoregressive pre-training |
| Vision Transformer (ViT) | Dosovitskiy et al. | 2021 | Transformers for vision |
| Layer Normalization | Ba et al. | 2016 | Essential for transformers |

---

## Chapter 5: ML Frameworks

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| TensorFlow | Abadi et al. | 2016 | Static graph execution model |
| PyTorch | Paszke et al. | 2019 | Dynamic graph, define-by-run |
| JAX/Autograd | Frostig et al. / Bradbury et al. | 2018 | Functional transformations |
| Theano | Bergstra et al. | 2010 | First symbolic computation + autodiff |
| Automatic Differentiation Survey | Baydin et al. | 2018 | Definitive autodiff reference |
| cuDNN | Chetlur et al. | 2014 | GPU primitives foundation |
| BLAS | Lawson et al. | 1979 | Linear algebra interface standard |
| Training with Sublinear Memory | Chen et al. | 2016 | Gradient checkpointing |

---

## Chapter 6: Training

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Learning Representations by Back-Propagating Errors | Rumelhart et al. | 1986 | Core training algorithm |
| Mixed Precision Training | Micikevicius et al. | 2017 | FP16/FP32 training |
| Training with Sublinear Memory | Chen et al. | 2016 | Gradient checkpointing |
| FlashAttention | Dao et al. | 2022 | IO-aware attention, O(n) memory |
| Accurate, Large Minibatch SGD | Goyal et al. | 2017 | Linear scaling rule for large batches |
| Large Scale Distributed Deep Networks | Dean et al. | 2012 | Parameter server architecture |
| Horovod | Sergeev & Del Balso | 2018 | Ring AllReduce for distributed training |
| SGDR: Warm Restarts | Loshchilov & Hutter | 2016 | Cosine annealing schedule |

---

## Chapter 7: Hardware Acceleration

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Scalable Parallel Programming with CUDA | Nickolls et al. | 2008 | GPU computing model |
| cuDNN | Chetlur et al. | 2014 | GPU deep learning primitives |
| In-Datacenter TPU Analysis | Jouppi et al. | 2017 | TPU architecture |
| Ten Lessons from Three TPU Generations | Jouppi et al. | 2021 | TPU evolution |
| Systolic Arrays for VLSI | Kung & Leiserson | 1979 | Systolic array concept |
| Why Systolic Architectures? | Kung | 1982 | Systolic design principles |
| Eyeriss | Chen et al. | 2016 | Dataflow taxonomy (weight/output/input stationary) |
| TVM | Chen et al. | 2018 | ML compiler with auto-tuning |
| MLIR | Lattner et al. | 2019 | Multi-level IR for ML |
| Roofline Model | Williams et al. | 2009 | Compute vs memory-bound analysis |
| Efficient Processing of DNNs Survey | Sze et al. | 2017 | Comprehensive accelerator survey |

---

## Chapter 8: Model Compression

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Quantization and Training for Efficient Inference | Jacob et al. | 2018 | Standard INT8 quantization |
| Deep Compression | Han et al. | 2015 | Pruning + quantization pipeline |
| Optimal Brain Damage | LeCun et al. | 1989 | First pruning formalization |
| Pruning Filters for Efficient ConvNets | Li et al. | 2017 | Structured pruning |
| Distilling Knowledge in a Neural Network | Hinton et al. | 2015 | Knowledge distillation |
| Neural Architecture Search with RL | Zoph & Le | 2017 | Automated architecture discovery |
| DARTS | Liu et al. | 2019 | Differentiable NAS |
| MobileNets | Howard et al. | 2017 | Depthwise separable convolutions |
| EfficientNet | Tan & Le | 2019 | Compound scaling |
| Lottery Ticket Hypothesis | Frankle & Carlin | 2019 | Sparse trainable subnetworks |

---

## Chapter 9: Benchmarking

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| MLPerf Training Benchmark | Mattson et al. | 2020 | Industry standard training benchmark |
| MLPerf Inference Benchmark | Reddi et al. | 2020 | Standardized inference evaluation |
| MLPerf Tiny Benchmark | Banbury et al. | 2021 | Microcontroller ML benchmark |
| DAWNBench | Coleman et al. | 2017 | Time-to-accuracy evaluation |
| ImageNet | Deng et al. | 2009 | Standard vision benchmark |
| COCO | Lin et al. | 2014 | Detection/segmentation benchmark |
| SQuAD | Rajpurkar et al. | 2016 | Reading comprehension benchmark |
| GLUE | Wang et al. | 2018 | Multi-task NLP benchmark |

---

## Chapter 10: Serving

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| TensorFlow Serving | Olston et al. | 2017 | Dynamic batching, model serving architecture |
| Clipper | Crankshaw et al. | 2017 | Low-latency prediction serving |
| The Tail at Scale | Dean & Barroso | 2013 | Tail latency in distributed systems |
| Orca | Yu et al. | 2022 | Continuous batching for LLMs |
| vLLM (PagedAttention) | Kwon et al. | 2023 | KV cache memory management |
| FlashAttention | Dao et al. | 2022 | Efficient attention for inference |
| Nexus | Shen et al. | 2019 | GPU cluster for DNN serving |
| Little's Law | Little | 1961 | Queuing theory foundation |

---

## Chapter 11: Data Engineering

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Data Cascades in High-Stakes AI | Sambasivan et al. | 2021 | Data quality as engineering concern |
| Hidden Technical Debt in ML Systems | Sculley et al. | 2015 | Training-serving skew |
| Datasheets for Datasets | Gebru et al. | 2021 | Dataset documentation standard |
| Survey on Concept Drift Adaptation | Gama et al. | 2014 | Drift detection taxonomy |
| Cheap and Fast—But is it Good? | Snow et al. | 2008 | Crowdsourcing quality |

---

## Chapter 12: Data Efficiency

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Scaling Laws for Neural Language Models | Kaplan et al. | 2020 | Power-law scaling relationships |
| Training Compute-Optimal LLMs (Chinchilla) | Hoffmann et al. | 2022 | Optimal data-to-parameter ratios |
| Curriculum Learning | Bengio et al. | 2009 | Easy-to-hard training order |
| Active Learning | Settles | 2009 | Query strategies book |
| FixMatch | Sohn et al. | 2020 | Semi-supervised learning |
| SimCLR | Chen et al. | 2020 | Contrastive self-supervised learning |
| MoCo | He et al. | 2020 | Momentum contrastive learning |
| mixup | Zhang et al. | 2018 | Data augmentation |

---

## Chapter 13: MLOps

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Hidden Technical Debt in ML Systems | Sculley et al. | 2015 | ML technical debt framework |
| Software Engineering for ML | Amershi et al. | 2019 | ML-specific SE practices |
| ML Test Score | Breck et al. | 2017 | Production readiness rubric |
| TFX | Baylor et al. | 2017 | End-to-end ML platform |
| MLflow | Zaharia et al. | 2018 | Experiment tracking standard |

---

## Chapter 14: Responsible Engineering

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| Model Cards for Model Reporting | Mitchell et al. | 2019 | Model documentation standard |
| Datasheets for Datasets | Gebru et al. | 2021 | Dataset documentation |
| "Why Should I Trust You?" (LIME) | Ribeiro et al. | 2016 | Model-agnostic explanations |
| SHAP | Lundberg & Lee | 2017 | Game-theoretic feature attribution |
| Gender Shades | Buolamwini & Gebru | 2018 | Bias audit methodology |
| Equality of Opportunity | Hardt et al. | 2016 | Fairness definitions |
| Inherent Trade-Offs in Fair Risk Scores | Kleinberg et al. | 2016 | Fairness impossibility results |
| Big Data's Disparate Impact | Barocas & Selbst | 2016 | Legal framework for algorithmic discrimination |

---

## Chapter 15: Workflow

| Paper | Authors | Year | Why Seminal |
|-------|---------|------|-------------|
| From Data Mining to KDD | Fayyad et al. | 1996 | KDD process methodology |
| CRISP-DM | Chapman et al. | 2000 | Industry-standard ML workflow |
| Software Engineering for ML | Amershi et al. | 2019 | ML lifecycle principles |

---

## Summary Statistics

| Chapter | Seminal Papers Listed |
|---------|----------------------|
| Introduction | 10 |
| ML Systems | 8 |
| DL Primer | 9 |
| DNN Architectures | 14 |
| Frameworks | 8 |
| Training | 8 |
| HW Acceleration | 11 |
| Model Compression | 10 |
| Benchmarking | 8 |
| Serving | 8 |
| Data Engineering | 5 |
| Data Efficiency | 8 |
| MLOps | 5 |
| Responsible Engr | 8 |
| Workflow | 3 |
| **TOTAL** | **~113 unique papers** |

---

## Next Steps

1. Cross-check each chapter against this corpus
2. Add missing citations where topics are discussed
3. Remove any citations that aren't justified by this list (clutter)

---

*This corpus represents the foundational literature for ML systems. Each paper was selected because it introduced a concept, technique, or result that shaped the field.*
