# Learning Resources

**TinyTorch teaches you to *build* ML systems. These resources help you understand the *why* behind what you're building.**

---

## Companion Textbook

### Machine Learning Systems
**[mlsysbook.ai](https://mlsysbook.ai)** by Prof. Vijay Janapa Reddi (Harvard University)

<div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 5px solid #1976d2; padding: 1.5rem; border-radius: 0.5rem; margin: 1.5rem 0;">
<p style="margin: 0; color: #0d47a1; font-size: 1.05rem; line-height: 1.6;">
<strong>TinyTorch began as hands-on labs for this textbook.</strong> While TinyTorch can be used standalone, the ML Systems book provides the theoretical depth and production context behind every module you build.
</p>
</div>

**What it teaches**: Systems engineering for production ML—memory hierarchies, performance optimization, deployment strategies, and the engineering decisions behind modern ML frameworks.

**How it connects to TinyTorch**:
- TinyTorch modules directly implement concepts from the book's chapters
- The book explains *why* PyTorch, TensorFlow, and JAX make certain design decisions
- Together, they provide both hands-on implementation and theoretical understanding

**When to use it**: Read in parallel with TinyTorch. When you implement Module 05 (Autograd), read the book's chapter on automatic differentiation to understand the systems engineering behind your code.

---

## Related Academic Courses

- **[CS 329S: Machine Learning Systems Design](https://stanford-cs329s.github.io/)** (Stanford)
  *Production ML systems and deployment*

- **[TinyML and Efficient Deep Learning](https://efficientml.ai)** (MIT 6.5940)
  *Edge computing, model compression, and efficient ML*

- **[CS 249r: Tiny Machine Learning](https://sites.google.com/g.harvard.edu/tinyml/home)** (Harvard)
  *TinyML systems and resource-constrained ML*

- **[CS 231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)** (Stanford)
  *Computer vision - complements TinyTorch Modules 08-09*

- **[CS 224n: Natural Language Processing](http://web.stanford.edu/class/cs224n/)** (Stanford)
  *Transformers and NLP - complements TinyTorch Modules 10-13*

---

## Other Textbooks

- **[Deep Learning](https://www.deeplearningbook.org/)** by Goodfellow, Bengio, Courville
  *Mathematical foundations behind what you implement in TinyTorch*

- **[Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)** by Aurélien Géron
  *Practical implementations using established frameworks*

---

## Minimal Frameworks

**Alternative approaches to building ML from scratch:**

- **[micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy
  *Autograd in 100 lines. Perfect 2-hour intro before TinyTorch.*

- **[nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy
  *Minimalist GPT implementation. Complements TinyTorch Modules 12-13.*

- **[tinygrad](https://github.com/geohot/tinygrad)** by George Hotz
  *Performance-focused educational framework with GPU acceleration.*

---

## Production Framework Internals

- **[PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)** by Edward Yang
  *How PyTorch actually works under the hood*

- **[PyTorch: Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.md)**
  *Custom operators and autograd functions*

---

**Ready to start?** See the **[Quick Start Guide](quickstart-guide)** for a 15-minute hands-on introduction.
