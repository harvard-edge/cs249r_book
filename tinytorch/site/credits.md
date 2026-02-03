# Acknowledgments

**TinyTorch stands on the shoulders of giants.**

This project draws inspiration from pioneering educational ML frameworks and owes its existence to the open source community's commitment to accessible ML education.


## Core Inspirations

### MiniTorch
**[minitorch.github.io](https://minitorch.github.io/)** by Sasha Rush (Cornell Tech)

TinyTorch's pedagogical DNA comes from MiniTorch's brilliant "build a framework from scratch" approach. MiniTorch pioneered teaching ML through implementation rather than usage, proving students gain deeper understanding by building systems themselves.

**What MiniTorch teaches**: Automatic differentiation through minimal, elegant implementations

**How TinyTorch differs**: Extends to full systems engineering including optimization, profiling, and production deployment across Foundation → Architecture → Optimization tiers

**When to use MiniTorch**: Excellent complement for deep mathematical understanding of autodifferentiation

**Connection to TinyTorch**: Modules 06-08 (Autograd, Optimizers, Training) share philosophical DNA with MiniTorch's core pedagogy


### micrograd
**[github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)** by Andrej Karpathy

Micrograd demonstrated that automatic differentiation—the heart of modern ML—can be taught in ~100 lines of elegant Python. Its clarity and simplicity inspired TinyTorch's emphasis on understandable implementations.

**What micrograd teaches**: Autograd engine in 100 beautiful lines of Python

**How TinyTorch differs**: Comprehensive framework covering vision, language, and production systems (20 modules vs. single-file implementation)

**When to use micrograd**: Perfect 2-hour introduction before starting TinyTorch

**Connection to TinyTorch**: Module 06 (Autograd) teaches the same core concepts with systems engineering focus


### nanoGPT
**[github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy

nanoGPT's minimalist transformer implementation showed how to teach modern architectures without framework abstraction. TinyTorch's transformer modules (12, 13) follow this philosophy: clear, hackable implementations that reveal underlying mathematics.

**What nanoGPT teaches**: Clean transformer implementation for understanding GPT architecture

**How TinyTorch differs**: Build transformers from tensors up, understanding all dependencies from scratch

**When to use nanoGPT**: Complement to TinyTorch Modules 10-13 for transformer-specific deep-dive

**Connection to TinyTorch**: Module 13 (Transformers) culminates in similar architecture built from your own tensor operations


### tinygrad
**[github.com/geohot/tinygrad](https://github.com/geohot/tinygrad)** by George Hotz

Tinygrad proves educational frameworks can achieve impressive performance. While TinyTorch optimizes for learning clarity over speed, tinygrad's emphasis on efficiency inspired our Optimization Tier's production-focused modules.

**What tinygrad teaches**: Performance-focused educational framework with actual GPU acceleration

**How TinyTorch differs**: Pedagogy-first with explicit systems thinking and scaffolding (educational over performant)

**When to use tinygrad**: After TinyTorch for performance optimization deep-dive and GPU programming

**Connection to TinyTorch**: Modules 14-19 (Optimization Tier) share production systems focus



## What Makes TinyTorch Unique

TinyTorch combines inspiration from these projects into a comprehensive ML systems course:

- **Comprehensive Scope**: Only educational framework covering Foundation → Architecture → Optimization
- **Systems Thinking**: Every module includes profiling, complexity analysis, production context
- **Historical Validation**: Milestone system proving implementations through ML history (1958 → 2018)
- **Pedagogical Scaffolding**: Progressive disclosure, Build → Use → Reflect methodology
- **Production Context**: Direct connections to PyTorch, TensorFlow, and industry practices




## ML Systems Book Contributors

TinyTorch is part of the broader [ML Systems Book](https://mlsysbook.ai) ecosystem. These contributors have helped build the educational foundation that TinyTorch extends.

```{raw} html
<style>
.contributor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(75px, 1fr));
  gap: 0.75rem;
  margin: 1.5rem 0;
}
.contributor {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-decoration: none;
  color: inherit;
  transition: transform 0.15s;
}
.contributor:hover {
  transform: translateY(-2px);
}
.contributor img {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  border: 2px solid #e5e7eb;
  transition: border-color 0.15s;
}
.contributor:hover img {
  border-color: #3b82f6;
}
.contributor .name {
  font-size: 0.6rem;
  font-weight: 500;
  margin-top: 0.35rem;
  text-align: center;
  max-width: 70px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>

<div class="contributor-grid">
  <a href="https://github.com/hzeljko" class="contributor" title="Zeljko Hrcek">
    <img src="https://avatars.githubusercontent.com/u/36451783?v=4&s=80" alt="hzeljko" />
    <span class="name">Zeljko Hrcek</span>
  </a>
  <a href="https://github.com/Mjrovai" class="contributor" title="Marcelo Rovai">
    <img src="https://avatars.githubusercontent.com/u/17109416?v=4&s=80" alt="Mjrovai" />
    <span class="name">Marcelo Rovai</span>
  </a>
  <a href="https://github.com/jasonjabbour" class="contributor" title="Jason Jabbour">
    <img src="https://avatars.githubusercontent.com/u/55008744?v=4&s=80" alt="jasonjabbour" />
    <span class="name">Jason Jabbour</span>
  </a>
  <a href="https://github.com/uchendui" class="contributor" title="Ikechukwu Uchendu">
    <img src="https://avatars.githubusercontent.com/u/14854496?v=4&s=80" alt="uchendui" />
    <span class="name">Ike Uchendu</span>
  </a>
  <a href="https://github.com/Naeemkh" class="contributor" title="Naeem Khoshnevis">
    <img src="https://avatars.githubusercontent.com/u/6773835?v=4&s=80" alt="Naeemkh" />
    <span class="name">Naeem K.</span>
  </a>
  <a href="https://github.com/Sara-Khosravi" class="contributor" title="Sara Khosravi">
    <img src="https://avatars.githubusercontent.com/u/76420116?v=4&s=80" alt="Sara-Khosravi" />
    <span class="name">Sara Khosravi</span>
  </a>
  <a href="https://github.com/didier-durand" class="contributor" title="Didier Durand">
    <img src="https://avatars.githubusercontent.com/u/2927957?v=4&s=80" alt="didier-durand" />
    <span class="name">Didier Durand</span>
  </a>
  <a href="https://github.com/18jeffreyma" class="contributor" title="Jeffrey Ma">
    <img src="https://avatars.githubusercontent.com/u/29385425?v=4&s=80" alt="18jeffreyma" />
    <span class="name">Jeffrey Ma</span>
  </a>
  <a href="https://github.com/V0XNIHILI" class="contributor" title="Douwe den Blanken">
    <img src="https://avatars.githubusercontent.com/u/24796206?v=4&s=80" alt="V0XNIHILI" />
    <span class="name">Douwe dB</span>
  </a>
  <a href="https://github.com/shanzehbatool" class="contributor" title="Shanzeh Batool">
    <img src="https://avatars.githubusercontent.com/u/66784337?v=4&s=80" alt="shanzehbatool" />
    <span class="name">Shanzeh B.</span>
  </a>
  <a href="https://github.com/eliasab16" class="contributor" title="Elias">
    <img src="https://avatars.githubusercontent.com/u/55062776?v=4&s=80" alt="eliasab16" />
    <span class="name">Elias</span>
  </a>
  <a href="https://github.com/JaredP94" class="contributor" title="Jared Ping">
    <img src="https://avatars.githubusercontent.com/u/13906915?v=4&s=80" alt="JaredP94" />
    <span class="name">Jared Ping</span>
  </a>
  <a href="https://github.com/ishapira1" class="contributor" title="Itai Shapira">
    <img src="https://avatars.githubusercontent.com/u/122899003?v=4&s=80" alt="ishapira1" />
    <span class="name">Itai Shapira</span>
  </a>
  <a href="https://github.com/jaysonzlin" class="contributor" title="Jayson Lin">
    <img src="https://avatars.githubusercontent.com/u/52141513?v=4&s=80" alt="jaysonzlin" />
    <span class="name">Jayson Lin</span>
  </a>
  <a href="https://github.com/sophiacho1" class="contributor" title="Sophia Cho">
    <img src="https://avatars.githubusercontent.com/u/67521139?v=4&s=80" alt="sophiacho1" />
    <span class="name">Sophia Cho</span>
  </a>
  <a href="https://github.com/alxrod" class="contributor" title="Alex Rodriguez">
    <img src="https://avatars.githubusercontent.com/u/11152802?v=4&s=80" alt="alxrod" />
    <span class="name">Alex Rodriguez</span>
  </a>
  <a href="https://github.com/korneelf1" class="contributor" title="Korneel Van den Berghe">
    <img src="https://avatars.githubusercontent.com/u/65716068?v=4&s=80" alt="korneelf1" />
    <span class="name">Korneel VdB</span>
  </a>
  <a href="https://github.com/colbybanbury" class="contributor" title="Colby Banbury">
    <img src="https://avatars.githubusercontent.com/u/17261463?v=4&s=80" alt="colbybanbury" />
    <span class="name">Colby Banbury</span>
  </a>
  <a href="https://github.com/zishenwan" class="contributor" title="Zishen Wan">
    <img src="https://avatars.githubusercontent.com/u/42975815?v=4&s=80" alt="zishenwan" />
    <span class="name">Zishen Wan</span>
  </a>
</div>
```

**[View all 40+ contributors on GitHub →](https://github.com/harvard-edge/cs249r_book/graphs/contributors)**


## License

TinyTorch is released under the MIT License, ensuring it remains free and open for educational use.


**Thank you to everyone building the future of accessible ML education.**
