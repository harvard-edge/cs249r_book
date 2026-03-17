# Volume I: Introduction to Machine Learning Systems

*Foundations for single-machine ML systems.*

[![Read Online](https://img.shields.io/badge/Read_Online-mlsysbook.ai-blue?logo=readthedocs)](https://mlsysbook.ai/vol1/)

---

## About This Volume

Volume I teaches how to build, optimize, and deploy machine learning systems on a single machine with one to eight accelerators. It covers the full stack from data engineering through model serving, grounding every concept in the physical constraints of real hardware: memory hierarchies, compute throughput, and power budgets.

This is the foundational volume. It establishes the quantitative frameworks (the Iron Law of ML Systems, the D.A.M Taxonomy) that Volume II builds on when scaling to fleets of machines.

## Status

Volume I content is **complete and undergoing final editorial polish.** It is ready for classroom use. Chapters are being reviewed for prose quality, figure consistency, and cross-reference accuracy, but the technical content is stable.

## What You Will Learn

<table>
  <thead>
    <tr>
      <th width="20%">Part</th>
      <th width="20%">Focus</th>
      <th width="60%">What You Will Learn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>I. Foundations</b></td>
      <td>Core concepts</td>
      <td>How ML systems differ from traditional software, the Iron Law framework, data engineering at scale</td>
    </tr>
    <tr>
      <td><b>II. Development</b></td>
      <td>Building blocks</td>
      <td>Neural computation, model architectures, framework internals, training systems</td>
    </tr>
    <tr>
      <td><b>III. Optimization</b></td>
      <td>Making it fast</td>
      <td>Data selection, model compression, hardware acceleration, benchmarking methodology</td>
    </tr>
    <tr>
      <td><b>IV. Deployment</b></td>
      <td>Making it work</td>
      <td>Model serving, MLOps, responsible engineering</td>
    </tr>
  </tbody>
</table>

## Chapter Map

<table>
  <thead>
    <tr>
      <th width="5%">#</th>
      <th width="20%">Chapter</th>
      <th width="20%">Directory</th>
      <th width="55%">Core Question</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><b>Introduction</b></td>
      <td><code>introduction/</code></td>
      <td>What are ML systems and why do they need engineering?</td>
    </tr>
    <tr>
      <td>2</td>
      <td><b>ML Systems</b></td>
      <td><code>ml_systems/</code></td>
      <td>How do deployment constraints shape system design?</td>
    </tr>
    <tr>
      <td>3</td>
      <td><b>ML Workflow</b></td>
      <td><code>ml_workflow/</code></td>
      <td>What does the end-to-end ML pipeline look like?</td>
    </tr>
    <tr>
      <td>4</td>
      <td><b>Data Engineering</b></td>
      <td><code>data_engineering/</code></td>
      <td>How do you build data pipelines that feed ML?</td>
    </tr>
    <tr>
      <td>5</td>
      <td><b>Neural Computation</b></td>
      <td><code>nn_computation/</code></td>
      <td>How does math become silicon operations?</td>
    </tr>
    <tr>
      <td>6</td>
      <td><b>Architectures</b></td>
      <td><code>nn_architectures/</code></td>
      <td>How do CNNs, RNNs, and Transformers differ as systems?</td>
    </tr>
    <tr>
      <td>7</td>
      <td><b>Frameworks</b></td>
      <td><code>frameworks/</code></td>
      <td>How do PyTorch and TensorFlow actually work inside?</td>
    </tr>
    <tr>
      <td>8</td>
      <td><b>Training</b></td>
      <td><code>training/</code></td>
      <td>How do you train models efficiently on real hardware?</td>
    </tr>
    <tr>
      <td>9</td>
      <td><b>Data Selection</b></td>
      <td><code>data_selection/</code></td>
      <td>How do you choose the right data for training?</td>
    </tr>
    <tr>
      <td>10</td>
      <td><b>Model Compression</b></td>
      <td><code>optimizations/</code></td>
      <td>How do you shrink models without losing accuracy?</td>
    </tr>
    <tr>
      <td>11</td>
      <td><b>HW Acceleration</b></td>
      <td><code>hw_acceleration/</code></td>
      <td>How do GPUs, TPUs, and accelerators execute neural networks?</td>
    </tr>
    <tr>
      <td>12</td>
      <td><b>Benchmarking</b></td>
      <td><code>benchmarking/</code></td>
      <td>How do you measure and compare ML system performance?</td>
    </tr>
    <tr>
      <td>13</td>
      <td><b>Model Serving</b></td>
      <td><code>model_serving/</code></td>
      <td>How do you deploy models to serve real users?</td>
    </tr>
    <tr>
      <td>14</td>
      <td><b>MLOps</b></td>
      <td><code>ml_ops/</code></td>
      <td>How do you keep ML systems running in production?</td>
    </tr>
    <tr>
      <td>15</td>
      <td><b>Responsible Engineering</b></td>
      <td><code>responsible_engr/</code></td>
      <td>How do you build ML systems that are fair, safe, and trustworthy?</td>
    </tr>
  </tbody>
</table>

## Prerequisites

Volume I assumes a CS/EE undergraduate background: operating systems, computer architecture, data structures and algorithms, linear algebra, calculus, and basic probability.

## How to Read

The chapters are designed to be read in order. Each chapter builds on the frameworks and vocabulary established in prior chapters. The Iron Law introduced in the opening chapter is used throughout the entire volume to reason about performance bottlenecks.

If you are an instructor adopting this for a course, the natural division is:
- **Parts I and II** (Chapters 1 through 8) for a foundations course
- **Parts III and IV** (Chapters 9 through 15) for an optimization and deployment course

## Feedback

If you spot an error, find an explanation that could be clearer, or have a suggestion, please [open an issue](https://github.com/harvard-edge/cs249r_book/issues) or [start a discussion](https://github.com/harvard-edge/cs249r_book/discussions). Even small corrections make the book better for every reader.

## Links

<table>
  <thead>
    <tr>
      <th width="25%">Resource</th>
      <th width="75%">Link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Read online</b></td>
      <td><a href="https://mlsysbook.ai/vol1/">mlsysbook.ai/vol1</a></td>
    </tr>
    <tr>
      <td><b>Volume II</b></td>
      <td><a href="../vol2/">book/quarto/contents/vol2/</a></td>
    </tr>
    <tr>
      <td><b>Main README</b></td>
      <td><a href="../../../../README.md">Repository root</a></td>
    </tr>
    <tr>
      <td><b>Full textbook</b></td>
      <td><a href="https://mlsysbook.ai/book/">mlsysbook.ai/book</a></td>
    </tr>
    <tr>
      <td><b>Discussions</b></td>
      <td><a href="https://github.com/harvard-edge/cs249r_book/discussions">GitHub Discussions</a></td>
    </tr>
  </tbody>
</table>
