#!/usr/bin/env python3
"""
TinyTorch Vector Diagram Generator
Produces publication-grade, modern SVGs, vector PDFs, and high-DPI PNGs
using the signature TinyTorch Palette (Torch Navy, Flame Orange, Warm Peach, Slate).
"""

import os
import subprocess
from pathlib import Path

DIAGRAM_DIR_QUARTO = Path("packages/tinytorch/quarto/assets/images/diagrams")
DIAGRAM_DIR_BOOK = Path("packages/tinytorch/book/assets/images/diagrams")

DIAGRAM_DIR_QUARTO.mkdir(parents=True, exist_ok=True)
DIAGRAM_DIR_BOOK.mkdir(parents=True, exist_ok=True)

# Common SVG Header & Styles
SVG_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#ffffff; font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <!-- Drop Shadow Filter -->
    <filter id="cardShadow" x="-10%" y="-10%" width="120%" height="125%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#1B3A5F" flood-opacity="0.08"/>
    </filter>
    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>

    <!-- Gradients -->
    <linearGradient id="torchGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1B3A5F" />
      <stop offset="100%" stop-color="#2E6F7E" />
    </linearGradient>
    <linearGradient id="flameGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF8246" />
      <stop offset="100%" stop-color="#FF5722" />
    </linearGradient>
    <linearGradient id="warmBgGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#FFFDF9" />
      <stop offset="100%" stop-color="#FFF5EC" />
    </linearGradient>
    <linearGradient id="highlightGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#FF8246" stop-opacity="0.15" />
      <stop offset="100%" stop-color="#FF8246" stop-opacity="0.02" />
    </linearGradient>

    <!-- Markers -->
    <marker id="arrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#FF8246" />
    </marker>
    <marker id="arrowNavy" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#1B3A5F" />
    </marker>
    <marker id="arrowTeal" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2E6F7E" />
    </marker>
  </defs>
"""

DIAGRAMS = {}

# 00: Journey Roadmap
DIAGRAMS["00_journey-diag-1.svg"] = """
  <!-- Header Bar -->
  <rect x="30" y="20" width="740" height="50" rx="8" fill="url(#torchGrad)" filter="url(#cardShadow)"/>
  <text x="50" y="52" fill="#FFFFFF" font-size="18" font-weight="700" letter-spacing="0.5">🔥 TINYTORCH 20-MODULE ARCHITECTURE ROADMAP</text>
  <text x="680" y="52" fill="#FF8246" font-size="14" font-weight="600">4 TIERS</text>

  <!-- Tier 1 -->
  <rect x="30" y="90" width="170" height="280" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.5" filter="url(#cardShadow)"/>
  <path d="M30 90 h170 v34 h-170 z" fill="#1B3A5F" />
  <text x="115" y="113" text-anchor="middle" fill="#FFFFFF" font-size="13" font-weight="700">Tier 1: Foundations</text>
  <g transform="translate(45, 140)" font-size="12" font-weight="500" fill="#1B3A5F">
    <rect x="0" y="0" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="16">01. Tensors &amp; Strides</text>
    <rect x="0" y="32" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="48">02. Activations (GELU)</text>
    <rect x="0" y="64" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="80">03. Linear Layers</text>
    <rect x="0" y="96" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="112">04. Cross-Entropy Loss</text>
    <rect x="0" y="132" width="140" height="32" rx="4" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.2"/>
    <text x="10" y="152" fill="#D22341" font-weight="700">Milestone 1: XOR</text>
  </g>

  <!-- Arrow 1->2 -->
  <path d="M200 230 H220" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

  <!-- Tier 2 -->
  <rect x="220" y="90" width="170" height="280" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.5" filter="url(#cardShadow)"/>
  <path d="M220 90 h170 v34 h-170 z" fill="#1B3A5F" />
  <text x="305" y="113" text-anchor="middle" fill="#FFFFFF" font-size="13" font-weight="700">Tier 2: Core Engine</text>
  <g transform="translate(235, 140)" font-size="12" font-weight="500" fill="#1B3A5F">
    <rect x="0" y="0" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="16">05. Async DataLoader</text>
    <rect x="0" y="32" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="48">06. Autograd Engine</text>
    <rect x="0" y="64" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="80">07. AdamW &amp; Cosine</text>
    <rect x="0" y="96" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="112">08. Training Engine</text>
    <rect x="0" y="132" width="140" height="32" rx="4" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.2"/>
    <text x="10" y="152" fill="#D22341" font-weight="700">Milestone 2: MNIST</text>
  </g>

  <!-- Arrow 2->3 -->
  <path d="M390 230 H410" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

  <!-- Tier 3 -->
  <rect x="410" y="90" width="170" height="280" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.5" filter="url(#cardShadow)"/>
  <path d="M410 90 h170 v34 h-170 z" fill="#1B3A5F" />
  <text x="495" y="113" text-anchor="middle" fill="#FFFFFF" font-size="13" font-weight="700">Tier 3: Architectures</text>
  <g transform="translate(425, 140)" font-size="12" font-weight="500" fill="#1B3A5F">
    <rect x="0" y="0" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="16">09. Spatial Convolutions</text>
    <rect x="0" y="30" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="46">10. Byte-Pair Encoding</text>
    <rect x="0" y="60" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="76">11. Token Embeddings</text>
    <rect x="0" y="90" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="106">12. Attention (SDPA)</text>
    <rect x="0" y="120" width="140" height="24" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="10" y="136">13. GPT-2 Transformer</text>
    <rect x="0" y="150" width="140" height="24" rx="4" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.2"/>
    <text x="10" y="166" fill="#D22341" font-weight="700">Milestone 3: GPT</text>
  </g>

  <!-- Arrow 3->4 -->
  <path d="M580 230 H600" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

  <!-- Tier 4 -->
  <rect x="600" y="90" width="170" height="280" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.5" filter="url(#cardShadow)"/>
  <path d="M600 90 h170 v34 h-170 z" fill="#1B3A5F" />
  <text x="685" y="113" text-anchor="middle" fill="#FFFFFF" font-size="13" font-weight="700">Tier 4: Acceleration</text>
  <g transform="translate(615, 134)" font-size="11" font-weight="500" fill="#1B3A5F">
    <rect x="0" y="0" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="14">14. Roofline Profiler</text>
    <rect x="0" y="24" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="38">15. INT8 Quantization</text>
    <rect x="0" y="48" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="62">16. Structured Pruning</text>
    <rect x="0" y="72" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="86">17. Fused Kernels</text>
    <rect x="0" y="96" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="110">18. KV-Cache Engine</text>
    <rect x="0" y="120" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="134">19. MLPerf Benchmark</text>
    <rect x="0" y="144" width="140" height="20" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="8" y="158">20. 16x Capstone Stack</text>
    <rect x="0" y="170" width="140" height="26" rx="4" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.2"/>
    <text x="8" y="187" fill="#D22341" font-weight="700">21. Triton &amp; JIT</text>
  </g>
"""

# 01: Tensor & Strides
DIAGRAMS["01_tensor-diag-1.svg"] = """
  <!-- Outer Frame -->
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  
  <!-- Header -->
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">TENSOR ARCHITECTURE: N-D LOGICAL METADATA VS 1D CONTIGUOUS STORAGE</text>

  <!-- Logical View Card -->
  <rect x="45" y="85" width="320" height="250" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="65" y="115" fill="#1B3A5F" font-size="15" font-weight="700">1. Logical View (shape: [2, 3])</text>
  <text x="65" y="135" fill="#4A5568" font-size="12">Multi-dimensional array indexable as A[i, j]</text>

  <!-- 2x3 Matrix Grid -->
  <g transform="translate(85, 155)" font-size="14" font-weight="600" text-anchor="middle">
    <!-- Row 0 -->
    <rect x="0" y="0" width="70" height="45" fill="#FFFFFF" stroke="#FF8246" stroke-width="1.5" rx="4"/>
    <text x="35" y="25" fill="#1B3A5F">1.0</text>
    <text x="35" y="38" fill="#FF8246" font-size="9">(0,0)</text>

    <rect x="80" y="0" width="70" height="45" fill="#FFFFFF" stroke="#FF8246" stroke-width="1.5" rx="4"/>
    <text x="115" y="25" fill="#1B3A5F">2.0</text>
    <text x="115" y="38" fill="#FF8246" font-size="9">(0,1)</text>

    <rect x="160" y="0" width="70" height="45" fill="#FFFFFF" stroke="#FF8246" stroke-width="1.5" rx="4"/>
    <text x="195" y="25" fill="#1B3A5F">3.0</text>
    <text x="195" y="38" fill="#FF8246" font-size="9">(0,2)</text>

    <!-- Row 1 -->
    <rect x="0" y="55" width="70" height="45" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.5" rx="4"/>
    <text x="35" y="80" fill="#1B3A5F">4.0</text>
    <text x="35" y="93" fill="#4A5568" font-size="9">(1,0)</text>

    <rect x="80" y="55" width="70" height="45" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.5" rx="4"/>
    <text x="115" y="80" fill="#1B3A5F">5.0</text>
    <text x="115" y="93" fill="#4A5568" font-size="9">(1,1)</text>

    <rect x="160" y="55" width="70" height="45" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.5" rx="4"/>
    <text x="195" y="80" fill="#1B3A5F">6.0</text>
    <text x="195" y="93" fill="#4A5568" font-size="9">(1,2)</text>
  </g>

  <!-- Arrow between cards -->
  <path d="M375 210 H425" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
  <text x="400" y="200" fill="#FF8246" font-size="11" font-weight="700" text-anchor="middle">Strides: (3, 1)</text>

  <!-- Physical 1D Storage Card -->
  <rect x="435" y="85" width="320" height="250" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="455" y="115" fill="#1B3A5F" font-size="15" font-weight="700">2. Physical Storage (1D Flat Buffer)</text>
  <text x="455" y="135" fill="#4A5568" font-size="12">Contiguous memory: offset = i × 3 + j × 1</text>

  <!-- 1D Buffer representation -->
  <g transform="translate(450, 160)" font-size="12" font-weight="600" text-anchor="middle">
    <rect x="0" y="0" width="45" height="50" fill="#FFFFFF" stroke="#FF8246" stroke-width="1.5" rx="3"/>
    <text x="22" y="28" fill="#1B3A5F">1.0</text>
    <text x="22" y="44" fill="#999" font-size="9">idx 0</text>

    <rect x="48" y="0" width="45" height="50" fill="#FFFFFF" stroke="#FF8246" stroke-width="1.5" rx="3"/>
    <text x="70" y="28" fill="#1B3A5F">2.0</text>
    <text x="70" y="44" fill="#999" font-size="9">idx 1</text>

    <rect x="96" y="0" width="45" height="50" fill="#FFFFFF" stroke="#FF8246" stroke-width="1.5" rx="3"/>
    <text x="118" y="28" fill="#1B3A5F">3.0</text>
    <text x="118" y="44" fill="#999" font-size="9">idx 2</text>

    <rect x="144" y="0" width="45" height="50" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.5" rx="3"/>
    <text x="166" y="28" fill="#1B3A5F">4.0</text>
    <text x="166" y="44" fill="#999" font-size="9">idx 3</text>

    <rect x="192" y="0" width="45" height="50" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.5" rx="3"/>
    <text x="214" y="28" fill="#1B3A5F">5.0</text>
    <text x="214" y="44" fill="#999" font-size="9">idx 4</text>

    <rect x="240" y="0" width="45" height="50" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.5" rx="3"/>
    <text x="262" y="28" fill="#1B3A5F">6.0</text>
    <text x="262" y="44" fill="#999" font-size="9">idx 5</text>
  </g>

  <!-- Formula Callout -->
  <rect x="450" y="235" width="290" height="75" rx="6" fill="#1B3A5F" />
  <text x="465" y="260" fill="#FF8246" font-size="12" font-weight="700">⚡ O(1) Zero-Copy Slicing Invariant</text>
  <text x="465" y="280" fill="#FFFFFF" font-size="11">Offset = StorageOffset + Σ (Coord[k] × Stride[k])</text>
  <text x="465" y="298" fill="#A0AEC0" font-size="10">Views alter metadata; DRAM buffer stays untouched.</text>
"""

# 02: Activations & GELU
DIAGRAMS["02_activations-diag-1.svg"] = """
  <!-- Outer Frame -->
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">NON-LINEAR ACTIVATIONS: BREAKING THE LINEAR COLLAPSE WALL</text>

  <!-- Card 1: ReLU -->
  <rect x="40" y="85" width="160" height="245" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="120" y="115" text-anchor="middle" fill="#1B3A5F" font-size="14" font-weight="700">ReLU</text>
  <path d="M60 210 H120 L180 150" stroke="#FF8246" stroke-width="3" fill="none"/>
  <line x1="60" y1="210" x2="180" y2="210" stroke="#CBD5E0" stroke-width="1" stroke-dasharray="3,3"/>
  <line x1="120" y1="230" x2="120" y2="140" stroke="#CBD5E0" stroke-width="1" stroke-dasharray="3,3"/>
  <rect x="55" y="240" width="130" height="70" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
  <text x="120" y="260" text-anchor="middle" fill="#1B3A5F" font-size="11" font-weight="600">f(x) = max(0, x)</text>
  <text x="120" y="280" text-anchor="middle" fill="#4A5568" font-size="10">Fast, piecewise linear</text>
  <text x="120" y="295" text-anchor="middle" fill="#D22341" font-size="9">Dying ReLU risk</text>

  <!-- Card 2: Sigmoid -->
  <rect x="220" y="85" width="160" height="245" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="300" y="115" text-anchor="middle" fill="#1B3A5F" font-size="14" font-weight="700">Sigmoid</text>
  <path d="M240 220 Q280 220 300 180 T360 140" stroke="#2E6F7E" stroke-width="3" fill="none"/>
  <line x1="240" y1="180" x2="360" y2="180" stroke="#CBD5E0" stroke-width="1" stroke-dasharray="3,3"/>
  <rect x="235" y="240" width="130" height="70" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
  <text x="300" y="260" text-anchor="middle" fill="#1B3A5F" font-size="11" font-weight="600">σ(x) = 1 / (1 + e⁻ˣ)</text>
  <text x="300" y="280" text-anchor="middle" fill="#4A5568" font-size="10">Range: (0, 1)</text>
  <text x="300" y="295" text-anchor="middle" fill="#D22341" font-size="9">Vanishing gradients</text>

  <!-- Card 3: Tanh -->
  <rect x="400" y="85" width="160" height="245" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="480" y="115" text-anchor="middle" fill="#1B3A5F" font-size="14" font-weight="700">Tanh</text>
  <path d="M420 225 Q460 225 480 180 T540 135" stroke="#1B3A5F" stroke-width="3" fill="none"/>
  <line x1="420" y1="180" x2="540" y2="180" stroke="#CBD5E0" stroke-width="1" stroke-dasharray="3,3"/>
  <rect x="415" y="240" width="130" height="70" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
  <text x="480" y="260" text-anchor="middle" fill="#1B3A5F" font-size="11" font-weight="600">tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)</text>
  <text x="480" y="280" text-anchor="middle" fill="#4A5568" font-size="10">Zero-centered: (-1, 1)</text>
  <text x="480" y="295" text-anchor="middle" fill="#2E6F7E" font-size="9">Smooth saturation</text>

  <!-- Card 4: GELU (Modern Foundation) -->
  <rect x="580" y="85" width="180" height="245" rx="8" fill="#FFF8F0" stroke="#FF8246" stroke-width="2" filter="url(#cardShadow)"/>
  <text x="670" y="115" text-anchor="middle" fill="#D22341" font-size="14" font-weight="700">GELU (GPT-2 Standard)</text>
  <path d="M600 215 Q635 210 655 220 T740 145" stroke="#FF8246" stroke-width="3.5" fill="none"/>
  <line x1="600" y1="210" x2="740" y2="210" stroke="#CBD5E0" stroke-width="1" stroke-dasharray="3,3"/>
  <rect x="590" y="240" width="160" height="75" rx="4" fill="#1B3A5F"/>
  <text x="670" y="260" text-anchor="middle" fill="#FF8246" font-size="11" font-weight="700">x · Φ(x) Probabilistic Gate</text>
  <text x="670" y="278" text-anchor="middle" fill="#FFFFFF" font-size="10">0.5x(1 + tanh(√(2/π)(x+0.0447x³)))</text>
  <text x="670" y="295" text-anchor="middle" fill="#A0AEC0" font-size="9">Smooth gradient everywhere</text>
"""

# 06: Autograd Tape DAG
DIAGRAMS["06_autograd-diag-1.svg"] = """
  <!-- Outer Frame -->
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">DYNAMIC AUTOGRAD TAPE: REVERSE-MODE TOPOLOGICAL ACCUMULATION</text>

  <!-- Left: Forward Pass Recording -->
  <rect x="40" y="85" width="340" height="250" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="60" y="115" fill="#1B3A5F" font-size="14" font-weight="700">1. Forward Pass (Tape Construction)</text>
  <text x="60" y="135" fill="#4A5568" font-size="11">Operation nodes recorded dynamically in tape history</text>

  <!-- Forward Graph Nodes -->
  <g transform="translate(60, 160)" font-size="11" font-weight="600" text-anchor="middle">
    <!-- Node A & B -->
    <circle cx="30" cy="20" r="18" fill="#1B3A5F" />
    <text x="30" y="24" fill="#FFFFFF">x</text>

    <circle cx="30" cy="80" r="18" fill="#1B3A5F" />
    <text x="30" y="84" fill="#FFFFFF">W</text>

    <!-- MatMul Op -->
    <rect x="100" y="35" width="50" height="30" rx="4" fill="#FF8246" />
    <text x="125" y="54" fill="#FFFFFF" font-weight="700">MatMul</text>

    <!-- ReLU Op -->
    <rect x="190" y="35" width="45" height="30" rx="4" fill="#FF8246" />
    <text x="212" y="54" fill="#FFFFFF" font-weight="700">ReLU</text>

    <!-- Loss -->
    <circle cx="280" cy="50" r="18" fill="#D22341" />
    <text x="280" y="54" fill="#FFFFFF">Loss</text>

    <!-- Forward Arrows -->
    <path d="M48 25 L100 45" stroke="#1B3A5F" stroke-width="1.8" marker-end="url(#arrowNavy)"/>
    <path d="M48 75 L100 55" stroke="#1B3A5F" stroke-width="1.8" marker-end="url(#arrowNavy)"/>
    <path d="M150 50 H190" stroke="#FF8246" stroke-width="1.8" marker-end="url(#arrowOrange)"/>
    <path d="M235 50 H262" stroke="#FF8246" stroke-width="1.8" marker-end="url(#arrowOrange)"/>
  </g>

  <!-- Right: Backward Pass Execution -->
  <rect x="400" y="85" width="360" height="250" rx="8" fill="#FFF8F0" stroke="#FF8246" stroke-width="1.5" filter="url(#cardShadow)"/>
  <text x="420" y="115" fill="#D22341" font-size="14" font-weight="700">2. Backward Pass (Topological Chain Rule)</text>
  <text x="420" y="135" fill="#4A5568" font-size="11">Gradients propagate in reverse: dL/dx = (dL/dy) · (dy/dx)</text>

  <!-- Backward Graph Nodes -->
  <g transform="translate(420, 160)" font-size="11" font-weight="600" text-anchor="middle">
    <!-- Seed -->
    <circle cx="300" cy="50" r="18" fill="#D22341" />
    <text x="300" y="54" fill="#FFFFFF">dL/dL=1</text>

    <!-- ReLU Backward -->
    <rect x="195" y="35" width="65" height="30" rx="4" fill="#1B3A5F" />
    <text x="227" y="54" fill="#FFFFFF">dReLU/dy</text>

    <!-- MatMul Backward -->
    <rect x="95" y="35" width="65" height="30" rx="4" fill="#1B3A5F" />
    <text x="127" y="54" fill="#FFFFFF">dMM/dx,dW</text>

    <!-- Target Gradients -->
    <circle cx="20" cy="20" r="18" fill="#2E6F7E" />
    <text x="20" y="24" fill="#FFFFFF">grad_x</text>

    <circle cx="20" cy="80" r="18" fill="#2E6F7E" />
    <text x="20" y="84" fill="#FFFFFF">grad_W</text>

    <!-- Backward Arrows -->
    <path d="M282 50 H260" stroke="#D22341" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
    <path d="M195 50 H160" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
    <path d="M95 45 L38 25" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
    <path d="M95 55 L38 75" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>
  </g>

  <!-- Key Rule Invariant -->
  <rect x="420" y="265" width="320" height="55" rx="5" fill="#1B3A5F"/>
  <text x="435" y="288" fill="#FF8246" font-size="11" font-weight="700">⚡ Invariant: In-place Gradient Accumulation</text>
  <text x="435" y="306" fill="#FFFFFF" font-size="10">param.grad += grad (Accumulates branches; zero_grad() clears)</text>
"""

# 12: Attention & SDPA
DIAGRAMS["12_attention-diag-1.svg"] = """
  <!-- Outer Frame -->
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">SCALED DOT-PRODUCT ATTENTION: O(N²) COMPUTE &amp; CAUSAL MASKING</text>

  <!-- Q, K, V Inputs -->
  <g transform="translate(45, 95)" font-size="13" font-weight="700" text-anchor="middle">
    <rect x="0" y="0" width="65" height="40" rx="6" fill="#1B3A5F"/>
    <text x="32" y="25" fill="#FFFFFF">Q (Query)</text>

    <rect x="85" y="0" width="65" height="40" rx="6" fill="#1B3A5F"/>
    <text x="117" y="25" fill="#FFFFFF">K (Key)</text>

    <rect x="200" y="0" width="65" height="40" rx="6" fill="#1B3A5F"/>
    <text x="232" y="25" fill="#FFFFFF">V (Value)</text>
  </g>

  <!-- MatMul Q·K^T -->
  <rect x="80" y="160" width="130" height="45" rx="6" fill="#FFF8F0" stroke="#FF8246" stroke-width="1.5" filter="url(#cardShadow)"/>
  <text x="145" y="188" text-anchor="middle" fill="#1B3A5F" font-size="13" font-weight="700">MatMul (Q · Kᵀ)</text>
  <path d="M77 135 L120 160" stroke="#1B3A5F" stroke-width="2" marker-end="url(#arrowNavy)"/>
  <path d="M152 135 L160 160" stroke="#1B3A5F" stroke-width="2" marker-end="url(#arrowNavy)"/>

  <!-- Scale Factor -->
  <rect x="80" y="230" width="130" height="40" rx="6" fill="#FFFFFF" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="145" y="255" text-anchor="middle" fill="#1B3A5F" font-size="12" font-weight="600">Scale: 1 / √dₖ</text>
  <path d="M145 205 V230" stroke="#FF8246" stroke-width="2" marker-end="url(#arrowOrange)"/>

  <!-- Mask Box -->
  <rect x="250" y="230" width="130" height="40" rx="6" fill="#FFF0E6" stroke="#D22341" stroke-width="1.2"/>
  <text x="315" y="255" text-anchor="middle" fill="#D22341" font-size="12" font-weight="700">Causal Mask (-inf)</text>
  <path d="M210 250 H250" stroke="#D22341" stroke-width="2" marker-end="url(#arrowOrange)"/>

  <!-- Softmax -->
  <rect x="420" y="230" width="110" height="40" rx="6" fill="url(#torchGrad)"/>
  <text x="475" y="255" text-anchor="middle" fill="#FFFFFF" font-size="13" font-weight="700">Softmax</text>
  <path d="M380 250 H420" stroke="#FF8246" stroke-width="2" marker-end="url(#arrowOrange)"/>

  <!-- Final MatMul with V -->
  <rect x="570" y="225" width="160" height="50" rx="8" fill="#FFF8F0" stroke="#FF8246" stroke-width="2" filter="url(#cardShadow)"/>
  <text x="650" y="255" text-anchor="middle" fill="#D22341" font-size="14" font-weight="700">Output = Attention · V</text>
  <path d="M530 250 H570" stroke="#FF8246" stroke-width="2" marker-end="url(#arrowOrange)"/>
  <path d="M280 135 C350 160 500 160 620 225" stroke="#1B3A5F" stroke-width="2" fill="none" marker-end="url(#arrowNavy)"/>

  <!-- Invariant Footer -->
  <rect x="45" y="295" width="690" height="45" rx="6" fill="#1B3A5F"/>
  <text x="60" y="322" fill="#FF8246" font-size="12" font-weight="700">⚡ Attention Scaling Law: Var(q · k) = dₖ</text>
  <text x="330" y="322" fill="#FFFFFF" font-size="11">Dividing by √dₖ preserves variance = 1, preventing softmax saturation &amp; dead gradients.</text>
"""

# 13: Transformers Architecture
DIAGRAMS["13_transformers-diag-1.svg"] = """
  <!-- Outer Frame -->
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">GPT-2 TRANSFORMER BLOCK: PRE-LN RESIDUAL HIGHWAYS</text>

  <!-- Input Tokens -->
  <g transform="translate(45, 180)" font-size="12" font-weight="700">
    <rect x="0" y="0" width="120" height="50" rx="6" fill="#1B3A5F"/>
    <text x="60" y="28" text-anchor="middle" fill="#FFFFFF">Tokens + Pos</text>
    <text x="60" y="42" text-anchor="middle" fill="#FF8246" font-size="10">[B, T, D]</text>
  </g>

  <!-- Arrow to Block -->
  <path d="M165 205 H200" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

  <!-- Transformer Block Container -->
  <rect x="200" y="80" width="410" height="250" rx="10" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.8" filter="url(#cardShadow)"/>
  <text x="220" y="105" fill="#1B3A5F" font-size="13" font-weight="700">TransformerBlock (Repeated L times)</text>

  <!-- Pre-LN Sub-layer 1 (Attention) -->
  <g transform="translate(220, 120)">
    <!-- Residual Highway 1 -->
    <path d="M0 85 C20 20 180 20 200 85" stroke="#FF8246" stroke-width="2.5" fill="none" stroke-dasharray="4,4"/>
    <text x="100" y="35" text-anchor="middle" fill="#FF8246" font-size="10" font-weight="700">Residual Highway (+)</text>

    <!-- LayerNorm 1 -->
    <rect x="15" y="70" width="65" height="30" rx="4" fill="#FFFFFF" stroke="#1B3A5F"/>
    <text x="47" y="89" text-anchor="middle" fill="#1B3A5F" font-size="10" font-weight="600">LayerNorm</text>

    <!-- Multi-Head Attention -->
    <rect x="95" y="65" width="85" height="40" rx="6" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.5"/>
    <text x="137" y="89" text-anchor="middle" fill="#D22341" font-size="11" font-weight="700">Causal MHA</text>

    <!-- Add 1 -->
    <circle cx="200" cy="85" r="12" fill="#1B3A5F"/>
    <text x="200" y="89" text-anchor="middle" fill="#FFFFFF" font-weight="700">+</text>

    <path d="M80 85 H95" stroke="#1B3A5F" stroke-width="1.5" marker-end="url(#arrowNavy)"/>
    <path d="M180 85 H188" stroke="#FF8246" stroke-width="1.5" marker-end="url(#arrowOrange)"/>
  </g>

  <!-- Pre-LN Sub-layer 2 (MLP) -->
  <g transform="translate(435, 120)">
    <!-- Residual Highway 2 -->
    <path d="M0 85 C20 20 150 20 160 85" stroke="#FF8246" stroke-width="2.5" fill="none" stroke-dasharray="4,4"/>
    <text x="80" y="35" text-anchor="middle" fill="#FF8246" font-size="10" font-weight="700">Residual Highway (+)</text>

    <!-- LayerNorm 2 -->
    <rect x="15" y="70" width="55" height="30" rx="4" fill="#FFFFFF" stroke="#1B3A5F"/>
    <text x="42" y="89" text-anchor="middle" fill="#1B3A5F" font-size="10" font-weight="600">LayerNorm</text>

    <!-- MLP FFN -->
    <rect x="80" y="65" width="65" height="40" rx="6" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.5"/>
    <text x="112" y="89" text-anchor="middle" fill="#D22341" font-size="11" font-weight="700">MLP (4d)</text>

    <!-- Add 2 -->
    <circle cx="160" cy="85" r="12" fill="#1B3A5F"/>
    <text x="160" y="89" text-anchor="middle" fill="#FFFFFF" font-weight="700">+</text>

    <path d="M70 85 H80" stroke="#1B3A5F" stroke-width="1.5" marker-end="url(#arrowNavy)"/>
    <path d="M145 85 H148" stroke="#FF8246" stroke-width="1.5" marker-end="url(#arrowOrange)"/>
  </g>

  <!-- Arrow to Output -->
  <path d="M610 205 H640" stroke="#FF8246" stroke-width="2.5" marker-end="url(#arrowOrange)"/>

  <!-- Output Logits -->
  <g transform="translate(640, 180)" font-size="12" font-weight="700">
    <rect x="0" y="0" width="115" height="50" rx="6" fill="url(#torchGrad)"/>
    <text x="57" y="28" text-anchor="middle" fill="#FFFFFF">LM Head</text>
    <text x="57" y="42" text-anchor="middle" fill="#FF8246" font-size="10">[B, T, Vocab]</text>
  </g>
"""

# 18: KV Cache
DIAGRAMS["18_memoization-diag-1.svg"] = """
  <!-- Outer Frame -->
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">KV CACHE MEMOIZATION: TRANSFORMING O(N) DECODE BANDWIDTH INTO O(1)</text>

  <!-- Without KV Cache (O(N) redundant compute) -->
  <rect x="40" y="85" width="335" height="245" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="60" y="115" fill="#1B3A5F" font-size="14" font-weight="700">1. Without Cache: Full Recompute</text>
  <text x="60" y="135" fill="#D22341" font-size="11">Every new token re-computes Keys &amp; Values for all past tokens</text>

  <!-- Step N vs Step N+1 recompute box -->
  <g transform="translate(55, 150)" font-size="11">
    <rect x="0" y="0" width="300" height="35" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="15" y="22" fill="#4A5568">Token 1: [Compute K₁, V₁]</text>

    <rect x="0" y="45" width="300" height="35" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/>
    <text x="15" y="67" fill="#4A5568">Token 2: [Recompute K₁, V₁] + [Compute K₂, V₂]</text>

    <rect x="0" y="90" width="300" height="35" rx="4" fill="#FFF0E6" stroke="#D22341" stroke-width="1.2"/>
    <text x="15" y="112" fill="#D22341" font-weight="600">Token N: Recomputes all N-1 keys/values! ❌</text>
  </g>
  <text x="60" y="305" fill="#D22341" font-size="11" font-weight="700">Total Compute per generation: O(N²) FLOPs</text>

  <!-- With KV Cache (O(1) incremental) -->
  <rect x="405" y="85" width="355" height="245" rx="8" fill="#FFF8F0" stroke="#FF8246" stroke-width="1.8" filter="url(#cardShadow)"/>
  <text x="425" y="115" fill="#D22341" font-size="14" font-weight="700">2. With TinyTorch KVCache (O(1))</text>
  <text x="425" y="135" fill="#2E6F7E" font-size="11">Append only the newest token's (K_new, V_new) to static buffer</text>

  <!-- Dynamic buffer box -->
  <g transform="translate(425, 150)" font-size="11">
    <!-- Past tokens in DRAM Cache -->
    <rect x="0" y="0" width="200" height="75" rx="6" fill="#1B3A5F"/>
    <text x="15" y="25" fill="#FFFFFF" font-weight="700">Cached Past Keys &amp; Values</text>
    <text x="15" y="45" fill="#FF8246">K_past: [B, H, T_past, D_h]</text>
    <text x="15" y="62" fill="#FF8246">V_past: [B, H, T_past, D_h]</text>

    <!-- New single token appended -->
    <rect x="210" y="0" width="105" height="75" rx="6" fill="#FFF0E6" stroke="#FF8246" stroke-width="1.5"/>
    <text x="218" y="25" fill="#D22341" font-weight="700">+ New Token</text>
    <text x="218" y="45" fill="#1B3A5F">K_new [B,H,1,D]</text>
    <text x="218" y="62" fill="#1B3A5F">V_new [B,H,1,D]</text>
  </g>

  <rect x="425" y="240" width="315" height="70" rx="6" fill="#1B3A5F"/>
  <text x="440" y="265" fill="#FF8246" font-size="12" font-weight="700">⚡ 4.2x Throughput Speedup</text>
  <text x="440" y="285" fill="#FFFFFF" font-size="11">Replaces full sequence compute with single token matrix-vector</text>
  <text x="440" y="300" fill="#A0AEC0" font-size="10">KVCache.update(k, v) amortizes memory roundtrips.</text>
"""

# Let's generate remaining diagrams programmatically with unified design patterns!
def generate_generic_diagram(name, title, subtitle, box1_title, box1_items, box2_title, box2_items, callout_title, callout_desc):
    return f"""
  <rect x="20" y="20" width="760" height="340" rx="10" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.5" filter="url(#cardShadow)"/>
  <rect x="20" y="20" width="760" height="45" rx="10" fill="url(#torchGrad)"/>
  <text x="40" y="49" fill="#FFFFFF" font-size="16" font-weight="700">{title}</text>

  <!-- Left Box -->
  <rect x="45" y="85" width="330" height="245" rx="8" fill="url(#warmBgGrad)" stroke="#1B3A5F" stroke-width="1.2"/>
  <text x="65" y="115" fill="#1B3A5F" font-size="14" font-weight="700">{box1_title}</text>
  <g transform="translate(65, 135)" font-size="12" fill="#4A5568">
    {''.join([f'<rect x="0" y="{i*38}" width="290" height="30" rx="4" fill="#FFFFFF" stroke="#E2E8F0"/><text x="12" y="{i*38+20}">{item}</text>' for i, item in enumerate(box1_items)])}
  </g>

  <!-- Right Box -->
  <rect x="405" y="85" width="350" height="245" rx="8" fill="#FFF8F0" stroke="#FF8246" stroke-width="1.8" filter="url(#cardShadow)"/>
  <text x="425" y="115" fill="#D22341" font-size="14" font-weight="700">{box2_title}</text>
  <g transform="translate(425, 135)" font-size="12" fill="#1B3A5F">
    {''.join([f'<rect x="0" y="{i*38}" width="310" height="30" rx="4" fill="#FFFFFF" stroke="#FF8246" stroke-width="1"/><text x="12" y="{i*38+20}" font-weight="600">{item}</text>' for i, item in enumerate(box2_items)])}
  </g>

  <rect x="425" y="240" width="310" height="70" rx="6" fill="#1B3A5F"/>
  <text x="440" y="265" fill="#FF8246" font-size="12" font-weight="700">{callout_title}</text>
  <text x="440" y="285" fill="#FFFFFF" font-size="11">{callout_desc}</text>
"""

# Fill remaining diagrams
DIAGRAMS["01_tensor-diag-2.svg"] = generate_generic_diagram(
    "01_tensor-diag-2.svg",
    "TENSOR STRIDES &amp; ZERO-COPY TRANSPOSE",
    "Memory views versus physical allocations",
    "Row-Major Standard Layout",
    ["shape: (2, 3) | strides: (3, 1)", "DRAM: [1, 2, 3, 4, 5, 6]", "Continuous address stepping"],
    "Transposed View (Zero-Copy)",
    ["shape: (3, 2) | strides: (1, 3)", "DRAM: [1, 2, 3, 4, 5, 6] (Same!)", "Swapping stride indices creates transpose in O(1)"],
    "⚡ Zero Allocation Transpose Invariant",
    "Tensor.transpose() manipulates metadata strides; 0 bytes allocated."
)

DIAGRAMS["01_tensor-diag-3.svg"] = generate_generic_diagram(
    "01_tensor-diag-3.svg",
    "BROADCASTING RULES &amp; STRIDE EXPANSION",
    "Mathematical expansion without memory duplication",
    "Dimension Alignment",
    ["Tensor A: (3, 1) | Strides: (1, 1)", "Tensor B: (1, 4) | Strides: (4, 1)", "Prepend 1s to shorter shape"],
    "Broadcasted Output View",
    ["Output Shape: (3, 4)", "Stride set to 0 for repeated dimensions", "Memory read repeats element with zero byte copy"],
    "⚡ Stride-0 Trick Invariant",
    "Stride = 0 repeats data along an axis without allocating duplicate DRAM."
)

DIAGRAMS["03_layers-diag-1.svg"] = generate_generic_diagram(
    "03_layers-diag-1.svg",
    "LINEAR LAYER FORWARD PASS &amp; WEIGHT INITIALIZATION",
    "Affine transformations in deep architectures",
    "Forward Pass: Y = X · W^T + b",
    ["Input X: [Batch, In_Features]", "Weight W: [Out_Features, In_Features]", "Bias b: [Out_Features] broadcasted"],
    "Kaiming / He Initialization",
    ["W ~ Uniform(-sqrt(k), sqrt(k))", "k = 1 / in_features", "Preserves activation variance across deep stacks"],
    "⚡ Variance Preservation Invariant",
    "Proper scaling prevents exploding or vanishing activations in deep networks."
)

DIAGRAMS["04_losses-diag-1.svg"] = generate_generic_diagram(
    "04_losses-diag-1.svg",
    "CROSS-ENTROPY LOSS &amp; LOG-SUM-EXP STABILIZATION",
    "Probabilistic loss and gradient stability",
    "Softmax Probability",
    ["P_i = exp(z_i) / Σ exp(z_j)", "Raw exponentials risk IEEE 754 float overflow", "Subtract max(z) before exponentiation"],
    "Log-Sum-Exp Stabilization",
    ["LSE(z) = c + log(Σ exp(z_i - c))", "c = max(z)", "Stable loss: L = -z_target + LSE(z)"],
    "⚡ Numerical Stability Invariant",
    "Log-Sum-Exp prevents NaN/Inf during backprop on float32/float16."
)

DIAGRAMS["05_dataloader-diag-1.svg"] = generate_generic_diagram(
    "05_dataloader-diag-1.svg",
    "ASYNCHRONOUS DATALOADER &amp; BATCH PIPELINE",
    "Preventing GPU starvation with prefetch queues",
    "Synchronous Bottleneck",
    ["1. CPU loads batch from disk", "2. GPU waits idle (0% utilization)", "3. GPU computes forward/backward", "4. CPU waits idle"],
    "Asynchronous Prefetch Engine",
    ["Background worker threads prefetch Batch N+1", "Thread-safe queue feeds tensors to math core", "Zero compute stalls: GPU stays 100% saturated"],
    "⚡ Overlapping I/O with Compute",
    "Async batch preparation turns disk latency into zero amortized delay."
)

DIAGRAMS["07_optimizers-diag-1.svg"] = generate_generic_diagram(
    "07_optimizers-diag-1.svg",
    "OPTIMIZERS: SGD MOMENTUM VS ADAMW DECOUPLED WEIGHT DECAY",
    "Gradient descent optimization trajectories",
    "SGD with Momentum",
    ["v = β · v + (1 - β) · grad", "w = w - lr · v", "Momentum rolls past flat saddle points"],
    "AdamW (Decoupled Weight Decay)",
    ["m = β₁m + (1-β₁)g | v = β₂v + (1-β₂)g²", "w = w - lr · (m̂ / (√v̂ + ε) + λ · w)", "Decoupling prevents L2 variance distortion"],
    "⚡ Loshchilov &amp; Hutter Invariant",
    "Decoupling weight decay from adaptive gradients stabilizes Adam training."
)

DIAGRAMS["08_training-diag-1.svg"] = generate_generic_diagram(
    "08_training-diag-1.svg",
    "THE 5-STEP TRAINING LOOP ENGINE CONTRACT",
    "Deterministic state transitions in framework loops",
    "The Rigid 5-Step Order",
    ["1. optimizer.zero_grad()", "2. logits = model(inputs)", "3. loss = criterion(logits, targets)", "4. loss.backward()", "5. optimizer.step()"],
    "Validation &amp; Checkpointing",
    ["model.eval() disables dropout/batchnorm", "torch.no_grad() skips autograd graph memory", "Atomic serialization prevents corrupted weights"],
    "⚡ Loop Invariant Closure",
    "Step order is mathematically rigid; zero_grad must precede backward."
)

DIAGRAMS["09_convolutions-diag-1.svg"] = generate_generic_diagram(
    "09_convolutions-diag-1.svg",
    "SPATIAL CONVOLUTIONS &amp; IM2COL GEMM EXPANSION",
    "Converting sliding 2D windows into BLAS GEMMs",
    "Sliding Spatial Window",
    ["Input: [C, H, W] | Kernel: [K, K]", "Sliding across height and width", "Direct loops cause non-contiguous DRAM hops"],
    "im2col Systolic GEMM",
    ["Unroll each receptive field into matrix columns", "GEMM: Output = KernelMatrix · ColMatrix", "Executes at 95%+ peak FLOP/s on hardware"],
    "⚡ Chellapilla GEMM Invariant",
    "im2col trades temporary buffer memory for dense matrix core throughput."
)

DIAGRAMS["10_tokenization-diag-1.svg"] = generate_generic_diagram(
    "10_tokenization-diag-1.svg",
    "BYTE-PAIR ENCODING (BPE) TOKENIZATION",
    "Subword vocabulary induction from byte frequencies",
    "Base Byte Vocabulary",
    ["256 UTF-8 byte tokens", "Zero out-of-vocabulary (OOV) tokens", "Character frequency counting on corpus"],
    "Iterative Pair Merges",
    ["Find most frequent pair: ('t', 'h') → 'th'", "Add merge rule to rank table", "Compresses text into informative tokens"],
    "⚡ Information Entropy Invariant",
    "BPE maximizes token information density while eliminating OOV errors."
)

DIAGRAMS["11_embeddings-diag-1.svg"] = generate_generic_diagram(
    "11_embeddings-diag-1.svg",
    "TOKEN EMBEDDINGS &amp; SINUSOIDAL POSITIONAL ENCODINGS",
    "Projecting discrete IDs into continuous semantic space",
    "Token Embedding Table",
    ["Matrix W_embed: [VocabSize, HiddenDim]", "Forward pass: zero-compute pointer indexing", "Gradients update only accessed rows"],
    "Sinusoidal Positional Encoding",
    ["PE(pos, 2i) = sin(pos / 10000^(2i/d))", "PE(pos, 2i+1) = cos(pos / 10000^(2i/d))", "Injects relative order into permutation-invariant attention"],
    "⚡ Vaswani Relative Shift Invariant",
    "Linear transformation can project PE(pos+k) from PE(pos) via rotation."
)

DIAGRAMS["14_profiling-diag-1.svg"] = generate_generic_diagram(
    "14_profiling-diag-1.svg",
    "ROOFLINE PROFILING &amp; ARITHMETIC INTENSITY",
    "Identifying compute-bound versus memory-bound bottlenecks",
    "Arithmetic Intensity",
    ["I = FLOPs / Memory_Bytes_Transferred", "Low intensity: DRAM bandwidth bottleneck", "High intensity: Tensor Core ALUs saturated"],
    "Roofline Execution Zones",
    ["Attainable Performance = min(Peak_FLOPs, I × Bandwidth)", "Operators below ridge point need kernel fusion", "Operators above ridge point saturate ALUs"],
    "⚡ Williams Roofline Invariant",
    "Optimizing math on memory-bound kernels yields 0x speedup; reduce DRAM traffic."
)

DIAGRAMS["15_quantization-diag-1.svg"] = generate_generic_diagram(
    "15_quantization-diag-1.svg",
    "INT8 QUANTIZATION &amp; SYMMETRIC AFFINE MAPPING",
    "4x memory footprint reduction and INT8 tensor core acceleration",
    "Quantization Mapping",
    ["Scale: S = max(|X|) / 127", "q = clamp(round(X / S), -128, 127)", "32-bit floats compressed into 8-bit integers"],
    "Dequantization Recovery",
    ["X_approx = q × S", "Maintains accuracy within 0.1% perplexity", "INT8 DP4A tensor instructions run 4x faster"],
    "⚡ Dynamic Range Invariant",
    "Symmetric scaling eliminates zero-point arithmetic overhead in GEMM."
)

DIAGRAMS["16_compression-diag-1.svg"] = generate_generic_diagram(
    "16_compression-diag-1.svg",
    "MODEL COMPRESSION: PRUNING, SVD &amp; DISTILLATION",
    "Squeezing deep networks into constrained hardware",
    "Structured vs Unstructured Pruning",
    ["Unstructured: Zero out individual small weights (sparse)", "Structured: Remove entire channels/heads (dense GEMM speed)"],
    "Low-Rank SVD &amp; Distillation",
    ["Factor W [M, N] into U [M, r] · V [r, N] (r &lt;&lt; min(M, N))", "Distillation: Soft student loss against teacher logits"],
    "⚡ Compression Trade-off Invariant",
    "Structured pruning preserves dense hardware matrix core throughput."
)

DIAGRAMS["17_acceleration-diag-1.svg"] = generate_generic_diagram(
    "17_acceleration-diag-1.svg",
    "HARDWARE ACCELERATION &amp; KERNEL FUSION",
    "Eliminating DRAM roundtrips via SRAM register residency",
    "Unfused Operations",
    ["Op 1: Load X → Add → Store Temp in DRAM", "Op 2: Load Temp → GELU → Store Result in DRAM", "DRAM memory bandwidth saturated by temporary writes"],
    "Fused Triton / CUDA Kernel",
    ["Single kernel launch executes Add + GELU in SRAM", "Temporary values kept in fast register memory", "Memory roundtrips cut from 3 to 1 (3x speedup)"],
    "⚡ SRAM Fusion Invariant",
    "Fusing elementwise operations eliminates memory bandwidth stalls."
)

DIAGRAMS["19_benchmarking-diag-1.svg"] = generate_generic_diagram(
    "19_benchmarking-diag-1.svg",
    "BENCHMARKING ENGINE &amp; LATENCY DISTRIBUTIONS",
    "Statistically rigorous performance measurement",
    "Measurement Rigor",
    ["Warmup passes discard cold JIT &amp; cache misses", "Synchronize GPU device before reading timers", "Measure 100+ iterations for stability"],
    "Latency Distribution (Tail Metrics)",
    ["P50 (Median): Typical user experience", "P95 / P99: Tail latency SLA violations", "Variance &amp; Standard deviation confidence bounds"],
    "⚡ Benchmarking Invariant",
    "Without GPU synchronization and warmup, benchmark numbers are meaningless."
)

DIAGRAMS["20_capstone-diag-1.svg"] = generate_generic_diagram(
    "20_capstone-diag-1.svg",
    "16X CAPSTONE PERFORMANCE STACK: CUMULATIVE ACCELERATION",
    "Multiplying optimization gains across the deep learning stack",
    "Individual Optimizations",
    ["1. INT8 Quantization: 2.0x (DRAM memory size cut)", "2. Operator Fusion: 1.7x (SRAM register reuse)", "3. KV Cache Memoization: 4.2x (O(1) decode)"],
    "Cumulative Multiplicative Speedup",
    ["S_total = S_quant × S_fusion × S_kvcache", "S_total = 2.0 × 1.7 × 4.2 ≈ 14.3x to 16.0x Speedup", "Fully verified in TinyTorch Capstone benchmark"],
    "⚡ Amdahl's Law Multiplier Invariant",
    "Addressing bottlenecks at every layer delivers cumulative multiplicative speedups."
)

# Output generation
for fname, body in DIAGRAMS.items():
    svg_content = SVG_HEAD.format(width=800, height=380) + body + "\n</svg>\n"
    
    # Write SVG to quarto and book
    p_quarto = DIAGRAM_DIR_QUARTO / fname
    p_book = DIAGRAM_DIR_BOOK / fname
    
    with open(p_quarto, "w", encoding="utf-8") as f:
        f.write(svg_content)
    with open(p_book, "w", encoding="utf-8") as f:
        f.write(svg_content)
        
    # Generate Vector PDF
    pdf_name = fname.replace(".svg", ".pdf")
    pdf_quarto = DIAGRAM_DIR_QUARTO / pdf_name
    pdf_book = DIAGRAM_DIR_BOOK / pdf_name
    
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "pdf", "-o", str(pdf_quarto), str(p_quarto)], check=True)
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "pdf", "-o", str(pdf_book), str(p_book)], check=True)

    # Generate High-DPI PNG
    png_name = fname.replace(".svg", ".png")
    png_quarto = DIAGRAM_DIR_QUARTO / png_name
    png_book = DIAGRAM_DIR_BOOK / png_name
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "png", "-d", "150", "-p", "150", "-o", str(png_quarto), str(p_quarto)], check=True)
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-f", "png", "-d", "150", "-p", "150", "-o", str(png_book), str(p_book)], check=True)

    print(f"✓ Generated SVG, PDF, and PNG for {fname}")

print(f"\n🎉 Successfully created all {len(DIAGRAMS)} publication-grade vector diagrams!")
