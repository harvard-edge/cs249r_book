# Physical AI SVG Figure Design & Color Standards

Every diagram in the Physical AI Systems project is an engineering blueprint. Figures must communicate architectural privilege, physical consequences, and multi-rate cadences with absolute clarity, crisp vector rendering, and publication-grade aesthetics.

All figures in this project are authored directly as **pure SVG (`.svg`) vector graphics** (no TikZ/LaTeX compilation dependencies).

---

## 1. The Complementary Systems Color Palette

Rather than relying solely on a two-tone crimson and blue motif, our diagrams use a **rich, harmonious, and complementary color system**. The palette uses **Harvard Crimson** and **Deep Slate Navy** as anchor points, enriched with a vibrant set of complementary hues (Warm Amber, Emerald/Teal, Cobalt, Amethyst, and Coral) to distinguish architectural organs, timing cadences, and authority levels.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE PHYSICAL AI FIGURE COLOR PALETTE                               │
├───────────────────┬──────────────┬──────────────┬──────────────┬─────────────────────────────────┤
│ Functional Role   │ Accent (Hex) │ Surface Tint │ Stroke Color │ Architectural Meaning           │
├───────────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 1. Cognitive /    │ Amber / Gold │ #FFFBEB      │ #D97706      │ System 2 Intent (0.5–2 Hz VLM), │
│    Proposal       │ #D97706      │ (Light Gold) │ (Dark Gold)  │ reasoning, expiring leases,     │
│                   │              │              │              │ untrusted candidate proposals   │
├───────────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 2. Trajectory /   │ Cobalt Blue  │ #EFF6FF      │ #2563EB      │ System 1.5 Planning (20–50 Hz), │
│    Planning       │ #2563EB      │ (Light Sky)  │ (Vibrant)    │ action chunks, temporal belief, │
│                   │              │              │              │ SE(3) frame trees, JEPAs        │
├───────────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 3. Deterministic /│ Emerald Teal │ #ECFDF5      │ #059669      │ System 1 Reflex (1000 Hz MCU),  │
│    Safety Reflex  │ #0D9488      │ (Pale Sage)  │ (Forest)     │ Control Barrier Functions (CBF),│
│                   │              │              │              │ PWM gate outputs, verified safe │
├───────────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 4. Physical /     │ Deep Crimson │ #FEF2F2      │ #A51C30      │ Physical reality (W_t -> W_t+1),│
│    Consequence    │ #A51C30      │ (Pale Rose)  │ (Crimson)    │ kinetic momentum, Joule heat,   │
│                   │              │              │              │ hardware interlocks, e-stops    │
├───────────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 5. Governance /   │ Amethyst     │ #F5F3FF      │ #7C3AED      │ Human supervisor, arbitration,  │
│    Arbitration    │ #7C3AED      │ (Pale Violet)│ (Violet)     │ bumpless transfer, provenance,  │
│                   │              │              │              │ cryptographic logs, ODD release │
├───────────────────┼──────────────┼──────────────┼──────────────┼─────────────────────────────────┤
│ 6. Alert /        │ Coral Red    │ #FFF1F2      │ #E11D48      │ Latency spikes (P99), bus stalls│
│    Fault Stress   │ #E11D48      │ (Soft Pink)  │ (Coral)      │ watchdog faults, UMA contention │
└───────────────────┴──────────────┴──────────────┴──────────────┴─────────────────────────────────┘
```

### Neutral Ink & Canvas Hierarchy

| Token | HEX Code | Usage |
| :--- | :--- | :--- |
| **Primary Ink (`ink`)** | `#0F172A` / `#1E293B` | High-contrast node titles, key equations, and primary labels |
| **Secondary Ink (`muted`)** | `#475569` / `#64748B` | Subtitles, descriptions, frequency tags, and bus specifications |
| **Hairline Border (`border`)**| `#CBD5E1` / `#E2E8F0` | Structural card borders (`stroke-width="1.2"`), axes, dividers |
| **Default Card Fill (`card`)**| `#F8FAFC` / `#FFFFFF` | Neutral card container background |
| **Chassis / Frame (`frame`)** | `#1F407A` (fill: `#F0F4FA`) | Top title banner, overall system enclosure frame |

---

## 2. SVG Authoring Rules & Technical Standards

### A. Pure Vector XML Structure
* Author diagrams directly in standard SVG format (`.svg`).
* Always include `xmlns="http://www.w3.org/2000/svg"`, `viewBox="0 0 W H"`, `width="100%"`, and `height="auto"`.
* Do NOT include raster images (`<image>`) inside SVG diagrams; all geometry must be clean vector paths, rects, circles, and text.

### B. Typography & Fonts in SVG
* **Main Font Stack:** Use a clean, universally rendered sans-serif stack:
  ```css
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  ```
* **Code / Peripheral Stack:**
  ```css
  font-family: "SF Mono", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  ```
* **Font Sizing Guide:**
  - **Diagram Title / Main Banner:** `14px`–`16px` (`font-weight: 700`)
  - **Card Header / Organ Name:** `12px`–`14px` (`font-weight: 700`)
  - **Badge / Cadence Pill:** `10px`–`11px` (`font-weight: 600`)
  - **Body Text / Bullet Item:** `11px`–`12px` (`font-weight: 400` or `500`)
  - **Secondary Annotation / Footnote:** `10px`–`11px` (`fill: #64748B`)

### C. Visual Card & Pill Patterns
* **Card Containers:** Use `<rect rx="6" ry="6" fill="..." stroke="..." stroke-width="1.2" />` for rounded, modern cards.
* **Badge Pills:** Use `<rect rx="10" ry="10" fill="#FFFFFF" stroke="..." stroke-width="1" />` for cadence labels (e.g., `1000 Hz · 1 ms`, `20 Hz · 50 ms`, `1 Hz · 1000 ms`).
* **Arrows & Connectors:** Define standard arrowheads in `<defs>`:
  ```xml
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569" />
    </marker>
    <marker id="arrow-crimson" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#A51C30" />
    </marker>
    <marker id="arrow-teal" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#0D9488" />
    </marker>
  </defs>
  ```

---

## 3. Canonical SVG Starter Template

Copy and adapt this starter template for new figures:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 480" width="100%" height="auto" role="img" aria-label="Physical AI Systems Architecture">
  <defs>
    <style>
      .title { font-family: system-ui, -apple-system, sans-serif; font-size: 15px; font-weight: 700; fill: #1F407A; }
      .subtitle { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; font-weight: 500; fill: #64748B; }
      .card-title { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; font-weight: 700; }
      .body-text { font-family: system-ui, -apple-system, sans-serif; font-size: 11px; fill: #334155; line-height: 1.4; }
      .badge-text { font-family: system-ui, -apple-system, sans-serif; font-size: 10px; font-weight: 600; }
      .code-text { font-family: 'SF Mono', Menlo, monospace; font-size: 10.5px; fill: #0F172A; }
      .flow-line { stroke: #64748B; stroke-width: 1.5; fill: none; stroke-linejoin: round; }
    </style>
    <marker id="arr" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748B" />
    </marker>
  </defs>

  <!-- Background Canvas -->
  <rect width="900" height="480" fill="#FFFFFF" rx="8"/>

  <!-- Top Title Banner -->
  <rect x="20" y="16" width="860" height="44" rx="6" fill="#F0F4FA" stroke="#1F407A" stroke-width="1.2"/>
  <text class="title" x="450" y="38" text-anchor="middle">THE PROPOSAL–PERMISSION PRIVILEGE SPLIT</text>
  <text class="subtitle" x="450" y="52" text-anchor="middle">Asynchronous Multi-Rate Runtimes: Untrusted Cognitive Proposals vs. Deterministic Real-Time Safety Vetoes</text>

  <!-- Column 1: System 2 (Amber / Proposal) -->
  <g transform="translate(20, 75)">
    <rect width="270" height="385" rx="6" fill="#FFFBEB" stroke="#D97706" stroke-width="1.2"/>
    <rect x="12" y="12" width="130" height="20" rx="10" fill="#FFFFFF" stroke="#D97706" stroke-width="1"/>
    <text class="badge-text" x="77" y="26" text-anchor="middle" fill="#D97706">SYSTEM 2 · 1 Hz</text>
    <text class="card-title" x="12" y="54" fill="#B45309">Untrusted Proposal Engine</text>
    <text class="body-text" x="12" y="78">
      <tspan x="12" dy="0">• Multimodal VLM (7B–70B parameters)</tspan>
      <tspan x="12" dy="18">• Open-vocabulary semantic reasoning</tspan>
      <tspan x="12" dy="18">• Emits Expiring Intent Lease (L_intent)</tspan>
      <tspan x="12" dy="18">• Tail latency: P99 ≈ 800–1500 ms</tspan>
    </text>
  </g>

  <!-- Column 2: System 1.5 (Cobalt / Planning) -->
  <g transform="translate(315, 75)">
    <rect width="270" height="385" rx="6" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.2"/>
    <rect x="12" y="12" width="140" height="20" rx="10" fill="#FFFFFF" stroke="#2563EB" stroke-width="1"/>
    <text class="badge-text" x="82" y="26" text-anchor="middle" fill="#2563EB">SYSTEM 1.5 · 20–50 Hz</text>
    <text class="card-title" x="12" y="54" fill="#1D4ED8">Trajectory Action Chunking</text>
    <text class="body-text" x="12" y="78">
      <tspan x="12" dy="0">• Diffusion Policy / ACT model</tspan>
      <tspan x="12" dy="18">• Generates H-step action trajectory</tspan>
      <tspan x="12" dy="18">• C^2 jerk-continuous spline blend</tspan>
      <tspan x="12" dy="18">• Shared lock-free SRAM ring buffer</tspan>
    </text>
  </g>

  <!-- Column 3: System 1 (Emerald / Safety Reflex) -->
  <g transform="translate(610, 75)">
    <rect width="270" height="385" rx="6" fill="#ECFDF5" stroke="#059669" stroke-width="1.2"/>
    <rect x="12" y="12" width="150" height="20" rx="10" fill="#FFFFFF" stroke="#059669" stroke-width="1"/>
    <text class="badge-text" x="87" y="26" text-anchor="middle" fill="#059669">SYSTEM 1 · 1000 Hz MCU</text>
    <text class="card-title" x="12" y="54" fill="#047857">Safety Permission Reflex</text>
    <text class="body-text" x="12" y="78">
      <tspan x="12" dy="0">• Bare-metal FreeRTOS supervisor</tspan>
      <tspan x="12" dy="18">• Control Barrier Functions: h(x) ≥ 0</tspan>
      <tspan x="12" dy="18">• Dynamic stopping veto: d_stop ≤ d_clear</tspan>
      <tspan x="12" dy="18">• Sole authority over motor gate PWM</tspan>
    </text>
  </g>
</svg>
```

---

## 4. Markdown Integration Rules

1. **Direct SVG Links in Quarto:** Always reference the `.svg` directly in `.qmd` files:
   ```markdown
   ![**Figure Title.** Caption explaining the systems mechanism.](figures/figXX_name.svg){#fig-name width=100%}
   ```
2. **Automated PDF Compilation:** When building PDF documents, Quarto/Pandoc automatically converts `.svg` files to high-resolution vector PDF using `pdftocairo` / `rsvg-convert`.
3. **Location:** Store all figure source files in the local chapter figure folder: `book/chapters/XX-name/figures/figXX_name.svg`.
