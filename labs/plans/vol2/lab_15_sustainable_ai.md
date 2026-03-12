# Mission Plan: lab_15_sustainable_ai (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Sustainable AI (`@sec-sustainable-ai`)
- **Core Invariant:** The **Jevons Paradox of AI** -- making models 10x more efficient leads to 100x more usage, not 10x energy savings. Per-unit efficiency gains are necessary but insufficient; absolute carbon budgets and carbon-aware scheduling are the only levers that guarantee net emission reductions at fleet scale.
- **Central Tension:** Students believe that algorithmic efficiency improvements directly translate into proportional energy savings ("if I halve inference cost, total energy drops by half"). The chapter's data demolishes this: a 50% cost reduction with elastic demand (300% usage increase) produces a *net 50% increase* in total energy consumption. Meanwhile, geographic site selection alone creates a 40x difference in carbon emissions (Quebec hydro at 20 g CO2/kWh vs Poland coal at 800 g CO2/kWh) -- larger than any possible algorithmic speedup. The real sustainability lever is not FLOPS/Watt but *Carbon-per-FLOP*, and that is dominated by when and where the job runs, not how efficiently the model computes.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students confront the Geography of Carbon. They predict the carbon footprint difference between training in a coal-heavy grid versus a hydro-powered grid, expecting the difference to be modest (perhaps 2--5x, similar to algorithmic speedups they have encountered throughout the book). The instrument reveals a 40x difference, establishing that site selection is the single highest-leverage sustainability intervention -- larger than pruning, quantization, and distillation combined. This calibrates their intuition away from "efficiency solves sustainability" toward "geography dominates."

**Act 2 (Design Challenge, 23 min):** Students design a carbon-aware scheduling strategy that must hit a 50% emission reduction target without violating a 48-hour project delay constraint. The instrument introduces Jevons Paradox directly: as students optimize efficiency (reducing cost per query), elastic demand increases total usage, and the total carbon footprint can *increase* despite per-unit improvements. Students must find the equilibrium point where efficiency gains, scheduling windows, and absolute carbon caps interact, discovering that governance (hard caps on total compute) is the only mechanism that guarantees net reduction when demand is elastic.

---

## 3. Act 1: The Geography of Carbon (Calibration -- 12 minutes)

### Pedagogical Goal
Students systematically underestimate the impact of grid carbon intensity on total emissions. Having spent the entire book optimizing algorithms and hardware, they assume the dominant sustainability lever is computational efficiency. The chapter's "Geography of Carbon" notebook demonstrates that a 10,000 MWh training run produces 200 tonnes CO2 in Quebec (hydro, 20 g/kWh) versus 8,000 tonnes in Poland (coal, 800 g/kWh) -- a 40x difference. No algorithmic optimization in history has delivered a 40x improvement in a single intervention. This act forces students to confront the scale mismatch between algorithmic gains and geographic gains.

### The Lock (Structured Prediction)

> "You are scheduling a 10,000 MWh frontier model training run. Site A uses hydropower (Quebec, 20 g CO2/kWh). Site B uses coal-heavy grid (Poland, 800 g CO2/kWh). How much more carbon does Site B emit compared to Site A?"

Options:
- A) About 2--3x more -- grids are not that different in practice
- B) About 5--10x more -- coal is dirtier but not by orders of magnitude
- **C) About 40x more -- grid carbon intensity spans two orders of magnitude** (correct)
- D) About 100x more -- coal is catastrophically worse than hydro

Common wrong answer: B) About 5--10x more. Students anchor on their experience with algorithmic speedups (2--10x is typical for pruning, quantization, etc.) and assume grid differences operate at the same scale. The 40x ratio is outside their calibrated range.

Why wrong: Grid carbon intensity is not a performance metric that degrades gracefully. It is a physical property of the energy source: coal releases 800 g CO2 per kWh of electricity generated; hydro releases 20 g. The ratio is fixed by thermochemistry, not engineering optimization.

### The Instrument: Carbon Geography Comparator

**Primary chart: Stacked bar chart -- "Carbon Footprint by Region"**
- X-axis: Region (Quebec Hydro / US Average / Texas Mixed / Poland Coal / Custom)
- Y-axis: Total CO2 emissions (tonnes), range 0--10,000
- Bars: Single bar per region, colored by carbon intensity band (GreenLine for <50 g/kWh, OrangeLine for 50--500, RedLine for >500)

Controls:
- **Training energy** slider: 1,000 -- 50,000 MWh (step 1,000; default 10,000)
- **Region selector**: dropdown with 5 options (Quebec: 20, US Avg: 429, Texas: 400, Poland: 800, Custom)
- **Custom carbon intensity** slider: 10 -- 1,000 g CO2/kWh (step 10; default 400; enabled only when Custom selected)
- **PUE** slider: 1.05 -- 2.00 (step 0.05; default 1.20)

The formula: `Carbon_tonnes = Energy_kWh * CI_g_per_kWh * PUE / 1,000,000`

A **ratio annotation** appears above the bars: "Site B emits [X]x more than Site A" dynamically updating as regions are changed.

### The Reveal
> "You predicted [X]x difference. The actual difference between Quebec (20 g/kWh) and Poland (800 g/kWh) is **40x**. Moving a training run from Poland to Quebec saves more carbon than any algorithmic optimization technique: pruning (2--4x), quantization (2--4x), and distillation (3--10x) combined yield at most ~160x compound savings -- but a single site selection decision achieves 40x with zero accuracy loss."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter claims site selection is the single most effective tool for sustainable AI. Based on Act 1, why?"

- A) Renewable energy is cheaper, so organizations naturally migrate to clean grids
- B) Algorithmic optimizations compound multiplicatively with site selection for even greater savings
- **C) Grid carbon intensity spans two orders of magnitude (20--800 g/kWh), creating a larger multiplier than any algorithmic intervention alone** (correct)
- D) Governments mandate clean energy for AI workloads, forcing relocation

### Math Peek

$$C_{operational} = E_{total} \times CI_{grid} \times PUE$$

where $E_{total}$ is IT equipment energy (kWh), $CI_{grid}$ is grid carbon intensity (g CO2/kWh), and $PUE$ accounts for cooling overhead. Quebec: $10{,}000{,}000 \times 20 \times 1.2 / 10^6 = 240$ tonnes. Poland: $10{,}000{,}000 \times 800 \times 1.2 / 10^6 = 9{,}600$ tonnes. Ratio: 40x.

---

## 4. Act 2: The Jevons Equilibrium (Design Challenge -- 23 minutes)

### Pedagogical Goal
Students believe that making inference more efficient automatically reduces total energy consumption. The chapter's Jevons Paradox section demolishes this: a 50% cost reduction with elastic demand (300% usage increase) produces a net 50% *increase* in total consumption. Students must design a carbon-aware scheduling strategy that achieves a 50% emission reduction target while accounting for induced demand. They discover that only absolute carbon caps (governance) guarantee net reduction when demand is elastic -- efficiency alone is insufficient.

### The Lock (Numeric Prediction)

> "Your team optimizes a translation service, reducing computational cost per query by 50% (2x efficiency gain). Demand is elastic: the cost reduction triggers a 300% increase in query volume. What is the net change in total energy consumption?"

Students enter a percentage (bounded: -100% to +200%). Expected wrong answers: -50% to -25% (students assume efficiency gains dominate). Actual: original cost = 1.0 per query, new cost = 0.5 per query, new volume = 4x original. Total energy = 0.5 * 4 = 2.0 = **+100% increase** (doubles).

The system shows: "You predicted [X]% change. Actual: **+100% increase**. Efficiency halved the per-query cost, but 4x demand more than offset it. This is Jevons Paradox."

### The Instrument: Carbon-Aware Fleet Scheduler

**Primary chart: Dual-axis time series -- "Fleet Carbon Over 24 Hours"**
- X-axis: Hour of day (0--24)
- Y-axis (left): Grid carbon intensity (g CO2/kWh), range 0--1000
- Y-axis (right): Total fleet carbon emissions (kg CO2/hr), range 0--5000
- Two lines: CI curve for coal-grid region (relatively flat ~800) and hydro-grid region (relatively flat ~20)
- Shaded bars: training job placement (movable across the 24-hour window)

**Secondary chart: Jevons Paradox Dashboard**
- X-axis: Efficiency improvement factor (1x--10x)
- Y-axis: Total fleet energy (normalized to baseline = 1.0)
- Three curves: Inelastic demand (total energy drops proportionally), Unit-elastic (total energy constant), Elastic demand (total energy *increases*)
- Student's operating point marked on the elastic curve

Controls:
- **Efficiency factor** slider: 1.0x -- 10.0x (step 0.5; default 1.0)
- **Demand elasticity** selector: Inelastic (0.5) / Unit-elastic (1.0) / Elastic (2.0) / Highly elastic (3.0)
- **Carbon cap** toggle: OFF / 50% of baseline / 25% of baseline
- **Scheduling window** selector: Coal-grid 24/7 / Hydro-grid 24/7 / Carbon-aware (shift to low-CI hours)
- **Deployment context** toggle: Coal-grid region (CI = 800 g/kWh) vs Hydro-grid region (CI = 20 g/kWh)

Formulas:
- `New_volume = Baseline_volume * (Efficiency_factor ^ Elasticity)`
- `Total_energy = (Baseline_energy / Efficiency_factor) * New_volume`
- `Total_carbon = Total_energy * CI_grid * PUE / 1e6`
- With carbon cap: `Actual_volume = min(New_volume, Cap_volume)` where `Cap_volume = Cap_carbon / (per_query_energy * CI * PUE)`

### The Scaling Challenge

**"Achieve a 50% carbon emission reduction (compared to baseline) for a fleet with elastic demand (elasticity = 2.0) without exceeding a 48-hour project delay."**

Students must combine multiple levers:
1. Efficiency improvements alone *increase* total carbon (Jevons effect)
2. Geographic shift (coal to hydro) provides 40x reduction but may add network latency
3. Carbon-aware scheduling (shifting jobs to low-CI hours) provides 50--80% reduction
4. Carbon caps enforce hard limits but may reject jobs

The solution requires: geographic shift to hydro + carbon cap at 50% baseline. Pure efficiency without a cap fails. Students discover that governance (the cap) is the indispensable lever.

### The Failure State

**Trigger condition:** `Total_carbon > 2.0 * Baseline_carbon` (Jevons rebound exceeds 2x)

**Visual change:** The Jevons Dashboard bar chart turns RedLine. The total carbon bar grows beyond the chart boundary.

**Banner text:** "JEVONS REBOUND -- Total fleet carbon INCREASED by [X]% despite [Y]x per-unit efficiency gain. Efficiency without governance amplifies consumption. Enable a carbon cap to constrain total demand."

The failure state is reversible: enabling the carbon cap immediately constrains volume and pulls carbon below the threshold.

### Structured Reflection

Four-option multiple choice:

> "The chapter states: 'Making models 10x more efficient will likely lead to 100x more usage, not 10x energy savings.' Which strategy guarantees net emission reduction regardless of demand elasticity?"

- A) Maximize per-unit efficiency (FLOPS/Watt) across the fleet
- B) Shift all workloads to renewable-powered regions during peak solar hours
- **C) Set absolute carbon budgets that cap total compute regardless of per-unit cost improvements** (correct)
- D) Deploy smaller models that inherently consume less energy per query

### Math Peek

$$E_{total} = \frac{E_{baseline}}{\text{Efficiency}} \times V_{baseline} \times \left(\text{Efficiency}\right)^{\text{Elasticity}}$$

For Efficiency = 2x, Elasticity = 2.0: $E_{total} = \frac{1}{2} \times 1 \times 4 = 2.0$ (100% increase).

The Jevons breakeven condition: $\text{Elasticity} < 1$ for efficiency to reduce total consumption. When $\text{Elasticity} \geq 1$, only absolute caps guarantee reduction.

---

## 5. Visual Layout Specification

### Act 1: Carbon Geography Comparator
- **Primary:** Stacked bar chart -- regions on X, tonnes CO2 on Y. Bars colored by CI band (green < 50, orange 50--500, red > 500 g/kWh). Dynamic ratio annotation between selected pair.
- **Chart type:** Vertical bar
- **X-axis:** Region name (categorical)
- **Y-axis:** Total CO2 emissions (tonnes), range 0--10,000
- **Failure state:** N/A for Act 1

### Act 2: Jevons Paradox Dashboard
- **Primary:** Dual-axis time series -- CI curve + fleet carbon per hour, with movable job placement shading. X: Hour (0--24), Y-left: CI (g/kWh, 0--1000), Y-right: Fleet carbon (kg/hr, 0--5000).
- **Secondary:** Jevons rebound curve -- efficiency factor on X (1--10x), normalized total energy on Y (0--3x baseline). Three demand curves (inelastic, unit-elastic, elastic). Student's operating point marked.
- **Failure state:** When total carbon exceeds 2x baseline, both charts turn RedLine with Jevons Rebound banner.

---

## 6. Deployment Context Definitions

| Context | Region | Grid CI (g/kWh) | PUE | Key Constraint |
|---|---|---|---|---|
| **Coal-grid region** | Poland / US Midwest | 800 | 1.40 | High operational carbon dominates lifecycle; geographic shift is the primary lever |
| **Hydro-grid region** | Quebec / Norway | 20 | 1.10 | Embodied carbon becomes dominant (~52% of lifecycle); hardware longevity is the lever |

The two contexts demonstrate that the *binding sustainability constraint* shifts depending on grid carbon intensity. In coal-grid regions, operational emissions dominate and geographic scheduling is the highest-leverage intervention. In hydro-grid regions, operational emissions are negligible and embodied carbon from hardware manufacturing becomes the dominant term, making hardware utilization and longevity the primary levers.

---

## 7. Design Ledger Output

```json
{
  "chapter": 15,
  "deployment_region": "coal_grid | hydro_grid",
  "grid_ci_g_per_kwh": 800,
  "efficiency_factor": 2.0,
  "demand_elasticity": 2.0,
  "carbon_cap_enabled": true,
  "carbon_reduction_achieved_pct": 50,
  "jevons_rebound_experienced": true,
  "scheduling_strategy": "carbon_aware | fixed_region"
}
```

The `carbon_cap_enabled` and `jevons_rebound_experienced` fields feed forward to:
- **Lab 16 (Responsible AI):** The responsible AI overhead budget must account for sustainability constraints from this lab's carbon cap
- **Lab 17 (Conclusion):** The fleet synthesis radar includes sustainability as one of the 6 principle dimensions, reading `carbon_reduction_achieved_pct` to set the baseline

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| 40x carbon difference (Quebec vs Poland) | `@sec-sustainable-ai-scale-environmental-impact-ac9a`, CarbonFrontier class | "Site A (Quebec): 20 g CO2/kWh. Site B (Poland): 800 g CO2/kWh. Ratio: 40x difference" |
| Training energy 1,287 MWh (GPT-3) | `@sec-sustainable-ai-scale-environmental-impact-ac9a`, CarbonCostGPT3 class | "Training GPT-3 consumed approximately 1,287 MWh of energy" |
| 552 trans-Atlantic flights equivalent | `@sec-sustainable-ai-scale-environmental-impact-ac9a`, CarbonCostGPT3 class | "552 flights" (552,123 kg / 1,000 kg per flight) |
| PUE range 1.08--1.67 | `@sec-sustainable-ai-energy-wall`, PueEfficiency class | "industry average PUE (1.67) to state-of-the-art (1.19)" and "Google's best facilities achieve PUE 1.08" |
| Jevons Paradox: 10x efficient -> 100x usage | `@sec-sustainable-ai-part-iii-implementation-solutions-232d` | "making models 10x more efficient will likely lead to 100x more usage, not 10x energy savings" |
| 50% cost reduction + 300% demand increase | `@sec-sustainable-ai-part-iii-implementation-solutions-232d`, checkpoint | "50% cost reduction leads to a 300% increase in query volume" |
| Carbon-aware scheduling: 50--80% reduction | `@sec-sustainable-ai-geographic-temporal-optimization-492c` | "Temporal scheduling can reduce emissions by 50-80% by aligning compute workloads with renewable energy availability" |
| Training: 60--80% of lifecycle emissions | `@sec-sustainable-ai-threephase-lifecycle-assessment-framework-883a` | "training phase (60--80% of emissions)" |
| Embodied carbon: 150--200 kg CO2 per H100 | `@sec-sustainable-ai-embodied-carbon-assessment-9de0` | "A single NVIDIA H100 GPU embodies approximately 150 to 200 kg CO2eq from manufacturing" |
| Compute growth 350,000x (2012--2019) | `@sec-sustainable-ai-exponential-growth-vs-physical-constraints-0f4e` | "compute requirements increasing 350,000x from 2012 to 2019" |
| Embodied carbon 52% of lifecycle in clean grid | `@sec-sustainable-ai-embodied-carbon-assessment-9de0`, EmbodiedCarbonAmort | "embodied carbon a significant fraction (~52%) of the total footprint" in clean grids |
