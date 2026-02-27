# 📐 Mission Plan: 15_sustainable_ai (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Sustainable AI: The Energy Wall & Jevons Paradox.
*   **Core Invariant:** The Sustainability Invariant (Efficiency enables Scale: **Jevons Paradox**) and the **Energy Wall**.
*   **The Struggle:** Understanding that "Efficiency is a Double-Edged Sword." Students must navigate the trade-off between **Hardware Efficiency** (TFLOPS/W) and **Total Carbon Footprint**, specifically focusing on how making training 2x cheaper often leads to 10x more training runs.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Sustainability Missions)

| Track | Persona | Fixed North Star Mission | The "Sustainability" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Grid Wall.** Your 100MW datacenter is causing local grid instability. The utility provider has capped you at 80MW. You must implement 'Carbon-Aware Scheduling' without violating user SLAs. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Fleet Thermal Drift.** A heatwave across 10,000 AVs is causing widespread thermal throttling. You must choose between 'Reducing Model Resolution' or 'Increasing Fan Energy' (which kills range). |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Phantom Load.** Even when 'Idle', your background fleet of 1 billion glasses is pulling 10mW each. This 'Phantom Load' is equivalent to a coal power plant. You must optimize the 'Fleet Sleep' policy. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Embodied Carbon Wall.** The carbon cost of MANUFACTURING the hearable's battery is higher than the carbon it saves over its lifetime. You must extend 'Device Longevity' to 5 years. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Energy Roofline (Exploration - 15 Mins)
*   **Objective:** Quantify the "Joules per Inference" across different hardware efficiency tiers.
*   **The "Lock" (Prediction):** "If you switch from a GPU to a specialized NPU that is 10x more efficient, will your total FLEET energy consumption drop by 10x?"
*   **The Workbench:**
    *   **Action:** Toggle between **Baseline Hardware** and **Green Hardware**. Adjust **Efficiency Factor** ($\eta_{energy}$).
    *   **Observation:** The **Energy-Performance Plot**. Watch the "Energy-per-Sample" drop while the "Inferences-per-Second" rises.
*   **Reflect:** "Patterson asks: 'Why is the Energy Wall harder to solve than the Power Wall?' (Reference the **Jevons Paradox** effect)."

### Part 2: Carbon-Aware Scheduling (Trade-off - 15 Mins)
*   **Objective:** Balance "Training Cost" vs. "Carbon Intensity" by shifting workloads across time/regions.
*   **The "Lock" (Prediction):** "Is it more 'Sustainable' to train a model in a Coal-powered region at night, or a Solar-powered region during the day?"
*   **The Workbench:**
    *   **Sliders:** Time-of-Day (0-24h), Grid Mix (Renewable %), Regional PUE.
    *   **Instruments:** **Carbon Intensity Gauge**. **TCO-vs-Carbon Seesaw**.
    *   **The 10-Iteration Rule:** Students must find the exact "Scheduling Window" that hits the 30% Carbon Reduction target without letting the "Project Delay" exceed 48 hours.
*   **Reflect:** "Jeff Dean observes: 'Your carbon-aware scheduler moved all jobs to Norway, but now the network latency is killing the Serving SLA.' Propose a 'Hybrid Regional' strategy."

### Part 3: The Jevons Equilibrium (Synthesis - 15 Mins)
*   **Objective:** Design a "Sustainable Scale Plan" that accounts for Induced Demand.
*   **The "Lock" (Prediction):** "If you optimize your model to be 50% smaller, will your organization use 50% less compute, or train a model that is 2x larger?"
*   **The Workbench:** 
    *   **Interaction:** **Efficiency Slider**. **Organization Growth Scrubber**. **Total Fleet Footprint Monitor**.
    *   **The "Stakeholder" Challenge:** The **Sustainability Lead** (or Board) demands a "Net-Zero" path. You must prove that using **Embodied Carbon Credits** (device recycling) is more impactful than another 5% gain in NPU efficiency.
*   **Reflect (The Ledger):** "Defend your final 'Sustainability Strategy.' Did you prioritize 'Operational Carbon' (Energy) or 'Embodied Carbon' (Hardware)? Justify using the Jevons Paradox math."

---

## 4. Visual Layout Specification
*   **Primary:** `EnergyRooflinePlot` (Throughput vs. Energy Efficiency).
*   **Secondary:** `CarbonWaterfall` (Embodied Carbon vs. Grid Energy vs. Cooling Overhead).
*   **Math Peek:** Toggle for `CUE = \frac{	ext{Carbon Emissions}}{	ext{IT Equipment Power}}` and `Jevons Elasticity`.
