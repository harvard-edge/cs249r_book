# MLSysBook LEGO Unit Hardening — Progress

**Branch:** `fmt-fix`
**Worktree:** `/Users/VJ/GitHub/MLSysBook-fmt-fix`
**Started:** 2026-05-31

## Checklist

| Step | Status | Notes |
|------|--------|-------|
| 1 — Baseline + PROGRESS | DONE | SHA `34e4f12ace80`; pytest green |
| 2–10 — MLSysIM infra | DONE | units, aliases, physics, domain formatters |
| 11–13 — Docs + lint + CI | DONE | lego-units.md; lint_lego_units.py |
| A′ — LOAD registry-first | IN PROGRESS | GPT-3 energy, flight CO₂e in registry |
| 14+ — QMD migration | IN PROGRESS | sustainable_ai.qmd nearly complete |

## Current

- **Chapter in flight:** `vol2/sustainable_ai/sustainable_ai.qmd`
- **Next after commit:** Vol I book order per `_quarto-html-vol1.yml` (introduction.qmd first unmigrated cell)

## sustainable_ai.qmd — migrated (committed + this batch)

CarbonCostGPT3, ArchetypeATdp, CarbonFrontier, AutoPlacement, GpuEmissionsScenario, PueEfficiency, LifecycleCarbonEstimate, TrainingEmissions, A100PowerScenario, EmbodiedCarbonAmort, TrainingEmbodiedRecap, InferenceLifecycleExample, TrainingEmissionsRecap, GridQueue, H100TdpRackRecap, RackPowerBudget, SustainableMobilePowerEnvelope, MemoryHierarchyEnergy, MatMulEnergyAnalysis, SustainableCoolingRackPowerRecap, PueSavings, H100TdpCoolingRecap, RackPowerCoolingRecap, OnDeviceLearningEnergy, WakeWordPower, InfraFrontier* param recap cells

## Remaining in sustainable_ai (low priority)

- Code listings (~1995 `calculate_carbon_footprint`, ~1439 energy measurement) — not LEGO classes
- Layer A′-4: grid CI as Quantity at LOAD (GpuEmissionsScenario, TrainingEmissions still use `fmt_qty(..., kg/kWh)`)
- EnergyEfficiencyRange, ComputeGrowthScenario, EnergyWallScenario — dimensionless / literature floats

## Step log (recent commits on fmt-fix)

| SHA | Summary |
|-----|---------|
| (pending) | Cooling/PUE/mobile/on-device LEGO quantity-first batch |
| `9f9d64f09a` | GridQueue, RackPowerBudget |
| `96397f8efe` | Embodied carbon recap cells |
| `3c4c8edbbd` | AutoPlacement through A100PowerScenario |
| `49e1cff3c1` | CarbonCostGPT3 |
| `17b7eca96d` | A′ registry + vol1 GPT-3 Quantity fixes |
