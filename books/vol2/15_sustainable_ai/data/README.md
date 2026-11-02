# Chapter 15 Figure and Table Data Provenance (`15_sustainable_ai`)

Every Tier-2 CSV in this directory carries structured `# Source:`, `# URL:`, `# Verified:`, `# Epistemic:`, `# Computation:`, and `# Notes:` header fields parsed at load time by `mlsysim.core.series_loader.LocalSeries.provenance` and verified by `./binder/binder check figures --scope data-provenance`.

## Epoch AI Compute and Energy Series (`@fig-ai-compute-growth`, `@fig-frontier-training-energy`)

Both historical scaling figures in Section 15.1 derive deterministically from the Epoch AI Notable AI Models snapshot (`notable_ai_models.csv`, CC BY 4.0, snapshot `2026-10-12`, raw SHA-256 `bacde4f9af388703ceebd01cf71a14854f93b8995e425141b016e48f21d4e7df`):

```bash
python3 binder/tools/scripts/extract_epoch_sustainability_data.py \
  --download --write-compute-csv \
  --manifest binder/tools/audits/epoch_sustainability_manifest.json
```

| Dataset | Consuming LEGO / Figure | Epistemic Status | Selection & Computation Recipe |
| :--- | :--- | :--- | :--- |
| `books/vol2/01_introduction/data/notable_model_training_runs.csv` | `ComputeGrowthTrend` / `@fig-ai-compute-growth` (shared with Ch. 1 `@fig-training-compute-trend`) | Modeled (Epoch AI training-compute estimates and builder-disclosed FLOP counts; `@epochai2026notable`, `@sevilla2022`) | Rows with `Publication date >= 2012-01-01` and non-empty `Training compute (FLOP)` or `Hardware quantity` (`501` rows total; `N = 468` with positive `training_compute_flop`, `n = 69` with `Frontier model == True`). Fits ordinary least squares (OLS) with White HC3 heteroskedasticity-consistent 95% confidence bounds on $\log_2(\text{FLOPs})$ vs. decimal publication year (`4.58×/yr`, HC3 95% CI `4.27×–4.91×/yr`, $R^2 = 0.97$, doubling time `5.47` months). |
| `notable_model_training_energy.csv` | `FrontierTrainingEnergy` / `@fig-frontier-training-energy` | Modeled peak-capacity facility upper bound (`@epochai2026notable`, `@cottier2024rising` App. C) | Standalone pretraining runs with `Publication date >= 2016-01-01`, `Training power draw (W) > 0`, `Training time (hours) > 0`, empty `Base model` (excluding `22` fine-tune/checkpoint rows), and no hardware-quantity $\times$ runtime warning (excluding `OpenAI Five`, whose Epoch notes warn against multiplying peak 1,536-GPU count by 10-month runtime). Yields `N = 122` standalone runs (`n = 23` with `Frontier model == True`). Upper-bound energy is $E_{\text{MWh}} = P_{\text{draw,W}} \times t_{\text{hours}} / 10^6$, where $P_{\text{draw,W}} = N_{\text{chips}} \times \text{TDP}_{\text{W}} \times 1.82 \times \text{PUE}(t)$ for $N_{\text{chips}} > 1$ and $\text{TDP}_{\text{W}} \times \text{PUE}(t)$ for $N_{\text{chips}} = 1$ (without Cottier et al. App. A.5 average-utilization derating). Fits OLS + White HC3 95% CI on $\log_{10}(E_{\text{MWh}})$ across standalone frontier runs (`3.01×/yr`, HC3 95% CI `2.52×–3.59×/yr`, $R^2 = 0.91$). Modeled upper-bound callouts round to 2 significant figures matching hardware/duration input precision. |

### Builder-Based Anchors Contrasted on `@fig-frontier-training-energy`

- **GPT-3 175B (`1,287 MWh`, measured facility boundary)**: Loaded from `Models.Language.GPT3.training_energy_mwh` and cross-checked against `patterson_training_footprints.csv` (`330 W` metered system average per V100 including host CPU, DRAM, fans, and NIC $\times$ `1.10` PUE; `@patterson2021carbon`, Table 4). Contrasted against Epoch AI's `~1,800 MWh` (`1,812 MWh` unrounded, `1.41×`) peak-capacity estimate.
- **Llama 3.1-405B (`21,588 MWh`, constructed accelerator-TDP boundary)**: Computed in `FrontierTrainingEnergy` as `Literature.TrainingRuns.Llama31_405B_TrainingGpuHours` (`30.84M` H100 GPU-hours; `@meta2024llama31card`) $\times$ `Hardware.Cloud.H100.tdp` (`700 W`; `@dubey2024llama`, §3.3.1), pre-host and pre-PUE (`~24,000 MWh` at `1.10` PUE). Contrasted against Epoch AI's `~48,000 MWh` (`48,266 MWh` unrounded, `2.24× = 1.97×` server-and-PUE multiplier $\times$ `1.14×` inferred `35.1M` wall-clock GPU-hour window).
- **Grid-scale reference (`24,000 MWh`, assumed scenario)**: `1 GW` rated power-plant output integrated over `24 hours`.

## Other Chapter 15 Datasets

| Dataset | Epistemic Status | Primary Source & Derivation |
| :--- | :--- | :--- |
| `patterson_training_footprints.csv` | Measured | Patterson et al. (2021), Table 4 (`@patterson2021carbon`): metered system power per chip, PUE, data-center $\text{CO}_2\text{e}$/kWh, facility MWh, and gross/net $\text{tCO}_2\text{e}$ across five NLP training runs (`T5`, `Meena`, `GShard-600B`, `Switch Transformer`, `GPT-3`). |
| `europe_generation_2023w47.csv` | Measured | Fraunhofer ISE Energy-Charts (`api.energy-charts.info/public_power?country=eu`): hourly EU net generation and load in MW (`2023-11-20` through `2023-11-27` CET), grouping hard coal, lignite, and oil into `coal_oil`. |
| `gemini_prompt_energy.csv` | Measured | Elsworth et al. (2025), Table 1 (`@elsworth2025measuring`): per-component Wh for the median Gemini Apps text prompt (May 2025) under the narrow (`0.10 Wh`) vs. comprehensive (`0.24 Wh`) fleet accounting boundary. |
| `google_serving_factors.csv` | Measured | Elsworth et al. (2025), Sections 3.3, 4.3, and Table 2 (`@elsworth2025measuring`): Google fleet PUE (`1.09`), ISO Category 2 WUE (`1.15 L/kWh`), location/market-based grid factors, and per-prompt water (`0.26 mL`) and $\text{CO}_2\text{e}$ (`0.03 g`). |
| `meta_operational_carbon.csv` | Measured | Wu et al. (2022), Figure 3 (`@wu2022sustainable`): operational $\text{CO}_2\text{e}$ (`10^6 kg`) split into offline training, online training, and inference across Meta production models (`LM`, `RM-1..RM-5`) vs. public training-only baselines. |
| `iot_device_growth.csv` | Modeled | Statista (2022): global connected IoT devices in billions, separating historical estimates (`2019–2023`) from projected counts (`2024–2030`). |
