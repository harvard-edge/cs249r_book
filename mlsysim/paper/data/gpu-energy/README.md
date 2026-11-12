# Measured GPU board energy

MLSys·im's solvers accept a device power, and the package fills it in one of
two ways. The sustainability scenarios (and `InferenceScalingModel`'s
reasoning energy) pass the registry TDP. `Engine.solve`, and so
`SingleNodeModel`'s `energy`, uses an energy-proportional rule,
TDP × (0.30 + 0.70 × utilization), with the fractions in
`engine/calibration.py`. Neither had been checked against a meter
(CONTRIBUTING.md: "TDP-based energy is an estimate, not meter data"). This
directory has the declared protocol, the measurements, and their comparison
with both assumptions.

| File | Role |
| --- | --- |
| `plan.json` | The configuration matrix, trial counts, phase lengths, and every efficiency, declared before measuring. |
| `runs/<run-id>/` | One measurement session on one GPU: `trials.jsonl`, one line per trial (warm-up and failed trials included), and a `manifest.json` that hashes it and indexes every trial. Six runs are committed: four are used and two are excluded, each with its reason in `summary.provenance.json`. |
| `summary.csv`, `summary.provenance.json` | Written by `../../scripts/analyze_gpu_energy.py` from every verified run. |
| `device_choice.csv` | Also written by the analysis: for each configuration measured on both GPUs, the lower-energy device by the meter and by each estimate. |

## Results (measured 2026-11-12)

Four runs are used: two on H100 SXM boards (700 W limit) and two on
A100-SXM4-80GB boards, each run on a different physical GPU in a separate Modal
container. Every configuration has 20 measured trials (10 per run, after one
warm-up), and none failed. Two more runs are kept and excluded: the pilot, which
was measured under the earlier plan, and one container that Modal served with an
`A100 80GB PCIe` (300 W limit), which the runner refused to measure. Figures
below are ranges over the configurations in `summary.csv`. Ratios are estimate
over measured energy per unit of work.

- **Instrument.** On both GPUs the energy counter and the power reading change
  every 100 ms. Integrated polled power agrees with the counter within 10%
  (`sampled_over_counter` 0.91–1.04).
- **Idle power** with the workload resident is 21% of TDP on both GPUs
  (H100 146–150 W, A100 82–86 W). The proportional rule assumes 30%.
- **Decode** draws 69–77% of TDP on the H100 and 75–85% on the A100, while its
  FLOP utilization is 0.2–6%. The proportional rule ties power to that
  utilization, so over the measured time it gives 0.40–0.44 of the measured
  decode energy. TDP over the measured time gives 1.18–1.44.
- **Prefill** above one 128-token prompt ends every trial at the power cap (the
  clock-event reasons report the software power cap). On the H100 it draws
  96–99% of TDP, and on both GPUs TDP over the measured time is within 8% of the
  meter (0.92–1.04). For a single 128-token prompt, engine overhead dominates and
  TDP overstates energy by up to 1.68×.
- **Training** (ResNet-50) draws 89% of TDP on the H100 and 94% on the A100. TDP
  over the measured time gives 1.07–1.13, and the proportional rule 0.43–0.47.
- **Time model.** `ServingModel`'s decode step is 0.62–0.70 of the measured one.
  Its TTFT at batch 16 or with 2,048-token prompts is 1.07–1.75 of measured,
  because these prefills run at a higher utilization than the declared 0.5
  efficiency (the utilization counts attention over the full context, an upper
  bound).
- **Errors that cancel.** TDP over the modeled time gives 0.79–0.97 for decode
  only because a step time 30–38% too short meets a power 18–44% too high. The
  proportional rule over the modeled time gives 0.27–0.31.
- **Device choice.** The H100 uses less energy per prefill (by 43–47% above one
  128-token prompt) and per training step (by 11%). The A100 uses less per
  decode step in five of six configurations (by 4–10%). In 10 of 13
  configurations the same device wins on every pairing of one H100 and one A100
  board (`measured_every_run_pair`). On those 10, both estimates that use the
  measured time pick the meter's device every time. Each fully analytical
  estimate picks the wrong device once. TDP over the modeled time chooses the
  A100 for training, where the H100 uses 11% less energy per step, because the
  modeled A100 step is 20% shorter than measured against 6% on the H100. The
  proportional rule over the modeled time chooses the H100 for decode at batch 16
  with 128-token prompts, where the A100 uses 7.8% less.
- **Boards of one model differ.** The two A100-SXM4-80GB boards report the same
  name with different power limits, 400 W and 500 W. On the 500 W board, large
  prefill ran 8% faster for 14% more energy and drew more than the registry's
  400 W TDP. The two boards' decode energy differed by 7–10%, as much as the
  A100's decode advantage (the two H100 boards differed by 1–4%).
  `power_limits_w` shows the mix on every row.

These results cover two boards per GPU type, one engine (vLLM 0.6.4.post1), one
language model at batch 16 or less, and ResNet-50 at one batch size, at the
board boundary only. The 95% intervals describe trial-to-trial noise, and the
`runs`, `gpus`, and `measured_every_run_pair` columns show what two boards can
say about board-to-board spread. The data do not test a scheduling decision (see
below).

Changes made after measuring began are listed here and in the plan. After the
pilot measured the 100 ms counter refresh, training steps went from 50 to 150
(`revision_note` in the plan). After the A100 runs, two descriptive columns were
added to the analysis, `power_limits_w` and `measured_every_run_pair`. Neither
changes an estimate, a measurement, or the plan.

The committed runs came from three launches on 2026-11-12:

```sh
modal run paper/scripts/modal_gpu_energy.py --devices H100 --repeats 1 \
    --configs h100-resnet50-train-b256,h100-llama-b16-p2048 --label pilot
modal run paper/scripts/modal_gpu_energy.py                     # 2x H100, 2x A100
modal run paper/scripts/modal_gpu_energy.py --devices A100 --repeats 1   # replaced the PCIe container
```

## Running the measurements

On Modal, after `modal setup` (the weights need no Hugging Face token):

```sh
modal run paper/scripts/modal_gpu_energy.py      # every device, twice, in parallel
PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py
```

The launcher first downloads the pinned weights into a Modal volume on a
CPU-only container, then starts every run at once, each in its own GPU
container, reading the weights offline. `--devices`, `--repeats`, `--configs`
and `--label` select a subset, for example a pilot. Each device is measured at
least twice. The trials inside one run share a process, a GPU, and a thermal
state, so their interval describes trial-to-trial noise only. Separate runs
show the spread between sessions and between physical GPUs, which the `runs`
and `gpus` columns count.

On any other single NVIDIA GPU, install `vllm`, `torch`, `torchvision`,
`transformers`, and `nvidia-ml-py` at the versions the Modal image pins, then:

```sh
python paper/scripts/measure_gpu_energy.py --device H100 --out paper/data/gpu-energy/runs/<run-id>
```

The runner does not import mlsysim. It refuses to measure if the GPU reports a
name different from the plan's `expected_name`. On Modal, `H100!` already stops
an H100 request from being served by an H200. This check catches a substitution
on any platform.

## Protocol

- **Energy** is the difference of the driver's cumulative board-energy counter
  (`nvmlDeviceGetTotalEnergyConsumption`) read after `torch.cuda.synchronize()`
  at each phase boundary. Polled power (`nvmlDeviceGetPowerUsage`) is kept only
  as a cross-check, reported as `sampled_over_counter`.
- **Counter refresh**: before any workload loads, each run polls the counter and
  the power reading in a tight loop for `counter_probe_seconds` and records every
  change. The summary reports the median interval between counter changes as
  `counter_refresh_ms`. Each phase boundary can be off by up to one interval, so
  phases are kept several seconds long. The probe runs on an idle GPU, so it
  does not show whether the refresh rate changes under load.
- **Phases**: each trial measures an idle phase with the workload resident, then
  the workload. Language models record two phases, each repeated until it lasts
  at least `phase_min_seconds`: one-token generations (`prefill`, which is TTFT
  for the whole batch) and full generations of `output_tokens` tokens
  (`generate`). A full generation is one prefill plus `output_tokens - 1` decode
  steps, so the analysis takes a decode step as (generation − one-token
  generation) / (`output_tokens` − 1), in time and in energy, using the same
  trial's prefill. This is how NIM defines ITL, (e2e latency − TTFT) /
  (output tokens − 1), and decode never includes TTFT. ResNet-50: unmeasured
  warm-up steps, then `steps` training steps.
- **Weights**: `NousResearch/Meta-Llama-3.1-8B`, an ungated mirror of
  `meta-llama/Llama-3.1-8B`, used under the Llama 3.1 Community License. At the
  pinned revisions its config, generation config, tokenizer, and weight index are
  byte-identical to Meta's (equal git blob ids), and its four weight shards have
  Meta's sizes and the same SHA-256 as a second mirror. Meta does not publish the
  shard hashes without a license acceptance. The plan lists the shard hashes, and
  the launcher checks the download against them before any GPU starts.
- **Engine**: Llama runs in vLLM 0.6.4.post1's offline engine on one GPU, with
  CUDA graphs for decode, the FlashAttention backend, greedy sampling, and
  `ignore_eos`, so every sequence produces exactly `output_tokens` tokens (the
  runner fails the trial otherwise). Every request of a batch is submitted at
  once. Prefix caching is off because every trial repeats the same prompts,
  and chunked prefill is off so prefill never interleaves with decode. The
  engine settings are in the plan, and one engine runs every configuration
  of a run, as a server would. Prompts are random token ids from a fixed seed.
  Decode includes vLLM's per-step scheduling on the host CPU, so the Modal
  launcher reserves 8 CPUs and the manifest records the CPU count and the
  container's cgroup CPU limit.
- **Trials**: `warmup_trials` full trials are recorded and never averaged, then
  `trials` measured trials. Failed trials are kept with their traceback and
  counted in `trials_failed`.
- **Nuisance variables** recorded at every phase boundary: SM and memory clocks,
  temperature, the clock-event (throttle) reason bitmask, and the number of
  compute processes on the GPU. Each manifest records the device's name, UUID,
  PCI bus id, power limits, driver and CUDA versions, every installed Python
  package version, the CPUs available, the engine settings and KV-cache size,
  the SHA-256 of the plan and the runner, and the exact command. The plan pins
  the model weights to a commit hash.
- **Held-out validity**: every efficiency in `plan.json` is a value the package
  already uses, and none is fitted here. Each manifest records the plan's SHA-256,
  and the analysis excludes runs measured under any other plan. `split_note` in
  the plan gives the split: the largest H100 Llama configuration and every A100
  configuration are held out. A later change that fits a parameter should use
  only `split: calibration` rows and report `heldout` rows separately.

## Boundary

NVML reports board power. It excludes the host CPU and DRAM, NICs, chassis
fans, power-supply losses, and facility overhead. The comparison therefore
uses `RunFootprintModel` at `it_ratio = 1` and `pue = 1` (accelerator energy
only). It does not validate the package's IT ratios or PUE values.

## What the summary reports

One row per configuration and phase, per unit of work (one prefill, one decode
step, one training step). Each row has measured time, energy and mean power
with 95% t intervals, idle power in watts and as a fraction of TDP
(`idle_over_rated`, which the proportional rule sets to 0.30), and four
estimates:

| Estimate | Device power | Time |
| --- | --- | --- |
| `rated_measured_time` | TDP | measured |
| `rated_modeled_time` | TDP | `ServingModel` or `SingleNodeModel` |
| `proportional_measured_time` | TDP × (0.30 + 0.70 × u) | measured |
| `proportional_modeled_time` | TDP × (0.30 + 0.70 × u) | modeled |

u is the phase's FLOPs, counted as the package counts them, divided by the time
and the peak FLOP rate (`utilization_at_measured_time`,
`utilization_at_modeled_time`). For training, `proportional_modeled_time` is
`SingleNodeModel`'s own energy. Each estimate's `_over_measured` ratio is its
error. The measured-time columns isolate the power assumption, and the
difference from the modeled-time columns is the time model's contribution.

## Which device an estimate would pick

`device_choice.csv` takes each configuration measured on both GPUs (same
workload, phase, batch, and token counts) and names the device with the lower
energy per unit of work by the meter (`measured_preferred`, with
`measured_clear` when its 95% interval lies wholly below the other device's,
and `measured_every_run_pair` when it also wins for every pairing of one run per
device) and by each of the four estimates, with `_agrees` when the estimate picks the
meter's device. At one site, carbon is energy times the same intensity, so the
energy choice is also the carbon choice. The comparison was declared in the
plan before any run.

## What the rows do not show for scheduling

They do not show whether a power error changes a carbon-aware scheduling decision. In
`TemporalShiftModel` with zero standby power and zero deferral energy, scaling
the run power scales every candidate start equally, so the chosen start cannot
change. That question needs nonzero standby or deferral terms, or a choice
between devices, and then the ratio of run power to standby power decides. If
standby is the idle draw of a reserved GPU, the proportional rule puts that
ratio at (0.30 + 0.70 × u) / 0.30. The measured run and idle powers give the
measured ratio, which a follow-up study can feed to the solver. Do not report a
changed decision until that study runs.

## Committing a run

Commit the run directory unchanged, then regenerate the summary and commit it.
The analysis verifies `trials.jsonl` against its manifest, so an edited trial
excludes the whole run, and so does any other file in the run directory (hidden
files such as `.DS_Store` are ignored). Each
run is two files because the paper's timing receipt hashes every `.json` and
`.csv` under `paper/data`: one file per trial would add about 150 entries per
run to it. The receipt hashes `manifest.json`, and the manifest hashes
`trials.jsonl`. `summary.provenance.json` also says whether each run was
measured with the committed runner. `analyze_gpu_energy.py --check` (part of
`make publication-check`) fails when the committed summary is stale. New files
under `paper/data` also change the paper's timing-receipt source hashes, which
`make values` in `paper/` refreshes.
