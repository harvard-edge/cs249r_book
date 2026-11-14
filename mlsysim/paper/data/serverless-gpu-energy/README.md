# GPU waiting costs and power on serverless GPUs

These tables are the per-run measurements behind
`mlsysim/engine/data/measured_gpu_energy.yaml`, which the opt-in module
`mlsysim.engine.measured_energy` reads. They cover two things the package
did not have measured values for:

- **What waiting costs a GPU job.** This is the idle power of a container
  that keeps its model loaded, the energy of a cold start, the energy of a
  keep-warm window, and the energy of pausing and resuming a training job.
  `standby_power(gpu)` returns the reserved idle power, which is the
  `standby_power` input of `TemporalShiftModel`. That solver charges it at the
  intensity of each waiting step. `coldstart_energy(gpu)` is a standalone
  value. A released GPU pays it when the job restarts, but the solver charges
  `deferral_energy` at the release step, so it is not a `deferral_energy`
  input.
- **How board power depends on compute and memory use.** These are 148 runs
  of seven workloads used to fit and cross-validate
  `measured_power(gpu, hfu, bwu)`.

All runs were made on 2026-11-12 on Modal serverless GPUs: T4, L4, A10,
A100-40GB, H100, H200 and B200. `../gpu-energy/` is a separate, declared
measurement of Llama-3.1-8B serving and ResNet-50 training on H100 and A100-80GB
boards. Its plan, raw trials and analysis are complete in this repository.

Rebuild the package table from these files and check it:

```sh
python paper/scripts/build_measured_gpu_energy.py          # write
python paper/scripts/build_measured_gpu_energy.py --check  # also in make publication-check
```

`tests/test_measured_energy.py` checks that the committed table equals what
the script builds.

## Files

| File | Rows | One row per |
|---|---|---|
| `idle_runs.csv` | 21 | idle run: a 600 s idle period after a W1 job, 3 per GPU |
| `coldstart_runs.csv` | 21 | cold start in a fresh container, 3 per GPU |
| `pause_steps.csv` | 150 | step of a pause and resume (steady, save, idle, load, resume), W3 and W5 on 5 GPUs, 3 repeats |
| `power_runs.csv` | 148 | 180 s measurement window of one workload at one batch size |
| `excluded_runs.csv` | 4 | complete run left out, with the reason |
| `failed_runs.csv` | 10 | failed, preempted or quarantined call, with the logged error |
| `runs_manifest.csv` | 254 | raw run behind any row above, with its device, software and file hashes |

**What can be reproduced from this repository.** The committed per-run
tables are reproducible into the package table. The original raw samples
are not independently replayable from this repository. The raw 100 ms NVML
and 1 s nvidia-smi samples and each run's `meta.json` (about 90 MB) are kept
with the study and are not published yet. `runs_manifest.csv` records, for
every run:

- the GPU class requested from Modal, and the GPU name, UUID, enforced power
  limit and driver version NVML reported;
- the torch and transformers versions and the Modal image id;
- the model revision and the sampling intervals;
- the SHA-256 of each raw file.

Once the archive is published, each raw file can be checked against these
hashes. Every row in the other tables has `meta_sha256`, the hash of the
`meta.json` that has the counter readings at each phase boundary.

The 250 runs used span 4 (T4) to 20 (B200) physical boards per GPU type,
identified by NVML UUID. Within each type, every board had the same enforced
power limit, and every board ran driver 610.57.04 except the T4s
(620.70.06). The vLLM version is not recorded per run. It is fixed by the
second image's lock file, at vLLM 0.10.1.1.

## How each value is measured

Energy is the difference of NVML's total-energy counter
(`nvmlDeviceGetTotalEnergyConsumption`) between two marks. That counter is an
unsigned 64-bit millijoule count since the driver loaded. The analysis stops
on any decrease, and none occurred. The 100 ms power samples were also
integrated (trapezoidal, interpolated at the marks) and checked against what
the container wrote, but every value here uses the counter. Board energy
only: host CPU, memory, networking, cooling and facility energy (PUE) are not
included.

- **Loaded and reserved idle power** (`idle_runs.csv`). After a W1 job
  (120 s warm-up, 180 s window), the container is kept alive for 600 s with the
  model loaded and no work. `idle_power_w` is counter energy over all 600 s,
  so the decay right after the job is included. `idle_steady_power_w` is the
  last 300 s. The package table uses the mean steady value as
  `loaded_idle_power_w` and the mean whole-period value as
  `reserved_idle_power_w`.
- **Keep-warm window** (`keepwarm_60s_energy_j`). This is the counter energy
  in the first 60 s of the idle period. 60 s is Modal's default `scaledown_window`.
- **Cold start** (`coldstart_runs.csv`). Each is a separate call with
  `single_use_containers=True`, so every call gets a fresh container. Energy
  runs from the first NVML read inside the function to the first generated
  token of Qwen2.5-1.5B: CUDA initialization, loading the model from a Modal
  Volume, and a one-token generation. GPU energy before the function body
  runs is not visible to NVML and is not counted. `crossover_wait_s` in the
  package table is cold-start energy over reserved idle power. It is the
  wait at which keeping the container costs as much GPU energy as releasing
  it and starting again.
- **Pause** (`pause_steps.csv`). Container A trains for 120 s, measures 60 s
  of steady training, saves a checkpoint (model, optimizer, GradScaler and
  RNG state) to a Modal Volume, idles 60 s and exits. A fresh container B
  loads the checkpoint and trains 60 s. The pause overhead is save plus load
  plus the resume's energy above steady training at the same step count,
  floored at zero. W3 checkpoints include only the LoRA adapters and
  optimizer state.
- **Power runs** (`power_runs.csv`). Each run has a 120 s warm-up and then
  whole work units until 180 s have passed. `measured_power_w` is counter
  energy over the window.

## Workloads

| Id | Workload | Batch |
|---|---|---|
| W1 | Qwen2.5-1.5B-Instruct generation (transformers, greedy, exactly 128 new tokens) | 1 |
| W2 | the same | 32 |
| W1b | Qwen2.5-7B-Instruct generation, as W1 | 1 |
| W3 | Qwen2.5-1.5B LoRA fine-tuning (r 16, alpha 32, q/k/v/o, 512 tokens) | 2 |
| W4 | ResNet-50 training from random initialization | 128 |
| W5 | GPT-2-small-shaped training from random initialization (1024 tokens) | 4 |
| W6 | Qwen2.5-7B-Instruct offline batch inference in vLLM 0.10.1.1 (128 output tokens) | 256 requests |

All workloads ran in fp16 (autocast with a GradScaler for training). The
models are pinned by commit: Qwen2.5-1.5B-Instruct at `989aa798` and
Qwen2.5-7B-Instruct at `a09a3545`, both Apache-2.0. The 32 prompts are fixed.
T4 to H200 ran torch 2.5.1 and transformers 4.46.3. The B200 and W6 need a
newer stack, so they ran torch 2.7.1 (CUDA 12.8), transformers 4.55.2 and
vLLM 0.10.1.1. The software stack matters. Three H100 runs each of W1 and W4
were repeated in the second image. At the same throughput, energy per unit
was 9.4% lower for W1 and 6.0% lower for W4, so comparisons between the two
images include a stack effect of that size. W6 is not on the T4, because
vLLM 0.10's engine needs compute capability 8.0 or newer.

## Which runs are used

- **One GPU type per row, by NVML name.** Modal's A10 class gave an
  `NVIDIA A10` (150 W limit) for most runs and an `NVIDIA A10G` (300 W limit)
  for three. Each GPU class keeps the NVML name that ran most of its runs,
  and the A10G runs are listed in `excluded_runs.csv`. The H100 was requested
  as `H100!`, so Modal could not substitute an H200.
- **A run key Modal ran twice.** The first-started run is used, a rule fixed
  before either run's numbers were read.
- **Failed calls** (`failed_runs.csv`) left no partial data. Each run's files
  were written to local disk and copied to the Volume only when complete.
  The list includes two vLLM runs that failed because an earlier run's engine
  still had the GPU's memory. After those failures, every W6 run got its own
  container, and one stack-check run that had shared a container with a W6
  run was quarantined.

## The power model

`hfu` and `bwu` are MLSys·im's own roofline quantities scaled to the measured
throughput. `hfu` is `Engine.solve`'s HFU times measured over predicted
throughput. `bwu` is `Engine.solve`'s memory time per batch times the measured
batches per second, capped at 1. `engine_power_w` is `Engine.solve`'s power
as shipped: TDP x (0.30 + 0.70 x HFU) at the default efficiency 0.5. All
three were computed with MLSys·im at commit `94d14b58`. The only package
difference from this branch's base is that the base adds FP32 and INT4 rates
to the T4 entry, which these fp16 runs do not use. The L4 and A10 were not
in the registry then. For them, the study used the same datasheet values as
the registry entries this change adds.

The model is P = min(TDP, P_idle + (b x HFU + c x BWU) x TDP), with P_idle
the GPU's measured loaded-idle power. b and c are fitted by least squares on
a fixed grid over all 148 runs, which gives b = 5.05 and c = 0.65.

Within this dataset, refitting b and c with each GPU left out gives a 11.4%
median absolute power error, and with each workload left out, 14.2%. These
errors are conditional on two measured inputs: each GPU's idle power, and
HFU and BWU scaled by measured throughput. They are not a prospective
accuracy. With HFU and BWU from a roofline prediction, the prediction's
throughput error adds to the power error. On the same 148 runs,
`Engine.solve`'s shipped power has a 35.8% median error. That figure applies
to these runs only.

This analysis is exploratory. The form was chosen after looking at the first
runs, because memory-bound decode drew up to the power limit at HFU below
1%, so the held-out errors are optimistic about that choice. The H200, B200
and W6 runs were measured after the form was fixed. They are a later check,
not a preregistered test. A test on a different serving engine and model is
still to be done. `../gpu-energy/` has such data for the H100, but it counts
memory traffic through a different solver path, so the two utilization
definitions must be reconciled first.

## Findings the table encodes

- Loaded idle power is 14% (A100-40GB) to 45% (L4) of TDP, against the 30%
  Engine.solve assumes. It is 17.5% on the H100. `../gpu-energy/` measured
  21% on H100 and A100-80GB boards, with vLLM's engine and its pre-allocated
  KV cache resident.
- A cold start costs as much GPU energy as 7.3 to 11.6 s of keeping the
  container warm (`crossover_wait_s`). Modal's default 60 s keep-warm window
  costs 5.2 to 8.5 times a cold start's GPU energy.
- Pausing a training job costs 0.4 kJ (W3 on the L4) to 6.8 kJ (W5 on the
  B200).
