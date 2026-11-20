# GPU board energy on more workloads and a third GPU

This is a second declared test of MLSys·im's two device-power assumptions: rated
TDP, and `Engine.solve`'s energy-proportional rule, TDP × (0.30 + 0.70 ×
utilization). It uses the runner, protocol, trial counts, engine settings and
analysis of `../gpu-energy/`, whose README documents them. It adds Llama-2-7B
and Llama-2-13B, which use multi-head attention (one key-value head per query
head, where Llama-3.1-8B uses grouped-query attention), and Mixtral-8x7B, a
mixture-of-experts model that runs in bf16, its native format. It adds the H200
as a third GPU and measures Llama-3.1-8B and ResNet-50 training again on new
boards.

Apart from the new models and GPU, these runs differ from `../gpu-energy/` in
six ways:

- The seed is 2198 (it was 2193), so the random prompt tokens differ. The seed
  has nothing to do with the number of the pull request that added
  `../gpu-energy/`.
- Every configuration is held out, where the first plan marked six H100
  configurations as calibration rows.
- The configuration ids name the model, so `h100-llama-b1-p128` there is
  `h100-llama31-8b-b1-p128` here.
- A run measures its configurations in plan order. Here that goes through both
  prompt lengths at each batch size (batch 1 with 128 tokens, batch 1 with
  2,048, batch 8 with 128, and so on). `../gpu-energy/` went through the batch
  sizes at each prompt length.
- The container image was built again. Its base image had moved on, from
  Python 3.11.12 and glibc 2.36 to Python 3.11.18 and glibc 2.41, and 29
  unpinned packages (web and serialization libraries such as aiohttp, protobuf
  and pydantic) have other versions. vLLM, PyTorch, torchvision, Transformers,
  nvidia-ml-py, CUDA 12.4 and cuDNN are the same.
- Six of the 20 used runs report NVIDIA driver 620.70.06. Every run of
  `../gpu-energy/` reports 610.57.04.

The efficiencies are the `engine/empirical.py` anchors that `../gpu-energy/`
used: 0.5 for every language model (the Llama-3-8B decode anchor), and 0.13 and
0.26 for ResNet-50 training on the H100 and the A100. The H200 has no anchor, so
it uses the H100's 0.13, since the registry gives both the same 989 TFLOP/s
dense FP16 peak.

| File | Role |
| --- | --- |
| `plan.json` | The configuration matrix, declared before any run, with every efficiency and the questions the analysis answers. |
| `pilot-plan.json` | `plan.json` with two trials per configuration and a note saying so. The pilot runs used it. |
| `runs/<run-id>/` | One measurement session on one GPU, as in `../gpu-energy/`: `trials.jsonl` and a `manifest.json` that hashes it. 28 runs are committed. 20 are used and 8 are excluded, each with its reason in `summary.provenance.json`. |
| `summary.csv`, `summary.provenance.json`, `device_choice.csv` | Written from `mlsysim/` by `paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-workloads`. `make publication-check` runs the same command with `--check`. |

## Results (measured 2026-11-19)

**Runs.** 20 runs are used, two per model and GPU, and each of the 20 ran on
a different physical board (by NVML UUID). Every configuration has 20
measured trials (10 per run, after one warm-up), and none failed. One A100
board reports a 500 W power limit and the other five 400 W. `power_limits_w`
gives each row's limits, and only the A100 Llama-2-7B rows show 400/500. Six of
the 20 runs report NVIDIA driver 620.70.06. The other 14, like every run of
`../gpu-energy/`, report 610.57.04. Eight runs are kept and excluded:

- one container that Modal served with an `A100 80GB PCIe`, which the runner
  refused to measure;
- seven pilot runs, measured under `pilot-plan.json` (SHA-256
  `36f9d1c8864ebd9229a40a199e0fdf348cf4cd4549f59315f007a7bad6f88163`). The
  first pilot (`pilot-…`) measured Llama-2-7B on the H100, and Mixtral inference
  and ResNet-50 training on the H200. Its Llama-2-13B trials on the A100 and the
  H200 failed (see below). The second pilot (`pilot2-…`) measured Llama-2-13B on
  all three GPUs.

Figures below are ranges over the configurations in `summary.csv`. Energy
ratios are estimate over measured energy per unit of work.

The plan declared four questions. The answers:

1. **Decode power at low utilization.** For every model on every GPU, decode
   draws 65–98% of TDP while its FLOP utilization is 0.2–6%. Over the measured
   time, the proportional rule gives 0.32–0.46 of the measured decode energy,
   and TDP gives 1.02–1.54.
2. **Offsetting errors.** For the three dense models, `ServingModel`'s decode
   step is 0.54–0.74 of the measured one (0.63–0.74 on the H100, 0.65–0.74 on
   the A100, 0.54–0.67 on the H200). In all 54 dense decode rows the modeled
   step is shorter than the measured one and TDP is above the measured power.
   TDP over the modeled time gives 0.66–1.06 of the measured decode energy. In
   40 of the 54 rows it is closer to 1, by absolute difference, than either
   power assumption over the measured time: 16 of 18 on the H100, 13 of 18 on
   the A100 and 11 of 18 on the H200 (11 of 12 in `../gpu-energy/`). The 14
   other rows are at batch 16, or at batch 8 with 2,048-token prompts.
3. **Device choice.** `device_choice.csv` has 37 comparisons: prefill and
   decode for the six configurations of each dense model on three GPUs, plus
   ResNet-50 training. Over all 37, TDP over the measured time picks the
   meter's device 35 times, the proportional rule over the measured time 36
   times, and each modeled-time estimate 20 times. In 30 of the 37 the same
   device wins on every pairing of one board per GPU, and in all 30 it is the
   H200. The other seven are prefill comparisons where the H100 and the H200
   are within 0.17–1.8% of each other. On the 30:
   - TDP over the measured time picks the meter's device 29 times. The
     exception is Llama-2-7B prefill at batch 1 with a 128-token prompt,
     where it picks the A100.
   - The proportional rule over the measured time picks it all 30 times.
   - Each modeled-time estimate picks another device 12 times. For the
     proportional rule, all 12 are exact ties between the H100 and the H200,
     which the analysis breaks toward the H100, listed first in the plan. For
     TDP, 11 are such ties. The twelfth is ResNet-50 training, where it picks
     the A100, as it did in `../gpu-energy/`. The ties come from the model. Both
     modeled-time estimates give the H100 and the H200 the same energy in all 18
     prefill configurations and in training. `ServingModel`'s prefill time has
     no memory term: it is the FLOPs divided by the product of peak FLOP/s and
     the efficiency, plus a dispatch overhead. `SingleNodeModel` rates ResNet-50
     training compute-bound on both GPUs. The registry gives the two GPUs the
     same peak FLOP/s, dispatch overhead and TDP, and the plan gives them the
     same efficiencies. In the 12 comparisons that the modeled-time estimates
     miss, the meter shows the H200 using 1.6–12% less energy than the H100.
4. **Mixtral-8x7B** (H200 only). The pattern depends on the batch size:
   - At batch 1, the modeled decode step is 1.70–1.71 of the measured one.
     `decode_step` reads all resident weights for a mixture-of-experts model
     (its docstring calls this conservative). Reading all 93.4 GB of weights
     in the measured step of 11.6 ms would take 8.0 TB/s, more than the H200's
     4.8 TB/s, so the engine reads less than all of them, as it would if it
     reads only the experts each token selects.
     The time and power errors compound, and TDP over the modeled time gives
     2.42–2.48 of the measured decode energy.
   - At batch 8 and 16, the modeled step is 0.70–0.96 of the measured one,
     and TDP over the modeled time gives 0.71–1.01, close to the dense
     models' 0.66–1.06.
   - The modeled prefill is 1.39–1.97 of the measured one above one 128-token
     prompt, and 0.52 at one 128-token prompt. `ServingModel` counts prefill
     FLOPs over all 46.7B parameters, 3.6 times the 12.9B active ones. The
     analysis counts utilization with `decode_step`, which uses the active
     parameters, so for Mixtral prefill the two FLOP counts differ. For the
     dense models they are the same.

**Prefill.** For the dense models above one 128-token prompt, prefill draws
96–99.5% of TDP on the H100 and 97–99% on the H200. On the A100 it draws
96–98%, apart from the Llama-2-7B rows. They are the only rows that include the
500 W board, and they draw 104–109% of the 400 W TDP. TDP over the measured time
gives 0.92–1.04 of the measured energy, as in `../gpu-energy/`.

**Idle** power with the workload resident is 18–21% of TDP on every GPU. The
proportional rule assumes 30%.

**Repeated configurations.** Llama-3.1-8B and ResNet-50 training were measured
again on new boards, with the new seed. The ratios below compare the two
`summary.csv` files. The pairing counts use each run's mean energy from
`trials.jsonl`, as `measured_every_run_pair` does.

- **A100.** The earlier pair of boards had 400 W and 500 W limits, and both
  new boards have 400 W. Energy per unit of work is within 1% of the earlier
  measurement for decode and training. For prefill it is 0.93–1.03 of it, and
  the time is 3–6% longer. That fits the first test, where the 500 W board
  ran large prefill faster for more energy.
- **H100.** Energy per unit of work is 0.91–0.98 of the earlier measurement.
  The time is 2–3% shorter for Llama-3.1-8B and 0.6% shorter for training.

For Llama-3.1-8B decode, the H100's energy relative to the A100's is 5–9% lower
than in `../gpu-energy/` in all six configurations. Three comparisons change
sides. In `../gpu-energy/` the H100 used 3.7–7.8% more energy than the A100,
and here:

- batch 1 with 128-token prompts: the A100 was lower on all four board
  pairings there, and the H100 is lower on all four here;
- batch 1 with 2,048-token prompts: the A100 was lower on three of four
  pairings there, and the H100 is lower on all four here;
- batch 16 with 128-token prompts: the A100 was lower on all four pairings
  there. Here the H100 is lower on two of four, and 0.6% lower on average.

At batch 8 the A100 is still lower on average, by 0.1% with 128-token prompts
and 1.0% with 2,048-token prompts, and the H100 is lower on two of four
pairings in each, where it was lower on none.

Within each test, the two H100 boards differ by 0.1–3.9% in Llama-3.1-8B
decode energy, but both new boards are 4.5–10% below both earlier ones. The two
tests also differ in seed, and one new H100 Llama-3.1-8B run used the newer
driver, so these two tests alone cannot say what caused the change.
`../gpu-energy-boards/` and `../gpu-energy-seeds/` measured more boards to find
out. On the same board, the seed changes decode energy by less than 1% on
average. The two H100 boards of `../gpu-energy/` used the most decode energy of
all 16 H100 runs in the four tests, in every configuration, and those runs also
differ from the later ones in date, container image and configuration order.
Over the eight H100 and eight A100 boards of the first three tests, no
Llama-3.1-8B decode configuration has the H100 or the A100 lower on every
pairing.

These results cover two boards per model and GPU, one engine (vLLM
0.6.4.post1), language models of 7B to 47B parameters on one GPU, Mixtral on
the H200 only, and ResNet-50 at one batch size, at the board boundary only. The
data do not test a scheduling decision.

## Changes made after measuring began

The plan, the runner and the protocol did not change. The launcher,
`paper/scripts/modal_gpu_energy.py`, had three fixes during the measurements:

- **Tokenizer.** In the first pilot, every Llama-2-13B trial failed because
  `NousResearch/Llama-2-13b-hf` has no `tokenizer.json` and the download step
  skipped its `tokenizer.model`. The download step now fetches
  `tokenizer.model` too, and the second pilot ran Llama-2-13B on all three
  GPUs.
- **Copies in the volume.** Each run also saves its files in the Modal volume,
  so a lost connection to the local client does not lose a run.
- **One container per run.** In the matrix launch, Modal ran queued runs in
  containers that had finished a run, and for each the launcher returned the
  files the container still held. On the H200, five run folders held the same
  Llama-2-7B run. On the A100, six folders held the same refusal from the one
  container that was served a PCIe card. These copies, four on the H200 and
  five on the A100, were removed before the analysis. GPU functions now use
  single-use containers, and each call writes to a fresh folder.

After the matrix launch, the A100 runs were launched again, and the four H200
runs whose folders held copies were measured again with the fixed launcher. The
second A100 launch ran before the single-use fix. Its six runs have six
different Modal task ids, six different boards, and commands that match their
folders, so none of them is a copy. That launch gave two runs on SXM4 boards for
every model.

After the last run, these launcher changes followed. None of them changes a
measurement.

- In a plan with more than one language model, every run folder now names its
  model.
- A failed copy to the volume no longer loses the run.
- The volume copy keeps the runner's exit code beside the run folder. Inside
  the folder it would make a run copied back from the volume fail
  verification. No committed run was copied back.
- The runner gets its configurations in plan order, as in `../gpu-energy/`.
  The committed runs listed training first. The runner measures training first
  either way, so only the command recorded in the manifest differs.

## Running the measurements

From `mlsysim/`, after `modal setup`:

```sh
GPU_ENERGY_DATA=gpu-energy-workloads modal run --detach paper/scripts/modal_gpu_energy.py
PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-workloads
```

Each language model gets its own run and container on each GPU, and ResNet-50
training joins the Llama-3.1-8B run. The committed runs came from these
launches on 2026-11-19:

```sh
# pilots, from a folder gpu-energy-workloads-pilot that held pilot-plan.json as plan.json
GPU_ENERGY_DATA=gpu-energy-workloads-pilot modal run paper/scripts/modal_gpu_energy.py --devices H100,A100,H200 --repeats 1 \
    --configs h200-mixtral-8x7b-b16-p2048,h200-llama2-13b-b16-p2048,h200-resnet50-train-b256,a100-llama2-13b-b16-p2048,h100-llama2-7b-b16-p2048 --label pilot
GPU_ENERGY_DATA=gpu-energy-workloads-pilot modal run paper/scripts/modal_gpu_energy.py --devices H100,A100,H200 --repeats 1 \
    --configs h100-llama2-13b-b16-p2048,a100-llama2-13b-b16-p2048,h200-llama2-13b-b16-p2048 --label pilot2
# the matrix, then the A100 runs again, then the four H200 runs again
GPU_ENERGY_DATA=gpu-energy-workloads modal run --detach paper/scripts/modal_gpu_energy.py
GPU_ENERGY_DATA=gpu-energy-workloads modal run --detach paper/scripts/modal_gpu_energy.py --devices A100
GPU_ENERGY_DATA=gpu-energy-workloads modal run --detach paper/scripts/modal_gpu_energy.py --devices H200 --repeats 1
```

The pilot runs were then moved into `runs/`. With the current launcher, every
run folder of this plan names its model, so `pilot-h100-…` would now be
`pilot-h100-llama2-7b-…`. The protocol, the boundary, the summary columns and
the device comparison are described in `../gpu-energy/README.md`.
