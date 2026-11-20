# GPU board energy with two seeds on the same board

This is a fourth declared test of MLSys·im's two device-power assumptions,
rated TDP and `Engine.solve`'s energy-proportional rule. It measures
Llama-3.1-8B serving and ResNet-50 training with the configurations of
`../gpu-energy-boards/`, twice on each board: once with seed 2193 and once
with seed 2198. The runner draws the random prompt tokens from the seed alone,
so the two runs feed the GPU different prompts. For ResNet-50 training the seed
also sets the initial weights and the synthetic data.

The test was declared after a comparison in
`../gpu-energy-boards/board_energy.csv`. Measured the same day, the seed-2198
runs of `../gpu-energy-workloads/` used 1.2–4.1% less decode energy on average
than the seed-2193 runs of `../gpu-energy-boards/`, in every decode
configuration on all three GPUs. Those runs were on different boards, so the
comparison could not tell the seed from the boards. Here each container
measures both seeds on one GPU, one after the other.

| File | Role |
| --- | --- |
| `seed-2193/plan.json`, `seed-2198/plan.json` | The two plans, which differ only in seed, with the four questions. Both were written before any run, and every run manifest records its plan's SHA-256. |
| `seed-2193/runs/`, `seed-2198/runs/` | One run per container in each folder, under the same name. |
| `seed-*/summary.csv`, `seed-*/summary.provenance.json`, `seed-*/device_choice.csv` | Written from `mlsysim/` by `paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-seeds/seed-2193` (and `seed-2198`). |
| `seed_pairs.csv` | Written by `paper/scripts/compare_gpu_energy_seeds.py`: each pair's energy per unit of work with both seeds, and the ratio of seed 2198's to seed 2193's. |
| `seed_summary.csv` | Written by the same script: the comparisons the plans declare, one row per GPU and configuration. |

`make publication-check` runs all three with `--check`.

## Results (measured 2026-11-19)

**Runs.** 12 pairs are used, four per GPU, each on a different board that
appears in none of the earlier tests. Every configuration has 40 measured
trials per seed (10 per run, after one warm-up), and none failed. On each GPU,
two containers measured seed 2193 first and two measured seed 2198 first.
Every A100 used is an `A100-SXM4-80GB` with a 400 W limit. One H100 and one H200
report NVIDIA driver 620.70.06, and the other ten boards 610.57.04.

Two more things happened:
- One container was served an `A100 80GB PCIe`. The runner refused to measure
  with either seed, so both refusals are kept and excluded. A replacement
  container then measured both seeds, seed 2193 first like the one it
  replaced.
- Modal preempted the container of `h200-20261119T220545Z-4` while it was
  measuring, and at 22:38 UTC started the call again in a new container,
  which measured both seeds from the start. The launcher returns only the
  files of the container that finished, so every pair comes from one
  container, and `compare_gpu_energy_seeds.py` checks that both runs of each
  pair report the same GPU.

The plans declared four questions. The answers:

1. **The seed.** On the same board, seed 2198's decode energy over seed
   2193's is 0.990–1.008 on average for each GPU and configuration, and
   0.961–1.032 for single boards. 64 of the 72 decode pairs are within 2%,
   and seed 2198 is lower in 31 of the 72. In prefill the averages are
   0.991–1.013, and in ResNet-50 training 0.991–0.998.
2. **The order.** Averaged over each container's six decode configurations,
   the second run used 0.4% less decode energy than the first (standard error
   0.2% over the 12 containers), and less in 9 of the 12. Most of that comes
   from one container, the replacement A100, at 2.3%. The other 11 average
   0.2%.
3. **Boards in one session.** With the same prompts on the same day, the four
   boards of each GPU differ in decode energy by 1.2–8.8% on the H100,
   13.4–19.5% on the A100 and 4.7–10.1% on the H200. On the A100 most of the
   difference is in power: at batch 1 with 128-token prompts, its four boards
   drew 283–332 W, while their step times differed by 8%. The A100 boards keep
   the same order by energy with both seeds in all six decode configurations,
   and the H100 and H200 boards, which are closer together, in two and three of
   six.
4. **The gap between the earlier tests.** Averaged over the six decode
   configurations, the seed changes energy by +0.0% on the H100, −0.3% on the
   A100 and +0.1% on the H200, while the same-day gap is 2.4%, 3.8% and 2.2%.
   So the seed explains little of the gap. Most of it fits the differences
   between containers, whose boards differ by up to 19.5% within one session.
   In a few configurations the seed is a larger share: on the H100 at batch 8
   with 2,048-token prompts, the gap is 1.4% and the seed ratio 0.990.

The answers of `../gpu-energy-workloads/` repeat with both seeds. Decode draws
65–84% of TDP at 0.2–6% FLOP utilization, and TDP over the modeled time gives
0.72–1.05 of the measured decode energy. In all six decode comparisons the
H200 uses less energy on every pairing of one board per GPU, and all four
estimates pick it.

The driver pattern that `../gpu-energy-boards/` reports does not repeat here.
With seed 2193, the H100 board on 620.70.06 used 1.1–4.8% more decode energy
than the average of the three on 610.57.04, and the H200 board on 620.70.06
used 0.9–2.6% more.

These results cover four boards per GPU, one engine (vLLM 0.6.4.post1), one
language model and ResNet-50 at one batch size, at the board boundary only.

## Running the measurements

From `mlsysim/`, after `modal setup`:

```sh
GPU_ENERGY_DATA=gpu-energy-seeds/seed-2193 GPU_ENERGY_PAIR=gpu-energy-seeds/seed-2198 \
    modal run --detach paper/scripts/modal_gpu_energy.py --repeats 4
PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-seeds/seed-2193
PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-seeds/seed-2198
PYTHONPATH=. python paper/scripts/compare_gpu_energy_seeds.py
```

The committed runs came from the first command on 2026-11-19, after both plans
were written, and from the same command with `--devices A100 --repeats 1`
for the replacement container. The protocol, the boundary and the summary
columns are described in `../gpu-energy/README.md`.
