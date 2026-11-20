# GPU board energy on more boards

This is a third declared test of MLSys·im's two device-power assumptions,
rated TDP and `Engine.solve`'s energy-proportional rule. It measures
Llama-3.1-8B serving and ResNet-50 training on four more boards each of the
H100, A100-SXM4-80GB and H200. The configurations, engine settings and
efficiencies are those of `../gpu-energy-workloads/`, and the seed is 2193, the
seed of `../gpu-energy/`. The runner draws the random prompt tokens from the
seed alone, so these runs and `../gpu-energy/` feed the GPU the same prompts.

`../gpu-energy-workloads/` found that both of its H100 boards used less decode
energy than both H100 boards of `../gpu-energy/`, and that the two tests also
differ in seed. This test asks how much energy differs between boards of one
GPU, and whether the boards or the seed caused that change.
`../gpu-energy-seeds/` answers the seed part more directly, by measuring both
seeds on the same boards.

| File | Role |
| --- | --- |
| `plan.json` | The configuration matrix and the four questions, declared before any run. |
| `runs/<run-id>/` | One measurement session on one GPU, as in `../gpu-energy/`. 12 runs, all used. |
| `summary.csv`, `summary.provenance.json`, `device_choice.csv` | Written from `mlsysim/` by `paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-boards`. |
| `board_energy.csv` | Written by `paper/scripts/compare_gpu_energy_boards.py`: each verified run's mean energy and time per unit of work, from all three tests, for the configurations of this plan. |
| `board_summary.csv` | Written by the same script: the comparisons the plan declares, one row per configuration. |

`make publication-check` runs both scripts with `--check`.

## Results (measured 2026-11-19)

**Runs.** All 12 runs were used, four per GPU. Each ran on a different
physical board (by NVML UUID), and none of the 12 boards appears in the other
two tests. Every configuration has 40 measured trials (10 per run, after one
warm-up), and none failed. Every A100 is an `A100-SXM4-80GB` with a 400 W
limit. Two H100 and two H200 runs report NVIDIA driver 620.70.06, and the
other eight runs report 610.57.04. `board_energy.csv` gives each run's board,
power limit and driver.

Figures below come from `board_summary.csv` and `board_energy.csv`, and for
question 4 and idle power from `summary.csv` and `device_choice.csv`. A spread
is the highest run's energy per unit of work over the lowest run's, minus one.
Each run is on its own board (by NVML UUID). The manifests do not identify the
host, but they show several cloud providers and regions.

The plan declared four questions. The answers:

1. **Spread between boards.** With the same prompts on the same day, the four
   boards of this test differ by:
   - decode: 2.7–5.8% on the H100, 6.0–7.4% on the A100 and 13.1–15.6% on the
     H200.
   - prefill above one 128-token prompt: 1.4–4.6% on the H100, 1.7–2.4% on
     the A100 and 3.4–5.9% on the H200. For one 128-token prompt, where the
     engine's work on the host is a large part of each prefill, the time per
     prefill differs by 81% between H100 runs (16.6–30.0 ms) and by 43%
     between H200 runs, and the energy by 18.5% and 27%.
   - ResNet-50 training: 3.6% on the H100, 2.7% on the A100 and 11.8% on the
     H200.

   With the boards of `../gpu-energy/`, measured a week earlier with the same
   prompts, the decode spread is 7.7–10.9% on six H100 boards and 10.3–11.3%
   on six A100 boards. Its 500 W A100 board is the lowest A100 board in every
   decode configuration.
2. **The seed.** On the A100 and the H200, every run of
   `../gpu-energy-workloads/` (seed 2198) falls inside the range of the runs
   measured with seed 2193. That range includes the 500 W board on the A100.
   On the H100, 12 of the 26 seed-2198 run values fall below it, by 0.1–2.1%,
   and none above. Measured the same day, the seed-2198 runs average 1.2–4.1%
   below this test's runs in every decode configuration on all three GPUs.
   Two boards per GPU cannot separate that from the boards drawn, so
   `../gpu-energy-seeds/` measured both seeds on the same boards. There the
   seed changes decode energy by less than 1% on average, so most of the gap
   comes from differences between containers, such as their boards.

   On the H100, the two boards of `../gpu-energy/` are the highest two of all
   eight in every decode configuration and in training, and the four boards
   of this test are 1.3–9.8% below them in decode with the same prompts. Those
   runs also differ in date, container image and the order of configurations
   within a run (see `../gpu-energy-workloads/README.md`), so this test cannot
   say whether the boards or those differences made them high.
3. **H100 against A100 in Llama-3.1-8B decode.** Pairings of one H100 board
   and one A100 board on which the H100 uses less energy:

   | Batch, prompt tokens | This test (4 × 4 boards) | All three tests (8 × 8 boards) |
   | --- | --- | --- |
   | 1, 128 | 15 of 16 | 47 of 64 |
   | 1, 2,048 | 15 of 16 | 48 of 64 |
   | 8, 128 | 10 of 16 | 25 of 64 |
   | 8, 2,048 | 10 of 16 | 26 of 64 |
   | 16, 128 | 13 of 16 | 31 of 64 |
   | 16, 2,048 | 16 of 16 | 58 of 64 |

   No decode configuration has one of the two GPUs lower on all 64 pairings.
   At batch 8, and at batch 16 with 128-token prompts, the split is close to
   even. In ResNet-50 training the H100 uses less on all 64 pairings.
4. **The earlier answers on these boards.** For Llama-3.1-8B decode they
   repeat those of `../gpu-energy-workloads/`:
   - Decode draws 67–86% of TDP at 0.2–6% FLOP utilization. Over the measured
     time, the proportional rule gives 0.38–0.46 of the measured decode
     energy, and TDP gives 1.17–1.50.
   - `ServingModel`'s decode step is 0.55–0.69 of the measured one. TDP over
     the modeled time gives 0.71–1.02, and in 16 of 18 decode rows it is
     closer to 1, by absolute difference, than either power assumption over
     the measured time (16 of 18 for Llama-3.1-8B in
     `../gpu-energy-workloads/`).
   - In all six decode comparisons the H200 uses less energy on every pairing
     of one board per GPU, and all four estimates pick it. In ResNet-50
     training the H200 also wins on every pairing. TDP over the modeled time
     picks the A100 there, and the proportional rule over the modeled time
     picks the H100 on an exact tie with the H200.

   Prefill does not repeat. Here no GPU wins on every pairing in prefill. The
   H200 is lowest on average, and the next GPU uses 0.4–6.8% more. In
   `../gpu-energy-workloads/` the H200 won Llama-3.1-8B prefill on every pairing
   at batch 1 and batch 8 with 128-token prompts. In `../gpu-energy-seeds/`, on
   four other boards per GPU, it wins five of six prefill comparisons on every
   pairing with seed 2193, and four of six with seed 2198. TDP over the measured
   time picks a GPU other than the lowest on average in three of the six prefill
   comparisons here: the A100 at batch 1 with 128-token prompts, and the H100 at
   batch 8 and batch 16 with 2,048-token prompts. Both modeled-time estimates
   give the H100 and the H200 the same prefill energy.

**Idle** power with the workload resident is 18.5–22% of TDP.

**Not a declared question: the driver.** In this test, both H100 boards on
NVIDIA driver 620.70.06 used less decode energy than both on 610.57.04 in five
of six configurations (2.7–4.1% less on average, and 0.1% in the sixth). Both
H200 boards on 620.70.06 used 6.4–7.7% less in all six. The pattern does not
hold in the other tests. In `../gpu-energy-workloads/` the H200 board on
620.70.06 used 5.8–6.4% more than the one on 610.57.04, and the two H100
boards split three and three. In `../gpu-energy-seeds/` the boards on
620.70.06 used 1–5% more. So the pattern here fits the boards drawn.

These results cover four to eight boards per GPU, one engine (vLLM
0.6.4.post1), one language model and ResNet-50 at one batch size, at the board
boundary only.

## Running the measurements

From `mlsysim/`, after `modal setup`:

```sh
GPU_ENERGY_DATA=gpu-energy-boards modal run --detach paper/scripts/modal_gpu_energy.py --repeats 4
PYTHONPATH=. python paper/scripts/analyze_gpu_energy.py --root paper/data/gpu-energy-boards
PYTHONPATH=. python paper/scripts/compare_gpu_energy_boards.py
```

The committed runs came from the first command on 2026-11-19, after the plan was
written. Every run manifest records the plan's SHA-256, which matches
`plan.json`. The runner and protocol are those of `../gpu-energy-workloads/`,
and the launcher includes the changes its README lists after its last run.
`compare_gpu_energy_boards.py` was changed after these runs, to match whole
workload entries and to report the runs it excludes. Its output on these data is
the same as that of the version written before the runs. The protocol, the
boundary and the summary columns are described in `../gpu-energy/README.md`.
