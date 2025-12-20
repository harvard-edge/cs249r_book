# TinyTorch Milestones

Milestones are capstone experiences that bring together everything you've built in the TinyTorch modules. Each milestone recreates a pivotal moment in ML history using YOUR implementations.

## How Milestones Work

After completing a set of modules, you unlock the ability to run a milestone. Each milestone:

1. **Uses YOUR code** - Every tensor operation, gradient computation, and layer runs on code YOU wrote
2. **Recreates history** - Experience the same breakthroughs researchers achieved decades ago
3. **Proves understanding** - If it works, you truly understand how these systems function

## Available Milestones

| ID | Name | Year | Required Modules | What You'll Do |
|----|------|------|------------------|----------------|
| 01 | Perceptron | 1957 | Part 1: 01-04, Part 2: 01-08 | Build Rosenblatt's first neural network |
| 02 | XOR Crisis | 1969 | Part 1: 01-03, Part 2: 01-08 | Experience and solve the XOR impossibility |
| 03 | MLP Revival | 1986 | 01-08 | Train MLPs on TinyDigits with backpropagation |
| 04 | CNN Revolution | 1998 | 01-09 | Build LeNet for image recognition |
| 05 | Transformer Era | 2017 | 01-13 | Build attention and generate text |
| 06 | MLPerf Benchmarks | 2018 | 01-19 | Optimize and benchmark your neural networks |

## Running Milestones

```bash
# List available milestones and your progress
tito milestone list

# Run a specific milestone (all parts)
tito milestone run 03

# Run a specific part of a multi-part milestone
tito milestone run 03 --part 1  # Part 1: XOR Solved
tito milestone run 03 --part 2  # Part 2: TinyDigits

# Get detailed info about a milestone
tito milestone info 05
```

## Directory Structure

```
milestones/
├── 01_1957_perceptron/     # Milestone 01: Rosenblatt's Perceptron
├── 02_1969_xor/            # Milestone 02: XOR Problem
├── 03_1986_mlp/            # Milestone 03: Backpropagation MLP
├── 04_1998_cnn/            # Milestone 04: LeNet CNN
├── 05_2017_transformer/    # Milestone 05: Attention Mechanism
├── 06_2018_mlperf/         # Milestone 06: Optimization Olympics
├── extras/                 # Additional demos and variants (see extras/README.md)
└── data_manager.py         # Shared dataset management utility
```

## The Journey

```
Module 01-03          Module 04-06           Module 08-09
    │                     │                      │
    ▼                     ▼                      ▼
┌─────────┐         ┌─────────┐            ┌─────────┐
│ MS 01   │ ──────► │ MS 02   │ ─────────► │ MS 03   │
│ 1957    │         │ 1969    │            │ 1986    │
│ Forward │         │ XOR     │            │ Backprop│
└─────────┘         └─────────┘            └─────────┘
                                                │
                    Module 11-13                │  Module 09
                        │                       ▼      │
                        ▼                 ┌─────────┐  │
                  ┌──────────┐            │ MS 04   │◄─┘
                  │ MS 05    │            │ 1998    │
                  │ 2017     │            │ CNN     │
                  │ Attention│            └─────────┘
                  └──────────┘
                        │
                        │  Module 14-19
                        ▼
                  ┌─────────┐
                  │ MS 06   │
                  │ 2018    │
                  │ Optimize│
                  └─────────┘
```

## Success Criteria

Each milestone has specific success criteria. Passing means your implementation is correct:

- **Milestone 01**: Forward pass produces reasonable outputs
- **Milestone 02**: Demonstrates XOR is unsolvable with single layer (75% max accuracy)
- **Milestone 03**: Part 1 solves XOR (100% accuracy), Part 2 achieves 85%+ on TinyDigits
- **Milestone 04**: TinyDigits achieves 90%+ accuracy with CNN
- **Milestone 05**: Pass all three attention challenges (95%+ accuracy)
- **Milestone 06**: Part 1 completes optimization pipeline, Part 2 shows KV cache speedup

## Troubleshooting

If a milestone fails:

1. Check that all required modules are completed: `tito module status`
2. Run the module tests: `tito test <module_number>`
3. Look at the specific error message for debugging hints
4. Review the milestone's docstring for implementation requirements
