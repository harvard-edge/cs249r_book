# Milestone Extras

This directory contains additional milestone variants and demos that are not part of the core curriculum. These scripts demonstrate alternative applications of the TinyTorch modules but are not required for course completion.

## Status

These scripts are provided as-is for exploration and self-study. They may:
- Require additional setup or dependencies
- Have different accuracy expectations than core milestones
- Serve as inspiration for your own experiments

## Available Extras

### Perceptron Variants
| File | Description | Based On |
|------|-------------|----------|
| `02_rosenblatt_trained.py` | Full perceptron training with learning | Milestone 01 |

### XOR Variants
| File | Description | Based On |
|------|-------------|----------|
| `01_xor_crisis.py` | Demonstrates why single-layer networks fail on XOR | Milestone 02 |

### MLP Variants
| File | Description | Based On |
|------|-------------|----------|
| `02_rumelhart_mnist.py` | MLP on full MNIST dataset (60K images) | Milestone 03 |

### CNN Variants
| File | Description | Based On |
|------|-------------|----------|
| `02_lecun_cifar10.py` | LeNet on CIFAR-10 natural images | Milestone 04 |

### Transformer Demos
| File | Description | Based On |
|------|-------------|----------|
| `01_tinytalks.py` | Conversational pattern learning | Milestone 05 |
| `01_vaswani_generation.py` | Text generation demo | Milestone 05 |
| `02_vaswani_dialogue.py` | CodeBot - Python autocomplete | Milestone 05 |
| `03_quickdemo.py` | Quick transformer demo | Milestone 05 |

### Optimization Demos
| File | Description | Based On |
|------|-------------|----------|
| `01_baseline_profile.py` | Profiling baseline measurements | Milestone 06 |
| `02_compression.py` | Model compression techniques | Milestone 06 |
| `03_generation_opts.py` | Generation optimization options | Milestone 06 |

## Running Extras

These are standalone Python scripts. Run them directly:

```bash
cd tinytorch
python3 milestones/extras/02_vaswani_dialogue.py
```

Note: Ensure you have completed the relevant modules first, as these scripts import from your TinyTorch implementations.

## Contributing

If you create an interesting variant or demo, consider adding it here! Good extras:
- Demonstrate a concept not covered in core milestones
- Use existing TinyTorch modules in creative ways
- Have clear documentation and success criteria
