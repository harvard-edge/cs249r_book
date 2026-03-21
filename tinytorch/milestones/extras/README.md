# Milestone Extras

This directory contains additional milestone variants and demos that are not part of the core curriculum. These scripts demonstrate alternative applications of the TinyTorch modules but are not required for course completion.

## Status

These scripts are provided as-is for exploration and self-study. They may:
- Require additional setup or dependencies
- Have different accuracy expectations than core milestones
- Serve as inspiration for your own experiments

## Available Extras

### Perceptron Variants
<table>
<thead>
<tr>
<th width="35%"><b>File</b></th>
<th width="45%">Description</th>
<th width="20%">Based On</th>
</tr>
</thead>
<tbody>
<tr><td><b>`02_rosenblatt_trained.py`</b></td><td>Full perceptron training with learning</td><td>Milestone 01</td></tr>
</tbody>
</table>

### XOR Variants
<table>
<thead>
<tr>
<th width="35%"><b>File</b></th>
<th width="45%">Description</th>
<th width="20%">Based On</th>
</tr>
</thead>
<tbody>
<tr><td><b>`01_xor_crisis.py`</b></td><td>Demonstrates why single-layer networks fail on XOR</td><td>Milestone 02</td></tr>
</tbody>
</table>

### MLP Variants
<table>
<thead>
<tr>
<th width="35%"><b>File</b></th>
<th width="45%">Description</th>
<th width="20%">Based On</th>
</tr>
</thead>
<tbody>
<tr><td><b>`02_rumelhart_mnist.py`</b></td><td>MLP on full MNIST dataset (60K images)</td><td>Milestone 03</td></tr>
</tbody>
</table>

### CNN Variants
<table>
<thead>
<tr>
<th width="35%"><b>File</b></th>
<th width="45%">Description</th>
<th width="20%">Based On</th>
</tr>
</thead>
<tbody>
<tr><td><b>`02_lecun_cifar10.py`</b></td><td>LeNet on CIFAR-10 natural images</td><td>Milestone 04</td></tr>
</tbody>
</table>

### Transformer Demos
<table>
<thead>
<tr>
<th width="35%"><b>File</b></th>
<th width="45%">Description</th>
<th width="20%">Based On</th>
</tr>
</thead>
<tbody>
<tr><td><b>`01_tinytalks.py`</b></td><td>Conversational pattern learning</td><td>Milestone 05</td></tr>
<tr><td><b>`01_vaswani_generation.py`</b></td><td>Text generation demo</td><td>Milestone 05</td></tr>
<tr><td><b>`02_vaswani_dialogue.py`</b></td><td>CodeBot - Python autocomplete</td><td>Milestone 05</td></tr>
<tr><td><b>`03_quickdemo.py`</b></td><td>Quick transformer demo</td><td>Milestone 05</td></tr>
</tbody>
</table>

### Optimization Demos
<table>
<thead>
<tr>
<th width="35%"><b>File</b></th>
<th width="45%">Description</th>
<th width="20%">Based On</th>
</tr>
</thead>
<tbody>
<tr><td><b>`01_baseline_profile.py`</b></td><td>Profiling baseline measurements</td><td>Milestone 06</td></tr>
<tr><td><b>`02_compression.py`</b></td><td>Model compression techniques</td><td>Milestone 06</td></tr>
<tr><td><b>`03_generation_opts.py`</b></td><td>Generation optimization options</td><td>Milestone 06</td></tr>
</tbody>
</table>

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
