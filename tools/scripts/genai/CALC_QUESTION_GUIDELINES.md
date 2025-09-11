# CALC Question Guidelines for ML Systems

## What Makes a BAD CALC Question

### Examples of Poor CALC Questions (DO NOT USE)

#### Example 1: Trivial Data Volume Multiplication
```json
{
  "question_type": "CALC",
  "question": "Waymo's autonomous vehicles generate 1 TB of data per hour. If a vehicle operates for 8 hours a day, calculate the total data generated per day and discuss the implications for data management.",
  "answer": "Total data generated per day is 8 TB (1 TB/hour × 8 hours)..."
}
```
**Why it's bad:** This is just 1 × 8 = 8. It's trivial multiplication that doesn't test ML knowledge.

#### Example 2: Simple Subtraction
```json
{
  "question_type": "CALC",
  "question": "A cloud-based ML system processes 1 TB of data daily, while an edge system processes 100 GB daily. Calculate the data volume difference over a week...",
  "answer": "The cloud system processes 1 TB × 7 = 7 TB weekly..."
}
```
**Why it's bad:** Just multiplication and subtraction. No ML-specific calculations.

#### Example 3: Trivial Cost Multiplication
```json
{
  "question_type": "CALC",
  "question": "A machine learning model requires 1,000 GPU-hours for training. If using a cloud provider with a rate of $0.50 per GPU-hour, calculate the total cost of training this model.",
  "answer": "The total cost is calculated by multiplying the number of GPU-hours by the rate per GPU-hour: 1,000 GPU-hours × $0.50/GPU-hour = $500..."
}
```
**Why it's bad:** This is elementary school math (1000 × 0.5). It doesn't test any ML systems knowledge - just basic multiplication that anyone could do.

## What Makes a GOOD CALC Question

Good CALC questions should:
1. **Use actual ML formulas** from the chapter content
2. **Test understanding of the formula**, not just arithmetic
3. **Require interpretation** of the results in an ML context
4. **Build on concepts** taught in the chapter

### Examples of Good CALC Questions

#### Example 1: Neural Network Parameter Calculation
```json
{
  "question_type": "CALC",
  "question": "A fully connected layer has 512 input neurons and 256 output neurons. Calculate the total number of parameters (weights + biases) in this layer.",
  "answer": "Parameters = (input_size × output_size) + output_size = (512 × 256) + 256 = 131,072 + 256 = 131,328 parameters. The weights account for 131,072 parameters (connections between neurons) and biases add 256 parameters (one per output neuron).",
  "learning_objective": "Apply parameter counting formulas to understand memory requirements of neural network layers."
}
```

#### Example 2: Training Time Estimation
```json
{
  "question_type": "CALC",
  "question": "A dataset contains 50,000 samples. With a batch size of 32 and 100 epochs, how many gradient update steps will occur during training? If each step takes 50ms, what is the total training time?",
  "answer": "Steps per epoch = 50,000 / 32 = 1,562.5 ≈ 1,563 steps. Total steps = 1,563 × 100 = 156,300 steps. Training time = 156,300 × 0.05s = 7,815 seconds ≈ 2.17 hours. This shows how batch size affects training duration.",
  "learning_objective": "Calculate training iterations and time estimation for batch gradient descent."
}
```

#### Example 3: Compression Ratio
```json
{
  "question_type": "CALC",
  "question": "After pruning, a model's parameters are reduced from 10 million to 2.5 million. Calculate the compression ratio and sparsity percentage.",
  "answer": "Compression ratio = 10M / 2.5M = 4×. Sparsity = (10M - 2.5M) / 10M × 100% = 75%. This 4× compression with 75% sparsity significantly reduces memory footprint while potentially maintaining accuracy.",
  "learning_objective": "Calculate compression metrics to evaluate model optimization effectiveness."
}
```

#### Example 4: GPU Utilization
```json
{
  "question_type": "CALC",
  "question": "A GPU has 80 streaming multiprocessors (SMs). During training, only 64 SMs are active. Calculate the GPU utilization percentage and discuss the efficiency implications.",
  "answer": "GPU Utilization = (64 / 80) × 100% = 80%. This 80% utilization suggests the workload isn't fully saturating the GPU, potentially due to memory bottlenecks or insufficient parallelism. Optimizing batch size or model architecture could improve utilization.",
  "learning_objective": "Calculate hardware utilization metrics to identify optimization opportunities."
}
```

#### Example 5: Arithmetic Intensity
```json
{
  "question_type": "CALC",
  "question": "A convolution operation performs 2.3 GFLOPs and accesses 100 MB of memory. Calculate the arithmetic intensity and determine if this operation is compute-bound or memory-bound on a GPU with 10 TFLOPS compute and 900 GB/s bandwidth.",
  "answer": "Arithmetic Intensity = 2.3 GFLOPs / 0.1 GB = 23 FLOPs/byte. GPU compute/bandwidth ratio = 10,000 GFLOPs / 900 GB/s = 11.1 FLOPs/byte. Since 23 > 11.1, this operation is compute-bound on this GPU.",
  "learning_objective": "Apply roofline model analysis to identify performance bottlenecks."
}
```

## Formula Categories for CALC Questions

Based on the knowledge map, focus on these formula types:

### 1. **Network Architecture Formulas**
- Parameter counting
- Output dimensions (CNN, pooling)
- Receptive field calculations
- Memory requirements

### 2. **Training Formulas**
- Learning rate schedules
- Batch size effects
- Gradient updates (SGD, Adam)
- Convergence rates

### 3. **Performance Metrics**
- Throughput calculations
- Latency analysis
- Scaling efficiency
- Cost per inference

### 4. **Optimization Formulas**
- Compression ratios
- Quantization speedup
- Pruning/sparsity metrics
- Energy efficiency

### 5. **System Metrics**
- GPU/TPU utilization
- Memory bandwidth
- Cache hit rates
- Communication overhead

## Guidelines for Question Creation

### DO:
- Use formulas explicitly mentioned in the chapter
- Require multi-step calculations that build understanding
- Ask for interpretation of results
- Connect calculations to real-world implications
- Use realistic values from actual ML systems

### DON'T:
- Create simple arithmetic problems (multiply by 7, add two numbers)
- Use made-up formulas not in the content
- Focus only on the arithmetic without ML context
- Create questions that could be solved without ML knowledge
- Use unrealistic or arbitrary numbers

## Progressive Difficulty

### Early Chapters (1-5):
- Basic accuracy, error rate calculations
- Simple parameter counting
- Data size estimations
- Basic performance metrics

### Middle Chapters (6-10):
- Complex architecture calculations
- Training dynamics and convergence
- Multi-step optimization problems
- Hardware utilization analysis

### Advanced Chapters (11-20):
- System-level performance modeling
- Distributed training calculations
- Cost-benefit analysis
- Complex optimization trade-offs

## Remember
CALC questions should test **ML systems engineering knowledge through calculation**, not just arithmetic skills. Every CALC question should teach something about how ML systems work quantitatively.