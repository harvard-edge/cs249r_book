# Milestone System

<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center; color: white;">
<h2 style="margin: 0 0 1rem 0; color: white;">Recreate ML History with YOUR Code</h2>
<p style="margin: 0; font-size: 1.1rem; opacity: 0.95;">Run the algorithms that changed the world using the TinyTorch you built from scratch</p>
</div>

**Purpose**: The milestone system lets you run famous ML algorithms (1958-2018) using YOUR implementations. Every milestone validates that your code can recreate a historical breakthrough.

The milestone system lets you run famous ML algorithms using YOUR implementations.

## What Are Milestones?

Milestones are **runnable recreations of historical ML papers** that use YOUR TinyTorch implementations:

- **1958 - Rosenblatt's Perceptron**: The first trainable neural network
- **1969 - XOR Solution**: Solving the problem that stalled AI
- **1986 - Backpropagation**: The MLP revival (Rumelhart, Hinton & Williams)
- **1998 - LeNet**: Yann LeCun's CNN breakthrough
- **2017 - Transformer**: "Attention is All You Need" (Vaswani et al.)
- **2018 - MLPerf**: Production ML benchmarks

Each milestone script imports **YOUR code** from the TinyTorch package you built.

## Quick Start

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**Typical workflow:**

```bash
# 1. Build the required modules (e.g., Foundation Tier for Milestone 03)
tito module complete 01 # Tensor
tito module complete 02 # Activations
tito module complete 03 # Layers
tito module complete 04 # Losses
tito module complete 05 # DataLoader
tito module complete 06 # Autograd
tito module complete 07 # Optimizers
tito module complete 08 # Training

# 2. See what milestones you can run
tito milestone list

# 3. Get details about a specific milestone
tito milestone info 03

# 4. Run it!
tito milestone run 03
```

</div>

## Essential Commands

### Discover Milestones

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1rem 0;">

**List All Milestones**
```bash
tito milestone list
```

Shows all 6 historical milestones with status:
- **LOCKED** - Need to complete required modules first
- **READY TO RUN** - All prerequisites met!
- **COMPLETE** - You've already achieved this

**Simple View** (compact list):
```bash
tito milestone list --simple
```

</div>

### Learn About Milestones

<div style="background: #fff3e0; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #ff9800; margin: 1rem 0;">

**Get Detailed Information**
```bash
tito milestone info 03
```

Shows:
- Historical context (year, researchers, significance)
- Description of what you'll recreate
- Required modules with / status
- Whether you're ready to run it

</div>

### Run Milestones

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0; margin: 1rem 0;">

**Run a Milestone**
```bash
tito milestone run 03
```

What happens:
1. **Checks prerequisites** - Validates required modules are complete
2. **Tests imports** - Ensures YOUR implementations work
3. **Shows context** - Historical background and what you'll recreate
4. **Runs the script** - Executes the milestone using YOUR code
5. **Tracks achievement** - Records your completion
6. **Celebrates!** - Shows achievement message

**Skip prerequisite checks** (not recommended):
```bash
tito milestone run 03 --skip-checks
```

</div>

### Track Progress

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; margin: 1rem 0;">

**View Milestone Progress**
```bash
tito milestone status
```

Shows:
- How many milestones you've completed
- Your overall progress (%)
- Unlocked capabilities
- Next milestone ready to run

**Visual Timeline**
```bash
tito milestone timeline
```

See your journey through ML history in a visual tree format.

</div>

## The 6 Milestones

### Milestone 01: Perceptron (1958)

**What**: Frank Rosenblatt's first trainable neural network

**Requires**: Modules 01-03 (Tensor, Activations, Layers)

**What you'll do**: Implement and train the perceptron that proved machines could learn

**Historical significance**: First demonstration of machine learning

**Run it**:
```bash
tito milestone info 01
tito milestone run 01
```


### Milestone 02: XOR Crisis (1969)

**What**: Demonstrating the problem that stalled AI research

**Requires**: Modules 01-03 (Tensor, Activations, Layers)

**What you'll do**: Experience how single-layer perceptrons fail on XOR - the limitation that triggered the "AI Winter"

**Historical significance**: Minsky & Papert showed perceptron limitations; this milestone demonstrates the crisis before the solution

**Run it**:
```bash
tito milestone info 02
tito milestone run 02
```


### Milestone 03: MLP Revival (1986)

**What**: Backpropagation breakthrough - train deep networks on MNIST

**Requires**: Modules 01-08 (Complete Foundation Tier)

**What you'll do**: Train a multi-layer perceptron to recognize handwritten digits (95%+ accuracy)

**Historical significance**: Rumelhart, Hinton & Williams (Nature, 1986) - the paper that reignited neural network research

**Run it**:
```bash
tito milestone info 03
tito milestone run 03
```


### Milestone 04: CNN Revolution (1998)

**What**: LeNet - Computer Vision Breakthrough

**Requires**: Modules 01-09 (Foundation + Convolutions)

**What you'll do**: Build LeNet for digit recognition using convolutional layers

**Historical significance**: Yann LeCun's breakthrough that enabled modern computer vision

**Run it**:
```bash
tito milestone info 04
tito milestone run 04
```


### Milestone 05: Transformer Era (2017)

**What**: "Attention is All You Need"

**Requires**: Modules 01-08 + 11-13 (Foundation + Embeddings, Attention, Transformers)

**What you'll do**: Implement transformer architecture with self-attention mechanism

**Historical significance**: Vaswani et al. revolutionized NLP and enabled GPT/BERT/modern LLMs

**Run it**:
```bash
tito milestone info 05
tito milestone run 05
```


### Milestone 06: MLPerf Benchmarks (2018)

**What**: Production ML Systems

**Requires**: Modules 01-08 + 14-19 (Foundation + Optimization Tier)

**What you'll do**: Optimize for production deployment with quantization, compression, and benchmarking

**Historical significance**: MLPerf standardized ML system benchmarks for real-world deployment

**Run it**:
```bash
tito milestone info 06
tito milestone run 06
```


## Prerequisites and Validation

### How Prerequisites Work

Each milestone requires specific modules to be complete. The `run` command automatically validates:

**Module Completion Check**:
```bash
tito milestone run 03

 Checking prerequisites for Milestone 03...
 Module 01 - complete
 Module 02 - complete
 Module 03 - complete
 Module 04 - complete
 Module 05 - complete
 Module 06 - complete
 Module 07 - complete
 Module 08 - complete

 All prerequisites met!
```

**Import Validation**:
```bash
 Testing YOUR implementations...
 Tensor import successful
 Activations import successful
 Layers import successful

 YOUR TinyTorch is ready!
```

### If Prerequisites Are Missing

You'll see a helpful error:

```bash
 Missing Required Modules

Milestone 03 requires modules: 01, 02, 03, 04, 05, 06, 07, 08
Missing: 06, 07, 08

Complete the missing modules first:
 tito module start 06
 tito module start 07
 tito module start 08
```

## Achievement Celebration

When you successfully complete a milestone, you'll see:

```
╔════════════════════════════════════════════════╗
║ Milestone 03: MLP Revival (1986) ║
║ Backpropagation Breakthrough ║
╚════════════════════════════════════════════════╝

 MILESTONE ACHIEVED!

You completed Milestone 03: MLP Revival (1986)
Backpropagation Breakthrough

What makes this special:
• Every line of code: YOUR implementations
• Every tensor operation: YOUR Tensor class
• Every gradient: YOUR autograd

Achievement saved to your progress!

 What's Next:
Milestone 04: CNN Revolution (1998)
Unlock by completing module: 09
```

## Understanding Your Progress

### Three Tracking Systems

TinyTorch tracks progress in three ways (all are related but distinct):

<div style="background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;">

**1. Module Completion** (`tito module status`)
- Which modules (01-20) you've implemented
- Tracked in `.tito/progress.json`
- **Required** for running milestones

**2. Milestone Achievements** (`tito milestone status`)
- Which historical papers you've recreated
- Tracked in `.tito/milestones.json`
- Unlocked by completing modules + running milestones

**3. Overall Status**
- Check `tito module status` and `tito milestone status`
- Quick view of all progress
- Purely informational

</div>

### Relationship Between Systems

```
Complete Modules (01-08)
 ↓
Unlock Milestone 03
 ↓
Run: tito milestone run 03
 ↓
Achievement Recorded
 ↓
Capability Unlocked (optional checkpoint system)
```

## Tips for Success

### 1. Complete Modules in Order

While you can technically skip around, the tier structure is designed for progressive learning:

- **Foundation Tier (01-08)**: Required for first milestone
- **Architecture Tier (09-13)**: Build on Foundation
- **Optimization Tier (14-19)**: Build on Architecture

### 2. Test as You Go

Before running a milestone, make sure your modules work:

```bash
# After completing a module
tito module complete 05

# Test it works
python -c "from tinytorch import Tensor; print(Tensor([[1,2]]))"
```

### 3. Use Info Before Run

Learn what you're about to do:

```bash
tito milestone info 03 # Read the context first
tito milestone run 03 # Then run it
```

### 4. Celebrate Achievements

Share your milestones! Each one represents recreating a breakthrough that shaped modern AI.

## Troubleshooting

### "Import Error" when running milestone

**Problem**: Module not exported or import failing

**Solution**:
```bash
# Re-export the module
tito module complete XX

# Test import manually
python -c "from tinytorch import Tensor"
```

### "Prerequisites Not Met" but I completed modules

**Problem**: Progress not tracked correctly

**Solution**:
```bash
# Check module status
tito module status

# If modules show incomplete, re-run complete
tito module complete XX
```

### Milestone script fails during execution

**Problem**: Bug in your implementation

**Solution**:
1. Check error message for which module failed
2. Edit `modules/XX_name/XX_name.ipynb` (NOT `tinytorch/`)
3. Re-export: `tito module complete XX`
4. Run milestone again

## Next Steps

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Ready to Recreate ML History?</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Start with the Foundation Tier and work toward your first milestone</p>
<a href="../tiers/foundation.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Foundation Tier →</a>
<a href="../milestones/milestones_ABOUT.html" style="display: inline-block; background: #6f42c1; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Historical Context →</a>
</div>


*Every milestone uses YOUR code. Every achievement is proof you understand ML systems deeply. Build from scratch, recreate history, master the fundamentals.*
