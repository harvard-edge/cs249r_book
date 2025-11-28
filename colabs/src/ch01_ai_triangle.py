# %% [markdown]
# # The AI Triangle: Why You Can't Optimize in Isolation
#
# **Machine Learning Systems: Engineering Intelligence at Scale**
# _Chapter 1: Introduction - Understanding ML as a Systems Discipline_
#
# ---
#
# In Chapter 1, you learned that ML systems consist of three tightly coupled components: **models** (algorithms that learn patterns), **data** (examples that guide learning), and **infrastructure** (compute that enables training and inference). The AI Triangle framework shows these aren't independent—they shape each other's possibilities.
#
# Now you'll experience this interdependence hands-on. You'll attempt to improve a medical imaging classifier from 80% to 90% accuracy. As you try different approaches, you'll discover why optimizing one component creates bottlenecks in the others.
#
# **Why this matters**: This pattern repeats across every ML deployment. Google's diabetic retinopathy detector required coordinated scaling of all three components over 3+ years. Tesla's Autopilot balances model sophistication against real-time compute constraints. Understanding these trade-offs is the essence of ML systems engineering.

# %% [markdown]
# ## Setup
#
# Run this cell to load the AI Triangle simulator.

# %%
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

plt.style.use('seaborn-v0_8-whitegrid')
# %matplotlib inline

print("✓ Setup complete")

# %%
class AITriangleSimulator:
    """Simplified simulator showing model-data-compute interdependencies"""

    MODELS = {
        'Small CNN': 5,
        'ResNet-50': 25,
        'ResNet-101': 45,
        'ResNet-152': 60,
        'Large Model': 100
    }

    def __init__(self, model_name='ResNet-50', dataset_size=10000, num_gpus=1):
        self.model_name = model_name
        self.model_params = self.MODELS[model_name]
        self.dataset_size = dataset_size
        self.num_gpus = num_gpus

    def estimate_accuracy(self):
        model_factor = np.log10(self.model_params + 1) * 10
        data_factor = np.log10(self.dataset_size / 1000 + 1) * 15
        compute_factor = np.log10(self.num_gpus + 1) * 5

        data_per_param = self.dataset_size / (self.model_params * 1000)

        if data_per_param < 10:
            bottleneck = "⚠️ DATA BOTTLENECK: Not enough data for model size (overfitting risk)"
            accuracy = min(95, 60 + data_factor)
        elif self.num_gpus < np.log10(self.dataset_size):
            bottleneck = "⚠️ COMPUTE BOTTLENECK: Training will be very slow or incomplete"
            accuracy = min(95, 65 + model_factor + compute_factor)
        else:
            bottleneck = "✓ Balanced system"
            accuracy = min(95, 70 + model_factor + data_factor + compute_factor)

        return round(accuracy, 1), bottleneck

    def estimate_costs(self):
        baseline_data = 10000
        new_data = max(0, self.dataset_size - baseline_data)
        data_cost = new_data * 3
        data_collection_months = new_data / 3000

        base_hours = (self.model_params * self.dataset_size) / (self.num_gpus * 1000)
        training_hours = max(0.5, base_hours / 100)
        compute_cost_per_run = training_hours * self.num_gpus * 50
        memory_gb = self.model_params * 0.2

        return {
            'data_collection_cost': round(data_cost),
            'data_collection_months': round(data_collection_months, 1),
            'compute_cost_per_run': round(compute_cost_per_run),
            'training_hours': round(training_hours, 1),
            'memory_gb': round(memory_gb, 1)
        }

    def display_status(self):
        accuracy, bottleneck = self.estimate_accuracy()
        costs = self.estimate_costs()

        print("=" * 70)
        print("AI TRIANGLE - CURRENT SYSTEM")
        print("=" * 70)
        print(f"\nAccuracy: {accuracy}%")
        print(f"Status: {bottleneck}")
        print(f"\nModel: {self.model_name} ({self.model_params}M parameters)")
        print(f"Dataset: {self.dataset_size:,} images")
        print(f"Compute: {self.num_gpus} GPU(s)")
        print(f"\nData Cost: ${costs['data_collection_cost']:,} over {costs['data_collection_months']} months")
        print(f"Training: {costs['training_hours']}h at ${costs['compute_cost_per_run']:,}/run")
        print(f"Memory: {costs['memory_gb']} GB")
        print("=" * 70)

        return accuracy, costs

print("✓ AI Triangle simulator loaded")

# %% [markdown]
# ---
#
# ## The Challenge
#
# You're an ML engineer at a medical imaging startup. Your team has built an AI system to detect diseases from chest X-rays.
#
# **Current system:**
# - Model: ResNet-50 (25M parameters)
# - Data: 10,000 labeled medical images (6 months to collect)
# - Compute: 1 GPU
# - **Accuracy: 80%**
#
# Your boss: *"We need 90% accuracy for FDA approval. Fix it."*
#
# **The question**: How do you improve accuracy? Should you use a bigger model? Collect more data? Add more GPUs? All three?
#
# Let's find out what happens when you try each approach.

# %% [markdown]
# ---
#
# ## Baseline System

# %%
system = AITriangleSimulator(
    model_name='ResNet-50',
    dataset_size=10000,
    num_gpus=1
)

print("BASELINE:")
baseline_accuracy, baseline_costs = system.display_status()

# %% [markdown]
# ---
#
# ## Attempt 1: Use a Bigger Model
#
# **Intuition**: More parameters should mean better accuracy.
#
# Let's upgrade from ResNet-50 (25M params) to ResNet-152 (60M params).

# %%
system_v2 = AITriangleSimulator(
    model_name='ResNet-152',  # Bigger!
    dataset_size=10000,        # Same
    num_gpus=1                 # Same
)

print("ATTEMPT 1: Bigger Model")
accuracy_v2, costs_v2 = system_v2.display_status()

print(f"\nChange: {baseline_accuracy}% → {accuracy_v2}% ({accuracy_v2 - baseline_accuracy:+.1f}%)")

# %% [markdown]
# **What happened?** The bigger model hit a data bottleneck—60M parameters need more than 10K examples to avoid overfitting.

# %% [markdown]
# ---
#
# ## Attempt 2: Collect More Data
#
# **Intuition**: The bigger model needs more data. Let's collect 25,000 images (2.5x more).

# %%
system_v3 = AITriangleSimulator(
    model_name='ResNet-152',
    dataset_size=25000,         # More data!
    num_gpus=1
)

print("ATTEMPT 2: Bigger Model + More Data")
accuracy_v3, costs_v3 = system_v3.display_status()

print(f"\nChange: {baseline_accuracy}% → {accuracy_v3}% ({accuracy_v3 - baseline_accuracy:+.1f}%)")
print(f"Data cost: ${costs_v3['data_collection_cost']:,} over {costs_v3['data_collection_months']} months")

# %% [markdown]
# **What happened?** Accuracy improved, but now we hit a compute bottleneck—training 60M parameters on 25K images with 1 GPU takes too long.

# %% [markdown]
# ---
#
# ## Attempt 3: Add More Compute
#
# **Intuition**: Use 8 GPUs to speed up training and enable better hyperparameter search.

# %%
system_v4 = AITriangleSimulator(
    model_name='ResNet-152',
    dataset_size=25000,
    num_gpus=8                  # More compute!
)

print("ATTEMPT 3: All Three Components Scaled")
accuracy_v4, costs_v4 = system_v4.display_status()

print(f"\nFinal change: {baseline_accuracy}% → {accuracy_v4}% ({accuracy_v4 - baseline_accuracy:+.1f}%)")
print(f"\nTotal investment:")
print(f"  Data: ${costs_v4['data_collection_cost']:,} over {costs_v4['data_collection_months']} months")
print(f"  Compute: ${costs_v4['compute_cost_per_run']:,} per training run")
print(f"  Time: {costs_v4['training_hours']}h training")

# %% [markdown]
# **Key insight**: To improve from ~80% to ~90%, we had to change **all three components**. We couldn't just:
# - Make the model bigger (hit data bottleneck)
# - Add more data (hit compute bottleneck)
# - Add more compute alone (didn't help without model + data)
#
# **This is the AI Triangle in action.**

# %% [markdown]
# ---
#
# ## Your Turn: Explore the Trade-offs
#
# Try to find:
# 1. The **cheapest** way to get above 85% accuracy
# 2. The **fastest** way to get above 88% accuracy
# 3. What happens if you max out the model but keep minimal data/compute
#
# **Modify the parameters below and run:**

# %%
# YOUR EXPERIMENT
# Models: 'Small CNN', 'ResNet-50', 'ResNet-101', 'ResNet-152', 'Large Model'

my_system = AITriangleSimulator(
    model_name='ResNet-50',     # Change this
    dataset_size=10000,          # Change this (5000-50000)
    num_gpus=1                   # Change this (1-16)
)

my_accuracy, my_costs = my_system.display_status()

if my_accuracy >= 90:
    print("\n🎉 SUCCESS! 90%+ accuracy")
elif my_accuracy >= 85:
    print("\n✓ Good progress (85%+)")
else:
    print("\nKeep experimenting!")

# %% [markdown]
# ---
#
# ## Visualizing the Interdependencies

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Model size vs accuracy for different data amounts
model_sizes = [5, 25, 45, 60, 100]
for data_size in [5000, 10000, 25000, 50000]:
    accuracies = []
    for model_name, params in AITriangleSimulator.MODELS.items():
        if params in model_sizes:
            sim = AITriangleSimulator(model_name, data_size, num_gpus=4)
            acc, _ = sim.estimate_accuracy()
            accuracies.append(acc)
    ax1.plot(model_sizes, accuracies, marker='o', label=f'{data_size:,} images')

ax1.axhline(y=90, color='r', linestyle='--', label='Goal: 90%')
ax1.set_xlabel('Model Size (Million Parameters)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Bigger Models Need More Data')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Compute vs training time
gpu_counts = [1, 2, 4, 8, 16]
training_times = []
for gpus in gpu_counts:
    sim = AITriangleSimulator('ResNet-152', 25000, gpus)
    costs = sim.estimate_costs()
    training_times.append(costs['training_hours'])

ax2.plot(gpu_counts, training_times, marker='s', color='green', linewidth=2)
ax2.set_xlabel('Number of GPUs')
ax2.set_ylabel('Training Time (hours)')
ax2.set_title('More Compute Reduces Training Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Key insight: All three components must scale together")

# %% [markdown]
# ---
#
# ## Understanding the Interdependencies
#
# **Why does a bigger model need more data?**
#
# A model with 60M parameters has millions of "knobs to tune." With only 10K training examples, the model memorizes the training set (overfitting) rather than learning generalizable patterns. You need roughly 10+ examples per 1000 parameters to avoid this.
#
# **Why does more data need more compute?**
#
# Each training example must be processed through the model multiple times (epochs). If you 2.5x your dataset (10K → 25K images), training time increases proportionally unless you add more compute to parallelize.
#
# **What if you had unlimited budget?**
#
# Even with infinite money, you're constrained by:
# - Time to collect and label data (months to years)
# - Time to train models (physical limits)
# - Available expert labelers (for medical images)
# - Regulatory approval processes
#
# **This is why systems thinking matters**—you're always optimizing under constraints.

# %% [markdown]
# ---
#
# ## Real-World Examples
#
# **Google Health (Diabetic Retinopathy)**
# - Model: Deep CNN with millions of parameters
# - Data: 128,000 retinal images (years to collect)
# - Compute: Weeks of training on TPU clusters
# - Result: 94% sensitivity, but took 3+ years to deploy
#
# **Tesla Autopilot**
# - Model: Billions of parameters
# - Data: Millions of miles from fleet
# - Compute: 10,000+ GPUs ($10M+/year)
# - Trade-off: Massive infrastructure for continuous improvement
#
# **MobileNet (On-Device)**
# - Strategy: Optimized for opposite constraints
# - Model: Tiny (4M params vs 60M)
# - Data: Works with smaller datasets
# - Compute: Runs on phone CPUs
# - Trade-off: Lower accuracy (~85%) but instant, anywhere
#
# **AlexNet (2012) - The Deep Learning Revolution**
#
# The breakthrough didn't come from a new algorithm—CNNs existed since the 1980s. It happened because all three components came together:
# 1. Algorithm: CNNs (already known)
# 2. Data: ImageNet (1.2M labeled images)
# 3. Compute: GPUs made training feasible (2 GPUs, 6 days)
#
# Result: Accuracy jumped from 74% to 85% on ImageNet.

# %% [markdown]
# ---
#
# ## The Bitter Lesson
#
# Richard Sutton's observation from 70 years of AI research:
#
# > *"General methods that leverage computation are ultimately the most effective, and by a large margin."*
#
# Why? Because scaling compute enables:
# - Bigger models (more parameters for complex patterns)
# - More data processing (train on larger datasets)
# - Which together beat clever algorithms with limited resources
#
# This is why **systems engineering** (knowing how to scale compute + data + models together) matters more than algorithmic tricks. The rest of this book teaches you how.

# %% [markdown]
# ---
#
# ## Summary
#
# **Key takeaways:**
#
# 1. **The AI Triangle shows interdependence** - You can't optimize models, data, or compute in isolation. Changes to one create bottlenecks in others.
#
# 2. **Every system is a compromise** - Google can spend millions on all three. Startups choose carefully. Mobile apps work with tiny models. Context determines trade-offs.
#
# 3. **Systems engineering matters most** - The Bitter Lesson: scaling compute + data beats clever algorithms. Knowing HOW to scale is ML systems engineering.
#
# 4. **Constraints propagate** - Limited budget → smaller model → less data → lower accuracy. Every choice has ripple effects.
#
# **What's next:** Throughout this book, you'll learn how to navigate these trade-offs across deployment paradigms (Chapter 2), optimize each component (Chapters 3-10), scale systems (Chapters 11-14), and build robust, fair, sustainable ML systems (Chapters 15-20).
#
# Welcome to ML systems engineering.
