"""
AI Triangle Simulator - Model-Data-Compute Interdependencies

This module provides a simplified analytical model demonstrating how model size,
dataset size, and compute resources interact to determine ML system performance.
Used in Chapter 1 to teach systems thinking fundamentals.
"""

import numpy as np


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
        """
        Initialize AI Triangle simulator

        Parameters:
        -----------
        model_name : str
            Model architecture name (from MODELS dict)
        dataset_size : int
            Number of training examples
        num_gpus : int
            Number of GPUs available for training
        """
        self.model_name = model_name
        self.model_params = self.MODELS[model_name]
        self.dataset_size = dataset_size
        self.num_gpus = num_gpus

    def estimate_accuracy(self):
        """
        Estimate accuracy and identify bottlenecks

        Returns:
        --------
        tuple: (accuracy, bottleneck_message)
        """
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
        """
        Estimate data collection and compute costs

        Returns:
        --------
        dict: Cost breakdown with keys:
            - data_collection_cost (int): Dollar cost for data
            - data_collection_months (float): Time to collect data
            - compute_cost_per_run (int): Dollar cost per training run
            - training_hours (float): Hours per training run
            - memory_gb (float): GPU memory required
        """
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
        """
        Print current system configuration and performance

        Returns:
        --------
        tuple: (accuracy, costs_dict)
        """
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
