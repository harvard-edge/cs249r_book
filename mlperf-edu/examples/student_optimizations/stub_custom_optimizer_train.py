"""Incomplete custom-optimizer assignment starter.

This file is not a benchmark baseline or a valid submission. Instructors must
define the exercise's data, run budget, quality threshold, and grading path
before assigning it. See the README in this directory.
"""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox


def execute_student_training_optimization():
    """
    Demonstrate where a student-defined parameter update can be inserted.

    This short synthetic loop does not implement benchmark data loading,
    provenance, quality evaluation, or grading.
    """
    print("Custom optimizer assignment starter")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = NanoGPTWhiteBox().to(device)
    model.train()

    # ---------------------------------------------------------------------------------
    # ⚠️ YOUR CUSTOM SYSTEMS LOGIC GOES HERE
    # Do not use `torch.optim`. Write your own Stochastic Gradient Descent natively!
    # Tip: Evaluate memory overheads of storing custom momentums mathematically!
    # ---------------------------------------------------------------------------------

    # Minimal update for orientation. Replace it under the assignment rubric.
    def my_custom_backward_step(loss, parameters):
        loss.backward()
        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    # Implement your mathematical routing natively here!
                    p.sub_(p.grad * 0.001)
                    p.grad.zero_()

    # ---------------------------------------------------------------------------------

    # Synthetic orientation loop. It is not a benchmark workload.
    for epoch in range(10):
        dummy_data = torch.randint(0, 128, (1, 16), device=device)
        dummy_targets = torch.randint(0, 128, (1, 16), device=device)

        logits, loss = model(dummy_data, targets=dummy_targets)

        my_custom_backward_step(loss, model.parameters())

        print(f"Step {epoch} | synthetic loss: {loss.item():.4f}")


if __name__ == "__main__":
    execute_student_training_optimization()
