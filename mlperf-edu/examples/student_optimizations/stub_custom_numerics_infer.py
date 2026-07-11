"""Incomplete custom-numerics assignment starter.

The quantization function intentionally contains a student TODO. This file is
not a reference implementation, benchmark baseline, or valid submission. See
the README in this directory before using it in a course.
"""

import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from mlperf.reference.cloud.nanogpt_train import NanoGPTWhiteBox


def load_and_quantize_teacher_math(checkpoint_path: str, device: str):
    """
    Load a checkpoint before the student-defined numerics transformation.

    This starter does not measure latency, energy, memory, or model quality.
    """
    print(f"Loading assignment checkpoint: {checkpoint_path}")

    model = NanoGPTWhiteBox()
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "A compatible checkpoint is required before this starter can run"
        ) from exc

    print("Starting the student numerics transformation")

    # ---------------------------------------------------------------------------------
    # ⚠️ YOUR CUSTOM SYSTEMS LOGIC GOES HERE
    # The Teacher's Matrix is currently locked in FP32 precision.
    # Use your own logic to squash this into INT8/INT4 analytically safely!
    # ---------------------------------------------------------------------------------

    # Assignment hook. Students replace this function under the course rubric.
    def my_custom_quantization_pass(fp32_model):
        with torch.no_grad():
            for name, param in fp32_model.named_parameters():
                if "weight" in name and param.dim() > 1:
                    # TODO: implement the assignment's declared quantization rule.
                    pass
        return fp32_model

    quantized_model = my_custom_quantization_pass(model)
    # ---------------------------------------------------------------------------------

    return quantized_model.to(device)


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Instructors should replace this with their assignment checkpoint path.
    dummy_path = "instructor_baseline.pt"

    student_model = load_and_quantize_teacher_math(dummy_path, device)

    print(
        "Starter transformation returned a model. Implement the TODO and the "
        "assignment's benchmark checks before treating it as a result."
    )
