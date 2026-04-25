import pytest
from pydantic import ValidationError
from mlsysim.hardware import Hardware, HardwareNode
from mlsysim.core.constants import Q_, TPUV4_FLOPS_BF16

def test_hardware_registry():
    a100 = Hardware.A100
    assert a100.name == "NVIDIA A100"
    assert a100.release_year == 2020
    assert a100.compute.peak_flops.magnitude == 312.0  # Dense FP16 Tensor Core

    # Check ridge point calculation
    ridge = a100.ridge_point()
    assert "flop/B" in str(ridge.units) or "flop / byte" in str(ridge.units)
    assert 100 < ridge.magnitude < 200  # ~153 flop/byte (312 TFLOPS / 2.039 TB/s)

def test_hardware_validation():
    # Should raise error on invalid quantity string
    with pytest.raises(ValidationError):
        HardwareNode(
            name="Broken",
            release_year=2025,
            compute={"peak_flops": "not a number"},
            memory={"capacity": "10 GiB", "bandwidth": "100 GB/s"}
        )

def test_json_serialization():
    a100 = Hardware.A100
    json_data = a100.model_dump_json()
    assert "NVIDIA A100" in json_data
    assert "312" in json_data  # FP16 Tensor Core peak


def test_tpuv4_is_not_alias_to_tpuv5p():
    assert Hardware.TPUv4.name == "Google TPU v4"
    assert Hardware.TPUv4 is not Hardware.TPUv5p
    assert Hardware.TPUv4.compute.peak_flops == TPUV4_FLOPS_BF16
