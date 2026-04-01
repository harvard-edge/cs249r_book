import pytest
from pydantic import ValidationError
from mlsysim.hardware import Hardware, HardwareNode
from mlsysim.core.constants import Q_

def test_hardware_registry():
    a100 = Hardware.A100
    assert a100.name == "NVIDIA A100"
    assert a100.release_year == 2020
    assert a100.compute.peak_flops.magnitude == 156.0  # Dense FP16 (312 with sparsity)

    # Check ridge point calculation
    ridge = a100.ridge_point()
    assert "flop/B" in str(ridge.units) or "flop / byte" in str(ridge.units)
    assert 50 < ridge.magnitude < 120  # ~76 flop/byte (156 TFLOPS / 2.039 TB/s)

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
    assert "156" in json_data  # Dense FP16 peak
