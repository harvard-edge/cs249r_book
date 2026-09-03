import pytest
from pydantic import ValidationError

from mlsysim.core.units import Q_, ureg
from mlsysim.hardware.types import ComputeCore, MemoryHierarchy
from mlsysim.models.types import TransformerWorkload
from mlsysim.systems.types import NetworkFabric


def test_memory_capacity_requires_storage_units():
    with pytest.raises(ValidationError, match="capacity"):
        MemoryHierarchy(capacity=80, bandwidth=Q_("1 TB/s"))


def test_memory_bandwidth_requires_rate_units():
    with pytest.raises(ValidationError, match="bandwidth"):
        MemoryHierarchy(capacity=Q_("80 GB"), bandwidth=Q_("1 second"))


def test_memory_bandwidth_rejects_operation_rate_units():
    with pytest.raises(ValidationError, match="bandwidth"):
        MemoryHierarchy(capacity=Q_("80 GB"), bandwidth=Q_("1 TFLOP/s"))


def test_compute_flops_rejects_data_rate_units():
    with pytest.raises(ValidationError, match="peak_flops"):
        ComputeCore(peak_flops=Q_("900 GB/s"))


def test_network_bandwidth_accepts_bit_and_byte_rates():
    bit_rate = NetworkFabric(name="IB", bandwidth=Q_("400 Gbit/s"))
    byte_rate = NetworkFabric(name="NVLink", bandwidth=Q_("900 GB/s"))

    assert bit_rate.bandwidth.check(ureg.bit / ureg.second)
    assert byte_rate.bandwidth.check(ureg.bit / ureg.second)


def test_network_fabric_rejects_unknown_fields():
    with pytest.raises(ValidationError, match="Extra inputs|extra_forbidden"):
        NetworkFabric(name="IB", bandwidth=Q_("400 Gbit/s"), stray_field=True)


def test_workload_parameters_require_count_units():
    with pytest.raises(ValidationError, match="parameters"):
        TransformerWorkload(
            name="Bad LLM",
            architecture="Transformer",
            parameters=70e9,
            layers=80,
        )


def test_workload_flops_rejects_data_units():
    with pytest.raises(ValidationError, match="inference_flops"):
        TransformerWorkload(
            name="Bad LLM",
            architecture="Transformer",
            parameters=Q_("70e9 param"),
            layers=80,
            inference_flops=Q_("80 GB"),
        )
