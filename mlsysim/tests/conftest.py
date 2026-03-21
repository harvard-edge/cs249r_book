"""Shared fixtures for mlsysim test suite.

All hardware, model, fleet, and solver fixtures are defined here.
Pytest discovers this file automatically — no imports needed in test files.
"""

import pytest

from mlsysim.core.constants import Q_
from mlsysim.core.solver import (
    CheckpointModel,
    ContinuousBatchingModel,
    CompressionModel,
    DataModel,
    DistributedModel,
    EconomicsModel,
    EfficiencyModel,
    InferenceScalingModel,
    OrchestrationModel,
    ReliabilityModel,
    ResponsibleEngineeringModel,
    ScalingModel,
    SensitivitySolver,
    ServingModel,
    SingleNodeModel,
    SustainabilityModel,
    SynthesisSolver,
    TailLatencyModel,
    TopologyModel,
    TransformationModel,
    WeightStreamingModel,
)
from mlsysim.hardware.registry import Hardware
from mlsysim.infra.registry import Grids, Infra
from mlsysim.models.registry import Models
from mlsysim.models.types import TransformerWorkload
from mlsysim.systems.registry import Systems
from mlsysim.systems.types import NetworkFabric


# ── Hardware Fixtures ────────────────────────────────────────────

@pytest.fixture(scope="session")
def h100():
    return Hardware.Cloud.H100


@pytest.fixture(scope="session")
def a100():
    return Hardware.Cloud.A100


@pytest.fixture(scope="session")
def cerebras_cs3():
    return Hardware.Cloud.Cerebras_CS3


@pytest.fixture(scope="session")
def jetson():
    return Hardware.Edge.JetsonOrinNX


@pytest.fixture(scope="session")
def esp32():
    return Hardware.Tiny.ESP32_S3


# ── Model / Workload Fixtures ───────────────────────────────────

@pytest.fixture(scope="session")
def resnet():
    return Models.ResNet50


@pytest.fixture(scope="session")
def gpt3():
    return Models.GPT3


@pytest.fixture(scope="session")
def llama3_8b():
    return Models.Language.Llama3_8B


@pytest.fixture(scope="session")
def llama3_70b():
    return TransformerWorkload(
        name="Llama-3-70B",
        architecture="Transformer",
        parameters=Q_("70e9 param"),
        layers=80,
        hidden_dim=8192,
        heads=64,
    )


# ── Fleet / System Fixtures ─────────────────────────────────────

@pytest.fixture(scope="session")
def research_cluster():
    return Systems.Clusters.Research_256


@pytest.fixture(scope="session")
def frontier_cluster():
    return Systems.Clusters.Frontier_8K


@pytest.fixture(scope="session")
def grid_quebec():
    return Infra.Quebec


@pytest.fixture(scope="session")
def grid_us_avg():
    return Infra.Grids.US_Avg


# ── Solver Fixtures ─────────────────────────────────────────────
# Each solver is stateless, so session scope is safe.

@pytest.fixture(scope="session")
def single_node_solver():
    return SingleNodeModel()


@pytest.fixture(scope="session")
def serving_solver():
    return ServingModel()


@pytest.fixture(scope="session")
def continuous_batching_solver():
    return ContinuousBatchingModel()


@pytest.fixture(scope="session")
def weight_streaming_solver():
    return WeightStreamingModel()


@pytest.fixture(scope="session")
def tail_latency_solver():
    return TailLatencyModel()


@pytest.fixture(scope="session")
def checkpoint_solver():
    return CheckpointModel()


@pytest.fixture(scope="session")
def distributed_solver():
    return DistributedModel()


@pytest.fixture(scope="session")
def reliability_solver():
    return ReliabilityModel()


@pytest.fixture(scope="session")
def economics_solver():
    return EconomicsModel()


@pytest.fixture(scope="session")
def sustainability_solver():
    return SustainabilityModel()


@pytest.fixture(scope="session")
def data_solver():
    return DataModel()


@pytest.fixture(scope="session")
def scaling_solver():
    return ScalingModel()


@pytest.fixture(scope="session")
def orchestration_solver():
    return OrchestrationModel()


@pytest.fixture(scope="session")
def compression_solver():
    return CompressionModel()


@pytest.fixture(scope="session")
def efficiency_solver():
    return EfficiencyModel()


@pytest.fixture(scope="session")
def transformation_solver():
    return TransformationModel()


@pytest.fixture(scope="session")
def topology_solver():
    return TopologyModel()


@pytest.fixture(scope="session")
def inference_scaling_solver():
    return InferenceScalingModel()


@pytest.fixture(scope="session")
def sensitivity_solver():
    return SensitivitySolver()


@pytest.fixture(scope="session")
def synthesis_solver():
    return SynthesisSolver()


@pytest.fixture(scope="session")
def responsible_engineering_solver():
    return ResponsibleEngineeringModel()
