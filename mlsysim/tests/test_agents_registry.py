"""Unit tests for mlsysim.agents registry and AgentArchitecture models."""

from __future__ import annotations

import pytest
from mlsysim import Agents
from mlsysim.agents.types import AgentArchitecture


def test_agents_registry_architectures_exist():
    assert Agents.Coding.SWE_Bench_Runner.name == "SWE-bench Coding Agent Harness"
    assert Agents.Deliberation.TreeSearch.name == "Deliberative MCTS Reasoning Agent"
    assert Agents.MultiAgent.SupervisorWorker.name == "Supervisor-Worker Multi-Agent Fleet"
    assert Agents.Interactive.StreamingVoice.name == "Real-Time Streaming Voice Agent"


def test_agents_registry_units():
    swe = Agents.Coding.SWE_Bench_Runner
    assert swe.context_window.to("token").magnitude == pytest.approx(128_000)
    assert swe.sandbox_startup_latency.to("ms").magnitude == pytest.approx(5.0)

    tree = Agents.Deliberation.TreeSearch
    assert tree.deliberation_branches == 8
    assert tree.verifier_cost_ratio == 0.15

    multi = Agents.MultiAgent.SupervisorWorker
    assert multi.coordination_overhead_beta == 0.03


def test_agents_registry_provenance():
    for agent in [
        Agents.Coding.SWE_Bench_Runner,
        Agents.Coding.SWE_Bench_Workstation,
        Agents.Deliberation.TreeSearch,
        Agents.MultiAgent.SupervisorWorker,
        Agents.Interactive.StreamingVoice,
    ]:
        assert isinstance(agent, AgentArchitecture)
        assert agent.metadata.provenance is not None
        assert agent.metadata.provenance.ref != ""


def test_agent_platforms_hierarchy():
    from mlsysim import AgentPlatforms, Systems, Hardware, Models

    cloud = AgentPlatforms.Cloud_DGX_H100
    assert cloud is Agents.Coding.SWE_Bench_Runner
    assert cloud.serving_node is Systems.Nodes.HGX_H100_EPYC
    assert cloud.serving_node.accelerator is Hardware.Cloud.H100
    assert cloud.serving_node.host_cpu == "Dual AMD EPYC 9654"
    assert cloud.serving_node.host_cpu_cores == 192
    assert cloud.sandbox_startup_latency.to("ms").magnitude == pytest.approx(5.0)
    assert cloud.sandbox_snapshot_restore_latency.to("ms").magnitude == pytest.approx(15.0)
    assert cloud.sandbox_memory_footprint.to("MiB").magnitude == pytest.approx(512.0)
    assert cloud.max_trajectory_steps == 30
    assert cloud.reference_model is Models.Language.Llama3_70B

    ws = AgentPlatforms.Workstation_Apple
    assert ws is Agents.Coding.SWE_Bench_Workstation
    assert ws.serving_node is Systems.Nodes.Workstation_M3Max
    assert ws.serving_node.accelerator is Hardware.Workstation.MacBookM3Max
    assert ws.serving_node.host_cpu == "Apple M3 Max 16-Core"
    assert ws.serving_node.host_cpu_cores == 16
    assert ws.reference_model is Models.Language.Llama3_8B
    assert ws.sandbox_startup_latency.to("ms").magnitude == pytest.approx(25.0)

