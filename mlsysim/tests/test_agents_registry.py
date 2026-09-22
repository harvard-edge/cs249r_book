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


def test_calc_kv_cache_bytes_per_token():
    from mlsysim import Models
    from mlsysim.physics import calc_kv_cache_bytes_per_token
    from mlsysim.core.units import ureg

    # From Llama 3 70B reference model: 80 layers, 8 KV heads, head_dim 128, 2 bytes/elem
    # 2 * 80 * 8 * 128 * 2 = 327,680 bytes = 320 KiB
    m_token = calc_kv_cache_bytes_per_token(model=Models.Language.Llama3_70B)
    assert m_token.to("byte").magnitude == pytest.approx(327_680)
    assert m_token.to("KiB").magnitude == pytest.approx(320.0)

    # From CodingAgents.SWE_Bench_Runner architecture
    swe_token = calc_kv_cache_bytes_per_token(model=Agents.Coding.SWE_Bench_Runner)
    assert swe_token.to("KiB").magnitude == pytest.approx(320.0)

    # Explicit parameters
    expl_token = calc_kv_cache_bytes_per_token(n_layers=80, n_kv_heads=8, head_dim=128, bytes_per_elem=2)
    assert expl_token.to("KiB").magnitude == pytest.approx(320.0)


def test_calc_tool_wait_stranded_tax():
    from mlsysim.physics import calc_tool_wait_stranded_tax
    from mlsysim.core.units import ureg

    GB = ureg.gigabyte
    second = ureg.second

    # Chapter 01 numbers: 8x H100 (640 GB total, 160 GB static weights -> 480 GB dynamic)
    # Context 160k tokens, m_kv = 51.2 GB
    # 60s wait @ $24/hr, 4 concurrent agents, 20 pipeline tools
    res = calc_tool_wait_stranded_tax(
        node_hbm_capacity=640.0 * GB,
        model_weights_memory=160.0 * GB,
        kv_memory=51.2 * GB,
        tool_wait_duration=60 * second,
        hourly_node_cost=24.0,
        num_gpus=8,
        num_concurrent_agents=4,
        num_pipeline_tools=20,
    )
    assert res["m_hbm_dynamic"].to(GB).magnitude == pytest.approx(480.0)
    assert res["m_hbm_dynamic_per_gpu"].to(GB).magnitude == pytest.approx(60.0)
    assert res["pct_stranded_single"] == pytest.approx(0.106666, rel=1e-3)
    assert res["m_stranded_concurrent"].to(GB).magnitude == pytest.approx(204.8)
    assert res["pct_stranded_concurrent"] == pytest.approx(0.426666, rel=1e-3)
    assert res["cost_turn"] == pytest.approx(0.40, rel=1e-3)
    assert res["cost_pipeline"] == pytest.approx(8.00, rel=1e-3)


