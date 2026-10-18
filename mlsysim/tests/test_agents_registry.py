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


def test_lighthouse_workload_profiles():
    """Research, remediation, and support workloads carry one book-wide set of inputs."""
    from mlsysim.agents.types import RemediationWorkload, ResearchWorkload, SupportWorkload

    research = Agents.Research.OrchestratorWorker
    assert isinstance(research, ResearchWorkload)
    assert research.fanout_token_multiple == 15  # anthropic2025multiagent
    assert research.agent_token_multiple == 4  # anthropic2025multiagent
    assert research.chat_input_tokens.to("token").magnitude == pytest.approx(16_000)
    assert research.untrusted_reads_per_task == 100

    remediation = Agents.Remediation.OnCall
    assert isinstance(remediation, RemediationWorkload)
    assert remediation.approval_decision_time.to("minute").magnitude == pytest.approx(15)
    assert remediation.approval_request_rate.to("1/hour").magnitude == pytest.approx(2)
    # Reviewer effort is part of, and shorter than, the elapsed decision time.
    assert remediation.approver_review_time < remediation.approval_decision_time
    assert remediation.approval_cost_per_attempt.to("USD").magnitude == pytest.approx(20.0)

    support = Agents.Support.CustomerOps
    assert isinstance(support, SupportWorkload)
    assert support.shared_prefix < support.session_context
    assert support.session_context.to("token").magnitude == pytest.approx(8_192)
    assert support.customer_reply_time > support.generation_residence

    for workload in (research, remediation, support):
        assert 0 < workload.acceptance_probability <= 1
        assert workload.metadata.provenance is not None
        assert workload.metadata.provenance.notes


def test_workload_rejects_wrong_units():
    from pydantic import ValidationError
    from mlsysim.agents.types import SupportWorkload
    from mlsysim.core.units import ureg

    base = Agents.Support.CustomerOps.model_dump()
    kwargs = {k: getattr(Agents.Support.CustomerOps, k) for k in base}
    kwargs["handoff_cost"] = 4 * ureg.second
    with pytest.raises(ValidationError):
        SupportWorkload(**kwargs)
    kwargs["handoff_cost"] = 4 * ureg.USD
    kwargs["shared_prefix"] = 9_000 * ureg.token
    with pytest.raises(ValidationError):
        SupportWorkload(**kwargs)
