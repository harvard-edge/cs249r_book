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
        Agents.Deliberation.TreeSearch,
        Agents.MultiAgent.SupervisorWorker,
        Agents.Interactive.StreamingVoice,
    ]:
        assert isinstance(agent, AgentArchitecture)
        assert agent.metadata.provenance is not None
        assert agent.metadata.provenance.ref != ""
