"""Registry of agent system architectures, deliberation profiles, and execution harnesses."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from ..models.registry import Models
from ..systems.registry import Systems
from .types import AgentArchitecture


class CodingAgents(Registry):
    """Hermetic coding and tool-use agents (SWE-bench / repository level)."""

    SWE_Bench_Runner = AgentArchitecture(
        name="SWE-bench Coding Agent Harness",
        paradigm="ReAct / Tool-Use",
        context_window=128_000 * ureg.token,
        working_memory=32_000 * ureg.token,
        sandbox_startup_latency=5.0 * ureg.ms,
        sandbox_snapshot_restore_latency=15.0 * ureg.ms,
        sandbox_memory_footprint=512 * ureg.MiB,
        max_trajectory_steps=30,
        reference_model=Models.Language.Llama3_70B,
        serving_node=Systems.Nodes.HGX_H100_EPYC,
        metadata=Metadata(provenance=pc.SWE_BENCH_HARNESS),
    )
    SWE_Bench_Workstation = AgentArchitecture(
        name="SWE-bench Workstation Developer Harness",
        paradigm="ReAct / Tool-Use",
        context_window=128_000 * ureg.token,
        working_memory=32_000 * ureg.token,
        sandbox_startup_latency=25.0 * ureg.ms,
        sandbox_snapshot_restore_latency=45.0 * ureg.ms,
        sandbox_memory_footprint=512 * ureg.MiB,
        max_trajectory_steps=30,
        reference_model=Models.Language.Llama3_8B,
        serving_node=Systems.Nodes.Workstation_M3Max,
        metadata=Metadata(
            provenance=pc.DEPLOYMENT_ENVELOPES,
            description="Local workstation developer baseline running on Apple Silicon unified memory.",
        ),
    )


class DeliberationAgents(Registry):
    """Test-time deliberation and tree-search reasoning architectures."""

    TreeSearch = AgentArchitecture(
        name="Deliberative MCTS Reasoning Agent",
        paradigm="TreeSearch / PRM Verifier",
        context_window=64_000 * ureg.token,
        working_memory=16_000 * ureg.token,
        deliberation_branches=8,
        verifier_cost_ratio=0.15,
        reference_model=Models.Language.Llama3_70B,
        serving_node=Systems.Nodes.DGX_H100,
        metadata=Metadata(provenance=pc.TEST_TIME_DELIBERATION),
    )


class MultiAgents(Registry):
    """Cooperative multi-agent fleet configurations."""

    SupervisorWorker = AgentArchitecture(
        name="Supervisor-Worker Multi-Agent Fleet",
        paradigm="Hierarchical Orchestrator-Worker",
        context_window=32_000 * ureg.token,
        working_memory=8_000 * ureg.token,
        coordination_overhead_beta=0.03,
        reference_model=Models.Language.Llama3_70B,
        serving_node=Systems.Nodes.DGX_H100,
        metadata=Metadata(provenance=pc.MULTI_AGENT_ORCHESTRATION),
    )


class InteractiveAgents(Registry):
    """Real-time streaming and multimodal interactive agents."""

    StreamingVoice = AgentArchitecture(
        name="Real-Time Streaming Voice Agent",
        paradigm="Audio-to-Audio Streaming Pipeline",
        context_window=8_000 * ureg.token,
        working_memory=2_000 * ureg.token,
        sandbox_startup_latency=0.0 * ureg.ms,
        reference_model=Models.Language.Llama3_70B,
        serving_node=Systems.Nodes.DGX_H100,
        metadata=Metadata(provenance=pc.STREAMING_VOICE_AGENT_PROFILE),
    )


class ReferencePlatforms(Registry):
    """Canonical reference execution platforms for Volume III: The Stochastic Computer."""

    Cloud_DGX_H100 = CodingAgents.SWE_Bench_Runner
    Workstation_Apple = CodingAgents.SWE_Bench_Workstation


class Agents(Registry):
    """Authoritative registry of agent system profiles and execution harnesses."""

    Coding = CodingAgents
    Deliberation = DeliberationAgents
    MultiAgent = MultiAgents
    Interactive = InteractiveAgents
    Platforms = ReferencePlatforms

