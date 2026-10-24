"""Registry of agent system architectures, recurring workload profiles, and execution harnesses."""

from ..core.registry import Registry
from ..core.types import Metadata
from ..core.units import ureg
from ..core import provenance_catalog as pc
from ..models.registry import Models
from ..systems.registry import Systems
from .types import AgentArchitecture, RemediationWorkload, ResearchWorkload, SupportWorkload


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
        metadata=Metadata(
            provenance=pc.SWE_BENCH_HARNESS,
            description="Hermetic SWE-bench coding agent harness running on HGX H100 server.",
        ),
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
        metadata=Metadata(
            provenance=pc.TEST_TIME_DELIBERATION,
            description="Test-time deliberative MCTS reasoning agent with PRM verifier.",
        ),
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
        metadata=Metadata(
            provenance=pc.MULTI_AGENT_ORCHESTRATION,
            description="Hierarchical supervisor-worker cooperative multi-agent fleet configuration.",
        ),
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
        metadata=Metadata(
            provenance=pc.STREAMING_VOICE_AGENT_PROFILE,
            description="Real-time streaming audio-to-audio interactive voice agent pipeline.",
        ),
    )


class ResearchAgents(Registry):
    """Research agents: orchestrator-worker fan-out over untrusted external sources."""

    OrchestratorWorker = ResearchWorkload(
        name="Orchestrator-Worker Research Agent",
        archetype="Untrusted reader",
        horizon_turns=20,
        untrusted_reads_per_task=100,
        acceptance_probability=0.60,
        environment_cost_per_attempt=0.10 * ureg.USD,
        assessment_cost_per_attempt=0.05 * ureg.USD,
        chat_input_tokens=16_000 * ureg.token,
        chat_output_tokens=800 * ureg.token,
        agent_token_multiple=4,
        fanout_token_multiple=15,
        serial_fraction=0.10,
        coherency_coefficient=0.002,
        metadata=Metadata(
            provenance=pc.RESEARCH_AGENT_WORKLOAD,
            description="Fan-out orchestrator-worker research agent workload over untrusted external sources.",
        ),
    )


class RemediationAgents(Registry):
    """On-call remediation agents: approved changes to shared production state."""

    OnCall = RemediationWorkload(
        name="On-Call Remediation Agent",
        archetype="Irreversible actor",
        horizon_turns=10,
        untrusted_reads_per_task=10,
        acceptance_probability=0.70,
        model_cost_per_attempt=0.15 * ureg.USD,
        environment_cost_per_attempt=0.10 * ureg.USD,
        assessment_cost_per_attempt=0.20 * ureg.USD,
        approval_request_rate=2 / ureg.hour,
        diagnosis_time=6 * ureg.minute,
        approval_decision_time=15 * ureg.minute,
        apply_verify_time=9 * ureg.minute,
        approver_review_time=10 * ureg.minute,
        approver_hourly_rate=120 * ureg.USD / ureg.hour,
        wrong_change_probability=0.04,
        wrong_change_impact=12_000 * ureg.USD,
        canary_catch_rate=0.80,
        canary_recovery_fraction=0.90,
        metadata=Metadata(
            provenance=pc.REMEDIATION_AGENT_WORKLOAD,
            description="On-call production remediation agent workload executing approved mutations.",
        ),
    )


class SupportAgents(Registry):
    """Customer-support agents: many short interactive sessions under a latency deadline."""

    CustomerOps = SupportWorkload(
        name="Customer-Operations Support Agent",
        archetype="Interactive responder",
        horizon_turns=6,
        untrusted_reads_per_task=6,
        acceptance_probability=0.85,
        model_cost_per_attempt=0.03 * ureg.USD,
        environment_cost_per_attempt=0.01 * ureg.USD,
        assessment_cost_per_attempt=0.01 * ureg.USD,
        session_context=8_192 * ureg.token,
        shared_prefix=6_144 * ureg.token,
        reply_tokens=150 * ureg.token,
        generation_residence=2 * ureg.second,
        customer_reply_time=20 * ureg.second,
        handoff_cost=4 * ureg.USD,
        metadata=Metadata(
            provenance=pc.SUPPORT_AGENT_WORKLOAD,
            description="High-throughput interactive customer-support agent workload.",
        ),
    )


class ReferencePlatforms(Registry):
    """Canonical reference execution platforms for Agentic Machine Learning Systems."""

    Cloud_DGX_H100 = CodingAgents.SWE_Bench_Runner
    Workstation_Apple = CodingAgents.SWE_Bench_Workstation


class Agents(Registry):
    """Authoritative registry of agent system profiles and execution harnesses."""

    Coding = CodingAgents
    Deliberation = DeliberationAgents
    MultiAgent = MultiAgents
    Interactive = InteractiveAgents
    Research = ResearchAgents
    Remediation = RemediationAgents
    Support = SupportAgents
    Platforms = ReferencePlatforms

