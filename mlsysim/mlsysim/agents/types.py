"""Agent system profile, architecture, and recurring workload types."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.units import ureg
from ..core.types import Quantity, Metadata, require_dimensionality, require_unit_family
from ..models.types import Workload
from ..systems.types import Node


class AgentArchitecture(BaseModel):
    """Vetted agent system profile (SWE-bench coding, deliberation tree search, multi-agent fleet, streaming)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    paradigm: str  # ReAct, TreeSearch, MultiAgent, InteractiveStreaming, Flywheel
    context_window: Quantity
    working_memory: Optional[Quantity] = None
    sandbox_startup_latency: Optional[Quantity] = None
    sandbox_snapshot_restore_latency: Optional[Quantity] = None
    sandbox_memory_footprint: Optional[Quantity] = None
    max_trajectory_steps: Optional[int] = None
    reference_model: Optional[Workload] = None
    serving_node: Optional[Node] = None
    deliberation_branches: Optional[int] = None
    verifier_cost_ratio: Optional[float] = None
    coordination_overhead_beta: Optional[float] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("sandbox_startup_latency", mode="after")
    @classmethod
    def _validate_latency(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "sandbox_startup_latency")

    @field_validator("sandbox_snapshot_restore_latency", mode="after")
    @classmethod
    def _validate_restore_latency(cls, v):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, "sandbox_snapshot_restore_latency")

    @field_validator("sandbox_memory_footprint", mode="after")
    @classmethod
    def _validate_sandbox_memory(cls, v):
        if v is None:
            return v
        return require_unit_family(v, ureg.byte, "sandbox_memory_footprint", "data")


def _require_tokens(v, field_name: str):
    """Validate a token count carried as a Pint quantity in ``token`` units."""
    if v is None:
        return None
    q = require_dimensionality(v, ureg.token, field_name)
    if "token" not in set(q.units._units):
        raise ValueError(f"{field_name} must use token units; got {q.units}")
    return q


_CURRENCY_FIELDS = (
    "model_cost_per_attempt",
    "environment_cost_per_attempt",
    "assessment_cost_per_attempt",
)


class AgentWorkload(BaseModel):
    """Recurring agent workload profile: task-level scenario inputs for one archetype.

    An ``AgentArchitecture`` describes a harness (context window, sandbox,
    serving node). An ``AgentWorkload`` describes the task a deployment runs
    repeatedly: its horizon, how much untrusted content one task reads, what an
    attempt costs outside model tokens, and how often an attempt is accepted.
    Per-attempt charges are in USD; the horizon is in model turns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    archetype: str  # untrusted reader, irreversible actor, interactive responder
    horizon_turns: int = Field(gt=0)
    untrusted_reads_per_task: int = Field(ge=0)
    acceptance_probability: float = Field(gt=0, le=1)
    model_cost_per_attempt: Optional[Quantity] = None
    environment_cost_per_attempt: Quantity
    assessment_cost_per_attempt: Quantity
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator(*_CURRENCY_FIELDS, mode="after")
    @classmethod
    def _validate_currency(cls, v, info):
        if v is None:
            return v
        return require_unit_family(v, ureg.USD, info.field_name, "currency")


class ResearchWorkload(AgentWorkload):
    """Orchestrator-worker research workload: fan-out over untrusted sources.

    ``agent_token_multiple`` and ``fanout_token_multiple`` are token use of a
    single agent and of a multi-agent system relative to one chat answer;
    ``serial_fraction`` and ``coherency_coefficient`` are the Universal
    Scalability Law coefficients (sigma, kappa) of its fan-out.
    """

    chat_input_tokens: Quantity
    chat_output_tokens: Quantity
    agent_token_multiple: float = Field(gt=1)
    fanout_token_multiple: float = Field(gt=1)
    serial_fraction: float = Field(ge=0, le=1)
    coherency_coefficient: float = Field(ge=0)

    @field_validator("chat_input_tokens", "chat_output_tokens", mode="after")
    @classmethod
    def _validate_tokens(cls, v, info):
        return _require_tokens(v, info.field_name)


class RemediationWorkload(AgentWorkload):
    """On-call remediation workload: approved changes to shared production state.

    Timing fields decompose time to mitigate (diagnosis, approval decision,
    apply and verify). ``approver_review_time`` is reviewer effort per approval,
    a part of the elapsed ``approval_decision_time``. The pivot fields price
    expected irreversible loss around a canary-then-fleet rollout.
    """

    approval_request_rate: Quantity
    diagnosis_time: Quantity
    approval_decision_time: Quantity
    apply_verify_time: Quantity
    approver_review_time: Quantity
    approver_hourly_rate: Quantity
    wrong_change_probability: float = Field(ge=0, le=1)
    wrong_change_impact: Quantity
    canary_catch_rate: float = Field(ge=0, le=1)
    canary_recovery_fraction: float = Field(ge=0, le=1)

    @field_validator("approval_request_rate", mode="after")
    @classmethod
    def _validate_rate(cls, v):
        return require_dimensionality(v, 1 / ureg.second, "approval_request_rate")

    @field_validator("diagnosis_time", "approval_decision_time", "apply_verify_time", "approver_review_time", mode="after")
    @classmethod
    def _validate_time(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("approver_hourly_rate", mode="after")
    @classmethod
    def _validate_hourly(cls, v):
        q = require_dimensionality(v, ureg.USD / ureg.hour, "approver_hourly_rate")
        if "USD" not in set(q.units._units) and "dollar" not in set(q.units._units):
            raise ValueError(f"approver_hourly_rate must use USD per time; got {q.units}")
        return q

    @field_validator("wrong_change_impact", mode="after")
    @classmethod
    def _validate_impact(cls, v):
        return require_unit_family(v, ureg.USD, "wrong_change_impact", "currency")

    @property
    def approval_cost_per_attempt(self) -> Quantity:
        """Reviewer cost of one approval: review effort times the hourly rate."""
        return (self.approver_review_time * self.approver_hourly_rate).to(ureg.USD)


class SupportWorkload(AgentWorkload):
    """Customer-support workload: many short sessions against a response deadline.

    ``session_context`` is the context a session holds at its latest turn
    (policy prefix, customer record, conversation); ``shared_prefix`` is the
    part identical across sessions (system prompt, policy, tool schemas).
    """

    session_context: Quantity
    shared_prefix: Quantity
    reply_tokens: Quantity
    generation_residence: Quantity
    customer_reply_time: Quantity
    handoff_cost: Quantity

    @field_validator("session_context", "shared_prefix", "reply_tokens", mode="after")
    @classmethod
    def _validate_tokens(cls, v, info):
        return _require_tokens(v, info.field_name)

    @field_validator("generation_residence", "customer_reply_time", mode="after")
    @classmethod
    def _validate_time(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("handoff_cost", mode="after")
    @classmethod
    def _validate_handoff(cls, v):
        return require_unit_family(v, ureg.USD, "handoff_cost", "currency")

    @model_validator(mode="after")
    def _prefix_within_session(self):
        if self.shared_prefix > self.session_context:
            raise ValueError("shared_prefix cannot exceed session_context")
        return self
