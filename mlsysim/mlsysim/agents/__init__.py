"""Agents package for agentic ML system profiles and execution harnesses."""

from .types import (
    AgentArchitecture,
    AgentWorkload,
    RemediationWorkload,
    ResearchWorkload,
    SupportWorkload,
)
from .registry import Agents, ReferencePlatforms

AgentPlatforms = ReferencePlatforms

__all__ = [
    "AgentArchitecture",
    "AgentWorkload",
    "ResearchWorkload",
    "RemediationWorkload",
    "SupportWorkload",
    "Agents",
    "AgentPlatforms",
    "ReferencePlatforms",
]
