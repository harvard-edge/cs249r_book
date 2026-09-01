"""Base check definitions and registry for native Physical AI checks."""

from __future__ import annotations

import abc
from typing import List, Optional, Type

from ..context import BookContext
from ..report import LintReport


class BaseCheck(abc.ABC):
    """Abstract base class for all native book validation checks."""
    name: str = "base"
    description: str = "Base check description"
    category: str = "general"  # "semantic", "formatting", "editorial", "structure", "layout", "orphans"

    @abc.abstractmethod
    def run(self, ctx: BookContext, report: LintReport) -> None:
        """Executes check logic against the book context and registers issues."""
        pass


class CheckRegistry:
    """Registry that holds and organizes all active native checks."""
    _checks: List[BaseCheck] = []

    @classmethod
    def register(cls, check_cls: Type[BaseCheck]):
        cls._checks.append(check_cls())
        return check_cls

    @classmethod
    def get_checks(cls, category: Optional[str] = None) -> List[BaseCheck]:
        if category:
            return [c for c in cls._checks if c.category == category]
        return list(cls._checks)
