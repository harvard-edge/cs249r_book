"""Small educational SUT protocol types.

MLPerf EDU does not claim to implement the official MLPerf LoadGen API.  The
types in this module are used by Lab 2 to make query inputs explicit while the
lab drives its SUT locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QuerySample:
    """One locally generated educational query."""

    id: int
    index: int
    arrival_time: float


@dataclass(frozen=True)
class QuerySampleResponse:
    """One locally measured educational response."""

    id: int
    response_data: Any
    arrival_time: float
    completion_time: float
    latency: float
