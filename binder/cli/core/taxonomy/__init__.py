"""Taxonomy maintenance internals used by the existing binder commands.

See taxonomy/ARCHITECTURE.md for ownership, dependency direction, and checks.
No separate command-line application or installation is required.
"""

from .hierarchy import load_taxonomy, taxonomy_artifacts

__all__ = ["load_taxonomy", "taxonomy_artifacts"]
