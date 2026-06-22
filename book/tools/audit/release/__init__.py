"""MIT Press Release Deep Audit scripts.

Phase A scripts produce ground-truth ledgers from already-extracted editor
JSON inputs. By default, inputs are read from
``book/tools/audit/release/data/`` and outputs are written under
``book/quarto/_build/release_audit/``. Set ``MLSYSBOOK_RELEASE_AUDIT_DATA``
or ``MLSYSBOOK_RELEASE_AUDIT_OUT`` to override those locations.

These scripts only measure; they never edit the manuscript.
"""
