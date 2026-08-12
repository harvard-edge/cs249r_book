"""Audit-fix-verify loop for the MIT Press editorial standard.

Scans textbook content against the project's prose style guide (maintained
outside this repository), applies safe fixes under strict safety gates, and
verifies the result.

The five-stage cycle: SCAN -> PLAN -> FIX -> VERIFY -> REPORT.
Verification is the load-bearing stage; do not skip it.
"""
