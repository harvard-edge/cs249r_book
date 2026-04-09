"""Audit ledger model and JSON serialization.

The ledger is the single source of truth for what the scanner found,
what the fixer applied, and what the verifier confirmed. Format per
Pass 15 plan section 3.

Issue lifecycle:
  open                  - fresh from scanner, not yet acted on
  fixed-script-lane     - script lane applied the fix, awaiting verify
  fixed-subagent-lane   - subagent returned an edit, applied, awaiting verify
  verified              - verify stage passed
  committed-<sha>       - committed in this commit
  failed-script-lane    - script's safety check failed, rolled back
  failed-verification   - verify stage rejected
  failed-subagent-parse - subagent returned non-JSON or malformed output
  accepted              - protected context, never touch
  deferred              - out of scope for this run
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── Status constants ────────────────────────────────────────────────────────

STATUS_OPEN = "open"
STATUS_FIXED_SCRIPT = "fixed-script-lane"
STATUS_FIXED_SUBAGENT = "fixed-subagent-lane"
STATUS_VERIFIED = "verified"
STATUS_FAILED_SCRIPT = "failed-script-lane"
STATUS_FAILED_VERIFY = "failed-verification"
STATUS_FAILED_SUBAGENT = "failed-subagent-parse"
STATUS_ACCEPTED = "accepted"
STATUS_DEFERRED = "deferred"

ALL_STATUSES = (
    STATUS_OPEN,
    STATUS_FIXED_SCRIPT,
    STATUS_FIXED_SUBAGENT,
    STATUS_VERIFIED,
    STATUS_FAILED_SCRIPT,
    STATUS_FAILED_VERIFY,
    STATUS_FAILED_SUBAGENT,
    STATUS_ACCEPTED,
    STATUS_DEFERRED,
)


# ── Issue model ─────────────────────────────────────────────────────────────


@dataclass
class Issue:
    """A single editorial issue found by the scanner.

    Fields map 1:1 to the JSON schema in Pass 15 plan section 3.

    Required fields (scanner must set):
      id               - unique id within the ledger, e.g. "vol1-h3-titlecase-0042"
      category         - category name matching one of the check modules
      rule             - human reference like "book-prose-merged.md section 10.9"
      file             - absolute or repo-relative path to the source file
      line             - 1-indexed line number
      before           - exact source substring (for audit and verification)

    Optional fields (scanner may set):
      col              - column offset on the line (0-indexed)
      suggested_after  - what the fixer should replace `before` with
      rule_text        - short excerpt of the rule for readability
      reason           - free-text explanation, esp. for needs_subagent
      confidence       - "high" | "medium" | "low" - subagent output field

    Lane routing (scanner must set at least one):
      auto_fixable     - script lane can handle this deterministically
      needs_subagent   - requires per-chapter judgment review
      protected_context - marked accepted, never touch (e.g. inside index)

    Lifecycle (fixer / verifier / orchestrator update):
      status           - one of ALL_STATUSES
      commit_sha       - set when committed
      verified_at      - set when verified
      error            - free-text error if failed
    """

    # Required
    id: str
    category: str
    rule: str
    file: str
    line: int
    before: str

    # Optional content
    col: int = 0
    suggested_after: str = ""
    rule_text: str = ""
    reason: str = ""
    confidence: str = "high"

    # Lane routing
    auto_fixable: bool = False
    needs_subagent: bool = False
    protected_context: Optional[str] = None  # name of the protection if accepted

    # Lifecycle
    status: str = STATUS_OPEN
    commit_sha: str = ""
    verified_at: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output. Omits empty/default fields."""
        d = asdict(self)
        # Omit defaults to keep the JSON readable
        for key in list(d.keys()):
            if d[key] in ("", 0, False, None) and key not in (
                "id", "category", "rule", "file", "line", "before", "status",
            ):
                del d[key]
        return d


# ── Ledger container ────────────────────────────────────────────────────────


@dataclass
class Ledger:
    """The audit ledger: scanner output + lifecycle tracking."""

    scope: str  # "vol1", "vol2", or "vol1+vol2"
    scan_date: str = ""
    scanner_version: str = "1.0"
    rule_file_sha: str = ""
    issues: list[Issue] = field(default_factory=list)

    def __post_init__(self):
        if not self.scan_date:
            self.scan_date = datetime.now(timezone.utc).isoformat()

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def summary(self) -> dict[str, Any]:
        """Compute summary counts for the Pass 15 plan §3 schema."""
        by_category: dict[str, int] = {}
        by_status: dict[str, int] = {status: 0 for status in ALL_STATUSES}
        for issue in self.issues:
            by_category[issue.category] = (
                by_category.get(issue.category, 0) + 1
            )
            by_status[issue.status] = by_status.get(issue.status, 0) + 1
        return {
            "total_issues": len(self.issues),
            "by_category": by_category,
            "by_status": by_status,
        }

    def to_json(self) -> str:
        """Serialize the ledger to a formatted JSON string."""
        return json.dumps(
            {
                "scope": self.scope,
                "scan_date": self.scan_date,
                "scanner_version": self.scanner_version,
                "rule_file_sha": self.rule_file_sha,
                "issues": [i.to_dict() for i in self.issues],
                "summary": self.summary(),
            },
            indent=2,
            ensure_ascii=False,
        )

    def save(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        data = json.loads(path.read_text(encoding="utf-8"))
        issues = [Issue(**raw) for raw in data.get("issues", [])]
        return cls(
            scope=data["scope"],
            scan_date=data.get("scan_date", ""),
            scanner_version=data.get("scanner_version", "1.0"),
            rule_file_sha=data.get("rule_file_sha", ""),
            issues=issues,
        )

    # ── Query helpers ──

    def open_issues(self) -> list[Issue]:
        return [i for i in self.issues if i.status == STATUS_OPEN]

    def issues_by_category(self, category: str) -> list[Issue]:
        return [i for i in self.issues if i.category == category]

    def issues_by_file(self, file: str) -> list[Issue]:
        return [i for i in self.issues if i.file == file]

    def open_by_category(self) -> dict[str, list[Issue]]:
        result: dict[str, list[Issue]] = {}
        for issue in self.open_issues():
            result.setdefault(issue.category, []).append(issue)
        return result


def make_issue_id(scope: str, category: str, counter: int) -> str:
    """Generate a deterministic issue ID like 'vol1-h3-titlecase-0042'."""
    return f"{scope}-{category}-{counter:04d}"
