"""Central check catalog schema, issue templates, and catalog loader.

Provides Pydantic models for check groups, scopes, and error code definitions,
allowing all diagnostics, triggers, fix templates, and help panels to be
centrally defined in `binder/cli/data/check_catalog.yaml`.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class CheckErrorCode(BaseModel):
    """Template and metadata for a single check error code."""

    code: str
    trigger: str
    fix: str
    rule_doc: str = ""
    message_template: str = ""
    suggestion_template: str = ""
    auto_fix_cmd: str = ""


class CheckScopeDoc(BaseModel):
    """Documentation and runtime expectations for one check scope."""

    name: str
    when: str
    cost: str
    invariant: str
    needs: str = "—"


class CheckGroupDoc(BaseModel):
    """Catalog documentation and error code registry for one check group."""

    group: str
    title: str
    description: str = ""
    scopes: List[CheckScopeDoc] = Field(default_factory=list)
    error_codes: List[CheckErrorCode] = Field(default_factory=list)
    common_invocations: List[List[str]] = Field(default_factory=list)
    deep_dive: str = ""


class CheckCatalog(BaseModel):
    """Root check catalog containing all registered groups."""

    groups: Dict[str, CheckGroupDoc] = Field(default_factory=dict)

    def get_group(self, group: str) -> Optional[CheckGroupDoc]:
        """Return the documentation and error code registry for a group."""
        return self.groups.get(group)

    def get_error(self, code: str) -> Optional[CheckErrorCode]:
        """Look up an error code definition across all registered groups."""
        for grp in self.groups.values():
            for err in grp.error_codes:
                if err.code == code:
                    return err
        return None


_CATALOG_CACHE: Optional[CheckCatalog] = None


def get_check_catalog(catalog_path: Optional[Path] = None) -> CheckCatalog:
    """Load and return the central check catalog singleton.

    Loads from `binder/cli/data/check_catalog.yaml` by default.
    """
    global _CATALOG_CACHE
    if _CATALOG_CACHE is not None and catalog_path is None:
        return _CATALOG_CACHE

    if catalog_path is None:
        catalog_path = Path(__file__).resolve().parent.parent / "data" / "check_catalog.yaml"

    if not catalog_path.is_file():
        catalog = CheckCatalog()
        if catalog_path.name == "check_catalog.yaml":
            _CATALOG_CACHE = catalog
        return catalog

    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    groups_data = raw.get("groups", {})
    catalog = CheckCatalog(groups=groups_data)
    if catalog_path.name == "check_catalog.yaml":
        _CATALOG_CACHE = catalog
    return catalog
