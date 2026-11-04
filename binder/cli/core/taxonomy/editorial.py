"""Source-backed editorial corrections, without model quorum requirements.

Existing-record changes preserve exact baselines and evidence. Relationship
removal is explicit; concept lifecycle and release operations are separate decisions.
"""

import re
from copy import deepcopy
from urllib.parse import urlparse

from ..concept_graph import ConceptGraphError


def _require(ok, message):
    if not ok:
        raise ConceptGraphError(f"editorial review: {message}")


def _validate_sources(sources: list, ident: str) -> None:
    _require(isinstance(sources, list) and bool(sources), f"cite evidence for {ident}")
    for source in sources:
        _require(
            isinstance(source, dict) and set(source) == {"url", "locator", "supports"},
            "source needs url, locator, supports",
        )
        _require(
            all(isinstance(v, str) and v.strip() for v in source.values()),
            "empty source field",
        )
        parsed = urlparse(source["url"])
        _require(
            parsed.scheme in {"https", "http"} and bool(parsed.netloc),
            "source URL must be absolute",
        )


def _validate_terms(change: dict, record: dict, fields: dict, collection: str) -> None:
    terms = change.get("terms", {})
    _require(isinstance(terms, dict), "term decisions must be a mapping")
    final_aliases = fields.get("aliases", record.get("aliases", []))
    _require(
        isinstance(final_aliases, list)
        and all(isinstance(a, str) and a.strip() for a in final_aliases),
        "entry terms must be nonempty strings",
    )
    _require(len(final_aliases) == len(set(final_aliases)), "duplicate entry term")
    _require(
        set(terms) <= set(final_aliases),
        "term decision must name an active entry term",
    )
    additions = set(final_aliases) - set(record.get("aliases", []))
    _require(additions <= set(terms), "new aliases need an admission decision")
    for term, decision in terms.items():
        _require(
            collection == "concepts"
            and isinstance(decision, dict)
            and set(decision) == {"kind", "rationale"},
            "term decision needs kind and rationale",
        )
        _require(
            decision["kind"]
            in {
                "synonym",
                "acronym",
                "narrower",
                "broader",
                "implementation",
                "related",
            },
            "unknown entry-term relationship",
        )
        _require(
            isinstance(decision["rationale"], str)
            and len(decision["rationale"].strip()) >= 15,
            "explain term equivalence or difference",
        )


def _validate_impact(change: dict, fields: dict, operation: str) -> None:
    if (
        operation in {"add", "retire"}
        or "home" in fields
        or change.get("impact") is not None
    ):
        impact = change.get("impact")
        _require(
            isinstance(impact, dict)
            and set(impact) == {"classification", "assignments", "relationships"},
            "lifecycle changes need classification, assignments, and relationships impact notes",
        )
        _require(
            all(isinstance(v, str) and len(v.strip()) >= 15 for v in impact.values()),
            "explain lifecycle impact",
        )


def _apply_draft_release(doc: dict, review: dict, result: dict) -> None:
    if "release" in review:
        release = review["release"]
        _require(
            isinstance(release, dict) and set(release) == {"from", "to", "rationale"},
            "release needs from, to, rationale",
        )
        _require(release["from"] == doc["version"], "stale release baseline")
        _require(
            isinstance(release["to"], str)
            and re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?-draft", release["to"]),
            "editorial release must remain explicitly draft",
        )

        def version_key(value):
            parts = tuple(map(int, value.removesuffix("-draft").split(".")))
            return parts + (0,) * (3 - len(parts))

        _require(
            version_key(release["to"]) > version_key(release["from"]),
            "release version must advance",
        )
        _require(
            isinstance(release["rationale"], str)
            and len(release["rationale"].strip()) >= 15,
            "explain the release",
        )
        result["version"] = release["to"]


def apply_review(doc: dict, review: dict) -> dict:
    """Validate and replay a review on a copy; never partially mutate the input."""

    _require(
        str(doc.get("version", "")).endswith("-draft"), "corrections require a draft"
    )
    _require(
        isinstance(review, dict)
        and {"reviewer", "reviewer_kind", "decision", "changes"}
        <= set(review)
        <= {"reviewer", "reviewer_kind", "decision", "changes", "release"},
        "expected reviewer, reviewer_kind, decision, and changes",
    )
    _require(
        isinstance(review["reviewer"], str) and bool(review["reviewer"].strip()),
        "name the reviewer",
    )
    _require(
        review["reviewer_kind"] in {"human", "model"},
        "identify human or model review honestly",
    )
    _require(review["decision"] == "accept", "only accepted reviews can be applied")
    changes = review["changes"]
    _require(
        isinstance(changes, list) and (bool(changes) or "release" in review),
        "changes must be nonempty unless recording a release",
    )
    result = deepcopy(doc)
    seen = set()
    for change in changes:
        required = {
            "collection",
            "id",
            "before",
            "fields",
            "rationale",
            "sources",
            "incident_relationships",
        }
        _require(
            isinstance(change, dict)
            and required <= set(change) <= required | {"operation", "impact", "terms"},
            "unexpected change fields",
        )
        operation = change.get("operation", "update")
        _require(
            operation in {"update", "remove", "add", "retire"}, "unsupported operation"
        )
        collection, ident = change["collection"], change["id"]
        _require(
            collection in {"concepts", "relationships"} and isinstance(ident, str),
            "invalid record selector",
        )
        _require((collection, ident) not in seen, f"duplicate change for {ident}")
        seen.add((collection, ident))
        matches = [r for r in result[collection] if r["id"] == ident]
        if operation == "add":
            _require(
                not matches and change["before"] is None,
                f"addition requires an absent identity {ident}",
            )
            _require(
                ident not in {r["id"] for r in doc.get("retired", [])},
                "retired identities cannot be reused",
            )
            record = {}
        else:
            _require(len(matches) == 1, f"unknown or duplicate identity {ident}")
            record = matches[0]
        _require(
            operation == "add" or record == change["before"],
            f"stale baseline for {ident}",
        )
        fields = change["fields"]
        allowed = (
            {"label", "definition", "aliases", "home", "family", "scope"}
            if collection == "concepts"
            else {"predicate", "conditions", "relation", "kind", "source", "target"}
        )
        if operation == "add":
            _require(
                isinstance(fields, dict) and fields.get("id") == ident,
                "addition needs the complete record and matching id",
            )
        elif operation == "retire":
            _require(
                collection == "concepts"
                and isinstance(fields, dict)
                and set(fields) == {"replaced_by"},
                "retirement needs one active replacement",
            )
            _require(fields["replaced_by"] != ident, "cannot retire to self")
        elif operation == "remove":
            _require(
                collection == "relationships" and fields == {},
                "only relationships can be removed; fields must be empty",
            )
        else:
            _require(
                isinstance(fields, dict)
                and (bool(fields) or bool(change.get("terms")))
                and set(fields) <= allowed,
                f"unsupported correction on {ident}",
            )
            if collection == "relationships" and (
                "relation" in fields or "kind" in fields
            ):
                relation = fields.get("relation", record.get("relation"))
                kinds = {r["id"]: r["kind"] for r in doc.get("relations", [])}
                _require(
                    relation in kinds
                    and fields.get("kind", record.get("kind")) == kinds[relation],
                    "relation and kind must match the controlled vocabulary",
                )
        _require(
            isinstance(change["rationale"], str)
            and len(change["rationale"].strip()) >= 15,
            f"explain {ident}",
        )
        _validate_sources(change["sources"], ident)
        expected_edges = (
            sorted(
                e["id"]
                for e in doc["relationships"]
                if e["source"] == ident or e["target"] == ident
            )
            if collection == "concepts"
            else []
        )
        _require(
            change["incident_relationships"] == expected_edges,
            f"record every incident relationship for {ident}",
        )
        _validate_terms(change, record, fields, collection)
        _validate_impact(change, fields, operation)
        if operation == "add":
            result[collection].append(deepcopy(fields))
            continue
        if operation == "retire":
            result["concepts"].remove(record)
            result.setdefault("retired", []).append(
                {
                    "id": ident,
                    "replaced_by": fields["replaced_by"],
                    "version": doc["version"],
                    "reason": change["rationale"],
                }
            )
            continue
        if operation == "remove":
            result[collection].remove(record)
            continue
        for key, value in fields.items():
            _require(value != record.get(key), f"no-op field {key} on {ident}")
            if key == "aliases":
                _require(
                    isinstance(value, list) and all(isinstance(a, str) for a in value),
                    "aliases must be strings",
                )
            else:
                _require(
                    isinstance(value, str) and bool(value.strip()),
                    f"empty {key} on {ident}",
                )
            record[key] = deepcopy(value)
    active = {c["id"] for c in result["concepts"]}
    # Fail rather than silently retagging relationships or assignments.
    for change in changes:
        if change.get("operation") == "retire":
            _require(
                change["fields"]["replaced_by"] in active,
                "retirement replacement must remain active",
            )
            _require(
                not any(
                    change["id"] in (e["source"], e["target"])
                    for e in result["relationships"]
                ),
                "retirement leaves incident relationships; explicitly remove or rewire them",
            )
    _apply_draft_release(doc, review, result)
    return result


def advance_provenance(state: dict, entry: dict) -> None:
    """Keep correction decisions and removed-entry-term history in the replay."""
    if state["pending_campaign_ids"] or state["pending_pre_evaluation_override"]:
        raise ConceptGraphError(
            "editorial review: resolve pending historical campaigns first"
        )
    for change in entry["replayed_patch"]["changes"]:
        if change["collection"] != "concepts":
            continue
        ident = change["id"]
        if change.get("operation") == "retire":
            old = change["before"]
            state.setdefault("retired_concepts", {})[ident] = {
                "former_label": old["label"],
                "former_aliases": old.get("aliases", []),
                "replaced_by": change["fields"]["replaced_by"],
                "reason": change["rationale"],
                "editorial_review": entry["campaign_id"],
                "previous_provenance": state["concepts"].pop(ident, {}),
            }
            state["grandfathered_draft_alias_exceptions"].pop(ident, None)
            continue
        record = state["concepts"].setdefault(ident, {})
        if change.get("operation") == "add":
            record["editorial_admission"] = {
                "campaign_id": entry["campaign_id"],
                "sources": deepcopy(change["sources"]),
                "rationale": change["rationale"],
                "impact": deepcopy(change["impact"]),
            }
        record.setdefault("editorial_reviews", []).append(entry["campaign_id"])
        for term, decision in change.get("terms", {}).items():
            record.setdefault("entry_term_decisions", {})[term] = {
                **deepcopy(decision),
                "sources": deepcopy(change["sources"]),
                "campaign_id": entry["campaign_id"],
            }
        if "aliases" in change["fields"]:
            removed = set((change["before"] or {}).get("aliases", [])) - set(
                change["fields"]["aliases"]
            )
            for alias in sorted(removed):
                record.setdefault("removed_aliases", {})[alias] = change["rationale"]
                record.get("alias_metadata", {}).pop(alias, None)
                record.get("entry_term_decisions", {}).pop(alias, None)
            exception = state["grandfathered_draft_alias_exceptions"].get(ident)
            if exception:
                exception["aliases"] = [
                    a for a in exception["aliases"] if a not in removed
                ]
                if not exception["aliases"]:
                    del state["grandfathered_draft_alias_exceptions"][ident]
