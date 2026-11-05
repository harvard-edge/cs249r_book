"""Validate and reconstruct the taxonomy change-history chain."""
from __future__ import annotations

import hashlib
import re
import subprocess
from copy import deepcopy
from pathlib import Path

from ..concept_graph import ConceptGraphError
from .governance import (
    verify_release_governance_receipt,
)
from .history import (
    _compute_warrant_lineages,
    _load_burned_clusters,
    init_genesis_provenance_state,
    load_genesis_baseline_taxonomy,
    replay_patch_on_taxonomy_doc,
    resolve_cluster_merges,
    verify_burned_clusters_append_only,
)
from .legacy_policy import (
    _ALLOWED_GUIDES_OP_PREFIXES,
    _ALLOWED_OP_PREFIXES,
    _ALLOWED_PROV_ALIAS_TYPES,
    _PRE_SIDECAR_VERSIONS,
    _REQUIRED_WITNESS_KEYS,
    BURNED_CLUSTERS_REL_PATH,
    GENESIS_PRE_SIDECAR_NON_GUIDES_SHA256,
    GENESIS_PRE_SIDECAR_TAXONOMY_SHA256,
    PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS,
)
from .records import (
    _HEX64_RE,
    _norm_prov_tokens,
    canonical_patch_json_sha256,
    canonical_taxonomy_json,
    compute_campaign_entry_sha256,
    compute_implementation_sha256,
    compute_non_guides_taxonomy_sha256,
    compute_sidecar_content_sha256,
    expected_status_for_version,
    json_loads_strict,
)


def _verify_enrichment_alias_meta(raw_av: object, enriched_av: object, cid: str, ak_clean: str) -> dict:
    if not isinstance(raw_av, dict) or not isinstance(enriched_av, dict):
        raise ConceptGraphError(f"schema sync: missing or malformed alias_metadata enrichment on {cid} alias {ak_clean!r}")
    exp_type = str(raw_av.get("type") or "").strip()
    exp_src = " ".join(str(raw_av.get("source") or "").split())
    exp_snote = " ".join(str(raw_av.get("scope_note") or "").split())
    exp_cunits = [str(u).strip() for u in (raw_av.get("checked_dev_units") or [])]
    exp_vq = " ".join(str(raw_av.get("verbatim_quote") or "").split())
    exp_loc = str(raw_av.get("primary_source_locator") or "").strip()
    if (
        enriched_av.get("type") != exp_type
        or enriched_av.get("source") != exp_src
        or enriched_av.get("scope_note") != exp_snote
        or enriched_av.get("checked_dev_units") != exp_cunits
        or enriched_av.get("verbatim_quote") != exp_vq
        or (exp_loc and enriched_av.get("primary_source_locator") != exp_loc)
        or not _HEX64_RE.match(str(enriched_av.get("primary_source_sha256") or ""))
        or len(str(enriched_av.get("primary_source_locator") or "").strip()) < 3
    ):
        raise ConceptGraphError(
            f"schema sync: alias_metadata enrichment on {cid} alias {ak_clean!r} disagrees with reviewed replayed_patch"
        )
    if raw_av.get("primary_source_sha256") and enriched_av.get("primary_source_sha256") != str(raw_av["primary_source_sha256"]).strip():
        raise ConceptGraphError(
            f"schema sync: alias_metadata primary_source_sha256 on {cid} alias {ak_clean!r} disagrees with replayed_patch"
        )
    if exp_type == "upward-posting":
        if enriched_av.get("dev_lineages_requiring_separation") != raw_av.get("dev_lineages_requiring_separation"):
            raise ConceptGraphError(
                f"schema sync: upward-posting dev_lineages_requiring_separation mismatch on {cid} alias {ak_clean!r}"
            )
    return deepcopy(enriched_av)


def _verify_enrichment_warrant_sources(
    raw_ws: object,
    enriched_ws: object,
    cid: str,
    burned_doc: dict | None,
) -> list[dict]:
    if not isinstance(raw_ws, list) or not isinstance(enriched_ws, list) or len(raw_ws) != len(enriched_ws) or len(raw_ws) < 2:
        raise ConceptGraphError(f"schema sync: warrant_sources enrichment on {cid} disagrees with reviewed replayed_patch")
    hist_units = (burned_doc or {}).get("historical_phase3_units") or {}
    merges = resolve_cluster_merges((burned_doc or {}).get("cluster_merges") or {})
    out: list[dict] = []
    for r_item, e_item in zip(raw_ws, enriched_ws):
        if not isinstance(r_item, dict) or not isinstance(e_item, dict):
            raise ConceptGraphError(f"schema sync: malformed warrant_sources enrichment item on {cid}")
        uid = str(r_item.get("unit_id") or "").strip()
        loc = " ".join(str(r_item.get("locator") or "").split())
        if e_item.get("unit_id") != uid or e_item.get("locator") != loc:
            raise ConceptGraphError(f"schema sync: warrant_sources[{uid!r}] on {cid} disagrees with replayed_patch")
        if uid in hist_units:
            hu = hist_units[uid]
            exp_pf = merges.get(str(hu["provenance_family_id"]), str(hu["provenance_family_id"]))
            if (
                e_item.get("stratum") != hu.get("stratum")
                or e_item.get("provenance_family_id") != exp_pf
                or e_item.get("source_type") != hu.get("source_type")
                or e_item.get("origin") != hu.get("origin")
                or bool(e_item.get("editor_affiliated", False)) != bool(hu.get("editor_affiliated", False))
            ):
                raise ConceptGraphError(
                    f"schema sync: warrant_sources[{uid!r}] on {cid} disagrees with pinned historical_phase3_units metadata"
                )
        out.append(deepcopy(e_item))
    return out


def replay_provenance_step(
    prov_state: dict[str, object],
    pre_doc: dict,
    post_doc: dict,
    entry: dict,
    burned_doc: dict | None = None,
) -> None:
    """Deterministically advance top-level provenance state across one campaign_entry."""
    mode = str(entry.get("mode") or "")
    pending_ids: list[str] = list(prov_state["pending_campaign_ids"])  # type: ignore[arg-type]

    if mode == "editorial-review":
        from .editorial import advance_provenance
        advance_provenance(prov_state, entry)
        return

    if mode == "guides-apply":
        if pending_ids or bool(prov_state["pending_pre_evaluation_override"]):
            raise ConceptGraphError(
                "schema sync: guides-apply is forbidden while structural pre-evaluation campaigns remain pending"
            )
        return

    if mode == "finalize-structural-evaluation":
        if not pending_ids:
            raise ConceptGraphError("schema sync: finalize-structural-evaluation requires non-empty pending_campaign_ids")
        covered = entry.get("covered_pending_campaign_ids")
        if (
            not isinstance(covered, list)
            or not covered
            or sorted(str(x) for x in covered) != sorted(pending_ids)
        ):
            raise ConceptGraphError(
                "schema sync: finalize-structural-evaluation covered_pending_campaign_ids does not cover all pending campaigns"
            )
        prov_state["pending_campaign_ids"] = []
        prov_state["pending_pre_evaluation_override"] = False
        return

    if mode != "full-3family-reconcile":
        raise ConceptGraphError(f"schema sync: unsupported mode {mode!r} in replay_provenance_step")

    camp_id = str(entry.get("campaign_id") or "").strip()
    if len(camp_id) < 3:
        raise ConceptGraphError("schema sync: full-3family-reconcile entry missing valid campaign_id")

    if entry.get("allow_pre_evaluation_apply") is True:
        if camp_id in pending_ids:
            raise ConceptGraphError(f"schema sync: duplicate pending campaign_id {camp_id!r}")
        pending_ids.append(camp_id)
        prov_state["pending_campaign_ids"] = pending_ids
        prov_state["pending_pre_evaluation_override"] = True
    else:
        if pending_ids:
            raise ConceptGraphError(
                "schema sync: non-staged full-3family-reconcile is forbidden while pending_campaign_ids remain unfinalized"
            )
        prov_state["pending_campaign_ids"] = []
        prov_state["pending_pre_evaluation_override"] = False

    change = entry["replayed_patch"]
    enrichment = entry.get("provenance_evidence_enrichment")
    if not isinstance(enrichment, dict):
        raise ConceptGraphError("schema sync: full-3family-reconcile entry missing provenance_evidence_enrichment mapping")
    ws_enrich = enrichment.get("warrant_sources") or {}
    am_enrich = enrichment.get("alias_metadata") or {}
    if not isinstance(ws_enrich, dict) or not isinstance(am_enrich, dict):
        raise ConceptGraphError("schema sync: malformed provenance_evidence_enrichment sub-mappings")

    prov_concepts: dict[str, dict] = prov_state["concepts"]  # type: ignore[assignment]
    prov_subjects: dict[str, dict] = prov_state["subjects"]  # type: ignore[assignment]
    prov_retired: dict[str, dict] = prov_state["retired_concepts"]  # type: ignore[assignment]
    prov_gf: dict[str, dict] = prov_state["grandfathered_draft_alias_exceptions"]  # type: ignore[assignment]

    rows = {row["id"]: row for area in pre_doc["areas"] for row in area["rows"]}
    code2row = {row["code"]: row["id"] for row in rows.values() if "code" in row}
    pre_concepts = {c["id"]: c for c in pre_doc["concepts"]}
    post_concepts = {c["id"]: c for c in post_doc["concepts"]}

    def resolve_row_id(home_ref: str) -> str:
        return code2row.get(home_ref, home_ref)

    expected_ws_cids: set[str] = set()
    expected_am_cids: set[str] = set()
    removed_rel_ids = {str(x).strip() for x in (change.get("relationships_remove") or []) if str(x).strip()}

    for u in change.get("concepts_update", []) or []:
        cid = u["id"].strip()
        post_node = post_concepts.get(cid) or pre_concepts[cid]
        cprov = deepcopy(prov_concepts.get(cid) or {})
        existing_ameta = dict(cprov.get("alias_metadata") or {})

        rem_list = [str(x).strip() for x in u.get("remove_aliases", []) or []]
        rem_norms = {_norm_prov_tokens(x) for x in rem_list}
        if rem_norms:
            existing_ameta = {ak: av for ak, av in existing_ameta.items() if _norm_prov_tokens(ak) not in rem_norms}
            if isinstance(cprov.get("restored_aliases"), dict):
                cleaned_rest = {
                    rk: rv for rk, rv in cprov["restored_aliases"].items() if _norm_prov_tokens(rk) not in rem_norms
                }
                if cleaned_rest:
                    cprov["restored_aliases"] = cleaned_rest
                else:
                    cprov.pop("restored_aliases", None)
            rat_map_in = u.get("remove_aliases_rationales") or {}
            rat_single = " ".join(str(u.get("remove_aliases_rationale") or "").split())
            rem_map = dict(cprov.get("removed_aliases") or {})
            for ra in rem_list:
                per_a_rat = ""
                if isinstance(rat_map_in, dict):
                    for rk_in, rv_in in rat_map_in.items():
                        if _norm_prov_tokens(str(rk_in)) == _norm_prov_tokens(ra):
                            per_a_rat = " ".join(str(rv_in).split())
                            break
                chosen_rat = per_a_rat or rat_single
                if len(chosen_rat) < 15:
                    raise ConceptGraphError(f"schema sync: missing >=15-char removal rationale for alias {ra!r} on {cid}")
                rem_map[ra] = chosen_rat
            cprov["removed_aliases"] = rem_map
            if cid in prov_gf:
                rem_gf = [ga for ga in prov_gf[cid]["aliases"] if _norm_prov_tokens(ga) not in rem_norms]
                if rem_gf:
                    prov_gf[cid] = {**prov_gf[cid], "aliases": rem_gf}
                else:
                    prov_gf.pop(cid, None)

        added_alist = [str(x).strip() for x in (u.get("add_aliases") or [])]
        attested_alist = [str(x).strip() for x in (u.get("attest_existing_aliases") or [])]
        meta_targets = added_alist + attested_alist
        raw_am_u = u.get("alias_metadata")
        if meta_targets or raw_am_u:
            if not isinstance(raw_am_u, dict) or {str(k).strip() for k in raw_am_u} != set(meta_targets):
                raise ConceptGraphError(f"schema sync: alias_metadata keys on {cid} disagree with add_aliases/attest_existing_aliases")
            expected_am_cids.add(cid)
            enr_u = am_enrich.get(cid)
            if not isinstance(enr_u, dict) or set(enr_u.keys()) != set(meta_targets):
                raise ConceptGraphError(f"schema sync: provenance_evidence_enrichment alias_metadata mismatch on {cid}")
            raw_am_clean = {str(k).strip(): v for k, v in raw_am_u.items()}
            existing_rem_norms = {_norm_prov_tokens(rk): rk for rk in (cprov.get("removed_aliases") or {})}
            rest_map = dict(cprov.get("restored_aliases") or {})
            for ak_clean in meta_targets:
                existing_ameta[ak_clean] = _verify_enrichment_alias_meta(raw_am_clean[ak_clean], enr_u[ak_clean], cid, ak_clean)
            for ak_clean in added_alist:
                ak_norm = _norm_prov_tokens(ak_clean)
                if ak_norm in existing_rem_norms:
                    if u.get("restore_removed_alias") is not True:
                        raise ConceptGraphError(
                            f"schema sync: restoring tombstoned alias {ak_clean!r} on {cid} requires restore_removed_alias=true"
                        )
                    rest_rat = " ".join(str(u.get("restore_aliases_rationale") or "").split())
                    if len(rest_rat) < 15:
                        raise ConceptGraphError(f"schema sync: restore_aliases_rationale on {cid} must be >=15 chars")
                    rest_map = {rk: rv for rk, rv in rest_map.items() if _norm_prov_tokens(rk) != ak_norm}
                    orig_tomb_key = existing_rem_norms[ak_norm]
                    rest_map[orig_tomb_key] = rest_rat
                    if ak_clean != orig_tomb_key:
                        rest_map[ak_clean] = rest_rat
            for ak_clean in attested_alist:
                ak_norm = _norm_prov_tokens(ak_clean)
                if cid in prov_gf:
                    rem_gf = [ga for ga in prov_gf[cid]["aliases"] if _norm_prov_tokens(ga) != ak_norm]
                    if rem_gf:
                        prov_gf[cid] = {**prov_gf[cid], "aliases": rem_gf}
                    else:
                        prov_gf.pop(cid, None)
            if rest_map:
                cprov["restored_aliases"] = rest_map

        if existing_ameta:
            cprov["alias_metadata"] = existing_ameta
        else:
            cprov.pop("alias_metadata", None)
        if u.get("removed_aliases_errata") is not None:
            raw_errata = u.get("removed_aliases_errata")
            if not isinstance(raw_errata, dict) or not raw_errata:
                raise ConceptGraphError(f"schema sync: removed_aliases_errata on {cid} must be a non-empty mapping")
            rem_map_now = cprov.get("removed_aliases") or {}
            tomb_by_norm = {_norm_prov_tokens(str(rk)): str(rk) for rk in rem_map_now}
            post_active_norms = {_norm_prov_tokens(str(a)) for a in (post_node.get("aliases") or [])}
            errata_store: dict[str, list[str]] = {
                str(ek): list(ev_list) for ek, ev_list in (cprov.get("removed_aliases_errata") or {}).items()
            }
            for ek_raw, ev_raw in raw_errata.items():
                ek_norm = _norm_prov_tokens(str(ek_raw))
                ev_clean = " ".join(str(ev_raw or "").split())
                if not ek_norm or ek_norm not in tomb_by_norm or ek_norm in post_active_norms or len(ev_clean) < 15:
                    raise ConceptGraphError(
                        f"schema sync: removed_aliases_errata[{ek_raw!r}] on {cid} must target an inactive tombstoned alias with >=15-char errata note"
                    )
                orig_tomb_k = tomb_by_norm[ek_norm]
                errata_store.setdefault(orig_tomb_k, []).append(ev_clean)
            cprov["removed_aliases_errata"] = errata_store
        if u.get("definition"):
            cprov["definition_change_kind"] = str(u["definition_change_kind"]).strip()
            cprov["edge_impact_note"] = " ".join(str(u["edge_impact_note"]).split())
            raw_checked_inc = u.get("checked_incident_relationships")
            if not isinstance(raw_checked_inc, list):
                raise ConceptGraphError(
                    f"schema sync: concepts_update[{cid}] definition edit requires checked_incident_relationships list"
                )
            clean_inc = [str(x).strip() for x in raw_checked_inc if isinstance(x, str) and x.strip()]
            expected_inc = sorted(
                {
                    str(e["id"])
                    for e in pre_doc.get("relationships", [])
                    if str(e["id"]) not in removed_rel_ids and (e.get("source") == cid or e.get("target") == cid)
                }
            )
            if len(clean_inc) != len(raw_checked_inc) or len(set(clean_inc)) != len(clean_inc) or sorted(clean_inc) != expected_inc:
                raise ConceptGraphError(
                    f"schema sync: checked_incident_relationships on {cid} disagrees with surviving baseline incident relationships {expected_inc}"
                )
            cprov["checked_incident_relationships"] = sorted(clean_inc)
        if u.get("label"):
            cprov["updated_label"] = " ".join(str(u["label"]).split())
        if u.get("scope"):
            cprov["updated_scope"] = str(u["scope"]).strip()
        if cprov:
            prov_concepts[cid] = cprov
        else:
            prov_concepts.pop(cid, None)

    for mv in change.get("moves", []) or []:
        mcid = mv["id"].strip()
        mhome = resolve_row_id(mv["home"].strip())
        if mhome in prov_subjects and "secondary_concepts_rationale" in prov_subjects[mhome]:
            prov_subjects[mhome]["secondary_concepts_rationale"].pop(mcid, None)
            if not prov_subjects[mhome]["secondary_concepts_rationale"]:
                prov_subjects.pop(mhome, None)

    target_version = str(change.get("version") or pre_doc["version"]).strip()
    for c in change.get("concepts_add", []) or []:
        cid = c["id"].strip()
        expected_ws_cids.add(cid)
        cprov_new: dict[str, object] = {
            "added_in": target_version,
            "warrant_sources": _verify_enrichment_warrant_sources(c.get("warrant_sources"), ws_enrich.get(cid), cid, burned_doc),
            "witness_justification": deepcopy(c["witness_justification"]),
        }
        c_aliases = [str(x).strip() for x in (c.get("aliases") or [])]
        raw_cam = c.get("alias_metadata")
        if c_aliases or raw_cam:
            if not isinstance(raw_cam, dict) or {str(k).strip() for k in raw_cam} != set(c_aliases):
                raise ConceptGraphError(f"schema sync: concepts_add alias_metadata mismatch on {cid}")
            expected_am_cids.add(cid)
            enr_c = am_enrich.get(cid)
            if not isinstance(enr_c, dict) or set(enr_c.keys()) != set(c_aliases):
                raise ConceptGraphError(f"schema sync: provenance_evidence_enrichment concepts_add alias_metadata mismatch on {cid}")
            raw_cam_clean = {str(k).strip(): v for k, v in raw_cam.items()}
            cprov_new["alias_metadata"] = {
                ak_c: _verify_enrichment_alias_meta(raw_cam_clean[ak_c], enr_c[ak_c], cid, ak_c) for ak_c in c_aliases
            }
        prov_concepts[cid] = cprov_new

    if set(ws_enrich.keys()) != expected_ws_cids or set(am_enrich.keys()) != expected_am_cids:
        raise ConceptGraphError("schema sync: provenance_evidence_enrichment contains extraneous concept keys")

    # Resolve retirements using pre-step concept state (after concepts_update/moves/concepts_add).
    mid_concepts = dict(pre_concepts)
    for c_added_node in post_doc.get("concepts", []):
        if c_added_node["id"] not in mid_concepts:
            mid_concepts[c_added_node["id"]] = c_added_node
    for u in change.get("concepts_update", []) or []:
        ucid = u["id"].strip()
        if ucid in mid_concepts:
            tmp_node = deepcopy(mid_concepts[ucid])
            if u.get("label"):
                tmp_node["label"] = " ".join(str(u["label"]).split())
            rem_n = {_norm_prov_tokens(str(x)) for x in (u.get("remove_aliases") or [])}
            als = [str(a) for a in (tmp_node.get("aliases") or []) if _norm_prov_tokens(str(a)) not in rem_n]
            als.extend(str(a).strip() for a in (u.get("add_aliases") or []))
            if als:
                tmp_node["aliases"] = als
            else:
                tmp_node.pop("aliases", None)
            mid_concepts[ucid] = tmp_node

    for r in change.get("concepts_retire", []) or []:
        rcid = r["id"].strip()
        rep = r["replaced_by"].strip()
        former = mid_concepts.pop(rcid)
        prov_gf.pop(rcid, None)
        former_lbl = str(former.get("label") or "").strip()
        former_als = [str(a) for a in (former.get("aliases") or [])]
        for rid_k in list(prov_subjects.keys()):
            if "secondary_concepts_rationale" in prov_subjects[rid_k]:
                prov_subjects[rid_k]["secondary_concepts_rationale"].pop(rcid, None)
                if not prov_subjects[rid_k]["secondary_concepts_rationale"]:
                    prov_subjects.pop(rid_k, None)
        old_cprov = prov_concepts.pop(rcid, None) or {}
        prov_retired[rcid] = {
            **old_cprov,
            "former_label": former_lbl,
            "former_aliases": former_als,
            "replaced_by": rep,
            "retired_in": str(r.get("version") or target_version).strip(),
            "reason": " ".join(str(r["reason"]).split()),
        }

    if post_doc.get("retired"):
        rep_map = {rec_r["id"]: rec_r["replaced_by"] for rec_r in post_doc["retired"]}
        for rcid_k, rprov_v in prov_retired.items():
            if rcid_k in rep_map:
                rprov_v["replaced_by"] = rep_map[rcid_k]

    for su in change.get("subject_updates", []) or []:
        rid = resolve_row_id(su["row"].strip())
        if "remove_secondary_concepts" in su or "add_secondary_concepts" in su:
            rem_secs = [s.strip() for s in su.get("remove_secondary_concepts", []) or []]
            sprov = deepcopy(prov_subjects.get(rid) or {})
            sec_rat_map = dict(sprov.get("secondary_concepts_rationale") or {})
            for rs in rem_secs:
                sec_rat_map.pop(rs, None)
            rat_text = " ".join(str(su.get("rationale") or "").split())
            for raw_s in su.get("add_secondary_concepts", []) or []:
                s = raw_s.strip()
                if len(rat_text) < 15:
                    raise ConceptGraphError(f"schema sync: missing >=15-char rationale for secondary_concept {s} on {rid}")
                sec_rat_map[s] = rat_text
            if sec_rat_map:
                sprov["secondary_concepts_rationale"] = sec_rat_map
                prov_subjects[rid] = sprov
            else:
                prov_subjects.pop(rid, None)

    if entry.get("release_governance_receipt") is not None:
        prov_state["release_governance_receipt"] = deepcopy(entry["release_governance_receipt"])
    prov_state["last_reconcile_receipt"] = entry


def build_provenance_doc_from_state(
    prov_state: dict[str, object],
    *,
    taxonomy_version: str,
    taxonomy_sha256: str,
    non_guides_taxonomy_sha256: str,
    applied_campaigns: list[dict],
) -> dict[str, object]:
    """Serialize a canonical provenance.yml mapping with self-hash from replayed state."""
    prov_concepts: dict[str, dict] = prov_state["concepts"]  # type: ignore[assignment]
    prov_subjects: dict[str, dict] = prov_state["subjects"]  # type: ignore[assignment]
    prov_retired: dict[str, dict] = prov_state["retired_concepts"]  # type: ignore[assignment]
    prov_gf: dict[str, dict] = prov_state["grandfathered_draft_alias_exceptions"]  # type: ignore[assignment]

    prov_doc: dict[str, object] = {
        "schema_version": 2,
        "taxonomy_version": str(taxonomy_version),
        "taxonomy_sha256": str(taxonomy_sha256),
        "non_guides_taxonomy_sha256": str(non_guides_taxonomy_sha256),
        "pending_pre_evaluation_override": bool(prov_state["pending_pre_evaluation_override"]),
        "pending_campaign_ids": list(prov_state["pending_campaign_ids"]),  # type: ignore[arg-type]
        "last_reconcile_receipt": prov_state["last_reconcile_receipt"],
        "applied_campaigns": list(applied_campaigns),
        "concepts": {k: prov_concepts[k] for k in sorted(prov_concepts) if prov_concepts[k]},
        "subjects": {k: prov_subjects[k] for k in sorted(prov_subjects) if prov_subjects[k]},
    }
    if prov_retired:
        prov_doc["retired_concepts"] = {k: prov_retired[k] for k in sorted(prov_retired)}
    if prov_gf:
        prov_doc["grandfathered_draft_alias_exceptions"] = {k: prov_gf[k] for k in sorted(prov_gf)}
    if prov_state.get("release_governance_receipt") is not None:
        prov_doc["release_governance_receipt"] = deepcopy(prov_state["release_governance_receipt"])
    prov_doc["sidecar_content_sha256"] = compute_sidecar_content_sha256(prov_doc)
    return prov_doc


def validate_term_provenance_sidecar(root: Path, *, verify_live_script_digests: bool = False) -> None:
    """Verify hash-chained provenance, deterministic genesis replay, and Gate 2/5 records in provenance.yml.

    Security & integrity boundary: `sidecar_content_sha256`, `campaign_entry_sha256`, canonical patch digests,
    and deterministic genesis replay of both taxonomy and top-level provenance metadata detect un-reconciled
    YAML edits, partial writes, provenance tampering, or untracked edits to 0.4-draft concepts. External
    authorization against a committer modifying validator code is supplied by git history and CI diff review
    against `origin/dev`.
    """
    from ..field_graph import load_taxonomy_yaml

    prov_path = root / "taxonomy/provenance.yml"
    tax_path = root / "taxonomy/taxonomy.yml"
    if not prov_path.exists():
        raise ConceptGraphError("schema sync: taxonomy/provenance.yml is missing (run reconcile.py and apply.py)")
    if not tax_path.exists():
        raise ConceptGraphError("schema sync: taxonomy/taxonomy.yml is missing")

    prov_text = prov_path.read_text()
    local_path_text = re.sub(r"https?://[^\s\"<>]+", "", prov_text)
    if ("/" + "Users/") in local_path_text or ("/" + "home/") in local_path_text:
        raise ConceptGraphError("schema sync: machine-local path detected in taxonomy/provenance.yml")

    prov = load_taxonomy_yaml(prov_path)
    raw_tax = load_taxonomy_yaml(tax_path)
    if not isinstance(prov, dict) or prov.get("schema_version") != 2:
        raise ConceptGraphError("schema sync: taxonomy/provenance.yml must be a schema_version=2 mapping")
    if not isinstance(raw_tax, dict):
        raise ConceptGraphError("schema sync: taxonomy/taxonomy.yml must be a mapping")

    expected_tax_sha = hashlib.sha256(tax_path.read_bytes()).hexdigest()
    if prov.get("taxonomy_sha256") != expected_tax_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/provenance.yml taxonomy_sha256 disagrees with taxonomy/taxonomy.yml "
            "(re-run reconcile.py and apply.py)"
        )
    expected_non_guides_sha = compute_non_guides_taxonomy_sha256(raw_tax)
    if prov.get("non_guides_taxonomy_sha256") != expected_non_guides_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/provenance.yml non_guides_taxonomy_sha256 disagrees with taxonomy/taxonomy.yml"
        )
    if prov.get("taxonomy_version") != str(raw_tax.get("version")):
        raise ConceptGraphError("schema sync: taxonomy/provenance.yml taxonomy_version disagrees with taxonomy.yml")
    exp_live_status = expected_status_for_version(str(raw_tax.get("version") or ""))
    if raw_tax.get("status") != exp_live_status:
        raise ConceptGraphError(
            f"schema sync: taxonomy.yml status {raw_tax.get('status')!r} disagrees with expected {exp_live_status!r} for version {raw_tax.get('version')!r}"
        )
    if isinstance(raw_tax.get("skos_hierarchy"), dict) and "status" in raw_tax["skos_hierarchy"]:
        if raw_tax["skos_hierarchy"].get("status") != exp_live_status:
            raise ConceptGraphError(
                f"schema sync: taxonomy.yml skos_hierarchy.status disagrees with expected {exp_live_status!r}"
            )

    expected_sidecar_sha = compute_sidecar_content_sha256(prov)
    if prov.get("sidecar_content_sha256") != expected_sidecar_sha:
        raise ConceptGraphError(
            "schema sync: taxonomy/provenance.yml sidecar_content_sha256 mismatch (hand-edited outside apply.py)"
        )

    campaigns = prov.get("applied_campaigns")
    if not isinstance(campaigns, list) or not campaigns:
        raise ConceptGraphError("schema sync: taxonomy/provenance.yml must record non-empty applied_campaigns hash chain")

    if isinstance(campaigns[-1], dict):
        if campaigns[-1].get("post_apply_taxonomy_sha256") != expected_tax_sha:
            raise ConceptGraphError("schema sync: final applied_campaigns post_apply_taxonomy_sha256 disagrees with taxonomy_sha256")
        if campaigns[-1].get("post_non_guides_taxonomy_sha256") != expected_non_guides_sha:
            raise ConceptGraphError("schema sync: final applied_campaigns post_non_guides_taxonomy_sha256 disagrees with non_guides_taxonomy_sha256")

    genesis_doc = load_genesis_baseline_taxonomy(root)
    genesis_cids = {c["id"] for c in genesis_doc.get("concepts", [])}
    burned_doc = _load_burned_clusters(root)
    live_burned_path = root / BURNED_CLUSTERS_REL_PATH
    if live_burned_path.is_file():
        live_burned_sha = hashlib.sha256(live_burned_path.read_bytes()).hexdigest()
        latest_camp_burned_sha: str | None = None
        for camp_e in campaigns:
            if isinstance(camp_e, dict) and isinstance(camp_e.get("burned_clusters_sha256"), str):
                latest_camp_burned_sha = str(camp_e["burned_clusters_sha256"])
        if latest_camp_burned_sha and live_burned_sha != latest_camp_burned_sha:
            known_prior_bshas: set[str] = set()
            for l_key in ("evaluation_attempts", "stage1_human_registrations", "stage2_human_attempts", "refresh_cycle_resets"):
                for l_rec in burned_doc.get(l_key) or []:
                    if isinstance(l_rec, dict) and isinstance(l_rec.get("prior_burned_clusters_sha256"), str):
                        known_prior_bshas.add(str(l_rec["prior_burned_clusters_sha256"]))
            migration = burned_doc.get("historical_migration")
            if isinstance(migration, dict):
                known_prior_bshas.add(str(migration["prior_burned_clusters_sha256"]))
            arc_obj = burned_doc.get("active_refresh_cycle")
            if isinstance(arc_obj, dict) and isinstance(arc_obj.get("prior_burned_clusters_sha256"), str):
                known_prior_bshas.add(str(arc_obj["prior_burned_clusters_sha256"]))
            if latest_camp_burned_sha not in known_prior_bshas:
                raise ConceptGraphError(
                    "schema sync: live burned-clusters.json SHA-256 diverges from latest applied_campaigns.burned_clusters_sha256 "
                    "without a hash-chained unseal/registration/reset entry"
                )
        if (root / ".git").exists():
            proc_b = subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{BURNED_CLUSTERS_REL_PATH}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc_b.returncode == 0 and proc_b.stdout.strip():
                head_bdoc = json_loads_strict(proc_b.stdout, context=f"HEAD:{BURNED_CLUSTERS_REL_PATH}")
                if isinstance(head_bdoc, dict):
                    verify_burned_clusters_append_only(
                        burned_doc, head_bdoc, prior_label=f"HEAD:{BURNED_CLUSTERS_REL_PATH}", evidence_root=root,
                    )
    replayed_doc = deepcopy(genesis_doc)
    replayed_prov_state = init_genesis_provenance_state(genesis_doc)
    reconstructed_step_sidecar_shas: list[str] = []

    governed_added_cids: set[str] = set()
    governed_touched_cids: set[str] = set()
    governed_retired_cids: set[str] = set()
    governed_def_updated_cids: set[str] = set()
    net_aliases_added: dict[str, set[str]] = {}
    all_aliases_removed: dict[str, set[str]] = {}
    restored_alias_norms_by_cid: dict[str, set[str]] = {}
    global_tombstoned_norms: dict[str, set[str]] = {}
    net_sec_added: dict[str, set[str]] = {}
    last_full_rec: dict | None = None
    seen_campaign_ids: set[str] = set()
    version_setting_entry: dict | None = None

    for idx, entry in enumerate(campaigns):
        if not isinstance(entry, dict):
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must be a mapping")
        mode = entry.get("mode")
        base_sha = str(entry.get("baseline_taxonomy_sha256") or "")
        post_sha = str(entry.get("post_apply_taxonomy_sha256") or "")
        base_ng_sha = str(entry.get("baseline_non_guides_taxonomy_sha256") or "")
        post_ng_sha = str(entry.get("post_non_guides_taxonomy_sha256") or "")
        if (
            not _HEX64_RE.match(base_sha)
            or not _HEX64_RE.match(post_sha)
            or not _HEX64_RE.match(base_ng_sha)
            or not _HEX64_RE.match(post_ng_sha)
        ):
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing valid baseline/post taxonomy or non-guides SHA-256")

        entry_sha = str(entry.get("campaign_entry_sha256") or "")
        if not _HEX64_RE.match(entry_sha) or entry_sha != compute_campaign_entry_sha256(entry):
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] campaign_entry_sha256 mismatch")

        base_prov_sha = entry.get("baseline_term_provenance_sha256")
        prior_sidecar_sha = entry.get("prior_sidecar_content_sha256")
        prev_entry_sha = entry.get("previous_campaign_entry_sha256")
        if idx == 0:
            if mode != "full-3family-reconcile" or entry.get("bootstrap_provenance") is not True:
                raise ConceptGraphError("schema sync: applied_campaigns[0] must be a bootstrap full-3family-reconcile entry")
            if base_sha != GENESIS_PRE_SIDECAR_TAXONOMY_SHA256 or base_ng_sha != GENESIS_PRE_SIDECAR_NON_GUIDES_SHA256:
                raise ConceptGraphError(
                    "schema sync: applied_campaigns[0] baseline_taxonomy_sha256 does not match pinned genesis 0.4-draft taxonomy"
                )
            if base_prov_sha is not None or prior_sidecar_sha is not None or prev_entry_sha is not None:
                raise ConceptGraphError("schema sync: bootstrap applied_campaigns[0] must have null baseline_term_provenance_sha256 and prior_sidecar_content_sha256")
        else:
            if bool(entry.get("bootstrap_provenance")):
                raise ConceptGraphError(f"schema sync: bootstrap_provenance forbidden after genesis at applied_campaigns[{idx}]")
            if base_sha != campaigns[idx - 1].get("post_apply_taxonomy_sha256"):
                raise ConceptGraphError(f"schema sync: broken taxonomy SHA-256 chain at applied_campaigns[{idx}]")
            if base_ng_sha != campaigns[idx - 1].get("post_non_guides_taxonomy_sha256"):
                raise ConceptGraphError(f"schema sync: broken non_guides_taxonomy_sha256 chain at applied_campaigns[{idx}]")
            if not isinstance(base_prov_sha, str) or not _HEX64_RE.match(base_prov_sha):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] requires valid baseline_term_provenance_sha256")
            if not isinstance(prior_sidecar_sha, str) or not _HEX64_RE.match(prior_sidecar_sha):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] requires valid prior_sidecar_content_sha256")
            if prior_sidecar_sha != reconstructed_step_sidecar_shas[idx - 1]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] prior_sidecar_content_sha256 disagrees with replayed step {idx - 1} sidecar digest"
                )
            if prev_entry_sha != campaigns[idx - 1].get("campaign_entry_sha256"):
                raise ConceptGraphError(f"schema sync: broken previous_campaign_entry_sha256 chain at applied_campaigns[{idx}]")

        replayed_patch = entry.get("replayed_patch")
        if not isinstance(replayed_patch, dict) or not replayed_patch:
            raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing non-empty replayed_patch")

        pre_step_doc = deepcopy(replayed_doc)

        if mode == "editorial-review":
            from .editorial import apply_review
            camp_id = str(entry.get("campaign_id") or "").strip()
            if len(camp_id) < 3 or camp_id in seen_campaign_ids:
                raise ConceptGraphError("editorial review: missing or duplicate campaign identity")
            seen_campaign_ids.add(camp_id)
            replayed_doc = apply_review(replayed_doc, replayed_patch)
            if compute_non_guides_taxonomy_sha256(replayed_doc) != post_ng_sha:
                raise ConceptGraphError("editorial review: replay differs from candidate content hash")
            for change in replayed_patch["changes"]:
                if change["collection"] == "concepts" and "aliases" in change["fields"]:
                    cid = change["id"]
                    removed = set((change["before"] or {}).get("aliases", [])) - set(change["fields"]["aliases"])
                    for alias in removed:
                        if _norm_prov_tokens(alias) == _norm_prov_tokens(change["fields"].get("label", change["before"].get("label", ""))):
                            continue  # An entry term promoted to the preferred label is not withdrawn.
                        global_tombstoned_norms.setdefault(_norm_prov_tokens(alias), set()).add(cid)
                    net_aliases_added.get(cid, set()).difference_update(removed)
        elif mode == "full-3family-reconcile":
            if entry.get("receipt_schema_version") != 4:
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must have receipt_schema_version=4")
            camp_id = str(entry.get("campaign_id") or "").strip()
            if len(camp_id) < 3 or camp_id in seen_campaign_ids:
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing or duplicate campaign_id {camp_id!r}")
            seen_campaign_ids.add(camp_id)

            if entry.get("allow_pre_evaluation_apply") is True and replayed_patch.get("version"):
                raise ConceptGraphError(
                    f"schema sync: allow_pre_evaluation_apply is forbidden with a version bump at applied_campaigns[{idx}]"
                )
            for hkey in (
                "reconcile_script_sha256",
                "apply_script_sha256",
                "taxonomy_schema_script_sha256",
                "run_reviews_script_sha256",
                "stage_reviewer_workspace_script_sha256",
                "baseline_linkml_schema_sha256",
                "burned_clusters_sha256",
                "sampling_manifest_sha256",
                "workspace_isolation_r2_sha256",
                "run_reviews_r2_sha256",
                "patch_yaml_sha256",
                "final_yaml_sha256",
                "final_yaml_canonical_json_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing valid {hkey}")
            if canonical_patch_json_sha256(replayed_patch) != entry["final_yaml_canonical_json_sha256"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] replayed_patch canonical JSON SHA-256 disagrees with final_yaml_canonical_json_sha256"
                )
            r2_shas = entry.get("r2_reviews_sha256")
            if (
                not isinstance(r2_shas, dict)
                or set(r2_shas.keys()) != {"claude-r2.yaml", "codex-r2.yaml", "gemini-r2.yaml"}
                or any(not isinstance(v, str) or not _HEX64_RE.match(v) for v in r2_shas.values())
            ):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must bind all 3 *-r2.yaml hashes")
            ops = entry.get("operations_applied")
            if not isinstance(ops, list) or not ops or not all(isinstance(x, str) and x for x in ops):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] must list non-empty operations_applied")
            for op in ops:
                if op != "version" and not op.startswith(_ALLOWED_OP_PREFIXES):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] unrecognized operation prefix {op!r}")

            details = entry.get("operation_details")
            if not isinstance(details, dict):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] missing structured operation_details")

            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if sorted(ops) != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] operations_applied disagrees with replayed_patch"
                )
            if compute_non_guides_taxonomy_sha256(replayed_doc) != post_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] replayed non_guides_taxonomy_sha256 disagrees with post_non_guides_taxonomy_sha256"
                )
            if replayed_patch.get("version"):
                version_setting_entry = entry

            c_added = sorted(details.get("concepts_added") or [])
            c_retired = sorted(details.get("concepts_retired") or [])
            d_upd = sorted(details.get("definitions_updated") or [])
            a_added_map = {k: list(v) for k, v in sorted((details.get("aliases_added") or {}).items())}
            a_att_map = {k: list(v) for k, v in sorted((details.get("aliases_attested") or {}).items())}
            a_rem_map = {k: list(v) for k, v in sorted((details.get("aliases_removed") or {}).items())}
            a_rest_map = details.get("aliases_restored") or {}
            other_c_upd = sorted(details.get("other_concept_updates") or [])
            s_add_map = {k: list(v) for k, v in sorted((details.get("secondary_concepts_added") or {}).items())}
            s_rem_map = {k: list(v) for k, v in sorted((details.get("secondary_concepts_removed") or {}).items())}
            other_s_upd = sorted(details.get("other_subject_updates") or [])

            if (
                c_added != replay_meta["concepts_added"]
                or c_retired != replay_meta["concepts_retired"]
                or d_upd != replay_meta["definitions_updated"]
                or a_added_map != replay_meta["aliases_added"]
                or a_att_map != replay_meta["aliases_attested"]
                or a_rem_map != replay_meta["aliases_removed"]
                or other_c_upd != replay_meta["other_concept_updates"]
                or s_add_map != replay_meta["secondary_concepts_added"]
                or s_rem_map != replay_meta["secondary_concepts_removed"]
                or other_s_upd != replay_meta["other_subject_updates"]
                or bool(details.get("guides_replaced", False)) != replay_meta["guides_replaced"]
            ):
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] operation_details disagrees with replayed_patch"
                )

            for rcid_rest, rlist in a_rest_map.items():
                if not isinstance(rlist, list) or not rlist or not set(rlist).issubset(set(a_added_map.get(rcid_rest) or [])):
                    raise ConceptGraphError(
                        f"schema sync: applied_campaigns[{idx}] aliases_restored on {rcid_rest} must be a non-empty subset of aliases_added"
                    )

            for cid in c_added:
                governed_added_cids.add(cid)
                governed_touched_cids.add(cid)
            for cid in set(d_upd) | set(a_added_map) | set(a_att_map) | set(a_rem_map) | set(other_c_upd):
                governed_touched_cids.add(cid)
            for rcid in c_retired:
                governed_retired_cids.add(rcid)
                governed_added_cids.discard(rcid)
                governed_def_updated_cids.discard(rcid)
                net_aliases_added.pop(rcid, None)

            governed_def_updated_cids.update(d_upd)
            for cid, rlist in a_rem_map.items():
                all_aliases_removed.setdefault(cid, set()).update(rlist)
                rem_norms_step = {_norm_prov_tokens(ra) for ra in rlist}
                for rn in rem_norms_step:
                    global_tombstoned_norms.setdefault(rn, set()).add(cid)
                if cid in net_aliases_added:
                    net_aliases_added[cid] = {
                        ea for ea in net_aliases_added[cid] if _norm_prov_tokens(ea) not in rem_norms_step
                    }
                if cid in restored_alias_norms_by_cid:
                    restored_alias_norms_by_cid[cid].difference_update(rem_norms_step)
            for cid, alist in a_added_map.items():
                net_aliases_added.setdefault(cid, set()).update(alist)
            for cid, alist in a_att_map.items():
                net_aliases_added.setdefault(cid, set()).update(alist)
            for cid, rlist in a_rest_map.items():
                for ra in rlist:
                    rn = _norm_prov_tokens(ra)
                    restored_alias_norms_by_cid.setdefault(cid, set()).add(rn)
                    if rn in global_tombstoned_norms:
                        global_tombstoned_norms[rn].discard(cid)
            for rid, slist in s_add_map.items():
                net_sec_added.setdefault(rid, set()).update(slist)
            for rid, slist in s_rem_map.items():
                if rid in net_sec_added:
                    net_sec_added[rid].difference_update(slist)

            last_full_rec = entry
        elif mode == "guides-apply":
            for hkey in (
                "guides_patch_yaml_sha256",
                "guides_final_yaml_sha256",
                "guides_final_yaml_canonical_json_sha256",
                "guides_reconcile_receipt_sha256",
                "guides_apply_script_sha256",
                "apply_script_sha256",
                "reconcile_script_sha256",
                "taxonomy_schema_script_sha256",
                "workspace_isolation_guides_sha256",
                "run_reviews_g1_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply missing valid {hkey}")
            if canonical_patch_json_sha256(replayed_patch) != entry["guides_final_yaml_canonical_json_sha256"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides replayed_patch canonical JSON SHA-256 mismatch"
                )
            g1_shas = entry.get("g1_reviews_sha256")
            if (
                not isinstance(g1_shas, dict)
                or set(g1_shas.keys()) != {"claude-g1.yaml", "codex-g1.yaml", "gemini-g1.yaml"}
                or any(not isinstance(v, str) or not _HEX64_RE.match(v) for v in g1_shas.values())
            ):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply must bind all 3 *-g1.yaml hashes")
            gops = entry.get("guides_operations_applied")
            if not isinstance(gops, list) or not gops or not all(isinstance(x, str) and x for x in gops):
                raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] guides-apply must list non-empty guides_operations_applied")
            for gop in gops:
                if not gop.startswith(_ALLOWED_GUIDES_OP_PREFIXES):
                    raise ConceptGraphError(f"schema sync: applied_campaigns[{idx}] unrecognized guides operation {gop!r}")

            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if sorted(gops) != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides_operations_applied disagrees with replayed_patch"
                )
            if post_ng_sha != base_ng_sha or compute_non_guides_taxonomy_sha256(replayed_doc) != post_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] guides-apply modified non-guides taxonomy content"
                )
        elif mode == "finalize-structural-evaluation":
            for hkey in (
                "apply_script_sha256",
                "reconcile_script_sha256",
                "taxonomy_schema_script_sha256",
                "burned_clusters_sha256",
                "unseal_holdout_receipt_sha256",
                "evaluation_receipt_sha256",
                "audit_receipt_sha256",
                "workspace_isolation_a1_sha256",
                "run_reviews_a1_sha256",
            ):
                if not isinstance(entry.get(hkey), str) or not _HEX64_RE.match(str(entry[hkey])):
                    raise ConceptGraphError(
                        f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation missing valid {hkey}"
                    )
            a1_shas = entry.get("a1_reviews_sha256")
            if (
                not isinstance(a1_shas, dict)
                or set(a1_shas.keys()) != {"claude-a1.yaml", "codex-a1.yaml", "gemini-a1.yaml"}
                or any(not isinstance(v, str) or not _HEX64_RE.match(v) for v in a1_shas.values())
            ):
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation must bind all 3 *-a1.yaml hashes"
                )
            if entry.get("evaluation_gate_completed") is not True:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation requires evaluation_gate_completed=true"
                )
            if post_sha != base_sha or post_ng_sha != base_ng_sha:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation must not mutate taxonomy YAML"
                )
            replay_meta = replay_patch_on_taxonomy_doc(replayed_doc, replayed_patch, mode)
            if entry.get("operations_applied") != replay_meta["expected_ops"]:
                raise ConceptGraphError(
                    f"schema sync: applied_campaigns[{idx}] finalize-structural-evaluation operations_applied mismatch"
                )
        else:
            raise ConceptGraphError(f"schema sync: unsupported applied_campaigns mode {mode!r}")

        replay_provenance_step(replayed_prov_state, pre_step_doc, replayed_doc, entry, burned_doc=burned_doc)
        step_prov_doc = build_provenance_doc_from_state(
            replayed_prov_state,
            taxonomy_version=str(replayed_doc["version"]),
            taxonomy_sha256=post_sha,
            non_guides_taxonomy_sha256=post_ng_sha,
            applied_campaigns=campaigns[: idx + 1],
        )
        reconstructed_step_sidecar_shas.append(str(step_prov_doc["sidecar_content_sha256"]))

    if verify_live_script_digests and isinstance(campaigns[-1], dict):
        live_ts = root / "binder/cli/core/taxonomy_schema.py"
        if live_ts.exists() and campaigns[-1].get("taxonomy_schema_script_sha256") != hashlib.sha256(live_ts.read_bytes()).hexdigest():
            raise ConceptGraphError("schema sync: latest campaign taxonomy_schema_script_sha256 disagrees with live taxonomy_schema.py")

    # Historical receipts retain their original single-script fingerprint. New
    # editorial receipts also cover the extracted implementation package.
    if verify_live_script_digests and campaigns[-1].get("taxonomy_implementation_sha256") is not None:
        if campaigns[-1]["taxonomy_implementation_sha256"] != compute_implementation_sha256(root):
            raise ConceptGraphError("schema sync: taxonomy implementation fingerprint disagrees with live code")

    if last_full_rec is None or prov.get("last_reconcile_receipt") != last_full_rec:
        raise ConceptGraphError("schema sync: last_reconcile_receipt disagrees with latest full-3family-reconcile campaign entry")

    if canonical_taxonomy_json(replayed_doc) != canonical_taxonomy_json(raw_tax):
        raise ConceptGraphError(
            "schema sync: replaying applied_campaigns from genesis 0.4-draft does not reproduce taxonomy.yml "
            "(untracked edit outside governed patches)"
        )

    expected_final_prov = build_provenance_doc_from_state(
        replayed_prov_state,
        taxonomy_version=str(raw_tax["version"]),
        taxonomy_sha256=expected_tax_sha,
        non_guides_taxonomy_sha256=expected_non_guides_sha,
        applied_campaigns=campaigns,
    )
    if canonical_taxonomy_json(expected_final_prov) != canonical_taxonomy_json(prov):
        raise ConceptGraphError(
            "schema sync: replaying applied_campaigns provenance state from genesis does not match provenance.yml "
            "(unreconciled top-level provenance edit detected)"
        )

    unclosed_pre_eval = bool(replayed_prov_state["pending_pre_evaluation_override"])
    pending_ids_final = list(replayed_prov_state["pending_campaign_ids"])  # type: ignore[arg-type]
    tax_ver_str = str(raw_tax.get("version") or "")
    is_non_draft = not tax_ver_str.endswith("-draft")

    if unclosed_pre_eval and (is_non_draft or prov.get("release_candidate_ready") is True):
        raise ConceptGraphError(
            "schema sync: unclosed allow_pre_evaluation_apply override cannot ship in a release candidate or final release"
        )
    if bool(prov.get("pending_pre_evaluation_override", False)) != unclosed_pre_eval:
        raise ConceptGraphError(
            "schema sync: pending_pre_evaluation_override flag disagrees with applied_campaigns pre-evaluation state"
        )
    if (prov.get("pending_campaign_ids") or []) != pending_ids_final:
        raise ConceptGraphError(
            "schema sync: pending_campaign_ids disagrees with applied_campaigns pre-evaluation state"
        )

    if is_non_draft:
        if prov.get("grandfathered_draft_alias_exceptions"):
            raise ConceptGraphError(
                f"schema sync: non-draft release {tax_ver_str!r} cannot retain unresolved grandfathered_draft_alias_exceptions"
            )
        if (
            version_setting_entry is None
            or version_setting_entry.get("mode") != "full-3family-reconcile"
            or version_setting_entry.get("evaluation_gate_completed") is not True
        ):
            raise ConceptGraphError(
                f"schema sync: non-draft release {tax_ver_str!r} requires a full-3family-reconcile version transition with evaluation_gate_completed=true"
            )
    verify_release_governance_receipt(
        prov.get("release_governance_receipt"),
        tax_ver_str,
        expected_tax_sha,
        expected_non_guides_sha,
        root=root,
        candidate_tax_doc=raw_tax,
        burned_doc=burned_doc,
        prov_concepts=prov.get("concepts") if isinstance(prov.get("concepts"), dict) else None,
    )

    concepts_by_id = {c["id"]: c for c in raw_tax.get("concepts", [])}
    retired_by_id = {r["id"]: r for r in raw_tax.get("retired", []) or []}
    rows_by_id = {r["id"]: r for a in raw_tax.get("areas", []) for r in a.get("rows", [])}

    prov_concepts = prov.get("concepts")
    prov_subjects = prov.get("subjects")
    prov_retired = prov.get("retired_concepts") or {}
    if not isinstance(prov_concepts, dict) or not isinstance(prov_subjects, dict) or not isinstance(prov_retired, dict):
        raise ConceptGraphError("schema sync: term-provenance concepts, subjects, and retired_concepts must be mappings")

    for cid in sorted(governed_touched_cids):
        if cid in concepts_by_id and cid not in prov_concepts:
            raise ConceptGraphError(f"schema sync: governed concept {cid} missing provenance record in provenance.yml")
    for rcid in sorted(governed_retired_cids):
        if rcid not in retired_by_id or rcid not in prov_retired:
            raise ConceptGraphError(f"schema sync: retired concept {rcid} missing from retired_concepts in provenance.yml")

    retired_reserved_norms: dict[str, str] = {}
    for rcid, rprov in prov_retired.items():
        if (
            rcid not in retired_by_id
            or not isinstance(rprov, dict)
            or len(str(rprov.get("former_label") or "").strip()) < 2
            or not isinstance(rprov.get("former_aliases"), list)
            or rprov.get("replaced_by") not in concepts_by_id
            or len(str(rprov.get("reason") or "").strip()) < 10
        ):
            raise ConceptGraphError(f"schema sync: retired_concepts[{rcid}] missing former_label/former_aliases/replaced_by/reason")
        rep_cid = str(rprov["replaced_by"])
        for fstr in [rprov["former_label"], *rprov["former_aliases"]]:
            fnorm = _norm_prov_tokens(str(fstr))
            if fnorm:
                retired_reserved_norms[fnorm] = rep_cid

    for cid, cnode in concepts_by_id.items():
        c_terms = [str(cnode.get("label") or ""), *(str(a) for a in (cnode.get("aliases") or []))]
        rest_norms_here = restored_alias_norms_by_cid.get(cid, set())
        for tstr in c_terms:
            tnorm = _norm_prov_tokens(tstr)
            if not tnorm:
                continue
            if tnorm in retired_reserved_norms and retired_reserved_norms[tnorm] != cid:
                raise ConceptGraphError(
                    f"schema sync: term {tstr!r} on {cid} collides with permanently reserved retired string (allowed only on {retired_reserved_norms[tnorm]})"
                )
            tomb_owners = global_tombstoned_norms.get(tnorm)
            if tomb_owners and (cid not in tomb_owners or tnorm not in rest_norms_here):
                raise ConceptGraphError(
                    f"schema sync: globally tombstoned alias {tstr!r} appears on {cid} without reviewed restoration"
                )

    for cid, cnode in concepts_by_id.items():
        if cid not in genesis_cids or str(cnode.get("added_in") or "") not in _PRE_SIDECAR_VERSIONS or cid in governed_added_cids:
            cprov = prov_concepts.get(cid)
            if not isinstance(cprov, dict):
                raise ConceptGraphError(f"schema sync: post-bootstrap concept {cid} missing term-provenance entry")
            # New editorial admissions are validated during exact replay. Historical
            # campaigns retain their original warrant and lineage requirements.
            if cprov.get("editorial_admission"):
                continue
            ws = cprov.get("warrant_sources")
            wj = cprov.get("witness_justification")
            if not isinstance(ws, list) or len(ws) < 2 or not isinstance(wj, dict) or set(wj.keys()) != set(_REQUIRED_WITNESS_KEYS):
                raise ConceptGraphError(f"schema sync: post-bootstrap concept {cid} missing structured warrant_sources or 6-key witness_justification")
            has_prose = False
            seen_uids: set[str] = set()
            for src in ws:
                if (
                    not isinstance(src, dict)
                    or not str(src.get("unit_id") or "").strip()
                    or src.get("stratum") not in ("S1", "S2", "S3", "S4", "S5")
                    or not str(src.get("provenance_family_id") or "").strip()
                    or src.get("source_type") not in ("prose", "heading")
                    or src.get("origin") not in ("external", "internal")
                    or not str(src.get("locator") or "").strip()
                ):
                    raise ConceptGraphError(f"schema sync: malformed warrant_sources item on {cid}")
                seen_uids.add(str(src["unit_id"]).strip())
                if src.get("source_type") == "prose":
                    has_prose = True
            all_lin, ext_non_ed_lin = _compute_warrant_lineages(ws, burned_doc)
            if len(seen_uids) < 2 or len(all_lin) < 2 or len(ext_non_ed_lin) < 1 or not has_prose:
                raise ConceptGraphError(
                    f"schema sync: post-bootstrap concept {cid} warrant_sources require >=2 distinct units, >=2 independent lineages (<50% author overlap), >=1 non-editor-affiliated external lineage, and >=1 prose source"
                )
            comp_cid = str(wj.get("compared_concept") or "").strip()
            if (
                comp_cid == cid
                or (comp_cid not in concepts_by_id and comp_cid not in retired_by_id)
                or str(wj.get("scenario_a") or "").strip() == str(wj.get("scenario_b") or "").strip()
                or any(len(str(wj.get(k) or "").strip()) < 15 for k in ("criterion_a", "criterion_b", "criterion_c", "scenario_a", "scenario_b"))
            ):
                raise ConceptGraphError(f"schema sync: post-bootstrap concept {cid} witness_justification fails minimum completeness checks")

    for cid in sorted(governed_def_updated_cids):
        if cid in concepts_by_id:
            cprov = prov_concepts.get(cid) or {}
            if (
                len(str(cprov.get("edge_impact_note") or "").strip()) < 15
                or cprov.get("definition_change_kind") != "editorial-clarification"
                or not isinstance(cprov.get("checked_incident_relationships"), list)
            ):
                raise ConceptGraphError(
                    f"schema sync: updated definition on {cid} missing edge_impact_note, checked_incident_relationships, or editorial-clarification in provenance.yml"
                )

    for cid, expected_added in net_aliases_added.items():
        if cid in concepts_by_id and expected_added:
            cprov = prov_concepts.get(cid) or {}
            ameta_map = cprov.get("alias_metadata") or {}
            active_set = set(concepts_by_id[cid].get("aliases") or [])
            for ak in sorted(expected_added):
                if ak not in active_set or ak not in ameta_map:
                    raise ConceptGraphError(f"schema sync: added or attested alias {ak!r} on {cid} missing from active aliases or term-provenance alias_metadata")

    for cid, expected_removed in all_aliases_removed.items():
        if cid in concepts_by_id and expected_removed:
            cprov = prov_concepts.get(cid) or {}
            rem_map = cprov.get("removed_aliases") or {}
            rest_norms_c = restored_alias_norms_by_cid.get(cid, set())
            active_norms = {_norm_prov_tokens(a) for a in (concepts_by_id[cid].get("aliases") or [])}
            for rk in sorted(expected_removed):
                if rk not in rem_map:
                    raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} missing tombstone in term-provenance removed_aliases")
                rk_norm = _norm_prov_tokens(rk)
                if rk_norm in active_norms and rk_norm not in rest_norms_c:
                    raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} reappeared without reviewed restoration")

    for rid, expected_sec in net_sec_added.items():
        if rid in rows_by_id and expected_sec:
            sprov = prov_subjects.get(rid) or {}
            srat_map = sprov.get("secondary_concepts_rationale") or {}
            active_sec = set(rows_by_id[rid].get("secondary_concepts") or [])
            for scid in sorted(expected_sec):
                if scid not in active_sec or scid not in srat_map:
                    raise ConceptGraphError(f"schema sync: added secondary_concept {scid} on {rid} missing from active row or secondary_concepts_rationale")

    for cid, cprov in prov_concepts.items():
        if cid not in concepts_by_id or not isinstance(cprov, dict) or not cprov:
            raise ConceptGraphError(f"schema sync: invalid or inactive concept {cid} in provenance.yml")
        active_aliases = set(concepts_by_id[cid].get("aliases") or [])
        active_alias_norms = {_norm_prov_tokens(a) for a in active_aliases}
        for ak, ameta in (cprov.get("alias_metadata") or {}).items():
            if ak not in active_aliases or not isinstance(ameta, dict):
                raise ConceptGraphError(f"schema sync: term-provenance alias {ak!r} on {cid} missing from active aliases")
            atype = ameta.get("type")
            asrc = str(ameta.get("source") or "").strip()
            snote = str(ameta.get("scope_note") or "").strip()
            cunits = ameta.get("checked_dev_units")
            vq = str(ameta.get("verbatim_quote") or "").strip()
            psha = str(ameta.get("primary_source_sha256") or "").strip()
            ploc = str(ameta.get("primary_source_locator") or "").strip()
            ak_tok = _norm_prov_tokens(ak)
            vq_tok = _norm_prov_tokens(vq)
            if (
                atype not in _ALLOWED_PROV_ALIAS_TYPES
                or len(asrc) < 10
                or len(snote) < 15
                or not isinstance(cunits, list)
                or not cunits
                or any(not isinstance(u, str) or not u.strip() for u in cunits)
                or not _HEX64_RE.match(psha)
                or len(ploc) < 3
                or not ak_tok
                or f" {ak_tok} " not in f" {vq_tok} "
            ):
                raise ConceptGraphError(
                    f"schema sync: incomplete or non-word-bounded alias_metadata on {cid} alias {ak!r}"
                )
            if atype == "upward-posting":
                dlin = ameta.get("dev_lineages_requiring_separation")
                if not isinstance(dlin, int) or isinstance(dlin, bool) or dlin not in (0, 1):
                    raise ConceptGraphError(
                        f"schema sync: upward-posting alias {ak!r} on {cid} requires dev_lineages_requiring_separation in {{0, 1}}"
                    )
        rest_map = cprov.get("restored_aliases") or {}
        if rest_map and not isinstance(rest_map, dict):
            raise ConceptGraphError(f"schema sync: restored_aliases on {cid} must be a mapping")
        rest_map_norms = {_norm_prov_tokens(rk) for rk in rest_map}
        for rk, rrat in (cprov.get("removed_aliases") or {}).items():
            if not isinstance(rrat, str) or len(rrat.strip()) < 15:
                raise ConceptGraphError(f"schema sync: invalid removed_aliases entry {rk!r} on {cid}")
            if _norm_prov_tokens(rk) in active_alias_norms and _norm_prov_tokens(rk) not in rest_map_norms:
                raise ConceptGraphError(f"schema sync: removed alias {rk!r} on {cid} is still active without restored_aliases record")
        if "removed_aliases_errata" in cprov:
            err_map = cprov["removed_aliases_errata"]
            rem_keys_set = set((cprov.get("removed_aliases") or {}).keys())
            if (
                not isinstance(err_map, dict)
                or not err_map
                or not set(err_map.keys()).issubset(rem_keys_set)
                or any(
                    not isinstance(elist, list)
                    or not elist
                    or any(not isinstance(enote, str) or len(enote.strip()) < 15 for enote in elist)
                    for elist in err_map.values()
                )
            ):
                raise ConceptGraphError(f"schema sync: invalid removed_aliases_errata on {cid}")
        if "edge_impact_note" in cprov or "definition_change_kind" in cprov or "checked_incident_relationships" in cprov:
            if (
                len(str(cprov.get("edge_impact_note") or "").strip()) < 15
                or cprov.get("definition_change_kind") != "editorial-clarification"
                or not isinstance(cprov.get("checked_incident_relationships"), list)
                or any(not isinstance(x, str) or not x.strip() for x in cprov["checked_incident_relationships"])
            ):
                raise ConceptGraphError(
                    f"schema sync: concept {cid} definition provenance requires edge_impact_note, checked_incident_relationships, and definition_change_kind='editorial-clarification'"
                )

    for rid, sprov in prov_subjects.items():
        if rid not in rows_by_id or not isinstance(sprov, dict):
            raise ConceptGraphError(f"schema sync: unknown subject {rid} in provenance.yml")
        active_sec = set(rows_by_id[rid].get("secondary_concepts") or [])
        for scid, srat in (sprov.get("secondary_concepts_rationale") or {}).items():
            if scid not in active_sec or scid not in concepts_by_id or not isinstance(srat, str) or len(srat.strip()) < 15:
                raise ConceptGraphError(f"schema sync: invalid secondary_concepts_rationale for {scid} on {rid}")

    gf_exc = prov.get("grandfathered_draft_alias_exceptions")
    if gf_exc is not None:
        if not isinstance(gf_exc, dict):
            raise ConceptGraphError("schema sync: grandfathered_draft_alias_exceptions must be a mapping")
        for gcid, grec in gf_exc.items():
            if gcid not in PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS or gcid not in concepts_by_id or not isinstance(grec, dict):
                raise ConceptGraphError(
                    f"schema sync: unpinned or inactive grandfathered_draft_alias_exceptions target {gcid!r} "
                    f"(allowed bootstrap set: {sorted(PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS)})"
                )
            galiases = grec.get("aliases")
            active_galiases = set(concepts_by_id[gcid].get("aliases") or [])
            allowed_galiases = set(PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS[gcid])
            if (
                not isinstance(galiases, list)
                or not galiases
                or not set(galiases).issubset(allowed_galiases)
                or not set(galiases).issubset(active_galiases)
                or len(str(grec.get("reason") or "").strip()) < 15
            ):
                raise ConceptGraphError(
                    f"schema sync: invalid or stale grandfathered_draft_alias_exceptions on {gcid!r}"
                )
