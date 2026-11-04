"""Historical reviewed-release protocol verification. Preserve its recorded semantics."""
from __future__ import annotations

import collections
import datetime
import hashlib
import math
import random
from pathlib import Path

from ..concept_graph import ConceptGraphError
from .history import (
    _load_burned_clusters,
    _norm_work_id_token,
    compute_lineage_components,
    load_genesis_baseline_taxonomy,
    norm_person_or_org_token,
)
from .legacy_policy import (
    _ANCHOR_BOOTSTRAP_CI_KEYS,
    _BOARD_SEATS_EXPECTED,
    _CHAIR_SIGN_OFF_AREAS_TO_SEAT,
    _EXPECTED_STANDING_SLOTS,
    _PRIMARY_BOOTSTRAP_CI_KEYS,
    _SPECIALIST_SLOTS_EXPECTED,
    _STAGE2_PAIR_ORDER,
    PINNED_BASELINE_SPLIT_HIERARCHY_EDGES,
    PINNED_BASELINE_ZERO_WARRANT_CONCEPTS,
    PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS,
    REPO_ROOT_PATH,
)
from .metrics import (
    _bootstrap_percentile_95_ci,
    _canonicalize_pair_eval_columns,
    _fast_fleiss_kappa_8rater_from_precomputed,
    _fast_hooper_rolling_from_precomputed,
    _fast_masi_alpha_from_precomputed,
    _fast_nominal_alpha_from_precomputed,
    _fast_pair_stratified_kappa_from_precomputed,
    _precompute_fleiss_kappa_8rater,
    _precompute_hooper_rolling,
    _precompute_masi_alpha,
    _precompute_nominal_alpha,
    _precompute_pair_stratified_kappa,
    expected_stage2_anchor_schedule,
    run_stage2_estimator_self_tests,
)
from .records import (
    _HEX64_RE,
    _is_finite_num,
    _norm_prov_tokens,
    canonical_taxonomy_json,
    expected_status_for_version,
    json_loads_strict,
    parse_taxonomy_semver,
    validate_ordinal_coverage_judgment,
)


def _resolve_confined_method_study_file(root: Path, rel_str: str, label: str) -> Path:
    if not rel_str:
        raise ConceptGraphError(f"schema sync: missing {label}")
    rel_p = Path(rel_str)
    if rel_p.is_absolute() or ".." in rel_p.parts or len(rel_p.parts) < 3 or rel_p.parts[:2] != ("taxonomy", "method-study"):
        raise ConceptGraphError(f"schema sync: {label} {rel_str!r} must be a relative path confined under taxonomy/method-study/")
    cur = root
    if cur.is_symlink():
        raise ConceptGraphError(f"schema sync: symlinked root forbidden when resolving {label}")
    for part in rel_p.parts:
        cur = cur / part
        if cur.is_symlink():
            raise ConceptGraphError(f"schema sync: symlink component {cur} is forbidden in {label}")
    ms_dir = (root / "taxonomy" / "method-study").resolve()
    resolved = cur.resolve()
    if not resolved.is_relative_to(ms_dir) or not resolved.is_file():
        raise ConceptGraphError(f"schema sync: {label} {rel_str!r} does not resolve to a regular file inside taxonomy/method-study/")
    return resolved


def _area_to_stage2_pair(area_code: str) -> str:
    if area_code in ("T1", "T2"):
        return area_code
    if area_code and area_code[0] in ("F", "S", "P", "Q"):
        return area_code[0]
    raise ConceptGraphError(f"schema sync: unknown area_code {area_code!r} in Stage 2 verification")


def _parse_iso_utc(raw_ts: object, ctx: str) -> datetime.datetime:
    if not isinstance(raw_ts, str) or len(raw_ts.strip()) < 10:
        raise ConceptGraphError(f"schema sync: {ctx} must be an ISO-8601 timestamp string with timezone")
    ts_s = raw_ts.strip()
    if ts_s.endswith("Z"):
        ts_s = ts_s[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(ts_s)
    except ValueError as exc:
        raise ConceptGraphError(f"schema sync: invalid ISO-8601 timestamp at {ctx}: {raw_ts!r}") from exc
    if dt.tzinfo is None:
        raise ConceptGraphError(f"schema sync: {ctx} timestamp {raw_ts!r} must carry an explicit UTC/offset timezone")
    return dt.astimezone(datetime.timezone.utc)


def _canonical_unit_provenance_sig(u: dict, *, require_passage: bool = False) -> tuple:
    auths_raw = u.get("authors") or []
    auths = tuple(
        sorted(
            {
                norm_person_or_org_token(str(a), is_org=False)
                for a in auths_raw
                if isinstance(a, str) and norm_person_or_org_token(a, is_org=False)
            }
        )
    )
    orgs_in: list[object] = []
    if isinstance(u.get("institutional_organizations"), list):
        orgs_in.extend(u["institutional_organizations"])
    if isinstance(u.get("institutional_organization"), str) and u["institutional_organization"].strip():
        orgs_in.append(u["institutional_organization"])
    orgs = tuple(
        sorted(
            {
                norm_person_or_org_token(str(o), is_org=True)
                for o in orgs_in
                if isinstance(o, str) and norm_person_or_org_token(o, is_org=True)
            }
        )
    )
    norm_passage = " ".join(str(u.get("passage") or "").split())
    if require_passage and len(norm_passage) < 10:
        raise ConceptGraphError(
            f"schema sync: unit {u.get('unit_id')!r} requires non-empty passage (>=10 chars) for canonical routing digest"
        )
    passage_sha = hashlib.sha256(norm_passage.encode("utf-8")).hexdigest()
    return (
        str(u.get("unit_id") or "").strip(),
        _norm_work_id_token(str(u.get("work_id") or "")),
        str(u.get("provenance_family_id") or "").strip(),
        str(u.get("origin") or "").strip(),
        str(u.get("area_code") or "").strip(),
        str(u.get("source_type") or "").strip(),
        str(u.get("passage_kind") or "").strip(),
        bool(u.get("editor_affiliated", False)),
        bool(u.get("editor_coauthor_5yr", False)),
        bool(u.get("institutional_author", False)),
        auths,
        orgs,
        passage_sha,
    )


def compute_stage1_routing_digest(stage1_items: list[dict], pool_by_uid: dict[str, dict] | None = None) -> str:
    """Compute pre-calibration SHA-256 over the 30 Stage 1 items, pair assignments, and passage/provenance signatures."""
    if not isinstance(stage1_items, list) or len(stage1_items) != 30:
        raise ConceptGraphError("schema sync: Stage 1 routing digest requires 30 stage1_items")
    rows = []
    for idx_u, u in enumerate(stage1_items):
        if not isinstance(u, dict):
            raise ConceptGraphError(f"schema sync: Stage 1 item[{idx_u}] must be a mapping")
        uid = str(u.get("unit_id") or "").strip()
        pair_id = str(u.get("pair_id") or "").strip()
        if not uid or pair_id not in _STAGE2_PAIR_ORDER:
            raise ConceptGraphError(f"schema sync: Stage 1 item[{idx_u}] requires valid unit_id and pair_id")
        src_u = pool_by_uid[uid] if (pool_by_uid is not None and uid in pool_by_uid) else u
        rows.append([pair_id, uid, list(_canonical_unit_provenance_sig(src_u, require_passage=True))])
    rows.sort(key=lambda r: (r[0], r[1]))
    return hashlib.sha256(canonical_taxonomy_json(rows).encode("utf-8")).hexdigest()


def compute_stage2_frozen_routing_digest(
    *,
    stage1_registration_id: str,
    stage1_items: list[dict],
    stage2_items: list[dict],
    recalibration_items: list[dict],
    pool_by_uid: dict[str, dict] | None = None,
) -> str:
    """Compute pre-unseal SHA-256 locking Stage 1 registration, 240-item area/pair/anchor routing, recalibration items, and passages."""
    reg_id = str(stage1_registration_id or "").strip()
    if len(reg_id) < 3:
        raise ConceptGraphError("schema sync: compute_stage2_frozen_routing_digest requires stage1_registration_id")
    s1_digest = compute_stage1_routing_digest(stage1_items, pool_by_uid=pool_by_uid)
    if not isinstance(stage2_items, list) or len(stage2_items) != 240:
        raise ConceptGraphError("schema sync: Stage 2 routing digest requires 240 stage2_items")
    by_area: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for idx_2, u2 in enumerate(stage2_items):
        if not isinstance(u2, dict):
            raise ConceptGraphError(f"schema sync: Stage 2 item[{idx_2}] must be a mapping")
        uid2 = str(u2.get("unit_id") or "").strip()
        src_u2 = pool_by_uid[uid2] if (pool_by_uid is not None and uid2 in pool_by_uid) else u2
        acode = str(u2.get("area_code") or src_u2.get("area_code") or "").strip()
        if not uid2 or not acode:
            raise ConceptGraphError(f"schema sync: Stage 2 item[{idx_2}] requires unit_id and area_code")
        by_area[acode].append((uid2, src_u2))
    if len(by_area) != 20 or any(len(v) != 12 for v in by_area.values()):
        raise ConceptGraphError("schema sync: Stage 2 routing digest requires 12 items in each of the 20 areas")
    exp_sched = expected_stage2_anchor_schedule()
    group_anc_cursor: dict[str, int] = {g: 0 for g in _STAGE2_PAIR_ORDER}
    s2_rows = []
    for acode in sorted(by_area.keys()):
        prim_pair = _area_to_stage2_pair(acode)
        alist = sorted(by_area[acode], key=lambda t: (acode, t[0]))
        for pos_in_area, (uid2, src_u2) in enumerate(alist):
            is_anc = pos_in_area < 2
            if is_anc:
                k_pos = group_anc_cursor[prim_pair]
                sec_pairs = list(exp_sched[prim_pair][k_pos])
                group_anc_cursor[prim_pair] = k_pos + 1
            else:
                sec_pairs = []
            sig2 = list(_canonical_unit_provenance_sig({**src_u2, "area_code": acode}, require_passage=True))
            s2_rows.append([acode, uid2, prim_pair, is_anc, sec_pairs, sig2])
    recal_rows = []
    for idx_r, ur in enumerate(recalibration_items or []):
        if not isinstance(ur, dict):
            raise ConceptGraphError(f"schema sync: recalibration item[{idx_r}] must be a mapping")
        uid_r = str(ur.get("unit_id") or "").strip()
        if not uid_r:
            raise ConceptGraphError(f"schema sync: recalibration item[{idx_r}] missing unit_id")
        src_ur = pool_by_uid[uid_r] if (pool_by_uid is not None and uid_r in pool_by_uid) else ur
        recal_rows.append([uid_r, list(_canonical_unit_provenance_sig(src_ur, require_passage=True))])
    recal_rows.sort(key=lambda r: r[0])
    payload = {
        "stage1_registration_id": reg_id,
        "stage1_routing_digest_sha256": s1_digest,
        "stage2_routing": s2_rows,
        "recalibration_routing": recal_rows,
    }
    return hashlib.sha256(canonical_taxonomy_json(payload).encode("utf-8")).hexdigest()


def _validate_person_record(
    rec: object,
    ctx: str,
    seen_pids: set[str],
    seen_id_commits: set[str],
    seen_name_shas: set[str],
    *,
    editor_pid: str,
    require_non_editor: bool = False,
) -> dict:
    if not isinstance(rec, dict):
        raise ConceptGraphError(f"schema sync: {ctx} must be a person mapping")
    pid = str(rec.get("person_id") or "").strip()
    inst = norm_person_or_org_token(str(rec.get("institution") or ""), is_org=True)
    id_commit = str(rec.get("identity_commitment_sha256") or "").strip()
    name_sha = str(rec.get("normalized_name_sha256") or "").strip()
    consent = rec.get("public_name_consent", False)
    sec_raw = rec.get("secondary_institutions")
    if sec_raw is not None and (not isinstance(sec_raw, list) or any(not isinstance(x, str) or not x.strip() for x in sec_raw)):
        raise ConceptGraphError(f"schema sync: {ctx} secondary_institutions must be a list of non-empty strings")
    sec_insts = {
        norm_person_or_org_token(x, is_org=True)
        for x in (sec_raw or [])
        if norm_person_or_org_token(x, is_org=True)
    }
    if (
        len(pid) < 3
        or not inst
        or not _HEX64_RE.match(id_commit)
        or not _HEX64_RE.match(name_sha)
        or id_commit == name_sha
        or not isinstance(consent, bool)
        or rec.get("five_year_coi_attested") is not True
        or not isinstance(rec.get("editor_institution"), bool)
        or not isinstance(rec.get("editor_coauthor_5yr"), bool)
        or not isinstance(rec.get("editor_advisee_or_lab_or_report"), bool)
    ):
        raise ConceptGraphError(
            f"schema sync: {ctx} requires person_id, institution, distinct 64-hex identity_commitment_sha256 & "
            "normalized_name_sha256, boolean public_name_consent, five_year_coi_attested=true, and boolean COI flags"
        )
    if consent is False:
        if rec.get("full_name") is not None or rec.get("orcid_or_uri") is not None or rec.get("identity_commitment_salt") is not None:
            raise ConceptGraphError(
                f"schema sync: {ctx} forbids plaintext full_name, orcid_or_uri, or identity_commitment_salt unless public_name_consent=true"
            )
    else:
        raw_fn = rec.get("full_name")
        raw_uri = rec.get("orcid_or_uri")
        norm_fn = norm_person_or_org_token(str(raw_fn or ""), is_org=False) if isinstance(raw_fn, str) else ""
        norm_uri = str(raw_uri or "").strip().lower() if isinstance(raw_uri, str) else ""
        if not norm_fn or len(norm_uri) < 6:
            raise ConceptGraphError(f"schema sync: {ctx} with public_name_consent=true requires full_name and orcid_or_uri")
        raw_salt = rec.get("identity_commitment_salt")
        if raw_salt is not None and not isinstance(raw_salt, str):
            raise ConceptGraphError(f"schema sync: {ctx} identity_commitment_salt must be a string when provided")
        exp_name_pre = f"{raw_salt}:{norm_fn}" if raw_salt else norm_fn
        exp_id_pre = f"{raw_salt}:{norm_uri}" if raw_salt else norm_uri
        if (
            hashlib.sha256(exp_name_pre.encode("utf-8")).hexdigest() != name_sha
            or hashlib.sha256(exp_id_pre.encode("utf-8")).hexdigest() != id_commit
        ):
            raise ConceptGraphError(
                f"schema sync: {ctx} full_name/orcid_or_uri hashes disagree with normalized_name_sha256 or identity_commitment_sha256"
            )
    if pid == editor_pid:
        raise ConceptGraphError(f"schema sync: Editor ({editor_pid!r}) is excluded from governance/indexing role {ctx}")
    if pid in seen_pids or id_commit in seen_id_commits or name_sha in seen_name_shas:
        raise ConceptGraphError(
            f"schema sync: duplicate person_id, identity_commitment_sha256, or normalized_name_sha256 across governance/indexing pools at {ctx}"
        )
    if require_non_editor and (
        rec["editor_institution"] or rec["editor_coauthor_5yr"] or rec["editor_advisee_or_lab_or_report"]
    ):
        raise ConceptGraphError(
            f"schema sync: {ctx} ({pid!r}) must be independent of Editor institution, lab/advisee/report chain, and 5-year co-authorship"
        )
    seen_pids.add(pid)
    seen_id_commits.add(id_commit)
    seen_name_shas.add(name_sha)
    all_insts = frozenset({inst, *sec_insts})
    return {**rec, "_norm_inst": inst, "_all_insts": all_insts, "_pid": pid}


def _validate_ci_map(raw_ci_map: object, expected_keys: tuple[str, ...], ctx: str) -> dict[str, tuple[float, float]]:
    if not isinstance(raw_ci_map, dict) or set(raw_ci_map.keys()) != set(expected_keys):
        raise ConceptGraphError(f"schema sync: {ctx} must record 95% CIs for exact keys {list(expected_keys)}")
    out: dict[str, tuple[float, float]] = {}
    for k in expected_keys:
        pair = raw_ci_map.get(k)
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not _is_finite_num(pair[0])
            or not _is_finite_num(pair[1])
            or float(pair[0]) < -1.0
            or float(pair[1]) > 1.0
            or float(pair[0]) > float(pair[1]) + 1e-12
        ):
            raise ConceptGraphError(f"schema sync: {ctx}[{k!r}] must be [lo, hi] with -1.0 <= lo <= hi <= 1.0")
        out[k] = (float(pair[0]), float(pair[1]))
    return out


def verify_release_governance_receipt(
    rel_rec: object,
    target_version: str,
    expected_tax_sha: str,
    expected_non_guides_sha: str | None = None,
    *,
    root: Path | None = None,
    candidate_tax_doc: dict | None = None,
    burned_doc: dict | None = None,
    prov_concepts: dict[str, dict] | None = None,
) -> None:
    """Enforce candidate-experts.md §2–§6 Editorial Board and 12-human Stage 2 gates on non-draft releases."""
    major, minor, _patch_v, pre_type, _rc_num = parse_taxonomy_semver(target_version)
    if pre_type == 0:
        return
    if major < 1:
        raise ConceptGraphError(
            f"schema sync: 0.x version {target_version!r} must carry '-draft' suffix; citable releases and release candidates start at 1.0-rc1 / 1.0"
        )
    if not isinstance(rel_rec, dict) or rel_rec.get("schema_version") != 1:
        raise ConceptGraphError(
            f"schema sync: version {target_version!r} (major >= 1) requires schema_version=1 release_governance_receipt"
        )
    cycle_id = str(rel_rec.get("cycle_id") or "").strip()
    if len(cycle_id) < 3:
        raise ConceptGraphError("schema sync: release_governance_receipt requires non-empty cycle_id (>=3 chars)")
    if str(rel_rec.get("target_version") or "").strip() != target_version:
        raise ConceptGraphError("schema sync: release_governance_receipt target_version disagrees with taxonomy_version")
    if str(rel_rec.get("candidate_taxonomy_sha256") or "").strip() != expected_tax_sha:
        raise ConceptGraphError("schema sync: release_governance_receipt candidate_taxonomy_sha256 disagrees with taxonomy_sha256")
    if expected_non_guides_sha is not None:
        if str(rel_rec.get("candidate_non_guides_taxonomy_sha256") or "").strip() != expected_non_guides_sha:
            raise ConceptGraphError(
                "schema sync: release_governance_receipt candidate_non_guides_taxonomy_sha256 disagrees with non_guides_taxonomy_sha256"
            )

    board = rel_rec.get("board_signoff")
    if (
        not isinstance(board, dict)
        or board.get("seated_voting_members") != 7
        or not isinstance(board.get("affirmative_votes"), int)
        or isinstance(board.get("affirmative_votes"), bool)
        or not (5 <= board["affirmative_votes"] <= 7)
        or board.get("v1_approved") is not True
        or board.get("v2_approved") is not True
        or board.get("sub_area_specialist_reports") != 13
        or not isinstance(board.get("active_editor_institution_voters"), int)
        or isinstance(board.get("active_editor_institution_voters"), bool)
        or not (0 <= board["active_editor_institution_voters"] <= 1)
        or not isinstance(board.get("active_editor_coauthor_voters"), int)
        or isinstance(board.get("active_editor_coauthor_voters"), bool)
        or not (0 <= board["active_editor_coauthor_voters"] <= 2)
        or not _HEX64_RE.match(str(board.get("charter_sha256") or ""))
        or not _HEX64_RE.match(str(board.get("vote_record_sha256") or ""))
    ):
        raise ConceptGraphError(
            f"schema sync: version {target_version!r} requires board_signoff matching candidate-experts.md §2–§3 "
            "(7 seated voting seats, >=5 affirmative votes, v1_approved=true, v2_approved=true, 13 specialist reports, active caps <=1/<=2)"
        )
    hs = rel_rec.get("human_study_stage2")
    if (
        not isinstance(hs, dict)
        or not isinstance(hs.get("n_items"), int)
        or isinstance(hs.get("n_items"), bool)
        or hs["n_items"] != 240
        or hs.get("n_raters") != 12
        or not isinstance(hs.get("canonical_provenance_clusters"), int)
        or isinstance(hs.get("canonical_provenance_clusters"), bool)
        or hs["canonical_provenance_clusters"] < 120
        or not isinstance(hs.get("effective_lineage_components"), int)
        or isinstance(hs.get("effective_lineage_components"), bool)
        or hs["effective_lineage_components"] < 120
        or not isinstance(hs.get("max_areas_per_component"), int)
        or isinstance(hs.get("max_areas_per_component"), bool)
        or not (1 <= hs["max_areas_per_component"] <= 4)
        or not isinstance(hs.get("bootstrap_resamples"), int)
        or isinstance(hs.get("bootstrap_resamples"), bool)
        or hs["bootstrap_resamples"] != 10000
        or not _is_finite_num(hs.get("alpha_subject"))
        or not (0.65 <= float(hs["alpha_subject"]) <= 1.0)
        or not _is_finite_num(hs.get("kappa_bar_subject"))
        or not (0.65 <= float(hs["kappa_bar_subject"]) <= 1.0)
        or not _is_finite_num(hs.get("cond_hooper"))
        or not (0.45 <= float(hs["cond_hooper"]) <= 1.0)
        or not _is_finite_num(hs.get("all_hooper"))
        or not (0.40 <= float(hs["all_hooper"]) <= 1.0)
        or not _is_finite_num(hs.get("cond_rolling"))
        or not (0.60 <= float(hs["cond_rolling"]) <= 1.0)
        or not _is_finite_num(hs.get("all_rolling"))
        or not (0.55 <= float(hs["all_rolling"]) <= 1.0)
        or not _is_finite_num(hs.get("alpha_masi"))
        or not (0.45 <= float(hs["alpha_masi"]) <= 1.0)
        or not _is_finite_num(hs.get("empty_rate"))
        or not (0.0 <= float(hs["empty_rate"]) <= 0.20)
        or not isinstance(hs.get("min_area_both_non_empty"), int)
        or isinstance(hs.get("min_area_both_non_empty"), bool)
        or not (8 <= hs["min_area_both_non_empty"] <= 12)
        or not _is_finite_num(hs.get("p_max"))
        or not (0.0 < float(hs["p_max"]) <= 0.15)
        or not _is_finite_num(hs.get("leaf_specificity"))
        or not (0.0 <= float(hs["leaf_specificity"]) <= 1.0)
        or not _is_finite_num(hs.get("seeded_gap_sensitivity"))
        or not (0.85 <= float(hs["seeded_gap_sensitivity"]) <= 1.0)
        or not _is_finite_num(hs.get("alpha_subject_anchor"))
        or not (-1.0 <= float(hs["alpha_subject_anchor"]) <= 1.0)
        or not _is_finite_num(hs.get("alpha_masi_anchor"))
        or not (-1.0 <= float(hs["alpha_masi_anchor"]) <= 1.0)
        or not _is_finite_num(hs.get("kappa_fleiss_anchor"))
        or not (-1.0 <= float(hs["kappa_fleiss_anchor"]) <= 1.0)
        or not isinstance(hs.get("zero_variance_strata"), list)
        or any(not isinstance(z, str) or not z for z in hs["zero_variance_strata"])
        or not isinstance(hs.get("anchor_zero_variance_flags"), dict)
        or set(hs["anchor_zero_variance_flags"].keys()) != set(_ANCHOR_BOOTSTRAP_CI_KEYS)
        or any(not isinstance(v, bool) for v in hs["anchor_zero_variance_flags"].values())
        or not isinstance(hs.get("degenerate_resample_count"), int)
        or isinstance(hs.get("degenerate_resample_count"), bool)
        or not (0 <= hs["degenerate_resample_count"] <= 100)
        or not isinstance(hs.get("anchor_degenerate_resample_count"), int)
        or isinstance(hs.get("anchor_degenerate_resample_count"), bool)
        or not (0 <= hs["anchor_degenerate_resample_count"] <= 100)
        or not isinstance(hs.get("recalibration_rerun_index"), int)
        or isinstance(hs.get("recalibration_rerun_index"), bool)
        or hs["recalibration_rerun_index"] not in (0, 1)
        or not _HEX64_RE.match(str(hs.get("study_bundle_sha256") or ""))
    ):
        raise ConceptGraphError(
            f"schema sync: version {target_version!r} requires complete finite Stage 2 human study receipt matching candidate-experts.md §4.3 "
            "(n_items==240, n_raters==12, M_canon>=120, M_eff>=120, max_areas_per_component<=4, 10000 resamples, "
            "all 7 primary floors, leaf_specificity, seeded_gap_sensitivity>=0.85, 3 anchor metrics, zero_variance_strata, "
            "anchor_zero_variance_flags, 95% CIs, degenerate counts <=100, recalibration_rerun_index in {0, 1})"
        )
    decl_primary_cis = _validate_ci_map(
        hs.get("primary_bootstrap_95_ci"),
        _PRIMARY_BOOTSTRAP_CI_KEYS,
        "human_study_stage2.primary_bootstrap_95_ci",
    )
    decl_anchor_cis = _validate_ci_map(
        hs.get("anchor_bootstrap_95_ci"),
        _ANCHOR_BOOTSTRAP_CI_KEYS,
        "human_study_stage2.anchor_bootstrap_95_ci",
    )

    run_stage2_estimator_self_tests()

    eff_root = root if root is not None else REPO_ROOT_PATH
    gov_rel = str(rel_rec.get("governance_bundle_relpath") or "").strip()
    gov_sha_decl = str(rel_rec.get("governance_bundle_sha256") or "").strip()
    if not _HEX64_RE.match(gov_sha_decl) or gov_sha_decl != str(hs["study_bundle_sha256"]):
        raise ConceptGraphError(
            "schema sync: release_governance_receipt requires governance_bundle_sha256 matching human_study_stage2.study_bundle_sha256"
        )
    gov_path = _resolve_confined_method_study_file(eff_root, gov_rel, "governance_bundle_relpath")
    gov_bytes = gov_path.read_bytes()
    if hashlib.sha256(gov_bytes).hexdigest() != gov_sha_decl:
        raise ConceptGraphError("schema sync: governance_bundle_relpath SHA-256 disagrees with governance_bundle_sha256")
    gov_bundle = json_loads_strict(gov_bytes.decode("utf-8"), context="governance_bundle")
    if (
        not isinstance(gov_bundle, dict)
        or gov_bundle.get("schema_version") != 1
        or str(gov_bundle.get("cycle_id") or "").strip() != cycle_id
        or str(gov_bundle.get("target_version") or "").strip() != target_version
        or str(gov_bundle.get("candidate_taxonomy_sha256") or "").strip() != expected_tax_sha
    ):
        raise ConceptGraphError(
            "schema sync: governance bundle header disagrees with cycle_id, target_version, or candidate_taxonomy_sha256"
        )
    if expected_non_guides_sha is not None and str(gov_bundle.get("candidate_non_guides_taxonomy_sha256") or "").strip() != expected_non_guides_sha:
        raise ConceptGraphError("schema sync: governance bundle candidate_non_guides_taxonomy_sha256 mismatch")

    eff_burned = burned_doc if burned_doc is not None else _load_burned_clusters(eff_root)
    if candidate_tax_doc is not None:
        eff_cand = candidate_tax_doc
    else:
        from ..field_graph import load_taxonomy_yaml

        eff_cand = load_taxonomy_yaml(eff_root / "taxonomy/taxonomy.yml")

    exp_model_status = expected_status_for_version(target_version)
    if eff_cand.get("status") != exp_model_status:
        raise ConceptGraphError(
            f"schema sync: candidate taxonomy status {eff_cand.get('status')!r} disagrees with expected {exp_model_status!r} for {target_version!r}"
        )
    if isinstance(eff_cand.get("skos_hierarchy"), dict) and "status" in eff_cand["skos_hierarchy"]:
        if eff_cand["skos_hierarchy"].get("status") != exp_model_status:
            raise ConceptGraphError(
                f"schema sync: skos_hierarchy.status {eff_cand['skos_hierarchy'].get('status')!r} disagrees with {exp_model_status!r}"
            )

    # 1. Verify charter attestation, Editor exclusion, >=46 distinct persons, COI sub-caps, appointments, reports, & 7-seat votes.
    charter_att = gov_bundle.get("charter_attestation")
    vote_rec = gov_bundle.get("vote_record")
    if not isinstance(charter_att, dict) or not charter_att or not isinstance(vote_rec, dict) or not vote_rec:
        raise ConceptGraphError("schema sync: governance bundle missing charter_attestation or vote_record")
    if hashlib.sha256(canonical_taxonomy_json(charter_att).encode("utf-8")).hexdigest() != board["charter_sha256"]:
        raise ConceptGraphError("schema sync: charter_attestation SHA-256 disagrees with board_signoff.charter_sha256")
    if hashlib.sha256(canonical_taxonomy_json(vote_rec).encode("utf-8")).hexdigest() != board["vote_record_sha256"]:
        raise ConceptGraphError("schema sync: vote_record SHA-256 disagrees with board_signoff.vote_record_sha256")

    editor_pid = str(gov_bundle.get("editor_person_id") or "").strip()
    editor_id_commit = str(gov_bundle.get("editor_identity_commitment_sha256") or "").strip()
    editor_name_sha = str(gov_bundle.get("editor_normalized_name_sha256") or "").strip()
    if (
        len(editor_pid) < 3
        or not _HEX64_RE.match(editor_id_commit)
        or not _HEX64_RE.match(editor_name_sha)
        or editor_id_commit == editor_name_sha
    ):
        raise ConceptGraphError(
            "schema sync: governance bundle requires explicit editor_person_id and distinct 64-hex editor_identity_commitment_sha256 & editor_normalized_name_sha256"
        )
    seen_pids: set[str] = {editor_pid}
    seen_id_commits: set[str] = {editor_id_commit}
    seen_name_shas: set[str] = {editor_name_sha}
    all_persons_by_pid: dict[str, dict] = {}

    conv = _validate_person_record(
        gov_bundle.get("convenor"),
        "convenor",
        seen_pids,
        seen_id_commits,
        seen_name_shas,
        editor_pid=editor_pid,
        require_non_editor=True,
    )
    if conv.get("voting") is not False:
        raise ConceptGraphError("schema sync: convenor must be non-voting (voting=false)")
    all_persons_by_pid[conv["_pid"]] = conv

    nom_panel_raw = gov_bundle.get("nominating_panel")
    if not isinstance(nom_panel_raw, list) or len(nom_panel_raw) != 3:
        raise ConceptGraphError("schema sync: nominating_panel must list 3 independent members")
    nom_panel_vp: list[dict] = []
    nom_seen_insts: set[str] = set(conv["_all_insts"])
    for idx_np, np_rec in enumerate(nom_panel_raw):
        vp_np = _validate_person_record(
            np_rec,
            f"nominating_panel[{idx_np}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
            require_non_editor=True,
        )
        if vp_np.get("voting") is not False:
            raise ConceptGraphError(f"schema sync: nominating_panel[{idx_np}] must be non-voting (voting=false)")
        if not vp_np["_all_insts"].isdisjoint(nom_seen_insts):
            raise ConceptGraphError(
                f"schema sync: nominating_panel[{idx_np}] shares an institution with convenor or another nominating-panel member"
            )
        nom_seen_insts.update(vp_np["_all_insts"])
        nom_panel_vp.append(vp_np)
        all_persons_by_pid[vp_np["_pid"]] = vp_np

    steer = _validate_person_record(
        gov_bundle.get("steering_advisor"),
        "steering_advisor",
        seen_pids,
        seen_id_commits,
        seen_name_shas,
        editor_pid=editor_pid,
        require_non_editor=True,
    )
    if steer.get("voting") is not False:
        raise ConceptGraphError("schema sync: steering_advisor must be non-voting (voting=false)")
    all_persons_by_pid[steer["_pid"]] = steer

    temp_proxies_raw = gov_bundle.get("temporary_proxies") or []
    if not isinstance(temp_proxies_raw, list):
        raise ConceptGraphError("schema sync: temporary_proxies must be a list")
    temp_proxy_by_pid: dict[str, dict] = {}
    for idx_tp, tp_rec in enumerate(temp_proxies_raw):
        vp_tp = _validate_person_record(
            tp_rec,
            f"temporary_proxies[{idx_tp}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
            require_non_editor=True,
        )
        temp_proxy_by_pid[vp_tp["_pid"]] = vp_tp
        all_persons_by_pid[vp_tp["_pid"]] = vp_tp

    standing_raw = gov_bundle.get("standing_roster")
    if not isinstance(standing_raw, list) or len(standing_raw) != 16:
        raise ConceptGraphError("schema sync: standing_roster must list 16 standing board members/alternates")
    standing_by_pid: dict[str, dict] = {}
    slot_map: dict[tuple[str, str], dict] = {}
    for idx_sr, sr_rec in enumerate(standing_raw):
        seat_peek = str((sr_rec or {}).get("seat") or "").strip() if isinstance(sr_rec, dict) else ""
        vp = _validate_person_record(
            sr_rec,
            f"standing_roster[{idx_sr}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
            require_non_editor=(seat_peek in ("V1", "V2")),
        )
        seat = str(vp.get("seat") or "").strip()
        role = str(vp.get("role") or "").strip()
        if (seat, role) not in _EXPECTED_STANDING_SLOTS or (seat, role) in slot_map:
            raise ConceptGraphError(
                f"schema sync: standing_roster[{idx_sr}] invalid or duplicate (seat={seat!r}, role={role!r})"
            )
        standing_by_pid[vp["_pid"]] = vp
        slot_map[(seat, role)] = vp
        all_persons_by_pid[vp["_pid"]] = vp
    if set(slot_map.keys()) != _EXPECTED_STANDING_SLOTS:
        raise ConceptGraphError("schema sync: standing_roster must populate all 16 required Chair/Co-Chair/Alternate slots")

    for single_seat in ("F", "S", "P", "Q", "V1", "V2"):
        if not slot_map[(single_seat, "chair")]["_all_insts"].isdisjoint(slot_map[(single_seat, "alternate")]["_all_insts"]):
            raise ConceptGraphError(f"schema sync: seat {single_seat} Chair and Alternate must have disjoint institutions")
    t_holders = [slot_map[("T", r)] for r in ("t1_co_chair", "t2_co_chair", "t1_alternate", "t2_alternate")]
    for i_t in range(len(t_holders)):
        for j_t in range(i_t + 1, len(t_holders)):
            if not t_holders[i_t]["_all_insts"].isdisjoint(t_holders[j_t]["_all_insts"]):
                raise ConceptGraphError("schema sync: all 4 Seat T Co-Chairs and Specialty Alternates must have distinct institutions")
    vocab_holders = [slot_map[(s_v, r_v)] for s_v in ("V1", "V2") for r_v in ("chair", "alternate")]
    for i_v in range(len(vocab_holders)):
        for j_v in range(i_v + 1, len(vocab_holders)):
            if not vocab_holders[i_v]["_all_insts"].isdisjoint(vocab_holders[j_v]["_all_insts"]):
                raise ConceptGraphError("schema sync: all 4 V1/V2 Chairs and Alternates must come from distinct non-Editor institutions")

    chair_holders = [m for m in standing_by_pid.values() if m["role"] in ("chair", "t1_co_chair", "t2_co_chair")]
    alt_holders = [m for m in standing_by_pid.values() if m["role"] in ("alternate", "t1_alternate", "t2_alternate")]
    if sum(1 for m in chair_holders if m["editor_institution"]) > 1 or sum(1 for m in alt_holders if m["editor_institution"]) > 1:
        raise ConceptGraphError("schema sync: standing_roster exceeds role COI cap (<=1 Chair/Co-Chair and <=1 Alternate from Editor institution)")
    if sum(1 for m in chair_holders if m["editor_coauthor_5yr"]) > 2 or sum(1 for m in alt_holders if m["editor_coauthor_5yr"]) > 1:
        raise ConceptGraphError("schema sync: standing_roster exceeds role COI cap (<=2 Chairs/Co-Chairs and <=1 Alternate 5-year Editor co-authors)")

    chair_inst_counts: collections.Counter[str] = collections.Counter()
    standing_inst_counts: collections.Counter[str] = collections.Counter()
    for m in chair_holders:
        for inst_tok in m["_all_insts"]:
            chair_inst_counts[inst_tok] += 1
    for m in standing_by_pid.values():
        for inst_tok in m["_all_insts"]:
            standing_inst_counts[inst_tok] += 1
    if any(cnt > 1 for cnt in chair_inst_counts.values()) or any(cnt > 2 for cnt in standing_inst_counts.values()):
        raise ConceptGraphError(
            "schema sync: standing_roster exceeds single-institution concentration cap (<=1 Chair/Co-Chair, <=2 total standing)"
        )

    sub_spec_subs_raw = gov_bundle.get("substitute_specialists") or []
    if not isinstance(sub_spec_subs_raw, list):
        raise ConceptGraphError("schema sync: substitute_specialists must be a list")
    sub_spec_sub_by_pid: dict[str, dict] = {}
    for idx_ss, ss_rec in enumerate(sub_spec_subs_raw):
        vp_ss = _validate_person_record(
            ss_rec,
            f"substitute_specialists[{idx_ss}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
            require_non_editor=True,
        )
        ss_slot = str(vp_ss.get("specialist_slot") or "").strip()
        if ss_slot not in _SPECIALIST_SLOTS_EXPECTED:
            raise ConceptGraphError(f"schema sync: substitute_specialists[{idx_ss}] invalid specialist_slot {ss_slot!r}")
        sub_spec_sub_by_pid[vp_ss["_pid"]] = vp_ss
        all_persons_by_pid[vp_ss["_pid"]] = vp_ss

    specs_raw = gov_bundle.get("sub_area_specialists")
    if not isinstance(specs_raw, list) or len(specs_raw) != 13:
        raise ConceptGraphError("schema sync: sub_area_specialists must list all 13 mandatory sub-area specialists")
    seen_spec_slots: set[str] = set()
    specialist_covered_areas: set[str] = set()
    specs_by_slot: dict[str, dict] = {}
    spec_inst_counts: collections.Counter[str] = collections.Counter()
    for idx_sp, sp_rec in enumerate(specs_raw):
        vp = _validate_person_record(
            sp_rec,
            f"sub_area_specialists[{idx_sp}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
        )
        slot = str(vp.get("specialist_slot") or "").strip()
        if slot not in _SPECIALIST_SLOTS_EXPECTED or slot in seen_spec_slots:
            raise ConceptGraphError(f"schema sync: invalid or duplicate specialist_slot {slot!r}")
        if vp.get("report_submitted") is not True or not _HEX64_RE.match(str(vp.get("report_sha256") or "")):
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} missing submitted report_sha256")
        seen_spec_slots.add(slot)
        specs_by_slot[slot] = vp
        all_persons_by_pid[vp["_pid"]] = vp
        specialist_covered_areas.add("Q4" if slot in ("Q4a", "Q4b") else slot)
        for inst_tok in vp["_all_insts"]:
            spec_inst_counts[inst_tok] += 1
    if (
        sum(1 for m in specs_by_slot.values() if m["editor_institution"]) > 1
        or sum(1 for m in specs_by_slot.values() if m["editor_coauthor_5yr"]) > 3
        or any(cnt > 3 for cnt in spec_inst_counts.values())
    ):
        raise ConceptGraphError(
            "schema sync: sub_area_specialists exceeds COI or single-institution cap (<=1 Editor institution, <=3 5-year Editor co-authors, <=3 per institution)"
        )

    indexers_raw = gov_bundle.get("human_indexers")
    if not isinstance(indexers_raw, list) or len(indexers_raw) != 12:
        raise ConceptGraphError("schema sync: human_indexers must list 12 distinct human indexers across 6 pairs")
    indexers_by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    indexer_inst_counts: collections.Counter[str] = collections.Counter()
    indexers_vp: list[dict] = []
    for idx_hi, hi_rec in enumerate(indexers_raw):
        vp = _validate_person_record(
            hi_rec,
            f"human_indexers[{idx_hi}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
            require_non_editor=True,
        )
        pair_id = str(vp.get("pair_id") or "").strip()
        if pair_id not in _STAGE2_PAIR_ORDER or vp.get("pair_coauthored_5yr") is not False:
            raise ConceptGraphError(f"schema sync: human_indexers[{idx_hi}] requires valid pair_id and pair_coauthored_5yr=false")
        indexers_by_pair[pair_id].append(vp)
        indexers_vp.append(vp)
        all_persons_by_pid[vp["_pid"]] = vp
        for inst_tok in vp["_all_insts"]:
            indexer_inst_counts[inst_tok] += 1
    if any(cnt > 2 for cnt in indexer_inst_counts.values()):
        raise ConceptGraphError("schema sync: human_indexers exceeds <=2 indexers per institution cap")
    for p_id in _STAGE2_PAIR_ORDER:
        pair_members = indexers_by_pair.get(p_id, [])
        if len(pair_members) != 2 or not pair_members[0]["_all_insts"].isdisjoint(pair_members[1]["_all_insts"]):
            raise ConceptGraphError(f"schema sync: pair {p_id} must have 2 indexers from distinct institutions")

    adhoc_raw = gov_bundle.get("adhoc_voters") or []
    if not isinstance(adhoc_raw, list):
        raise ConceptGraphError("schema sync: adhoc_voters must be a list")
    adhoc_by_pid: dict[str, dict] = {}
    for idx_av, av_rec in enumerate(adhoc_raw):
        vp = _validate_person_record(
            av_rec,
            f"adhoc_voters[{idx_av}]",
            seen_pids,
            seen_id_commits,
            seen_name_shas,
            editor_pid=editor_pid,
            require_non_editor=True,
        )
        av_seat = str(vp.get("seat") or "").strip()
        if av_seat not in _BOARD_SEATS_EXPECTED:
            raise ConceptGraphError(f"schema sync: adhoc_voters[{idx_av}] missing valid seat")
        if av_seat == "T" and str(vp.get("t_half") or "").strip() not in ("T1", "T2"):
            raise ConceptGraphError(f"schema sync: adhoc_voters[{idx_av}] for seat T must specify t_half in {{'T1', 'T2'}}")
        adhoc_by_pid[vp["_pid"]] = vp
        all_persons_by_pid[vp["_pid"]] = vp

    if len(seen_pids) - 1 < 46 or len(seen_id_commits) != len(seen_pids) or len(seen_name_shas) != len(seen_pids):
        raise ConceptGraphError(
            "schema sync: governance bundle must have >=46 distinct non-Editor natural persons across person_id, identity_commitment_sha256, and normalized_name_sha256"
        )

    # Verify charter_attestation against live candidate-experts.md and content-addressed pre-outreach selection evidence.
    charter_rel = str(charter_att.get("charter_relpath") or "").strip()
    if charter_rel != "taxonomy/method-study/candidate-experts.md":
        raise ConceptGraphError("schema sync: charter_attestation.charter_relpath must equal 'taxonomy/method-study/candidate-experts.md'")
    charter_live_path = _resolve_confined_method_study_file(eff_root, charter_rel, "charter_attestation.charter_relpath")
    if str(charter_att.get("charter_file_sha256") or "").strip() != hashlib.sha256(charter_live_path.read_bytes()).hexdigest():
        raise ConceptGraphError("schema sync: charter_attestation.charter_file_sha256 disagrees with live candidate-experts.md")
    if not _HEX64_RE.match(str(charter_att.get("convenor_identity_escrow_attestation_sha256") or "")):
        raise ConceptGraphError("schema sync: charter_attestation requires 64-hex convenor_identity_escrow_attestation_sha256")
    conv_trace = charter_att.get("convenor_selection_trace")
    nom_trace = charter_att.get("nominating_panel_selection_trace")
    signed_pids = charter_att.get("signed_by_person_ids")
    _conv_prec_order = ("NISO_Z39_19", "ISO_TC46_SC9_WG8", "W3C_SKOS", "DCMI_USAGE_BOARD", "ISKO_ASIST_FALLBACK")
    if (
        not isinstance(conv_trace, dict)
        or conv_trace.get("precedence_committee") not in _conv_prec_order
        or not _HEX64_RE.match(str(conv_trace.get("signed_coi_declaration_sha256") or ""))
        or not _HEX64_RE.match(str(conv_trace.get("selection_evidence_sha256") or ""))
        or not isinstance(nom_trace, list)
        or len(nom_trace) != 3
        or not isinstance(signed_pids, list)
    ):
        raise ConceptGraphError(
            "schema sync: charter_attestation requires convenor_selection_trace (with selection_evidence_sha256), 3-entry nominating_panel_selection_trace, and signed_by_person_ids"
        )
    conv_ev_bytes = _resolve_confined_method_study_file(
        eff_root,
        str(conv_trace.get("selection_evidence_relpath") or "").strip(),
        "convenor_selection_trace.selection_evidence_relpath",
    ).read_bytes()
    if hashlib.sha256(conv_ev_bytes).hexdigest() != str(conv_trace["selection_evidence_sha256"]):
        raise ConceptGraphError("schema sync: convenor_selection_trace.selection_evidence_relpath SHA-256 mismatch")
    conv_ev_doc = json_loads_strict(conv_ev_bytes.decode("utf-8"), context="convenor_selection_evidence")
    prec_comm = str(conv_trace["precedence_committee"])
    prec_idx = _conv_prec_order.index(prec_comm)
    if (
        not isinstance(conv_ev_doc, dict)
        or conv_ev_doc.get("person_id") != conv["_pid"]
        or conv_ev_doc.get("precedence_committee") != prec_comm
        or len(str(conv_ev_doc.get("selection_basis") or "").strip()) < 15
    ):
        raise ConceptGraphError("schema sync: convenor_selection_evidence header/person_id/selection_basis mismatch")
    if prec_idx > 0:
        req_higher = set(_conv_prec_order[:prec_idx])
        decl_higher = conv_ev_doc.get("prior_precedence_declines_or_conflicts")
        if not isinstance(decl_higher, list):
            raise ConceptGraphError(
                f"schema sync: convenor from {prec_comm} requires prior_precedence_declines_or_conflicts covering {sorted(req_higher)}"
            )
        seen_higher: set[str] = set()
        for dh in decl_higher:
            if (
                not isinstance(dh, dict)
                or dh.get("committee") not in req_higher
                or dh["committee"] in seen_higher
                or len(str(dh.get("reason") or "").strip()) < 15
            ):
                raise ConceptGraphError("schema sync: invalid or duplicate prior_precedence_declines_or_conflicts entry")
            seen_higher.add(str(dh["committee"]))
        if seen_higher != req_higher:
            raise ConceptGraphError(
                f"schema sync: prior_precedence_declines_or_conflicts {sorted(seen_higher)} does not cover {sorted(req_higher)}"
            )

    t_conv_pub = _parse_iso_utc(conv_trace.get("pre_outreach_published_at"), "convenor_selection_trace.pre_outreach_published_at")
    t_nom_outreach = _parse_iso_utc(conv_trace.get("panel_outreach_started_at"), "convenor_selection_trace.panel_outreach_started_at")
    if t_conv_pub > t_nom_outreach:
        raise ConceptGraphError("schema sync: convenor pre_outreach_published_at must precede or equal panel_outreach_started_at")
    nom_pids_set = {m["_pid"] for m in nom_panel_vp}
    nom_by_pid = {m["_pid"]: m for m in nom_panel_vp}
    nom_by_slot: dict[str, dict] = {}
    for idx_nt, nt in enumerate(nom_trace):
        if not isinstance(nt, dict):
            raise ConceptGraphError(f"schema sync: nominating_panel_selection_trace[{idx_nt}] must be a mapping")
        vslot = str(nt.get("venue_slot") or "").strip()
        nt_pid = str(nt.get("person_id") or "").strip()
        nt_ev_sha = str(nt.get("selection_evidence_sha256") or "").strip()
        if (
            vslot not in ("MLSys", "OSDI_SOSP", "KOS_SKOS")
            or vslot in nom_by_slot
            or nt_pid not in nom_by_pid
            or not _HEX64_RE.match(nt_ev_sha)
        ):
            raise ConceptGraphError(
                f"schema sync: nominating_panel_selection_trace[{idx_nt}] invalid/duplicate venue_slot {vslot!r}, person_id, or selection_evidence_sha256"
            )
        nt_ev_bytes = _resolve_confined_method_study_file(
            eff_root,
            str(nt.get("selection_evidence_relpath") or "").strip(),
            f"nominating_panel_selection_trace[{idx_nt}].selection_evidence_relpath",
        ).read_bytes()
        if hashlib.sha256(nt_ev_bytes).hexdigest() != nt_ev_sha:
            raise ConceptGraphError(f"schema sync: nominating_panel_selection_trace[{idx_nt}] selection_evidence_relpath SHA-256 mismatch")
        nt_ev_doc = json_loads_strict(nt_ev_bytes.decode("utf-8"), context=f"nominating_panel_selection_evidence[{idx_nt}]")
        if (
            not isinstance(nt_ev_doc, dict)
            or nt_ev_doc.get("person_id") != nt_pid
            or nt_ev_doc.get("venue_slot") != vslot
            or len(str(nt_ev_doc.get("selection_basis") or "").strip()) < 15
        ):
            raise ConceptGraphError(f"schema sync: nominating_panel_selection_evidence[{idx_nt}] header/selection_basis mismatch")
        nom_by_slot[vslot] = nom_by_pid[nt_pid]

    if set(nom_by_slot.keys()) != {"MLSys", "OSDI_SOSP", "KOS_SKOS"} or {m["_pid"] for m in nom_by_slot.values()} != nom_pids_set:
        raise ConceptGraphError("schema sync: nominating_panel_selection_trace must bind distinct MLSys, OSDI_SOSP, and KOS_SKOS scholars")
    if not ({conv["_pid"], *nom_pids_set}).issubset({str(x) for x in signed_pids}):
        raise ConceptGraphError("schema sync: charter_attestation signed_by_person_ids incomplete")

    # Verify deterministic 3-voter appointment_votes across all appointed governance & indexing roles.
    app_votes = gov_bundle.get("appointment_votes")
    if not isinstance(app_votes, list):
        raise ConceptGraphError("schema sync: governance bundle requires appointment_votes list")
    vocab_pids = {slot_map[(s_v, r_v)]["_pid"] for s_v in ("V1", "V2") for r_v in ("chair", "alternate")}
    nom_order_pids = [nom_by_slot["KOS_SKOS"]["_pid"], nom_by_slot["MLSys"]["_pid"], nom_by_slot["OSDI_SOSP"]["_pid"]]
    tp_order_pids = sorted(temp_proxy_by_pid.keys())
    all_comm_capable_pids = {*vocab_pids, steer["_pid"], *nom_pids_set, *tp_order_pids}
    expected_appointee_ids: set[str] = {
        steer["_pid"],
        *tp_order_pids,
        *standing_by_pid.keys(),
        *(m["_pid"] for m in specs_by_slot.values()),
        *sub_spec_sub_by_pid.keys(),
        *adhoc_by_pid.keys(),
        *(m["_pid"] for m in indexers_vp),
    }

    seen_appointed: set[str] = set()
    for idx_ap, ap_rec in enumerate(app_votes):
        if not isinstance(ap_rec, dict):
            raise ConceptGraphError(f"schema sync: appointment_votes[{idx_ap}] must be a mapping")
        ap_pid = str(ap_rec.get("appointee_person_id") or "").strip()
        voters_list = ap_rec.get("voters")
        if ap_pid not in expected_appointee_ids or ap_pid in seen_appointed or not isinstance(voters_list, list) or len(voters_list) != 3:
            raise ConceptGraphError(
                f"schema sync: appointment_votes[{idx_ap}] invalid/duplicate appointee {ap_pid!r} or non-3-member committee"
            )
        seen_appointed.add(ap_pid)
        appointee_obj = all_persons_by_pid[ap_pid]
        raw_rec_comm = ap_rec.get("recused_or_vacant_committee_members") or []
        if not isinstance(raw_rec_comm, list):
            raise ConceptGraphError(f"schema sync: appointment_votes[{idx_ap}] recused_or_vacant_committee_members must be a list")
        recused_comm_pids: set[str] = set()
        for rc_item in raw_rec_comm:
            if not isinstance(rc_item, dict):
                raise ConceptGraphError(f"schema sync: malformed recused_or_vacant_committee_members entry in appointment_votes[{idx_ap}]")
            rc_pid = str(rc_item.get("person_id") or "").strip()
            rc_reason = str(rc_item.get("reason") or "").strip()
            rc_note = str(rc_item.get("evidence_note") or "").strip()
            if (
                rc_pid not in all_comm_capable_pids
                or rc_pid in recused_comm_pids
                or rc_reason not in ("coauthored_5yr", "vacant_or_unavailable", "topic_or_consortium_coi")
                or len(rc_note) < 15
            ):
                raise ConceptGraphError(
                    f"schema sync: invalid recused_or_vacant_committee_members entry {rc_pid!r} in appointment_votes[{idx_ap}]"
                )
            recused_comm_pids.add(rc_pid)

        def _is_skipped_comm(cand_vpid: str) -> bool:
            if cand_vpid == ap_pid or cand_vpid in recused_comm_pids:
                return True
            return not all_persons_by_pid[cand_vpid]["_all_insts"].isdisjoint(appointee_obj["_all_insts"])

        if ap_pid in temp_proxy_by_pid:
            exp_comm_pids = [p for p in [*nom_order_pids, steer["_pid"]] if not _is_skipped_comm(p)][:3]
        elif ap_pid == steer["_pid"]:
            exp_comm_pids = [p for p in [*nom_order_pids, *tp_order_pids] if not _is_skipped_comm(p)][:3]
        elif ap_pid in vocab_pids:
            exp_comm_pids = [p for p in [steer["_pid"], *nom_order_pids, *tp_order_pids] if not _is_skipped_comm(p)][:3]
            avail_nom = [p for p in nom_order_pids if not _is_skipped_comm(p)]
            if len(avail_nom) >= 2 and len(set(exp_comm_pids) & nom_pids_set) < 2:
                raise ConceptGraphError(f"schema sync: V1/V2 appointment for {ap_pid!r} requires >=2 nominating-panel voters")
        else:
            v1_c = slot_map[("V1", "chair")]["_pid"]
            v1_a = slot_map[("V1", "alternate")]["_pid"]
            v2_c = slot_map[("V2", "chair")]["_pid"]
            v2_a = slot_map[("V2", "alternate")]["_pid"]
            s1_pick = v1_c if not _is_skipped_comm(v1_c) else (v1_a if not _is_skipped_comm(v1_a) else None)
            s2_pick = v2_c if not _is_skipped_comm(v2_c) else (v2_a if not _is_skipped_comm(v2_a) else None)
            s3_pick = steer["_pid"] if not _is_skipped_comm(steer["_pid"]) else None
            chosen_c = [x for x in (s1_pick, s2_pick, s3_pick) if x is not None]
            if s1_pick is None:
                for fb_p in [v2_a, *nom_order_pids, *tp_order_pids]:
                    if fb_p not in chosen_c and not _is_skipped_comm(fb_p):
                        chosen_c.append(fb_p)
                        break
            if s2_pick is None:
                for fb_p in [v1_a, *nom_order_pids, *tp_order_pids]:
                    if fb_p not in chosen_c and not _is_skipped_comm(fb_p):
                        chosen_c.append(fb_p)
                        break
            if s3_pick is None:
                for fb_p in [*nom_order_pids, *tp_order_pids]:
                    if fb_p not in chosen_c and not _is_skipped_comm(fb_p):
                        chosen_c.append(fb_p)
                        break
            exp_comm_pids = chosen_c

        if len(exp_comm_pids) != 3:
            raise ConceptGraphError(
                f"schema sync: fewer than 3 unconflicted committee members available for {ap_pid!r} in appointment_votes[{idx_ap}]"
            )
        seen_vp_here: set[str] = set()
        aff_ap = 0
        for v_entry in voters_list:
            if not isinstance(v_entry, dict):
                raise ConceptGraphError(f"schema sync: malformed voter entry in appointment_votes[{idx_ap}]")
            v_pid = str(v_entry.get("voter_person_id") or "").strip()
            v_choice = v_entry.get("vote")
            if (
                v_pid not in exp_comm_pids
                or v_pid == ap_pid
                or v_pid in seen_vp_here
                or v_pid in recused_comm_pids
                or v_choice not in ("approve", "reject", "abstain")
                or v_entry.get("coauthored_5yr_with_appointee") is not False
            ):
                raise ConceptGraphError(
                    f"schema sync: invalid, out-of-precedence, or conflicted voter {v_pid!r} in appointment_votes[{idx_ap}] for {ap_pid!r} "
                    f"(expected deterministic committee {exp_comm_pids})"
                )
            voter_obj = all_persons_by_pid[v_pid]
            if not voter_obj["_all_insts"].isdisjoint(appointee_obj["_all_insts"]):
                raise ConceptGraphError(
                    f"schema sync: appointment voter {v_pid!r} shares an institution with appointee {ap_pid!r}"
                )
            seen_vp_here.add(v_pid)
            if v_choice == "approve":
                aff_ap += 1
        if seen_vp_here != set(exp_comm_pids):
            raise ConceptGraphError(
                f"schema sync: appointment_votes[{idx_ap}] for {ap_pid!r} voters {sorted(seen_vp_here)} disagree with deterministic committee {sorted(exp_comm_pids)}"
            )
        if aff_ap * 3 < 2 * len(voters_list):
            raise ConceptGraphError(f"schema sync: appointment_votes[{idx_ap}] for {ap_pid!r} failed >=2/3 approval floor")
    if seen_appointed != expected_appointee_ids:
        raise ConceptGraphError("schema sync: appointment_votes missing required appointed governance/indexing roles")

    # Verify V1 standards text & COI disposition attestation, V2 SKOS/crosswalk verification, and exact queue dispositions.
    v1_std = gov_bundle.get("v1_standards_verification")
    v1_pids = {slot_map[("V1", "chair")]["_pid"], slot_map[("V1", "alternate")]["_pid"]} | {
        k for k, v in adhoc_by_pid.items() if v.get("seat") == "V1"
    }
    checked_stds = v1_std.get("checked_standards") if isinstance(v1_std, dict) else None
    if (
        not isinstance(v1_std, dict)
        or str(v1_std.get("signer_person_id") or "").strip() not in v1_pids
        or v1_std.get("primary_standard_texts_verified") is not True
        or v1_std.get("editor_coi_dispositions_confirmed") is not True
        or v1_std.get("compound_term_audit_completed") is not True
        or not isinstance(checked_stds, list)
        or not {"ANSI/NISO Z39.19-2005 (R2010)", "ISO 25964-1:2011"}.issubset({str(s) for s in checked_stds})
    ):
        raise ConceptGraphError(
            "schema sync: v1_standards_verification requires V1 signer, primary_standard_texts_verified=true, editor_coi_dispositions_confirmed=true, compound_term_audit_completed=true, and Z39.19/ISO 25964-1 checks"
        )

    v2_skos = gov_bundle.get("v2_skos_verification")
    v2_pids = {slot_map[("V2", "chair")]["_pid"], slot_map[("V2", "alternate")]["_pid"]} | {
        k for k, v in adhoc_by_pid.items() if v.get("seat") == "V2"
    }
    checked_xwalks = v2_skos.get("checked_crosswalks") if isinstance(v2_skos, dict) else None
    if (
        not isinstance(v2_skos, dict)
        or str(v2_skos.get("signer_person_id") or "").strip() not in v2_pids
        or v2_skos.get("ttl_and_dcat_verified") is not True
        or not isinstance(checked_xwalks, list)
        or not {"ACM CCS", "CS2023", "SFIA 9", "ESCO"}.issubset({str(x) for x in checked_xwalks})
    ):
        raise ConceptGraphError(
            "schema sync: v2_skos_verification requires V2 signer, ttl_and_dcat_verified=true, and ACM CCS/CS2023/SFIA 9/ESCO crosswalk checks"
        )

    cand_concepts_by_id = {str(c["id"]): c for c in eff_cand.get("concepts", [])}
    cand_retired_by_id = {str(r["id"]): r for r in (eff_cand.get("retired") or [])}
    cand_rels_by_id = {str(e["id"]): e for e in eff_cand.get("relationships", [])}
    genesis_rel_ids = {str(e["id"]) for e in load_genesis_baseline_taxonomy(eff_root).get("relationships", [])}
    post_gen_cross_subj_spec = {
        rid
        for rid, rel_e in cand_rels_by_id.items()
        if rid not in genesis_rel_ids
        and rel_e.get("relation") == "specializes"
        and (cand_concepts_by_id.get(str(rel_e.get("source"))) or {}).get("home")
        != (cand_concepts_by_id.get(str(rel_e.get("target"))) or {}).get("home")
    }
    req_split_edges = set(PINNED_BASELINE_SPLIT_HIERARCHY_EDGES) | post_gen_cross_subj_spec
    req_zero_warrant = set(PINNED_BASELINE_ZERO_WARRANT_CONCEPTS)
    eligible_board_voter_pids = set(standing_by_pid.keys()) | set(adhoc_by_pid.keys())

    q_disp = gov_bundle.get("queue_dispositions")
    split_h_q = q_disp.get("split_hierarchy_edges") if isinstance(q_disp, dict) else None
    zero_w_q = q_disp.get("zero_warrant_concepts") if isinstance(q_disp, dict) else None
    if not isinstance(q_disp, dict) or not isinstance(split_h_q, list) or not isinstance(zero_w_q, list):
        raise ConceptGraphError("schema sync: queue_dispositions requires split_hierarchy_edges and zero_warrant_concepts lists")
    for q_lbl, q_list, req_q_ids, active_lookup in (
        ("split_hierarchy_edges", split_h_q, req_split_edges, cand_rels_by_id),
        ("zero_warrant_concepts", zero_w_q, req_zero_warrant, cand_concepts_by_id),
    ):
        seen_q_ids: set[str] = set()
        for idx_q, q_rec in enumerate(q_list):
            if not isinstance(q_rec, dict):
                raise ConceptGraphError(f"schema sync: malformed queue_dispositions.{q_lbl}[{idx_q}]")
            qid = str(q_rec.get("item_id") or "").strip()
            q_d = q_rec.get("disposition")
            q_sig = str(q_rec.get("signer_person_id") or "").strip()
            q_rat = str(q_rec.get("rationale") or "").strip()
            if (
                not qid
                or qid in seen_q_ids
                or q_d not in ("uphold", "revised", "retired", "aliased")
                or q_sig not in eligible_board_voter_pids
                or len(q_rat) < 15
            ):
                raise ConceptGraphError(
                    f"schema sync: queue_dispositions.{q_lbl}[{idx_q}] ({qid!r}) requires unique item_id, valid disposition, board signer_person_id, and >=15-char rationale"
                )
            is_act_q = qid in active_lookup
            if is_act_q and q_d not in ("uphold", "revised"):
                raise ConceptGraphError(
                    f"schema sync: active item {qid!r} in queue_dispositions.{q_lbl} cannot carry disposition={q_d!r}"
                )
            if not is_act_q and q_d in ("uphold", "revised"):
                raise ConceptGraphError(
                    f"schema sync: inactive/removed item {qid!r} in queue_dispositions.{q_lbl} cannot carry disposition={q_d!r}"
                )
            seen_q_ids.add(qid)
        missing_q = sorted(req_q_ids - seen_q_ids)
        if missing_q:
            raise ConceptGraphError(
                f"schema sync: queue_dispositions.{q_lbl} omits required pinned/cross-subject queue items: {missing_q[:5]}"
            )

    # Verify content-addressed specialist reports (13) & chair_area_signoffs (8) across all active concepts per area.
    row_to_area: dict[str, str] = {}
    expected_20_areas: set[str] = set()
    for a_obj in eff_cand.get("areas", []):
        ac_str = str(a_obj["code"])
        expected_20_areas.add(ac_str)
        for r_obj in a_obj.get("rows", []):
            row_to_area[str(r_obj["id"])] = ac_str
            if "code" in r_obj:
                row_to_area[str(r_obj["code"])] = ac_str
    area_cids: dict[str, set[str]] = {ac: set() for ac in expected_20_areas}
    for c_obj in eff_cand.get("concepts", []):
        c_home = str(c_obj.get("home") or "")
        if c_home not in row_to_area:
            raise ConceptGraphError(f"schema sync: concept {c_obj.get('id')} has unknown home row {c_home!r}")
        area_cids[row_to_area[c_home]].add(str(c_obj["id"]))

    derived_unresolved_rejections: list[str] = []

    def _is_item_active_in_candidate(item_id_s: str) -> bool:
        if item_id_s in cand_concepts_by_id or item_id_s in cand_rels_by_id:
            return True
        if item_id_s.startswith("alias:"):
            parts_a = item_id_s.split(":", 2)
            if len(parts_a) == 3 and parts_a[1] in cand_concepts_by_id:
                act_anorms = {_norm_prov_tokens(str(a)) for a in (cand_concepts_by_id[parts_a[1]].get("aliases") or [])}
                return _norm_prov_tokens(parts_a[2]) in act_anorms
        return False

    def _verify_rejection_resolution(br_obj: object, req_approvers: set[str], rej_ctx: str) -> None:
        if not isinstance(br_obj, dict) or not str(br_obj.get("item_id") or "").strip():
            raise ConceptGraphError(f"schema sync: malformed rejection record in {rej_ctx}")
        it_id = str(br_obj["item_id"]).strip()
        if br_obj.get("resolved_or_excluded") is not True:
            derived_unresolved_rejections.append(f"{rej_ctx}:{it_id}")
            return
        res_mode = str(br_obj.get("resolution_mode") or "").strip()
        if res_mode == "excluded_from_candidate":
            if _is_item_active_in_candidate(it_id):
                raise ConceptGraphError(
                    f"schema sync: {rej_ctx} claims resolution_mode='excluded_from_candidate' for {it_id!r}, but item remains active in candidate taxonomy"
                )
        elif res_mode == "renewed_approval":
            res_rel = str(br_obj.get("resolution_relpath") or "").strip()
            res_sha = str(br_obj.get("resolution_sha256") or "").strip()
            if not _HEX64_RE.match(res_sha):
                raise ConceptGraphError(f"schema sync: {rej_ctx} renewed_approval for {it_id!r} requires 64-hex resolution_sha256")
            res_bytes = _resolve_confined_method_study_file(eff_root, res_rel, f"{rej_ctx}.resolution_relpath").read_bytes()
            if hashlib.sha256(res_bytes).hexdigest() != res_sha:
                raise ConceptGraphError(f"schema sync: {rej_ctx} resolution_relpath SHA-256 mismatch for {it_id!r}")
            res_doc = json_loads_strict(res_bytes.decode("utf-8"), context=f"{rej_ctx}.resolution")
            app_pids = (
                {str(x).strip() for x in res_doc.get("approver_person_ids")}
                if isinstance(res_doc, dict) and isinstance(res_doc.get("approver_person_ids"), list)
                else set()
            )
            if (
                not isinstance(res_doc, dict)
                or res_doc.get("item_id") != it_id
                or res_doc.get("candidate_taxonomy_sha256") != expected_tax_sha
                or res_doc.get("approved") is not True
                or len(str(res_doc.get("rationale") or "").strip()) < 15
                or not req_approvers.issubset(app_pids)
            ):
                raise ConceptGraphError(
                    f"schema sync: {rej_ctx} renewed_approval receipt for {it_id!r} fails item/taxonomy SHA/approver verification"
                )
        else:
            raise ConceptGraphError(
                f"schema sync: {rej_ctx} resolved rejection {it_id!r} requires resolution_mode in {{'excluded_from_candidate', 'renewed_approval'}}"
            )

    def _collect_report_rejections(rep_doc: dict, req_approvers: set[str], rep_ctx: str) -> None:
        brejs = rep_doc.get("binding_rejections") or []
        if not isinstance(brejs, list):
            raise ConceptGraphError(f"schema sync: {rep_ctx}.binding_rejections must be a list")
        for br in brejs:
            _verify_rejection_resolution(br, req_approvers, rep_ctx)

    sub_reports_raw = gov_bundle.get("substitute_specialist_reports") or []
    if not isinstance(sub_reports_raw, list):
        raise ConceptGraphError("schema sync: substitute_specialist_reports must be a list")
    sub_cov_by_slot: dict[str, set[str]] = collections.defaultdict(set)
    for idx_srp, srp_meta in enumerate(sub_reports_raw):
        if not isinstance(srp_meta, dict):
            raise ConceptGraphError(f"schema sync: substitute_specialist_reports[{idx_srp}] must be a mapping")
        srp_pid = str(srp_meta.get("reviewer_person_id") or "").strip()
        srp_slot = str(srp_meta.get("specialist_slot") or "").strip()
        srp_rel = str(srp_meta.get("report_relpath") or "").strip()
        srp_sha = str(srp_meta.get("report_sha256") or "").strip()
        if srp_pid not in sub_spec_sub_by_pid or sub_spec_sub_by_pid[srp_pid].get("specialist_slot") != srp_slot or not _HEX64_RE.match(srp_sha):
            raise ConceptGraphError(f"schema sync: invalid substitute_specialist_reports[{idx_srp}]")
        srp_bytes = _resolve_confined_method_study_file(eff_root, srp_rel, f"substitute_specialist_reports[{idx_srp}]").read_bytes()
        if hashlib.sha256(srp_bytes).hexdigest() != srp_sha:
            raise ConceptGraphError(f"schema sync: substitute_specialist_reports[{idx_srp}] SHA-256 mismatch")
        srp_doc = json_loads_strict(srp_bytes.decode("utf-8"), context=f"substitute_specialist_reports[{idx_srp}]")
        if (
            not isinstance(srp_doc, dict)
            or srp_doc.get("specialist_slot") != srp_slot
            or srp_doc.get("reviewer_person_id") != srp_pid
            or srp_doc.get("candidate_taxonomy_sha256") != expected_tax_sha
            or not isinstance(srp_doc.get("reviewed_concept_ids"), list)
        ):
            raise ConceptGraphError(f"schema sync: malformed substitute specialist report payload for {srp_slot}")
        _collect_report_rejections(srp_doc, {srp_pid}, f"sub_spec[{srp_slot}]")
        sub_cov_by_slot[srp_slot].update(str(x) for x in srp_doc["reviewed_concept_ids"])

    q4_split_cov: dict[str, set[str]] = {}
    for slot, vp in sorted(specs_by_slot.items()):
        rep_rel = str(vp.get("report_relpath") or "").strip()
        rep_sha = str(vp["report_sha256"])
        rep_bytes = _resolve_confined_method_study_file(eff_root, rep_rel, f"specialist[{slot}].report_relpath").read_bytes()
        if hashlib.sha256(rep_bytes).hexdigest() != rep_sha:
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} report_relpath SHA-256 mismatch")
        rep_doc = json_loads_strict(rep_bytes.decode("utf-8"), context=f"specialist[{slot}]")
        if (
            not isinstance(rep_doc, dict)
            or rep_doc.get("specialist_slot") != slot
            or rep_doc.get("reviewer_person_id") != vp["_pid"]
            or rep_doc.get("candidate_taxonomy_sha256") != expected_tax_sha
            or not isinstance(rep_doc.get("reviewed_concept_ids"), list)
        ):
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} report header/reviewer/taxonomy SHA mismatch")
        _collect_report_rejections(rep_doc, {vp["_pid"]}, f"specialist[{slot}]")
        rev_set = {str(x) for x in rep_doc["reviewed_concept_ids"]}
        recused_set = {str(x) for x in (rep_doc.get("recused_concept_ids") or [])}
        if not rev_set.isdisjoint(recused_set) or not recused_set.issubset(sub_cov_by_slot.get(slot, set())):
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} recused_concept_ids lack substitute coverage")
        combined_slot_cov = rev_set | sub_cov_by_slot.get(slot, set())
        target_area = "Q4" if slot in ("Q4a", "Q4b") else slot
        if not combined_slot_cov or not combined_slot_cov.issubset(area_cids[target_area]):
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} reviewed_concept_ids outside area {target_area}")
        if slot in ("Q4a", "Q4b"):
            q4_split_cov[slot] = combined_slot_cov
        elif combined_slot_cov != area_cids[target_area]:
            raise ConceptGraphError(f"schema sync: specialist_slot {slot!r} does not cover all active concepts in {target_area}")
    if (q4_split_cov.get("Q4a", set()) | q4_split_cov.get("Q4b", set())) != area_cids["Q4"]:
        raise ConceptGraphError("schema sync: Q4a + Q4b specialist reports do not jointly cover all active concepts in Q4")

    chair_signoffs = gov_bundle.get("chair_area_signoffs")
    if not isinstance(chair_signoffs, dict) or set(chair_signoffs.keys()) != set(_CHAIR_SIGN_OFF_AREAS_TO_SEAT.keys()):
        raise ConceptGraphError(
            "schema sync: chair_area_signoffs must cover exact 8 non-specialist areas {F1,F2,S1,S2,S3,P4,T1,T2}"
        )
    for acode, req_seat in _CHAIR_SIGN_OFF_AREAS_TO_SEAT.items():
        cs_entry = chair_signoffs[acode]
        if (
            not isinstance(cs_entry, dict)
            or cs_entry.get("signed_off") is not True
            or not _HEX64_RE.match(str(cs_entry.get("report_sha256") or ""))
        ):
            raise ConceptGraphError(f"schema sync: chair_area_signoffs[{acode!r}] requires signed_off=true, report_relpath, and report_sha256")
        cs_pid = str(cs_entry.get("signer_person_id") or "").strip()
        cs_person = standing_by_pid.get(cs_pid) or adhoc_by_pid.get(cs_pid)
        if cs_person is None or str(cs_person.get("seat")) != req_seat:
            raise ConceptGraphError(f"schema sync: chair_area_signoffs[{acode!r}] must be signed by an eligible holder of seat {req_seat}")
        if acode == "T1":
            if cs_pid in standing_by_pid and cs_person.get("role") not in ("t1_co_chair", "t1_alternate"):
                raise ConceptGraphError("schema sync: chair_area_signoffs['T1'] must be signed by a T1 Co-Chair or T1 Alternate")
            if cs_pid in adhoc_by_pid and cs_person.get("t_half") != "T1":
                raise ConceptGraphError("schema sync: chair_area_signoffs['T1'] ad-hoc signer must have t_half='T1'")
        elif acode == "T2":
            if cs_pid in standing_by_pid and cs_person.get("role") not in ("t2_co_chair", "t2_alternate"):
                raise ConceptGraphError("schema sync: chair_area_signoffs['T2'] must be signed by a T2 Co-Chair or T2 Alternate")
            if cs_pid in adhoc_by_pid and cs_person.get("t_half") != "T2":
                raise ConceptGraphError("schema sync: chair_area_signoffs['T2'] ad-hoc signer must have t_half='T2'")
        cs_bytes = _resolve_confined_method_study_file(
            eff_root, str(cs_entry.get("report_relpath") or "").strip(), f"chair_area_signoffs[{acode}].report_relpath"
        ).read_bytes()
        if hashlib.sha256(cs_bytes).hexdigest() != str(cs_entry["report_sha256"]):
            raise ConceptGraphError(f"schema sync: chair_area_signoffs[{acode!r}] report_relpath SHA-256 mismatch")
        cs_doc = json_loads_strict(cs_bytes.decode("utf-8"), context=f"chair_area_signoffs[{acode}]")
        if (
            not isinstance(cs_doc, dict)
            or cs_doc.get("area_code") != acode
            or cs_doc.get("signer_person_id") != cs_pid
            or cs_doc.get("candidate_taxonomy_sha256") != expected_tax_sha
            or not isinstance(cs_doc.get("reviewed_concept_ids"), list)
            or {str(x) for x in cs_doc["reviewed_concept_ids"]} != area_cids[acode]
        ):
            raise ConceptGraphError(f"schema sync: chair_area_signoffs[{acode!r}] report fails area/signer/concept-set check")
        _collect_report_rejections(cs_doc, {cs_pid}, f"chair_signoff[{acode}]")
    if (specialist_covered_areas | set(chair_signoffs.keys())) != expected_20_areas:
        raise ConceptGraphError("schema sync: 13 sub_area_specialists + 8 chair_area_signoffs do not cover all 20 taxonomy areas")

    for idx_dr, d_rec in enumerate(vote_rec.get("dissent_records") or []):
        if not isinstance(d_rec, dict) or not _HEX64_RE.match(str(d_rec.get("dissent_sha256") or "")):
            raise ConceptGraphError(f"schema sync: malformed dissent_records[{idx_dr}]")
        d_bytes = _resolve_confined_method_study_file(
            eff_root, str(d_rec.get("dissent_relpath") or "").strip(), f"dissent_records[{idx_dr}]"
        ).read_bytes()
        if hashlib.sha256(d_bytes).hexdigest() != str(d_rec["dissent_sha256"]):
            raise ConceptGraphError(f"schema sync: dissent_records[{idx_dr}] SHA-256 mismatch")
        d_doc = json_loads_strict(d_bytes.decode("utf-8"), context=f"dissent_records[{idx_dr}]")
        d_auth_pid = str(d_doc.get("author_person_id") or "").strip() if isinstance(d_doc, dict) else ""
        d_seat = str(d_doc.get("seat") or "").strip() if isinstance(d_doc, dict) else ""
        if (
            not isinstance(d_doc, dict)
            or d_auth_pid not in all_persons_by_pid
            or d_seat not in _BOARD_SEATS_EXPECTED
            or d_doc.get("candidate_taxonomy_sha256") != expected_tax_sha
        ):
            raise ConceptGraphError(f"schema sync: dissent_records[{idx_dr}] missing valid author_person_id, seat, or candidate_taxonomy_sha256")
        _collect_report_rejections(d_doc, {d_auth_pid}, f"dissent[{idx_dr}]")

    _domain_ring = ("F", "S", "P", "Q", "T")
    for idx_iv, iv_rec in enumerate(vote_rec.get("item_level_votes") or []):
        if not isinstance(iv_rec, dict) or not str(iv_rec.get("item_id") or "").strip():
            raise ConceptGraphError(f"schema sync: malformed item_level_votes[{idx_iv}]")
        iv_item_id = str(iv_rec["item_id"]).strip()
        iv_kind = iv_rec.get("vote_kind")
        iv_seat_votes = iv_rec.get("seat_votes")
        iv_outcome = iv_rec.get("outcome")
        if not isinstance(iv_seat_votes, dict) or not iv_seat_votes or iv_outcome not in ("approve", "binding_reject"):
            raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}] requires seat_votes mapping and outcome in {{'approve', 'binding_reject'}}")
        if not set(iv_seat_votes.keys()).issubset(_BOARD_SEATS_EXPECTED):
            raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}] contains unknown seat key")

        rec_prim_raw = iv_rec.get("recused_primary_holders") or []
        rec_seats_raw = iv_rec.get("recused_seats") or []
        if not isinstance(rec_prim_raw, list) or not isinstance(rec_seats_raw, list):
            raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}] recused_primary_holders/recused_seats must be lists")
        rec_prim_pids: set[str] = set()
        for rph in rec_prim_raw:
            if (
                not isinstance(rph, dict)
                or str(rph.get("person_id") or "").strip() not in standing_by_pid
                or len(str(rph.get("reason") or "").strip()) < 3
                or len(str(rph.get("evidence_note") or "").strip()) < 15
            ):
                raise ConceptGraphError(f"schema sync: malformed recused_primary_holders entry in item_level_votes[{idx_iv}]")
            rec_prim_pids.add(str(rph["person_id"]).strip())
        rec_seat_ids: set[str] = set()
        for rs_it in rec_seats_raw:
            if (
                not isinstance(rs_it, dict)
                or str(rs_it.get("seat") or "").strip() not in _BOARD_SEATS_EXPECTED
                or len(str(rs_it.get("reason") or "").strip()) < 3
                or len(str(rs_it.get("evidence_note") or "").strip()) < 15
            ):
                raise ConceptGraphError(f"schema sync: malformed recused_seats entry in item_level_votes[{idx_iv}]")
            rec_seat_ids.add(str(rs_it["seat"]).strip())

        iv_persons: list[dict] = []
        seen_iv_pids: set[str] = set()
        iv_rejecting_pids: set[str] = set()
        iv_aff = 0
        for s_k, sv_it in sorted(iv_seat_votes.items()):
            if (
                not isinstance(sv_it, dict)
                or sv_it.get("vote") not in ("approve", "reject", "abstain")
                or sv_it.get("item_coi_clear") is not True
                or s_k in rec_seat_ids
            ):
                raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}].seat_votes[{s_k!r}] invalid vote, COI flag, or recused seat collision")
            if s_k == "T":
                it_t1_pid = str(sv_it.get("t1_voter_person_id") or "").strip()
                it_t2_pid = str(sv_it.get("t2_voter_person_id") or "").strip()
                it_t1_v = sv_it.get("t1_vote")
                it_t2_v = sv_it.get("t2_vote")
                if (
                    not it_t1_pid
                    or not it_t2_pid
                    or it_t1_pid == it_t2_pid
                    or it_t1_v not in ("approve", "reject", "abstain")
                    or it_t2_v not in ("approve", "reject", "abstain")
                ):
                    raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}].seat_votes['T'] requires distinct T1/T2 voters and votes")
                for h_lbl, h_pid, c_role, a_role in (
                    ("T1", it_t1_pid, "t1_co_chair", "t1_alternate"),
                    ("T2", it_t2_pid, "t2_co_chair", "t2_alternate"),
                ):
                    if h_pid in seen_iv_pids or h_pid in rec_prim_pids:
                        raise ConceptGraphError(f"schema sync: duplicate or recused {h_lbl} voter {h_pid!r} in item_level_votes[{idx_iv}]")
                    h_obj = standing_by_pid.get(h_pid) or adhoc_by_pid.get(h_pid)
                    if h_obj is None or h_obj.get("seat") != "T":
                        raise ConceptGraphError(f"schema sync: ineligible {h_lbl} voter {h_pid!r} in item_level_votes[{idx_iv}]")
                    if h_pid in standing_by_pid:
                        if h_obj.get("role") not in (c_role, a_role):
                            raise ConceptGraphError(f"schema sync: {h_lbl} voter {h_pid!r} wrong role in item_level_votes[{idx_iv}]")
                        if h_obj.get("role") == a_role and slot_map[("T", c_role)]["_pid"] not in rec_prim_pids:
                            raise ConceptGraphError(f"schema sync: {h_lbl} Alternate substitution requires {c_role} in recused_primary_holders")
                    else:
                        if h_obj.get("t_half") != h_lbl or slot_map[("T", c_role)]["_pid"] not in rec_prim_pids or slot_map[("T", a_role)]["_pid"] not in rec_prim_pids:
                            raise ConceptGraphError(f"schema sync: {h_lbl} ad-hoc substitution requires both standing T holders in recused_primary_holders")
                    seen_iv_pids.add(h_pid)
                    iv_persons.append(h_obj)
                exp_it_tv = (
                    "approve"
                    if (it_t1_v == "approve" and it_t2_v == "approve")
                    else ("reject" if (it_t1_v == "reject" and it_t2_v == "reject") else "abstain")
                )
                if sv_it["vote"] != exp_it_tv:
                    raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}] Seat T joint vote disagrees with (t1_vote, t2_vote)")
                if iv_kind == "escalation_3voter" and it_t1_v != it_t2_v:
                    raise ConceptGraphError(f"schema sync: split Seat T cannot serve on escalation_3voter in item_level_votes[{idx_iv}]")
                if it_t1_v == "reject":
                    iv_rejecting_pids.add(it_t1_pid)
                if it_t2_v == "reject":
                    iv_rejecting_pids.add(it_t2_pid)
            else:
                vp_id = str(sv_it.get("voter_person_id") or "").strip()
                if vp_id in seen_iv_pids or vp_id in rec_prim_pids:
                    raise ConceptGraphError(f"schema sync: duplicate or recused voter {vp_id!r} in item_level_votes[{idx_iv}]")
                vp_obj = standing_by_pid.get(vp_id) or adhoc_by_pid.get(vp_id)
                if vp_obj is None or vp_obj.get("seat") != s_k:
                    raise ConceptGraphError(f"schema sync: ineligible voter {vp_id!r} for seat {s_k} in item_level_votes[{idx_iv}]")
                if vp_id in standing_by_pid:
                    if vp_obj.get("role") == "alternate" and slot_map[(s_k, "chair")]["_pid"] not in rec_prim_pids:
                        raise ConceptGraphError(f"schema sync: seat {s_k} Alternate substitution requires Chair in recused_primary_holders")
                else:
                    if slot_map[(s_k, "chair")]["_pid"] not in rec_prim_pids or slot_map[(s_k, "alternate")]["_pid"] not in rec_prim_pids:
                        raise ConceptGraphError(f"schema sync: seat {s_k} ad-hoc substitution requires Chair and Alternate in recused_primary_holders")
                seen_iv_pids.add(vp_id)
                iv_persons.append(vp_obj)
                if sv_it["vote"] == "reject":
                    iv_rejecting_pids.add(vp_id)
            if sv_it["vote"] == "approve":
                iv_aff += 1

        iv_ed_inst = sum(1 for p in iv_persons if p["editor_institution"])
        iv_ed_co = sum(1 for p in iv_persons if p["editor_coauthor_5yr"])
        iv_inst_counts: collections.Counter[str] = collections.Counter()
        for p_iv in iv_persons:
            for inst_tok in p_iv["_all_insts"]:
                iv_inst_counts[inst_tok] += 1
        if iv_kind == "escalation_3voter":
            orig_seat = str(iv_rec.get("originating_seat") or "").strip()
            if orig_seat not in _domain_ring or orig_seat in iv_seat_votes or orig_seat not in rec_seat_ids:
                raise ConceptGraphError(
                    f"schema sync: escalation_3voter item_level_votes[{idx_iv}] requires originating_seat in {_domain_ring} documented in recused_seats and excluded from seat_votes"
                )
            v_seats_in = [s for s in iv_seat_votes if s in ("V1", "V2")]
            d_seats_in = [s for s in iv_seat_votes if s in _domain_ring]
            if (
                len(iv_seat_votes) != 3
                or len(v_seats_in) != 1
                or len(d_seats_in) != 2
                or iv_ed_inst > 1
                or iv_ed_co > 1
                or any(cnt > 1 for cnt in iv_inst_counts.values())
            ):
                raise ConceptGraphError(
                    f"schema sync: escalation_3voter item_level_votes[{idx_iv}] requires 1 vocabulary seat + 2 domain seats, <=1 per-institution, and <=1/<=1 COI caps"
                )
            if v_seats_in[0] == "V2" and (
                slot_map[("V1", "chair")]["_pid"] not in rec_prim_pids or slot_map[("V1", "alternate")]["_pid"] not in rec_prim_pids
            ):
                raise ConceptGraphError(
                    f"schema sync: escalation_3voter item_level_votes[{idx_iv}] selected V2 without recusing both V1 Chair and V1 Alternate"
                )
            orig_idx = _domain_ring.index(orig_seat)
            circ_order = [_domain_ring[(orig_idx + k_off) % 5] for k_off in range(1, 5)]
            max_sel_pos = max(circ_order.index(ds) for ds in d_seats_in)
            for earlier_ds in circ_order[:max_sel_pos]:
                if earlier_ds not in d_seats_in and earlier_ds not in rec_seat_ids:
                    raise ConceptGraphError(
                        f"schema sync: escalation_3voter item_level_votes[{idx_iv}] skipped circular domain seat {earlier_ds!r} without recused_seats entry"
                    )
            exp_iv_out = "approve" if iv_aff >= 2 else "binding_reject"
        elif iv_kind == "full_board_item":
            missing_fb_seats = _BOARD_SEATS_EXPECTED - set(iv_seat_votes.keys())
            if (
                not (5 <= len(iv_seat_votes) <= 7)
                or not missing_fb_seats.issubset(rec_seat_ids)
                or iv_ed_inst > 1
                or iv_ed_co > 2
                or any(cnt > 2 for cnt in iv_inst_counts.values())
            ):
                raise ConceptGraphError(
                    f"schema sync: full_board_item item_level_votes[{idx_iv}] violates >=5 seat quorum, recused_seats accounting, <=2 per-institution, or <=1/<=2 COI cap"
                )
            exp_iv_out = "approve" if iv_aff >= math.ceil((len(iv_seat_votes) + 1) / 2) else "binding_reject"
        else:
            raise ConceptGraphError(f"schema sync: unknown vote_kind {iv_kind!r} in item_level_votes[{idx_iv}]")
        if iv_outcome != exp_iv_out:
            raise ConceptGraphError(f"schema sync: item_level_votes[{idx_iv}] outcome disagrees with seat tally")
        if iv_outcome == "binding_reject" or iv_rejecting_pids:
            _verify_rejection_resolution(iv_rec, iv_rejecting_pids or seen_iv_pids, f"item_vote[{iv_item_id}]")

    seat_votes = vote_rec.get("seat_votes")
    if not isinstance(seat_votes, dict) or set(seat_votes.keys()) != _BOARD_SEATS_EXPECTED:
        raise ConceptGraphError("schema sync: vote_record.seat_votes must record all 7 voting seats")
    active_voters: list[dict] = []
    seen_active_voters: set[str] = set()
    aff_count = 0
    for seat_k, sv in sorted(seat_votes.items()):
        if not isinstance(sv, dict) or sv.get("recused") is not False or sv.get("vote") not in ("approve", "reject", "abstain"):
            raise ConceptGraphError(f"schema sync: vote_record.seat_votes[{seat_k!r}] invalid vote or recusal flag")
        if seat_k == "T":
            t1_pid = str(sv.get("t1_voter_person_id") or "").strip()
            t2_pid = str(sv.get("t2_voter_person_id") or "").strip()
            t1_v = sv.get("t1_vote")
            t2_v = sv.get("t2_vote")
            if (
                not t1_pid
                or not t2_pid
                or t1_pid == t2_pid
                or t1_v not in ("approve", "reject", "abstain")
                or t2_v not in ("approve", "reject", "abstain")
            ):
                raise ConceptGraphError(
                    "schema sync: vote_record.seat_votes['T'] requires distinct t1_voter_person_id (t1_vote) and t2_voter_person_id (t2_vote)"
                )
            for half_lbl, h_pid, req_roles in (
                ("T1", t1_pid, ("t1_co_chair", "t1_alternate")),
                ("T2", t2_pid, ("t2_co_chair", "t2_alternate")),
            ):
                if h_pid in seen_active_voters:
                    raise ConceptGraphError(f"schema sync: person {h_pid!r} cannot cast multiple active votes")
                seen_active_voters.add(h_pid)
                h_person = standing_by_pid.get(h_pid) or adhoc_by_pid.get(h_pid)
                if h_person is None or str(h_person.get("seat")) != "T":
                    raise ConceptGraphError(f"schema sync: {half_lbl} voter {h_pid!r} is not an eligible Seat T holder")
                if h_pid in standing_by_pid and h_person.get("role") not in req_roles:
                    raise ConceptGraphError(f"schema sync: {half_lbl} voter {h_pid!r} must hold {req_roles}, got {h_person.get('role')!r}")
                if h_pid in adhoc_by_pid and h_person.get("t_half") != half_lbl:
                    raise ConceptGraphError(f"schema sync: {half_lbl} ad-hoc voter {h_pid!r} must have t_half={half_lbl!r}")
                active_voters.append(h_person)
            exp_t_vote = (
                "approve"
                if (t1_v == "approve" and t2_v == "approve")
                else ("reject" if (t1_v == "reject" and t2_v == "reject") else "abstain")
            )
            if (
                sv["vote"] != exp_t_vote
                or vote_rec.get("seat_t_t1_concurred") is not (t1_v == "approve")
                or vote_rec.get("seat_t_t2_concurred") is not (t2_v == "approve")
            ):
                raise ConceptGraphError(
                    "schema sync: Seat T joint vote or seat_t_t1_concurred/seat_t_t2_concurred disagrees with (t1_vote, t2_vote)"
                )
            for t_half_k, t_half_v, t_half_pid in (("t1", t1_v, t1_pid), ("t2", t2_v, t2_pid)):
                if t_half_v == "reject":
                    rej_list_t = sv.get(f"{t_half_k}_rejected_items")
                    if not isinstance(rej_list_t, list) or not rej_list_t:
                        derived_unresolved_rejections.append(f"seat_T:{t_half_k}_reject")
                    else:
                        for br_t in rej_list_t:
                            _verify_rejection_resolution(br_t, {t_half_pid}, f"seat_T.{t_half_k}")
        else:
            vpid = str(sv.get("voter_person_id") or "").strip()
            if not vpid or vpid in seen_active_voters:
                raise ConceptGraphError(f"schema sync: person {vpid!r} cannot cast votes for multiple seats")
            seen_active_voters.add(vpid)
            vperson = standing_by_pid.get(vpid) or adhoc_by_pid.get(vpid)
            if vperson is None or str(vperson.get("seat")) != seat_k:
                raise ConceptGraphError(f"schema sync: voter {vpid!r} is not an eligible member/replacement for seat {seat_k}")
            active_voters.append(vperson)
            if sv["vote"] == "reject":
                rej_list_s = sv.get("rejected_items")
                if not isinstance(rej_list_s, list) or not rej_list_s:
                    derived_unresolved_rejections.append(f"seat_{seat_k}:reject")
                else:
                    for br_s in rej_list_s:
                        _verify_rejection_resolution(br_s, {vpid}, f"seat_{seat_k}")
        if sv["vote"] == "approve":
            aff_count += 1

    active_inst_counts: collections.Counter[str] = collections.Counter()
    for av_p in active_voters:
        for inst_tok in av_p["_all_insts"]:
            active_inst_counts[inst_tok] += 1
    if any(cnt > 2 for cnt in active_inst_counts.values()):
        raise ConceptGraphError("schema sync: active voting holders exceed <=2 single-institution concentration cap")

    act_inst = sum(1 for v in active_voters if v["editor_institution"])
    act_coauth = sum(1 for v in active_voters if v["editor_coauthor_5yr"])
    if (
        len(active_voters) != 8
        or aff_count != board["affirmative_votes"]
        or aff_count < 5
        or seat_votes["V1"]["vote"] != "approve"
        or seat_votes["V2"]["vote"] != "approve"
        or act_inst != board["active_editor_institution_voters"]
        or act_inst > 1
        or act_coauth != board["active_editor_coauthor_voters"]
        or act_coauth > 2
        or derived_unresolved_rejections
        or vote_rec.get("unresolved_binding_rejections") != []
    ):
        raise ConceptGraphError(
            "schema sync: vote_record fails supermajority, V1/V2 concurrence, 8-holder COI cap, or unresolved binding rejection check"
        )

    # 2. Verify pre-1.0 legacy alias dispositions across all entries, all 7 pinned legacy concepts, and all unattested active legacy aliases.
    leg_disp = gov_bundle.get("v1_legacy_alias_dispositions")
    if not isinstance(leg_disp, dict) or not set(PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS.keys()).issubset(leg_disp.keys()):
        raise ConceptGraphError("schema sync: governance bundle must record v1_legacy_alias_dispositions covering all 7 pinned legacy concepts")
    for disp_cid, c_disp_map in leg_disp.items():
        if (disp_cid not in cand_concepts_by_id and disp_cid not in cand_retired_by_id) or not isinstance(c_disp_map, dict) or not c_disp_map:
            raise ConceptGraphError(f"schema sync: v1_legacy_alias_dispositions[{disp_cid!r}] must target a known concept with a non-empty alias mapping")
        act_norms_here = {_norm_prov_tokens(str(a)) for a in ((cand_concepts_by_id.get(disp_cid) or {}).get("aliases") or [])}
        for al_k, drec in c_disp_map.items():
            if (
                not isinstance(al_k, str)
                or not _norm_prov_tokens(al_k)
                or not isinstance(drec, dict)
                or drec.get("disposition") not in ("uphold", "remove", "reclassify-or-split")
                or drec.get("v1_signed_off") is not True
                or len(str(drec.get("rationale") or "").strip()) < 15
            ):
                raise ConceptGraphError(f"schema sync: invalid v1_legacy_alias_dispositions entry on {disp_cid} alias {al_k!r}")
            is_act_al = _norm_prov_tokens(al_k) in act_norms_here
            if is_act_al and drec["disposition"] != "uphold":
                raise ConceptGraphError(f"schema sync: active legacy alias {al_k!r} on {disp_cid} requires disposition='uphold'")
            if not is_act_al and drec["disposition"] == "uphold":
                raise ConceptGraphError(f"schema sync: inactive/removed legacy alias {al_k!r} on {disp_cid} cannot carry disposition='uphold'")

    for gcid, req_aliases in PINNED_GRANDFATHERED_ALIAS_EXCEPTIONS.items():
        c_disp = leg_disp.get(gcid)
        if not isinstance(c_disp, dict) or not set(req_aliases).issubset(c_disp.keys()):
            raise ConceptGraphError(f"schema sync: v1_legacy_alias_dispositions[{gcid!r}] must cover {list(req_aliases)}")

    for cid_cand, c_node in cand_concepts_by_id.items():
        att_meta = ((prov_concepts.get(cid_cand) if isinstance(prov_concepts, dict) else None) or {}).get("alias_metadata") or {}
        att_norms = {_norm_prov_tokens(str(ak)) for ak in att_meta} if isinstance(att_meta, dict) else set()
        for active_al in c_node.get("aliases") or []:
            al_str = str(active_al)
            if _norm_prov_tokens(al_str) in att_norms:
                continue
            c_disp_all = leg_disp.get(cid_cand)
            drec_all = c_disp_all.get(al_str) if isinstance(c_disp_all, dict) else None
            if (
                not isinstance(drec_all, dict)
                or drec_all.get("disposition") != "uphold"
                or drec_all.get("v1_signed_off") is not True
                or len(str(drec_all.get("rationale") or "").strip()) < 15
            ):
                raise ConceptGraphError(
                    f"schema sync: unattested legacy alias {al_str!r} on {cid_cand} requires v1_legacy_alias_dispositions[{cid_cand!r}][{al_str!r}] with disposition='uphold', v1_signed_off=true, and >=15-char rationale"
                )

    valid_subjects = {str(r["id"]) for a in eff_cand.get("areas", []) for r in a.get("rows", [])}
    valid_concepts = {str(c["id"]) for c in eff_cand.get("concepts", [])}
    non_leaf_concepts = {
        str(e["target"])
        for e in eff_cand.get("relationships", [])
        if e.get("relation") == "specializes" and str(e.get("source")) in valid_concepts
    }

    # 3. Verify calibrated seeded-gap sensitivity >= 85% over complete (unit_id, judge_id, ablated_concept_id) denominator.
    sg_audit = gov_bundle.get("seeded_gap_audit")
    if not isinstance(sg_audit, dict):
        raise ConceptGraphError("schema sync: non-draft release requires seeded_gap_audit mapping")
    sg_man_rel = str(sg_audit.get("source_manifest_relpath") or "").strip()
    sg_man_sha = str(sg_audit.get("source_manifest_sha256") or "").strip()
    sg_judge_prov = sg_audit.get("judge_provenance")
    sg_judges_raw = sg_audit.get("judge_ids")
    sg_ablated_raw = sg_audit.get("ablated_concept_ids")
    sg_base_cov = sg_audit.get("baseline_coverage_judgments")
    sg_items = sg_audit.get("seeded_gap_items")
    if (
        not _HEX64_RE.match(sg_man_sha)
        or not isinstance(sg_judge_prov, dict)
        or not isinstance(sg_judges_raw, list)
        or len(sg_judges_raw) < 2
        or any(not isinstance(j, str) or not j.strip() for j in sg_judges_raw)
        or len({j.strip() for j in sg_judges_raw}) != len(sg_judges_raw)
        or not isinstance(sg_ablated_raw, list)
        or len(sg_ablated_raw) < 10
        or any(not isinstance(c, str) or c.strip() not in valid_concepts for c in sg_ablated_raw)
        or len({c.strip() for c in sg_ablated_raw}) != len(sg_ablated_raw)
        or not isinstance(sg_base_cov, list)
        or not sg_base_cov
        or not isinstance(sg_items, list)
        or len(sg_items) < 10
    ):
        raise ConceptGraphError(
            "schema sync: seeded_gap_audit requires source_manifest_sha256, judge_provenance, >=2 distinct judge_ids, >=10 active ablated_concept_ids, baseline_coverage_judgments, and seeded_gap_items"
        )
    sg_man_bytes = _resolve_confined_method_study_file(eff_root, sg_man_rel, "seeded_gap_audit.source_manifest_relpath").read_bytes()
    if hashlib.sha256(sg_man_bytes).hexdigest() != sg_man_sha:
        raise ConceptGraphError("schema sync: seeded_gap_audit.source_manifest_relpath SHA-256 mismatch")
    sg_man_doc = json_loads_strict(sg_man_bytes.decode("utf-8"), context="seeded_gap_audit.source_manifest")
    sg_man_units = sg_man_doc.get("units") if isinstance(sg_man_doc, dict) else None
    if not isinstance(sg_man_units, list) or len(sg_man_units) < 10:
        raise ConceptGraphError("schema sync: seeded_gap_audit source_manifest must contain >=10 units")
    sg_closure = compute_lineage_components(
        sg_man_units,
        eff_burned,
        include_all_exposed_meta=True,
        strict_prospective_work_bijection=True,
    )
    sg_merges: dict[str, str] = sg_closure["merges"]  # type: ignore[assignment]
    sg_pf_to_comp: dict[str, str] = sg_closure["pf_to_component"]  # type: ignore[assignment]
    sg_comp_ed: set[str] = sg_closure["component_editor_affiliated"]  # type: ignore[assignment]
    sg_comp_ed_co: set[str] = sg_closure["component_independence_excluded"]  # type: ignore[assignment]
    sg_internal: set[str] = sg_closure["internal_clusters"]  # type: ignore[assignment]
    sg_units_by_id: dict[str, dict] = {}
    for idx_smu, smu in enumerate(sg_man_units):
        if not isinstance(smu, dict) or not str(smu.get("unit_id") or "").strip():
            raise ConceptGraphError(f"schema sync: malformed seeded_gap_audit source_manifest.units[{idx_smu}]")
        sm_uid = str(smu["unit_id"]).strip()
        if sm_uid in sg_units_by_id:
            raise ConceptGraphError(f"schema sync: duplicate unit_id {sm_uid!r} in seeded_gap_audit source_manifest")
        sg_units_by_id[sm_uid] = smu

    sg_judge_set = {j.strip() for j in sg_judges_raw}
    if set(sg_judge_prov.keys()) != sg_judge_set:
        raise ConceptGraphError("schema sync: seeded_gap_audit.judge_provenance keys must match judge_ids")
    sg_judge_families: set[str] = set()
    for j_id_k, jp_rec in sg_judge_prov.items():
        if not isinstance(jp_rec, dict):
            raise ConceptGraphError(f"schema sync: seeded_gap_audit.judge_provenance[{j_id_k!r}] must be a mapping")
        jfam = str(jp_rec.get("model_family") or "").strip()
        jmod = str(jp_rec.get("model_id") or "").strip()
        jout_rel = str(jp_rec.get("raw_output_relpath") or "").strip()
        jout_sha = str(jp_rec.get("raw_output_sha256") or "").strip()
        if jfam not in ("claude", "codex", "gemini") or len(jmod) < 3 or not _HEX64_RE.match(jout_sha):
            raise ConceptGraphError(
                f"schema sync: seeded_gap_audit.judge_provenance[{j_id_k!r}] requires model_family in {{'claude','codex','gemini'}}, model_id, and 64-hex raw_output_sha256"
            )
        jout_bytes = _resolve_confined_method_study_file(
            eff_root, jout_rel, f"seeded_gap_audit.judge_provenance[{j_id_k}].raw_output_relpath"
        ).read_bytes()
        if hashlib.sha256(jout_bytes).hexdigest() != jout_sha:
            raise ConceptGraphError(f"schema sync: seeded_gap_audit.judge_provenance[{j_id_k!r}] raw_output_relpath SHA-256 mismatch")
        sg_judge_families.add(jfam)
    if len(sg_judge_families) < 2:
        raise ConceptGraphError("schema sync: seeded_gap_audit requires >=2 distinct model families across judge_provenance")

    sg_ablated_set = {c.strip() for c in sg_ablated_raw}
    base_judgments_by_pair: dict[tuple[str, str], tuple[float, list[str]]] = {}
    sg_eval_units: set[str] = set()
    for idx_bc, bc_rec in enumerate(sg_base_cov):
        if not isinstance(bc_rec, dict):
            raise ConceptGraphError(f"schema sync: malformed baseline_coverage_judgments[{idx_bc}]")
        b_uid = str(bc_rec.get("unit_id") or "").strip()
        b_jid = str(bc_rec.get("judge_id") or "").strip()
        if not b_uid or b_uid not in sg_units_by_id or b_jid not in sg_judge_set or (b_uid, b_jid) in base_judgments_by_pair:
            raise ConceptGraphError(f"schema sync: invalid or unmanifested unit/judge in baseline_coverage_judgments[{idx_bc}]")
        smu_here = sg_units_by_id[b_uid]
        smu_pf = sg_merges.get(str(smu_here.get("provenance_family_id") or "").strip(), str(smu_here.get("provenance_family_id") or "").strip())
        smu_comp = sg_pf_to_comp.get(smu_pf, "")
        smu_pass = " ".join(str(smu_here.get("passage") or "").split())
        if (
            smu_here.get("origin") != "external"
            or bool(smu_here.get("editor_affiliated", False))
            or bool(smu_here.get("editor_coauthor_5yr", False))
            or smu_pf in sg_internal
            or not smu_comp
            or smu_comp == "eff_lineage:internal"
            or smu_comp in sg_comp_ed
            or smu_comp in sg_comp_ed_co
            or len(smu_pass) < 10
        ):
            raise ConceptGraphError(
                f"schema sync: seeded_gap_audit unit {b_uid!r} must be non-Editor external with >=10-char passage and zero 5-year Editor co-authorship"
            )
        b_sc, b_cids_clean = validate_ordinal_coverage_judgment(
            bc_rec.get("baseline_score"),
            bc_rec.get("assigned_concepts"),
            valid_concepts,
            f"baseline_coverage_judgments[{idx_bc}]",
            check_active=True,
        )
        sg_eval_units.add(b_uid)
        base_judgments_by_pair[(b_uid, b_jid)] = (b_sc, b_cids_clean)
    if len(sg_eval_units) < 10 or set(base_judgments_by_pair.keys()) != {(u, j) for u in sg_eval_units for j in sg_judge_set}:
        raise ConceptGraphError(
            "schema sync: seeded_gap_audit.baseline_coverage_judgments must cover full cross-product of >=10 eval units x judge_ids"
        )
    expected_sg_probes: dict[tuple[str, str, str], float] = {}
    probed_ablated_cids: set[str] = set()
    for (b_uid, b_jid), (b_sc, b_cids) in base_judgments_by_pair.items():
        if b_sc in (0.5, 1.0):
            for c_hit in b_cids:
                if c_hit in sg_ablated_set:
                    expected_sg_probes[(b_uid, b_jid, c_hit)] = b_sc
                    probed_ablated_cids.add(c_hit)
    if probed_ablated_cids != sg_ablated_set:
        raise ConceptGraphError(
            "schema sync: every concept in seeded_gap_audit.ablated_concept_ids must have >=1 eligible baseline probe"
        )
    seen_sg_keys: set[tuple[str, str, str]] = set()
    degraded_cnt = 0
    for idx_sg, sgi in enumerate(sg_items):
        if not isinstance(sgi, dict) or not _is_finite_num(sgi.get("baseline_score")):
            raise ConceptGraphError(f"schema sync: malformed seeded_gap_items[{idx_sg}]")
        sg_u = str(sgi.get("unit_id") or "").strip()
        sg_j = str(sgi.get("judge_id") or "").strip()
        sg_c = str(sgi.get("ablated_concept_id") or "").strip()
        sg_key = (sg_u, sg_j, sg_c)
        if sg_key not in expected_sg_probes or sg_key in seen_sg_keys:
            raise ConceptGraphError(
                f"schema sync: seeded_gap_items[{idx_sg}] {sg_key} is duplicate or outside baseline eligible probe denominator"
            )
        seen_sg_keys.add(sg_key)
        bs = float(sgi["baseline_score"])
        as_v, ab_c_list = validate_ordinal_coverage_judgment(
            sgi.get("ablated_score"),
            sgi.get("ablated_concepts"),
            valid_concepts,
            f"seeded_gap_items[{idx_sg}]",
            check_active=True,
        )
        if bs != expected_sg_probes[sg_key] or as_v > bs or sg_c in set(ab_c_list):
            raise ConceptGraphError(f"schema sync: invalid scores or ablated_concepts in seeded_gap_items[{idx_sg}]")
        if as_v < bs:
            degraded_cnt += 1
    if seen_sg_keys != set(expected_sg_probes.keys()):
        raise ConceptGraphError(
            "schema sync: seeded_gap_items omits eligible (unit_id, judge_id, ablated_concept_id) baseline probes"
        )
    sg_sens = degraded_cnt / len(sg_items)
    if sg_sens < 0.85:
        raise ConceptGraphError(f"schema sync: non-draft release seeded-gap sensitivity {sg_sens:.4f} < 0.85 floor")
    if abs(float(hs["seeded_gap_sensitivity"]) - sg_sens) > 1e-9:
        raise ConceptGraphError("schema sync: human_study_stage2.seeded_gap_sensitivity disagrees with recomputed seeded_gap_items")

    # 4. Verify Stage 1 pre-registration, Stage 2 protocol approval, candidate-pool closure, pre-unseal ledger, routing, & dual bootstrap.
    s2 = gov_bundle.get("stage2_study")
    if not isinstance(s2, dict):
        raise ConceptGraphError("schema sync: governance bundle missing stage2_study")
    proto = s2.get("protocol_approval")
    if not isinstance(proto, dict):
        raise ConceptGraphError("schema sync: stage2_study requires protocol_approval mapping")
    v1_app_pid = str(proto.get("v1_approver_person_id") or "").strip()
    guide_rel = str(proto.get("frozen_tagging_guide_relpath") or "").strip()
    guide_sha = str(proto.get("frozen_tagging_guide_sha256") or "").strip()
    pool_rel = str(proto.get("candidate_pool_manifest_relpath") or "").strip()
    pool_sha = str(proto.get("candidate_pool_manifest_sha256") or "").strip()
    s1_reg_id = str(s2.get("stage1_registration_id") or "").strip()
    if v1_app_pid not in v1_pids or len(s1_reg_id) < 3 or not _HEX64_RE.match(guide_sha) or not _HEX64_RE.match(pool_sha):
        raise ConceptGraphError(
            "schema sync: stage2_study requires V1 v1_approver_person_id, stage1_registration_id, frozen_tagging_guide_sha256, and candidate_pool_manifest_sha256"
        )
    t_proto = _parse_iso_utc(proto.get("protocol_approved_at"), "protocol_approval.protocol_approved_at")
    t_s1_reg = _parse_iso_utc(proto.get("stage1_registered_at"), "protocol_approval.stage1_registered_at")
    t_s1_done = _parse_iso_utc(proto.get("stage1_completed_at"), "protocol_approval.stage1_completed_at")
    t_guide = _parse_iso_utc(proto.get("guide_frozen_at"), "protocol_approval.guide_frozen_at")
    t_unseal = _parse_iso_utc(proto.get("unsealed_at"), "protocol_approval.unsealed_at")
    t_collect = _parse_iso_utc(proto.get("collection_completed_at"), "protocol_approval.collection_completed_at")
    if not (t_proto <= t_s1_reg <= t_s1_done <= t_guide <= t_unseal <= t_collect):
        raise ConceptGraphError(
            "schema sync: Stage 2 timestamps violate protocol_approved_at <= stage1_registered_at <= stage1_completed_at <= guide_frozen_at <= unsealed_at <= collection_completed_at"
        )
    guide_bytes = _resolve_confined_method_study_file(eff_root, guide_rel, "frozen_tagging_guide_relpath").read_bytes()
    if hashlib.sha256(guide_bytes).hexdigest() != guide_sha:
        raise ConceptGraphError("schema sync: frozen_tagging_guide_relpath SHA-256 mismatch")
    pool_bytes = _resolve_confined_method_study_file(eff_root, pool_rel, "candidate_pool_manifest_relpath").read_bytes()
    if hashlib.sha256(pool_bytes).hexdigest() != pool_sha:
        raise ConceptGraphError("schema sync: candidate_pool_manifest_relpath SHA-256 mismatch")
    pool_doc = json_loads_strict(pool_bytes.decode("utf-8"), context="candidate_pool_manifest")
    pool_units = pool_doc.get("units") if isinstance(pool_doc, dict) else None
    if not isinstance(pool_units, list) or len(pool_units) < 270:
        raise ConceptGraphError("schema sync: candidate_pool_manifest.units must list >=270 prospective external units")
    pool_by_uid: dict[str, dict] = {}
    for idx_pu, pu in enumerate(pool_units):
        if not isinstance(pu, dict) or not str(pu.get("unit_id") or "").strip():
            raise ConceptGraphError(f"schema sync: malformed candidate_pool_manifest.units[{idx_pu}]")
        p_uid = str(pu["unit_id"]).strip()
        if p_uid in pool_by_uid:
            raise ConceptGraphError(f"schema sync: duplicate unit_id {p_uid!r} in candidate_pool_manifest")
        pool_by_uid[p_uid] = pu

    s1_items = s2.get("stage1_calibration_items")
    s2_items = s2.get("stage2_items")
    recal_items = s2.get("recalibration_slice_items") or []
    if not isinstance(s1_items, list) or len(s1_items) != 30 or not isinstance(s2_items, list) or len(s2_items) != 240 or not isinstance(recal_items, list):
        raise ConceptGraphError("schema sync: stage2_study requires 30 stage1_calibration_items and 240 stage2_items")
    for grp_lbl, grp_list in (("stage1", s1_items), ("stage2", s2_items), ("recalibration", recal_items)):
        for idx_it, it_u in enumerate(grp_list):
            if not isinstance(it_u, dict):
                raise ConceptGraphError(f"schema sync: malformed {grp_lbl}[{idx_it}]")
            it_uid = str(it_u.get("unit_id") or "").strip()
            if it_uid not in pool_by_uid:
                raise ConceptGraphError(f"schema sync: {grp_lbl}[{idx_it}] ({it_uid!r}) missing from candidate_pool_manifest")
            pool_u_ref = pool_by_uid[it_uid]
            eff_it_u = {**it_u}
            if grp_lbl != "stage2" and "area_code" not in eff_it_u and "area_code" in pool_u_ref:
                eff_it_u["area_code"] = pool_u_ref["area_code"]
            if _canonical_unit_provenance_sig(eff_it_u, require_passage=True) != _canonical_unit_provenance_sig(
                {**pool_u_ref, "area_code": eff_it_u.get("area_code", pool_u_ref.get("area_code"))},
                require_passage=True,
            ):
                raise ConceptGraphError(
                    f"schema sync: {grp_lbl}[{idx_it}] ({it_uid!r}) metadata/passage disagrees with candidate_pool_manifest"
                )

    rerun_idx = hs["recalibration_rerun_index"]
    if s2.get("recalibration_rerun_index") != rerun_idx:
        raise ConceptGraphError("schema sync: stage2_study.recalibration_rerun_index disagrees with human_study_stage2")
    cycle_mm = f"{major}.{minor}"

    raw_s1_regs = eff_burned.get("stage1_human_registrations")
    if not isinstance(raw_s1_regs, list) or not raw_s1_regs:
        raise ConceptGraphError(
            "schema sync: burned-clusters.json must contain non-empty stage1_human_registrations (run reconcile.py --register-stage1-human)"
        )
    all_s1_regs: list[dict] = []
    seen_s1_reg_ids: set[str] = set()
    for idx_sr1, sr1 in enumerate(raw_s1_regs):
        if (
            not isinstance(sr1, dict)
            or len(str(sr1.get("registration_id") or "").strip()) < 3
            or len(str(sr1.get("cycle_id") or "").strip()) < 3
            or len(str(sr1.get("target_major_minor") or "").strip()) < 3
            or not isinstance(sr1.get("exposed_cluster_ids"), list)
            or not sr1["exposed_cluster_ids"]
        ):
            raise ConceptGraphError(f"schema sync: malformed burned-clusters stage1_human_registrations[{idx_sr1}]")
        rid_1 = str(sr1["registration_id"]).strip()
        if rid_1 in seen_s1_reg_ids:
            raise ConceptGraphError(f"schema sync: duplicate registration_id {rid_1!r} in stage1_human_registrations")
        seen_s1_reg_ids.add(rid_1)
        all_s1_regs.append(sr1)
    cycle_s1_regs = [
        r for r in all_s1_regs if str(r.get("cycle_id")) == cycle_id or str(r.get("target_major_minor")) == cycle_mm
    ]
    if len(cycle_s1_regs) != 1 or str(cycle_s1_regs[0]["registration_id"]).strip() != s1_reg_id:
        raise ConceptGraphError(
            f"schema sync: stage1_human_registrations for cycle_id={cycle_id!r} / {cycle_mm} must contain exact single registration {s1_reg_id!r}"
        )
    s1_reg_cur = cycle_s1_regs[0]
    exp_s1_digest = compute_stage1_routing_digest(s1_items, pool_by_uid=pool_by_uid)
    if (
        str(s1_reg_cur.get("cycle_id")) != cycle_id
        or str(s1_reg_cur.get("target_major_minor")) != cycle_mm
        or _parse_iso_utc(s1_reg_cur.get("registered_at"), "stage1_human_registrations.registered_at") != t_s1_reg
        or _parse_iso_utc(s1_reg_cur.get("protocol_approved_at"), "stage1_human_registrations.protocol_approved_at") != t_proto
        or sorted(str(x) for x in (s1_reg_cur.get("stage1_unit_ids") or [])) != sorted(str(u["unit_id"]).strip() for u in s1_items)
        or str(s1_reg_cur.get("stage1_routing_digest_sha256") or "") != exp_s1_digest
    ):
        raise ConceptGraphError(
            "schema sync: stage1_calibration_items/timestamps/routing digest disagree with pre-calibration stage1_human_registrations record"
        )

    raw_all_attempts = eff_burned.get("stage2_human_attempts")
    if not isinstance(raw_all_attempts, list) or not raw_all_attempts:
        raise ConceptGraphError(
            "schema sync: burned-clusters.json must contain non-empty pre-unseal stage2_human_attempts (run reconcile.py --unseal-stage2-human)"
        )
    all_s2_attempts: list[dict] = []
    seen_attempt_ids: set[str] = set()
    for idx_la, la in enumerate(raw_all_attempts):
        if (
            not isinstance(la, dict)
            or len(str(la.get("attempt_id") or "").strip()) < 3
            or len(str(la.get("cycle_id") or "").strip()) < 3
            or len(str(la.get("target_major_minor") or "").strip()) < 3
            or la.get("recalibration_rerun_index") not in (0, 1)
            or not isinstance(la.get("exposed_cluster_ids"), list)
            or not la["exposed_cluster_ids"]
        ):
            raise ConceptGraphError(f"schema sync: malformed burned-clusters stage2_human_attempts[{idx_la}]")
        aid = str(la["attempt_id"]).strip()
        if aid in seen_attempt_ids:
            raise ConceptGraphError(f"schema sync: duplicate attempt_id {aid!r} in stage2_human_attempts")
        seen_attempt_ids.add(aid)
        all_s2_attempts.append(la)

    ledger_s2_attempts = [
        a for a in all_s2_attempts
        if str(a.get("cycle_id")) == cycle_id or str(a.get("target_major_minor")) == cycle_mm
    ]
    if len(ledger_s2_attempts) != rerun_idx + 1:
        raise ConceptGraphError(
            f"schema sync: stage2_human_attempts for cycle_id={cycle_id!r} / {cycle_mm} has {len(ledger_s2_attempts)} entries "
            f"(expected {rerun_idx + 1} matching recalibration_rerun_index={rerun_idx})"
        )
    for exp_r_idx, att_e in enumerate(ledger_s2_attempts):
        if (
            str(att_e.get("cycle_id")) != cycle_id
            or str(att_e.get("target_major_minor")) != cycle_mm
            or att_e.get("recalibration_rerun_index") != exp_r_idx
            or str(att_e.get("stage1_registration_id") or "") != s1_reg_id
        ):
            raise ConceptGraphError(
                "schema sync: stage2_human_attempts violates (cycle_id, target_major_minor, stage1_registration_id) binding or monotonic rerun index [0..rerun_idx]"
            )

    att_cur = ledger_s2_attempts[rerun_idx]
    cur_attempt_id = str(att_cur["attempt_id"]).strip()
    exp_frozen_routing_sha = compute_stage2_frozen_routing_digest(
        stage1_registration_id=s1_reg_id,
        stage1_items=s1_items,
        stage2_items=s2_items,
        recalibration_items=recal_items,
        pool_by_uid=pool_by_uid,
    )
    if (
        str(att_cur.get("candidate_pool_manifest_sha256") or "") != pool_sha
        or str(att_cur.get("frozen_tagging_guide_sha256") or "") != guide_sha
        or str(att_cur.get("candidate_taxonomy_sha256") or "") != expected_tax_sha
        or (expected_non_guides_sha is not None and str(att_cur.get("candidate_non_guides_taxonomy_sha256") or "") != expected_non_guides_sha)
        or str(att_cur.get("frozen_routing_digest_sha256") or "") != exp_frozen_routing_sha
        or _parse_iso_utc(att_cur.get("unsealed_at"), "stage2_human_attempts.unsealed_at") != t_unseal
        or sorted(str(x) for x in (att_cur.get("stage1_unit_ids") or [])) != sorted(str(u["unit_id"]).strip() for u in s1_items)
        or sorted(str(x) for x in (att_cur.get("stage2_unit_ids") or [])) != sorted(str(u["unit_id"]).strip() for u in s2_items)
        or sorted(str(x) for x in (att_cur.get("recalibration_unit_ids") or [])) != sorted(str(u["unit_id"]).strip() for u in recal_items)
    ):
        raise ConceptGraphError(
            "schema sync: current stage2_study items/candidate hashes/routing digest/unsealed_at disagree with pre-unseal stage2_human_attempts record"
        )

    prior_attempt_units: list[dict] = []
    prior_pool_units: list[dict] = []
    if rerun_idx == 0:
        if recal_items:
            raise ConceptGraphError("schema sync: recalibration_rerun_index=0 cannot carry recalibration_slice_items")
    else:
        prior_rel = str(s2.get("prior_failed_bundle_relpath") or "").strip()
        prior_sha = str(s2.get("prior_failed_bundle_sha256") or "").strip()
        if not _HEX64_RE.match(prior_sha) or prior_sha == gov_sha_decl:
            raise ConceptGraphError("schema sync: recalibration_rerun_index=1 requires valid distinct prior_failed_bundle_sha256")
        prior_path = _resolve_confined_method_study_file(eff_root, prior_rel, "prior_failed_bundle_relpath")
        prior_bytes = prior_path.read_bytes()
        if hashlib.sha256(prior_bytes).hexdigest() != prior_sha:
            raise ConceptGraphError("schema sync: prior_failed_bundle_relpath SHA-256 mismatch")
        prior_bundle = json_loads_strict(prior_bytes.decode("utf-8"), context="prior_failed_bundle")
        prior_s2 = (prior_bundle.get("stage2_study") or {}) if isinstance(prior_bundle, dict) else {}
        prior_proto = (prior_s2.get("protocol_approval") or {}) if isinstance(prior_s2, dict) else {}
        if prior_s2.get("recalibration_rerun_index") != 0:
            raise ConceptGraphError("schema sync: prior_failed_bundle must have recalibration_rerun_index=0")
        prior_pool_rel = str(prior_proto.get("candidate_pool_manifest_relpath") or "").strip()
        prior_pool_sha = str(prior_proto.get("candidate_pool_manifest_sha256") or "").strip()
        prior_guide_sha = str(prior_proto.get("frozen_tagging_guide_sha256") or "").strip()
        att0 = ledger_s2_attempts[0]
        p_s1 = prior_s2.get("stage1_calibration_items") or []
        p_s2 = prior_s2.get("stage2_items") or []
        if (
            not _HEX64_RE.match(prior_pool_sha)
            or str(att0.get("candidate_pool_manifest_sha256") or "") != prior_pool_sha
            or str(att0.get("frozen_tagging_guide_sha256") or "") != prior_guide_sha
            or str(prior_s2.get("stage1_registration_id") or "") != s1_reg_id
            or not isinstance(p_s1, list)
            or len(p_s1) != 30
            or not isinstance(p_s2, list)
            or len(p_s2) != 240
            or sorted(str(x) for x in (att0.get("stage1_unit_ids") or [])) != sorted(str(u.get("unit_id") or "").strip() for u in p_s1)
            or sorted(str(x) for x in (att0.get("stage2_unit_ids") or [])) != sorted(str(u.get("unit_id") or "").strip() for u in p_s2)
        ):
            raise ConceptGraphError(
                "schema sync: prior_failed_bundle disagrees with burned-clusters stage2_human_attempts[0]"
            )
        p_pool_bytes = _resolve_confined_method_study_file(eff_root, prior_pool_rel, "prior_failed_bundle.candidate_pool_manifest").read_bytes()
        if hashlib.sha256(p_pool_bytes).hexdigest() != prior_pool_sha:
            raise ConceptGraphError("schema sync: prior_failed_bundle candidate_pool_manifest SHA-256 mismatch")
        p_pool_doc = json_loads_strict(p_pool_bytes.decode("utf-8"), context="prior_candidate_pool_manifest")
        for u_pp in (p_pool_doc.get("units") or []) if isinstance(p_pool_doc, dict) else []:
            if isinstance(u_pp, dict):
                prior_pool_units.append(u_pp)
        for lst_k in ("stage2_items", "recalibration_slice_items"):
            for u_p in (prior_s2.get(lst_k) or []):
                if isinstance(u_p, dict):
                    prior_attempt_units.append(u_p)
        if not prior_attempt_units or len(recal_items) != 12:
            raise ConceptGraphError(
                "schema sync: recalibration_rerun_index=1 requires non-empty prior_failed_bundle items and 12 recalibration_slice_items"
            )

    all_closure_units = [*pool_units, *prior_pool_units, *prior_attempt_units]
    closure = compute_lineage_components(
        all_closure_units,
        eff_burned,
        include_all_exposed_meta=True,
        strict_prospective_work_bijection=True,
    )
    merges: dict[str, str] = closure["merges"]  # type: ignore[assignment]
    pf_to_comp: dict[str, str] = closure["pf_to_component"]  # type: ignore[assignment]
    comp_ed: set[str] = closure["component_editor_affiliated"]  # type: ignore[assignment]
    comp_ed_co: set[str] = closure["component_independence_excluded"]  # type: ignore[assignment]
    internal_set: set[str] = closure["internal_clusters"]  # type: ignore[assignment]

    historically_tainted_pfs: set[str] = {
        merges.get(str(hu.get("provenance_family_id") or "").strip(), str(hu.get("provenance_family_id") or "").strip())
        for hu in (eff_burned.get("historical_phase3_units") or {}).values()
        if isinstance(hu, dict) and str(hu.get("provenance_family_id") or "").strip()
    }
    for ev_att in eff_burned.get("evaluation_attempts") or []:
        if isinstance(ev_att, dict):
            for raw_ep in ev_att.get("exposed_cluster_ids") or ev_att.get("unsealed_holdout_cluster_ids") or []:
                s_ep = str(raw_ep or "").strip()
                if s_ep:
                    historically_tainted_pfs.add(merges.get(s_ep, s_ep))
    for bc in eff_burned.get("burned_clusters") or []:
        if not isinstance(bc, dict):
            continue
        raw_bpf = str(bc.get("cluster_id") or "").strip()
        if raw_bpf:
            canon_bpf = merges.get(raw_bpf, raw_bpf)
            if (
                bc.get("burned_by_stage2_attempt_id") != cur_attempt_id
                and bc.get("burned_by_stage1_registration_id") != s1_reg_id
            ) or bc.get("burned_by_campaign_id"):
                historically_tainted_pfs.add(canon_bpf)
    for other_s1 in all_s1_regs:
        if str(other_s1["registration_id"]).strip() != s1_reg_id:
            for raw_op1 in other_s1.get("exposed_cluster_ids") or []:
                s_op1 = str(raw_op1 or "").strip()
                if s_op1:
                    historically_tainted_pfs.add(merges.get(s_op1, s_op1))
    for other_att in all_s2_attempts:
        if str(other_att["attempt_id"]).strip() != cur_attempt_id:
            for raw_op in other_att.get("exposed_cluster_ids") or []:
                s_op = str(raw_op or "").strip()
                if s_op:
                    historically_tainted_pfs.add(merges.get(s_op, s_op))

    pre_burned_comps: set[str] = set()
    exp_meta_map = eff_burned.get("exposed_cluster_metadata") or {}
    burned_or_exposed_canon: set[str] = set()
    for raw_b_str in eff_burned.get("burned_or_exposed_cluster_ids") or []:
        s_bpf = str(raw_b_str or "").strip()
        if not s_bpf:
            continue
        canon_bpf = merges.get(s_bpf, s_bpf)
        burned_or_exposed_canon.add(canon_bpf)
        meta_bpf = exp_meta_map.get(canon_bpf) if isinstance(exp_meta_map, dict) else None
        is_cur_study_fresh = (
            canon_bpf not in historically_tainted_pfs
            and isinstance(meta_bpf, dict)
            and not bool(meta_bpf.get("historical_multi_work_cluster", False))
            and (
                (
                    str(meta_bpf.get("first_burned_by_stage1_registration_id") or "") == s1_reg_id
                    and meta_bpf.get("first_burned_by_stage2_attempt_id") is None
                )
                or (
                    str(meta_bpf.get("first_burned_by_stage2_attempt_id") or "") == cur_attempt_id
                    and meta_bpf.get("first_burned_by_stage1_registration_id") is None
                )
            )
        )
        if not is_cur_study_fresh and canon_bpf in pf_to_comp:
            pre_burned_comps.add(pf_to_comp[canon_bpf])
    for t_pf in historically_tainted_pfs:
        if t_pf in pf_to_comp:
            pre_burned_comps.add(pf_to_comp[t_pf])

    s1_reg_exposed_canon = {
        merges.get(str(x).strip(), str(x).strip()) for x in s1_reg_cur["exposed_cluster_ids"] if str(x).strip()
    }
    att_cur_exposed_canon = {
        merges.get(str(x).strip(), str(x).strip()) for x in att_cur["exposed_cluster_ids"] if str(x).strip()
    }

    def _unit_ext_comp(u_obj: dict, label_ctx: str, *, is_stage1: bool = False) -> str:
        raw_pf = str(u_obj.get("provenance_family_id") or "").strip()
        if (
            not raw_pf
            or u_obj.get("origin") != "external"
            or bool(u_obj.get("editor_affiliated", False))
            or bool(u_obj.get("editor_coauthor_5yr", False))
        ):
            raise ConceptGraphError(f"schema sync: {label_ctx} must be non-editor-affiliated, non-5yr-coauthor origin='external'")
        if not (u_obj.get("authors") or u_obj.get("institutional_organizations") or u_obj.get("institutional_organization")):
            raise ConceptGraphError(f"schema sync: {label_ctx} missing authors/institutional_organizations metadata")
        canon_pf = merges.get(raw_pf, raw_pf)
        if canon_pf in internal_set:
            raise ConceptGraphError(f"schema sync: {label_ctx} uses internal cluster {canon_pf!r}")
        meta_here = exp_meta_map.get(canon_pf) if isinstance(exp_meta_map, dict) else None
        if not isinstance(meta_here, dict) or canon_pf not in burned_or_exposed_canon:
            raise ConceptGraphError(f"schema sync: {label_ctx} cluster {canon_pf!r} missing from burned-clusters.json")
        if is_stage1:
            if (
                canon_pf not in s1_reg_exposed_canon
                or str(meta_here.get("first_burned_by_stage1_registration_id") or "") != s1_reg_id
                or meta_here.get("first_burned_by_stage2_attempt_id") is not None
            ):
                raise ConceptGraphError(
                    f"schema sync: {label_ctx} cluster {canon_pf!r} was not pre-registered by Stage 1 registration {s1_reg_id!r}"
                )
        else:
            if (
                canon_pf not in att_cur_exposed_canon
                or str(meta_here.get("first_burned_by_stage2_attempt_id") or "") != cur_attempt_id
                or meta_here.get("first_burned_by_stage1_registration_id") is not None
            ):
                raise ConceptGraphError(
                    f"schema sync: {label_ctx} cluster {canon_pf!r} was not pre-burned by Stage 2 attempt {cur_attempt_id!r}"
                )
        comp_id = pf_to_comp[canon_pf]
        if comp_id == "eff_lineage:internal" or comp_id in comp_ed or comp_id in comp_ed_co:
            raise ConceptGraphError(
                f"schema sync: {label_ctx} resolves to internal, editor-affiliated, or 5-year Editor co-author lineage {comp_id!r}"
            )
        if comp_id in pre_burned_comps:
            raise ConceptGraphError(f"schema sync: {label_ctx} uses burned lineage component {comp_id!r}")
        return comp_id

    s1_comps: set[str] = set()
    s1_pair_counts: collections.Counter[str] = collections.Counter()
    seen_unit_ids: set[str] = set()
    for idx_u, u1 in enumerate(s1_items):
        uid1 = str(u1.get("unit_id") or "").strip()
        p1 = str(u1.get("pair_id") or "").strip()
        if not uid1 or uid1 in seen_unit_ids or p1 not in _STAGE2_PAIR_ORDER:
            raise ConceptGraphError(f"schema sync: invalid or duplicate unit_id/pair_id in stage1_calibration_items[{idx_u}]")
        seen_unit_ids.add(uid1)
        s1_pair_counts[p1] += 1
        s1_comps.add(_unit_ext_comp(u1, f"stage1_calibration_items[{idx_u}]", is_stage1=True))
    if any(s1_pair_counts[p] != 5 for p in _STAGE2_PAIR_ORDER):
        raise ConceptGraphError("schema sync: stage1_calibration_items must allocate 5 items to each of the 6 pairs")

    recal_comps: set[str] = set()
    for i_rc, ur in enumerate(recal_items):
        rc_uid = str(ur.get("unit_id") or "").strip()
        if not rc_uid or rc_uid in seen_unit_ids:
            raise ConceptGraphError(f"schema sync: invalid or duplicate unit_id in recalibration_slice_items[{i_rc}]")
        seen_unit_ids.add(rc_uid)
        recal_comps.add(_unit_ext_comp(ur, f"recalibration_slice_items[{i_rc}]", is_stage1=False))

    prior_comps: set[str] = set()
    for up in prior_attempt_units:
        raw_ppf = str(up.get("provenance_family_id") or "").strip()
        if raw_ppf:
            cppf = merges.get(raw_ppf, raw_ppf)
            if cppf in pf_to_comp:
                prior_comps.add(pf_to_comp[cppf])
    if not recal_comps.isdisjoint(s1_comps) or not recal_comps.isdisjoint(prior_comps):
        raise ConceptGraphError("schema sync: recalibration_slice_items share lineage components with Stage 1 or prior attempt")

    pair_rater_sets = {p: {m["_pid"] for m in indexers_by_pair[p]} for p in _STAGE2_PAIR_ORDER}
    canon_pair_raters: dict[str, tuple[str, str]] = {
        p: (sorted(pair_rater_sets[p])[0], sorted(pair_rater_sets[p])[1]) for p in _STAGE2_PAIR_ORDER
    }

    items_by_area: dict[str, list[dict]] = collections.defaultdict(list)
    for idx_2, u2 in enumerate(s2_items):
        uid2 = str(u2.get("unit_id") or "").strip()
        acode = str(u2.get("area_code") or "").strip()
        if not uid2 or uid2 in seen_unit_ids or acode not in expected_20_areas:
            raise ConceptGraphError(f"schema sync: invalid or duplicate unit_id/area_code in stage2_items[{idx_2}]")
        seen_unit_ids.add(uid2)
        comp2 = _unit_ext_comp(u2, f"stage2_items[{idx_2}]", is_stage1=False)
        if comp2 in s1_comps or comp2 in recal_comps or comp2 in prior_comps:
            raise ConceptGraphError(
                f"schema sync: stage2_items[{idx_2}] ({uid2}) shares effective_lineage_component_id {comp2} with Stage 1, recalibration, or prior attempt"
            )
        items_by_area[acode].append({**u2, "_comp": comp2})

    if set(items_by_area.keys()) != expected_20_areas:
        raise ConceptGraphError("schema sync: stage2_items must cover all 20 taxonomy areas")

    ordered_s2: list[dict] = []
    area_item_indices: list[list[int]] = []
    anchors_by_group: dict[str, list[dict]] = collections.defaultdict(list)
    min_area_both_non_empty = 12
    comp_areas_map: dict[str, set[str]] = collections.defaultdict(set)

    def _check_pair_eval(pe_obj: object, expected_pair: str, ctx_lbl: str) -> tuple[str, str, frozenset[str], frozenset[str]]:
        if not isinstance(pe_obj, dict) or str(pe_obj.get("pair_id") or "").strip() != expected_pair:
            raise ConceptGraphError(f"schema sync: {ctx_lbl} pair_id must equal {expected_pair!r}")
        r1 = str(pe_obj.get("rater_1") or "").strip()
        r2 = str(pe_obj.get("rater_2") or "").strip()
        if {r1, r2} != pair_rater_sets[expected_pair]:
            raise ConceptGraphError(f"schema sync: {ctx_lbl} raters {(r1, r2)} disagree with human_indexers[{expected_pair}]")
        sb1 = str(pe_obj.get("subject_1") or "").strip()
        sb2 = str(pe_obj.get("subject_2") or "").strip()
        if sb1 not in valid_subjects or sb2 not in valid_subjects:
            raise ConceptGraphError(f"schema sync: {ctx_lbl} invalid subject ID")
        raw_c1 = pe_obj.get("concepts_1")
        raw_c2 = pe_obj.get("concepts_2")
        if (
            not isinstance(raw_c1, list)
            or not isinstance(raw_c2, list)
            or len(raw_c1) > 3
            or len(raw_c2) > 3
            or len(set(raw_c1)) != len(raw_c1)
            or len(set(raw_c2)) != len(raw_c2)
            or not set(raw_c1).issubset(valid_concepts)
            or not set(raw_c2).issubset(valid_concepts)
        ):
            raise ConceptGraphError(f"schema sync: {ctx_lbl} concepts_1/concepts_2 must be 0..3 distinct active concept IDs")
        return _canonicalize_pair_eval_columns(
            canon_pair_raters[expected_pair],
            r1,
            r2,
            sb1,
            sb2,
            frozenset(str(x) for x in raw_c1),
            frozenset(str(x) for x in raw_c2),
        )

    for acode in sorted(expected_20_areas):
        alist = sorted(items_by_area[acode], key=lambda x: (str(x["area_code"]), str(x["unit_id"])))
        if len(alist) != 12:
            raise ConceptGraphError(f"schema sync: area {acode} has {len(alist)} Stage 2 items (expected 12)")
        if len({x["_comp"] for x in alist}) != 12:
            raise ConceptGraphError(f"schema sync: area {acode} fails <=1 item per effective_lineage_component_id cap")
        exp_pair = _area_to_stage2_pair(acode)
        idxs_here: list[int] = []
        area_non_empty = 0
        for pos_in_area, item in enumerate(alist):
            comp_areas_map[str(item["_comp"])].add(acode)
            exp_anchor = pos_in_area < 2
            if bool(item.get("is_anchor", False)) != exp_anchor:
                raise ConceptGraphError(
                    f"schema sync: item {item['unit_id']} in {acode} is_anchor={item.get('is_anchor')} disagrees with deterministic first-2-per-area rule"
                )
            sb1, sb2, fs1, fs2 = _check_pair_eval(item.get("primary_evaluation"), exp_pair, f"stage2[{item['unit_id']}].primary")
            if fs1 and fs2:
                area_non_empty += 1
            rec_clean = {
                **item,
                "_group": exp_pair,
                "_sb1": sb1,
                "_sb2": sb2,
                "_fs1": fs1,
                "_fs2": fs2,
            }
            if exp_anchor:
                anchors_by_group[exp_pair].append(rec_clean)
            else:
                if item.get("secondary_evaluations"):
                    raise ConceptGraphError(f"schema sync: non-anchor item {item['unit_id']} cannot carry secondary_evaluations")
            idxs_here.append(len(ordered_s2))
            ordered_s2.append(rec_clean)
        area_item_indices.append(idxs_here)
        if area_non_empty < min_area_both_non_empty:
            min_area_both_non_empty = area_non_empty

    exp_sched = expected_stage2_anchor_schedule()
    ordered_anchors: list[dict] = []
    anchor_8subjs: list[list[str]] = []
    anchor_8sets: list[list[frozenset[str]]] = []
    for g_id in _STAGE2_PAIR_ORDER:
        g_ancs = sorted(anchors_by_group[g_id], key=lambda x: (str(x["area_code"]), str(x["unit_id"])))
        if len(g_ancs) != len(exp_sched[g_id]):
            raise ConceptGraphError(f"schema sync: group {g_id} anchor count mismatch")
        for k_pos, anc_item in enumerate(g_ancs):
            sec_list = anc_item.get("secondary_evaluations")
            exp_receivers = exp_sched[g_id][k_pos]
            if not isinstance(sec_list, list) or len(sec_list) != 3:
                raise ConceptGraphError(f"schema sync: anchor {anc_item['unit_id']} requires 3 secondary_evaluations")
            sec_by_p = {str(se.get("pair_id") or "").strip(): se for se in sec_list if isinstance(se, dict)}
            if tuple(sorted(sec_by_p.keys(), key=lambda p: _STAGE2_PAIR_ORDER.index(p) if p in _STAGE2_PAIR_ORDER else 99)) != exp_receivers:
                raise ConceptGraphError(
                    f"schema sync: anchor {anc_item['unit_id']} secondary receiver pairs {list(sec_by_p.keys())} disagree with schedule {list(exp_receivers)}"
                )
            subjs_8 = [anc_item["_sb1"], anc_item["_sb2"]]
            sets_8 = [anc_item["_fs1"], anc_item["_fs2"]]
            for rec_p in exp_receivers:
                s_b1, s_b2, f_s1, f_s2 = _check_pair_eval(sec_by_p[rec_p], rec_p, f"anchor[{anc_item['unit_id']}].sec[{rec_p}]")
                subjs_8.extend([s_b1, s_b2])
                sets_8.extend([f_s1, f_s2])
            ordered_anchors.append(anc_item)
            anchor_8subjs.append(subjs_8)
            anchor_8sets.append(sets_8)

    # 5. Recompute primary 240-item point estimates, 40-anchor 8-rater point estimates, and 10,000-resample 95% CIs.
    w_240 = [1.0] * 240
    groups_240 = [x["_group"] for x in ordered_s2]
    s1_240 = [x["_sb1"] for x in ordered_s2]
    s2_240 = [x["_sb2"] for x in ordered_s2]
    fs1_240 = [x["_fs1"] for x in ordered_s2]
    fs2_240 = [x["_fs2"] for x in ordered_s2]
    both_non_empty_mask = [bool(a and b) for a, b in zip(fs1_240, fs2_240)]

    p_nom_240 = _precompute_nominal_alpha([[a, b] for a, b in zip(s1_240, s2_240)])
    p_kap_240 = _precompute_pair_stratified_kappa(groups_240, s1_240, s2_240)
    p_hr_240 = _precompute_hooper_rolling(fs1_240, fs2_240)
    p_masi_240 = _precompute_masi_alpha([[a, b] for a, b in zip(fs1_240, fs2_240)])

    pt_alpha_subj, zv_alpha_subj = _fast_nominal_alpha_from_precomputed(*p_nom_240, w_240)
    pt_kappa_bar, _any_zv_kap, zv_kap_strata = _fast_pair_stratified_kappa_from_precomputed(p_kap_240, w_240)
    pt_hr = _fast_hooper_rolling_from_precomputed(p_hr_240, w_240)
    pt_alpha_masi, zv_alpha_masi = _fast_masi_alpha_from_precomputed(*p_masi_240, w_240)

    expected_zv_strata = list(zv_kap_strata)
    if zv_alpha_subj:
        expected_zv_strata.append("alpha_subject")
    if zv_alpha_masi:
        expected_zv_strata.append("alpha_masi")

    tag_counts: collections.Counter[str] = collections.Counter()
    total_tags = 0
    leaf_tags = 0
    for fs_pair in (fs1_240, fs2_240):
        for fs in fs_pair:
            for cid_t in fs:
                tag_counts[cid_t] += 1
                total_tags += 1
                if cid_t not in non_leaf_concepts:
                    leaf_tags += 1
    if total_tags == 0:
        raise ConceptGraphError("schema sync: Stage 2 total concept tag assignments cannot be 0")
    pt_p_max = max(tag_counts.values()) / total_tags
    pt_leaf_spec = leaf_tags / total_tags

    w_40 = [1.0] * 40
    p_nom_40 = _precompute_nominal_alpha(anchor_8subjs)
    p_masi_40 = _precompute_masi_alpha(anchor_8sets)
    p_fl_40 = _precompute_fleiss_kappa_8rater(anchor_8subjs)
    pt_anc_alpha_subj, zv_anc_subj = _fast_nominal_alpha_from_precomputed(*p_nom_40, w_40)
    pt_anc_alpha_masi, zv_anc_masi = _fast_masi_alpha_from_precomputed(*p_masi_40, w_40)
    pt_anc_fleiss, zv_anc_fl = _fast_fleiss_kappa_8rater_from_precomputed(*p_fl_40, w_40)
    expected_anc_zv_flags = {
        "alpha_subject_anchor": zv_anc_subj,
        "alpha_masi_anchor": zv_anc_masi,
        "kappa_fleiss_anchor": zv_anc_fl,
    }

    m_canon_240 = len({merges.get(str(x["provenance_family_id"]), str(x["provenance_family_id"])) for x in ordered_s2})
    comp_list_240 = sorted(comp_areas_map.keys())
    m_eff = len(comp_list_240)
    max_areas_comp = max(len(aset) for aset in comp_areas_map.values())
    if (
        m_eff < 120
        or max_areas_comp > 4
        or hs["canonical_provenance_clusters"] != m_canon_240
        or hs["effective_lineage_components"] != m_eff
        or hs["max_areas_per_component"] != max_areas_comp
    ):
        raise ConceptGraphError(
            f"schema sync: Stage 2 lineage independence check failed (M_canon={m_canon_240}, M_eff={m_eff} [floor 120], "
            f"max_areas_per_component={max_areas_comp} [cap 4])"
        )

    comp_to_idx_240 = {c: idx for idx, c in enumerate(comp_list_240)}
    item_comp_idx_240 = [comp_to_idx_240[x["_comp"]] for x in ordered_s2]
    rng_primary = random.Random(20270101)
    degen_primary = 0
    valid_primary = 0
    prim_draws: dict[str, list[float]] = {k: [] for k in _PRIMARY_BOOTSTRAP_CI_KEYS}
    while valid_primary < 10000:
        mults = [0] * m_eff
        for _ in range(m_eff):
            mults[rng_primary.randrange(m_eff)] += 1
        if sum(1 for c in mults if c > 0) < 2:
            degen_primary += 1
            if degen_primary > 100:
                raise ConceptGraphError("schema sync: Stage 2 primary bootstrap exceeded degenerate_resample_count > 100")
            continue
        w_draw = [0.0] * 240
        area_zero = False
        for idxs_a in area_item_indices:
            raw_sum_a = sum(mults[item_comp_idx_240[i]] for i in idxs_a)
            if raw_sum_a == 0:
                area_zero = True
                break
            scale_a = 12.0 / raw_sum_a
            for i in idxs_a:
                m_i = mults[item_comp_idx_240[i]]
                if m_i:
                    w_draw[i] = m_i * scale_a
        if area_zero or sum(w_draw[i] for i in range(240) if both_non_empty_mask[i]) <= 0.0:
            degen_primary += 1
            if degen_primary > 100:
                raise ConceptGraphError("schema sync: Stage 2 primary bootstrap exceeded degenerate_resample_count > 100")
            continue
        bs_asub, _ = _fast_nominal_alpha_from_precomputed(*p_nom_240, w_draw)
        bs_kap, _, _ = _fast_pair_stratified_kappa_from_precomputed(p_kap_240, w_draw)
        bs_hr = _fast_hooper_rolling_from_precomputed(p_hr_240, w_draw)
        bs_masi, _ = _fast_masi_alpha_from_precomputed(*p_masi_240, w_draw)
        draw_vals = {
            "alpha_subject": bs_asub,
            "kappa_bar_subject": bs_kap,
            "cond_hooper": bs_hr["cond_hooper"],
            "all_hooper": bs_hr["all_hooper"],
            "cond_rolling": bs_hr["cond_rolling"],
            "all_rolling": bs_hr["all_rolling"],
            "alpha_masi": bs_masi,
        }
        if any(not math.isfinite(v) or v < -1.0 or v > 1.0 for v in draw_vals.values()):
            raise ConceptGraphError("schema sync: non-finite or out-of-bounds primary bootstrap draw")
        for k_ci, v_ci in draw_vals.items():
            prim_draws[k_ci].append(v_ci)
        valid_primary += 1

    # Separate pooled 40-anchor cluster bootstrap (seed = 20270102).
    anc_comp_list = sorted({x["_comp"] for x in ordered_anchors})
    m_anc = len(anc_comp_list)
    if m_anc < 2:
        raise ConceptGraphError("schema sync: 40 anchors require >=2 distinct effective_lineage_component_ids")
    anc_comp_to_idx = {c: idx for idx, c in enumerate(anc_comp_list)}
    anc_comp_idx = [anc_comp_to_idx[x["_comp"]] for x in ordered_anchors]
    rng_anc = random.Random(20270102)
    degen_anc = 0
    valid_anc = 0
    anc_draws: dict[str, list[float]] = {k: [] for k in _ANCHOR_BOOTSTRAP_CI_KEYS}
    while valid_anc < 10000:
        mults_a = [0] * m_anc
        for _ in range(m_anc):
            mults_a[rng_anc.randrange(m_anc)] += 1
        raw_anc = [mults_a[anc_comp_idx[i]] for i in range(40)]
        raw_anc_sum = sum(raw_anc)
        if sum(1 for c in mults_a if c > 0) < 2 or raw_anc_sum == 0:
            degen_anc += 1
            if degen_anc > 100:
                raise ConceptGraphError("schema sync: Stage 2 anchor bootstrap exceeded anchor_degenerate_resample_count > 100")
            continue
        scale_anc = 40.0 / raw_anc_sum
        w_anc_draw = [r * scale_anc for r in raw_anc]
        bs_anc_sub, _ = _fast_nominal_alpha_from_precomputed(*p_nom_40, w_anc_draw)
        bs_anc_masi, _ = _fast_masi_alpha_from_precomputed(*p_masi_40, w_anc_draw)
        bs_fl, _ = _fast_fleiss_kappa_8rater_from_precomputed(*p_fl_40, w_anc_draw)
        anc_vals = {
            "alpha_subject_anchor": bs_anc_sub,
            "alpha_masi_anchor": bs_anc_masi,
            "kappa_fleiss_anchor": bs_fl,
        }
        if any(not math.isfinite(v) or v < -1.0 or v > 1.0 for v in anc_vals.values()):
            raise ConceptGraphError("schema sync: non-finite or out-of-bounds anchor bootstrap draw")
        for k_ac, v_ac in anc_vals.items():
            anc_draws[k_ac].append(v_ac)
        valid_anc += 1

    expected_metrics = {
        "alpha_subject": pt_alpha_subj,
        "kappa_bar_subject": pt_kappa_bar,
        "cond_hooper": pt_hr["cond_hooper"],
        "all_hooper": pt_hr["all_hooper"],
        "cond_rolling": pt_hr["cond_rolling"],
        "all_rolling": pt_hr["all_rolling"],
        "alpha_masi": pt_alpha_masi,
        "empty_rate": pt_hr["empty_rate"],
        "p_max": pt_p_max,
        "leaf_specificity": pt_leaf_spec,
        "alpha_subject_anchor": pt_anc_alpha_subj,
        "alpha_masi_anchor": pt_anc_alpha_masi,
        "kappa_fleiss_anchor": pt_anc_fleiss,
    }
    for m_key, exp_val in expected_metrics.items():
        if abs(float(hs[m_key]) - exp_val) > 1e-9:
            raise ConceptGraphError(
                f"schema sync: human_study_stage2.{m_key}={hs[m_key]} disagrees with raw recomputation ({exp_val:.12f})"
            )
    if hs["min_area_both_non_empty"] != min_area_both_non_empty:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.min_area_both_non_empty={hs['min_area_both_non_empty']} disagrees with recomputed {min_area_both_non_empty}"
        )
    if hs["zero_variance_strata"] != expected_zv_strata:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.zero_variance_strata={hs['zero_variance_strata']} disagrees with recomputed {expected_zv_strata}"
        )
    if hs["anchor_zero_variance_flags"] != expected_anc_zv_flags:
        raise ConceptGraphError(
            "schema sync: human_study_stage2.anchor_zero_variance_flags disagrees with recomputed flags"
        )
    for k_ci in _PRIMARY_BOOTSTRAP_CI_KEYS:
        exp_lo, exp_hi = _bootstrap_percentile_95_ci(prim_draws[k_ci])
        dec_lo, dec_hi = decl_primary_cis[k_ci]
        if abs(dec_lo - exp_lo) > 1e-9 or abs(dec_hi - exp_hi) > 1e-9:
            raise ConceptGraphError(
                f"schema sync: primary_bootstrap_95_ci[{k_ci!r}]={[dec_lo, dec_hi]} disagrees with recomputed {[exp_lo, exp_hi]}"
            )
    for k_ac in _ANCHOR_BOOTSTRAP_CI_KEYS:
        exp_lo_a, exp_hi_a = _bootstrap_percentile_95_ci(anc_draws[k_ac])
        dec_lo_a, dec_hi_a = decl_anchor_cis[k_ac]
        if abs(dec_lo_a - exp_lo_a) > 1e-9 or abs(dec_hi_a - exp_hi_a) > 1e-9:
            raise ConceptGraphError(
                f"schema sync: anchor_bootstrap_95_ci[{k_ac!r}]={[dec_lo_a, dec_hi_a]} disagrees with recomputed {[exp_lo_a, exp_hi_a]}"
            )
    if hs["degenerate_resample_count"] != degen_primary:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.degenerate_resample_count={hs['degenerate_resample_count']} disagrees with recomputed {degen_primary}"
        )
    if hs["anchor_degenerate_resample_count"] != degen_anc:
        raise ConceptGraphError(
            f"schema sync: human_study_stage2.anchor_degenerate_resample_count={hs['anchor_degenerate_resample_count']} disagrees with recomputed {degen_anc}"
        )
