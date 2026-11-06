"""Historical campaign replay, source ledgers, and genesis reconstruction."""
from __future__ import annotations

import collections
import gzip
import hashlib
import re
import unicodedata
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from ..concept_graph import ConceptGraphError
from .legacy_policy import (
    _INTERNAL_LINEAGE_PREFIXES,
    BOOTSTRAP_GRANDFATHERED_ALIAS_EXCEPTIONS,
    BURNED_CLUSTERS_REL_PATH,
    GENESIS_BURNED_CLUSTER_IDS,
    GENESIS_CANONICAL_JSON_SHA256,
    GENESIS_CLUSTER_MERGES,
    GENESIS_EDITOR_AFFILIATED_CLUSTERS,
    GENESIS_EDITOR_COAUTHOR_5YR_CLUSTERS,
    GENESIS_HISTORICAL_PHASE3_UNITS_SHA256,
    GENESIS_INTERNAL_CLUSTERS,
    GENESIS_SNAPSHOT_GZ_SHA256,
    GENESIS_SNAPSHOT_REL_PATH,
    REPO_ROOT_PATH,
)
from .records import (
    _HEX64_RE,
    _norm_prov_tokens,
    canonical_taxonomy_json,
    expected_status_for_version,
    json_loads_strict,
)


@lru_cache(maxsize=2)
def _cached_genesis_snapshot(gz_path_str: str) -> dict:
    gz_bytes = Path(gz_path_str).read_bytes()
    if hashlib.sha256(gz_bytes).hexdigest() != GENESIS_SNAPSHOT_GZ_SHA256:
        raise ConceptGraphError(
            "schema sync: genesis-0.4-baseline.json.gz SHA-256 does not match pinned 0.4-draft snapshot"
        )
    canon_bytes = gzip.decompress(gz_bytes)
    if hashlib.sha256(canon_bytes).hexdigest() != GENESIS_CANONICAL_JSON_SHA256:
        raise ConceptGraphError(
            "schema sync: decompressed genesis-0.4-baseline.json.gz SHA-256 does not match pinned 0.4-draft JSON"
        )
    loaded = json_loads_strict(canon_bytes.decode("utf-8"), "genesis-0.4-baseline.json.gz")
    if not isinstance(loaded, dict):
        raise ConceptGraphError("schema sync: genesis-0.4-baseline.json.gz must contain a JSON object")
    return loaded


def load_genesis_baseline_taxonomy(root: Path | None = None) -> dict:
    """Load a fresh deepcopy of the pinned 0.4-draft genesis taxonomy for deterministic replay."""
    cand = (root / GENESIS_SNAPSHOT_REL_PATH) if root is not None else None
    path = cand if (cand is not None and cand.exists()) else (REPO_ROOT_PATH / GENESIS_SNAPSHOT_REL_PATH)
    if not path.exists():
        raise ConceptGraphError(f"schema sync: missing genesis snapshot {GENESIS_SNAPSHOT_REL_PATH}")
    return deepcopy(_cached_genesis_snapshot(str(path)))


@lru_cache(maxsize=4)
def _cached_burned_clusters(burned_path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    loaded = json_loads_strict(Path(burned_path_str).read_text(encoding="utf-8"), BURNED_CLUSTERS_REL_PATH)
    if not isinstance(loaded, dict):
        raise ConceptGraphError(f"schema sync: {BURNED_CLUSTERS_REL_PATH} must contain a JSON object")
    validate_burned_clusters_v3_ledger(loaded, burned_path_str, evidence_root=Path(burned_path_str).parents[2])
    return loaded


def _load_burned_clusters(root: Path | None = None) -> dict:
    if root is not None:
        cand = root / BURNED_CLUSTERS_REL_PATH
        if (root / ".git").exists() and not cand.is_file():
            raise ConceptGraphError(f"schema sync: missing burned-clusters ledger {cand}")
        path = cand if cand.is_file() else (REPO_ROOT_PATH / BURNED_CLUSTERS_REL_PATH)
    else:
        path = REPO_ROOT_PATH / BURNED_CLUSTERS_REL_PATH
    if not path.is_file():
        raise ConceptGraphError(f"schema sync: missing burned-clusters ledger {BURNED_CLUSTERS_REL_PATH}")
    bdoc = _cached_burned_clusters(str(path), path.stat().st_mtime_ns)
    validate_burned_clusters_v3_ledger(bdoc, str(path), evidence_root=path.parents[2])
    return bdoc


def resolve_cluster_merges(raw_merges: dict[str, str], src_label: str = "burned-clusters.json") -> dict[str, str]:
    """Resolve multi-hop cluster_merges transitively with cycle detection."""
    clean = {str(k).strip(): str(v).strip() for k, v in (raw_merges or {}).items() if str(k).strip() and str(v).strip()}
    resolved: dict[str, str] = {}
    for k in clean:
        seen = {k}
        cur = clean[k]
        while cur in clean:
            if cur in seen:
                raise ConceptGraphError(f"schema sync: cycle in cluster_merges starting at {k!r} in {src_label}")
            seen.add(cur)
            cur = clean[cur]
        resolved[k] = cur
    return resolved


_LATIN_FOLD_TABLE = str.maketrans({
    "ß": "ss",
    "ø": "o",
    "đ": "d",
    "ł": "l",
})


def norm_person_or_org_token(raw: str, *, is_org: bool = False) -> str:
    """Normalize personal author or institutional organization names identically across all lineage checks."""
    s = (raw or "").lower().translate(_LATIN_FOLD_TABLE).replace("æ", "ae").replace("œ", "oe")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    toks = re.findall(r"[a-z0-9]+", s)
    if not toks:
        return ""
    if is_org:
        return f"inst:{' '.join(toks)}"
    if len(toks) >= 3:
        toks = [toks[0], *(t for t in toks[1:-1] if len(t) > 1), toks[-1]]
    return " ".join(toks)


def _extract_auth_and_org_sets(obj: dict) -> tuple[set[str], set[str]]:
    auths = {norm_person_or_org_token(str(a), is_org=False) for a in (obj.get("authors") or [])}
    orgs: set[str] = set()
    if obj.get("institutional_organization"):
        orgs.add(norm_person_or_org_token(str(obj["institutional_organization"]), is_org=True))
    for org in obj.get("institutional_organizations") or []:
        orgs.add(norm_person_or_org_token(str(org), is_org=True))
    auths.discard("")
    orgs.discard("")
    return auths, orgs


def _norm_work_id_token(raw: object) -> str:
    return " ".join(str(raw or "").strip().lower().split())


def _extract_work_ids(obj: dict) -> set[str]:
    wids: set[str] = set()
    if obj.get("work_id") is not None:
        nw = _norm_work_id_token(obj.get("work_id"))
        if nw:
            wids.add(nw)
    for w in obj.get("work_ids") or []:
        nw = _norm_work_id_token(w)
        if nw:
            wids.add(nw)
    raw_wr = obj.get("work_rosters")
    if isinstance(raw_wr, dict):
        for wk in raw_wr:
            nw = _norm_work_id_token(wk)
            if nw:
                wids.add(nw)
    return wids


LEGACY_LEDGER_SHA256 = "cbc72fa6e769fec7aa8015fd6d154a425674e5124b379d35dd10724f4b48fef8"


LEGACY_EVALUATION_INVENTORY_SHA256 = "57ec682e9b5e75a2bbfa81d6957a84ee08cf417c48cff117f4dd1546d4374403"


LEGACY_LEDGER_REL_PATH = "taxonomy/method-study/legacy-burned-clusters-v3.json.gz"


def _historical_ledger_migration(bdoc: dict, label: str, evidence_root: Path) -> set[str]:
    """Validate the pinned historical migration; unknown COI never implies independence."""
    migration = bdoc.get("historical_migration")
    if not isinstance(migration, dict) or migration.get("kind") != "retrospective-evidence-recovery":
        raise ConceptGraphError(f"schema sync: {label} missing historical migration record")
    if migration.get("prior_burned_clusters_sha256") != LEGACY_LEDGER_SHA256:
        raise ConceptGraphError(f"schema sync: {label} historical migration predecessor mismatch")
    try:
        raw = gzip.decompress((evidence_root / LEGACY_LEDGER_REL_PATH).read_bytes())
    except (OSError, EOFError) as exc:
        raise ConceptGraphError(f"schema sync: {label} historical ledger archive unavailable or invalid") from exc
    if hashlib.sha256(raw).hexdigest() != LEGACY_LEDGER_SHA256:
        raise ConceptGraphError(f"schema sync: {label} historical ledger archive hash mismatch")
    prior = json_loads_strict(raw.decode(), LEGACY_LEDGER_REL_PATH)
    expected_unknown = set(prior["exposed_cluster_metadata"])
    unknown = bdoc.get("historical_unknown_coauthor_clusters")
    if not isinstance(unknown, list) or len(unknown) != len(set(unknown)) or set(unknown) != expected_unknown:
        raise ConceptGraphError(f"schema sync: {label} historical unknown COI inventory changed")
    # Historical records are immutable. New metadata can be appended, but every
    # existing value remains backed by the original archived bytes.
    def preserves(current, original):
        if isinstance(original, dict):
            return isinstance(current, dict) and all(k in current and preserves(current[k], v) for k, v in original.items())
        if isinstance(original, list):
            return isinstance(current, list) and current[:len(original)] == original
        return type(current) is type(original) and current == original
    for key, value in prior.items():
        if key != "schema_version" and not preserves(bdoc.get(key), value):
            raise ConceptGraphError(f"schema sync: {label} changed archived historical field {key}")
    for cluster in expected_unknown:
        metadata = bdoc["exposed_cluster_metadata"][cluster]
        if metadata.get("editor_coauthor_5yr") is not None or any(
            roster.get("editor_coauthor_5yr") is not None
            for roster in metadata["work_rosters"].values()
        ):
            raise ConceptGraphError(f"schema sync: {label} historical unknown COI requires an explicit evidence-backed resolution migration")
    if migration.get("evaluation_inventory_sha256") != LEGACY_EVALUATION_INVENTORY_SHA256:
        raise ConceptGraphError(f"schema sync: {label} pinned evaluation inventory changed")
    recovered = migration.get("evaluation_inventory")
    if not isinstance(recovered, dict) or recovered.get("record_kind") != "retrospective-evaluation-inventory":
        raise ConceptGraphError(f"schema sync: {label} missing retrospective evaluation inventory")
    if any(k.startswith("unsealed_") for k in recovered):
        raise ConceptGraphError(f"schema sync: {label} retrospective inventory cannot assert an unseal event")
    inventory_path = evidence_root / "taxonomy/method-study/evaluation-reconstruction.json"
    try:
        inventory_raw = inventory_path.read_bytes()
    except OSError as exc:
        raise ConceptGraphError(f"schema sync: {label} evaluation inventory unavailable") from exc
    if hashlib.sha256(inventory_raw).hexdigest() != migration.get("evaluation_inventory_sha256"):
        raise ConceptGraphError(f"schema sync: {label} evaluation inventory hash mismatch")
    if json_loads_strict(inventory_raw.decode(), str(inventory_path)) != recovered:
        raise ConceptGraphError(f"schema sync: {label} evaluation inventory differs from recovery artifact")
    for key in ("evaluated_coverage_unit_ids", "evaluated_indexing_unit_ids"):
        ids = recovered.get(key)
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or not set(ids) <= set(prior["historical_phase3_units"]):
            raise ConceptGraphError(f"schema sync: {label} invalid recovered evaluation identities")
    return expected_unknown


def validate_burned_clusters_v3_ledger(
    bdoc: object, label: str = "burned-clusters.json", *, evidence_root: Path | None = None,
) -> dict[str, object]:
    """Validate ledger invariants, including the explicit v4 historical migration.

    The function name is retained for existing callers. V4 preserves the pinned
    historical ledger and distinguishes observed evaluations from unseal events.
    """
    if not isinstance(bdoc, dict) or bdoc.get("schema_version") not in (3, 4):
        raise ConceptGraphError(f"schema sync: {label} must be a schema_version=3 or 4 JSON object")
    migrated = bdoc.get("schema_version") == 4
    unknown_coi = _historical_ledger_migration(
        bdoc, label, evidence_root if evidence_root is not None else REPO_ROOT_PATH,
    ) if migrated else set()
    raw_burned = bdoc.get("burned_or_exposed_cluster_ids")
    raw_merges_obj = bdoc.get("cluster_merges")
    raw_internal = bdoc.get("internal_clusters")
    raw_ed_aff = bdoc.get("editor_affiliated_clusters")
    raw_ed_co = bdoc.get("editor_coauthor_5yr_clusters")
    raw_exposed_meta = bdoc.get("exposed_cluster_metadata")
    raw_hist_units = bdoc.get("historical_phase3_units")
    eval_attempts = bdoc.get("evaluation_attempts")
    s1_regs = bdoc.get("stage1_human_registrations")
    s2_attempts = bdoc.get("stage2_human_attempts")
    cycle_resets = bdoc.get("refresh_cycle_resets")
    active_rc = bdoc.get("active_refresh_cycle")
    # Report every structural omission together so a ledger migration can be
    # planned without inventing defaults or iteratively hiding missing evidence.
    required_shapes = {
        "burned_or_exposed_cluster_ids": isinstance(raw_burned, list) and bool(raw_burned),
        "cluster_merges": isinstance(raw_merges_obj, dict),
        "internal_clusters": isinstance(raw_internal, list) and bool(raw_internal),
        "editor_affiliated_clusters": isinstance(raw_ed_aff, list),
        "editor_coauthor_5yr_clusters": isinstance(raw_ed_co, list),
        "exposed_cluster_metadata": isinstance(raw_exposed_meta, dict) and bool(raw_exposed_meta),
        "historical_phase3_units": isinstance(raw_hist_units, dict) and len(raw_hist_units) == 382,
        "evaluation_attempts": isinstance(eval_attempts, list) and (migrated or bool(eval_attempts)),
        "stage1_human_registrations": isinstance(s1_regs, list),
        "stage2_human_attempts": isinstance(s2_attempts, list),
        "refresh_cycle_resets": isinstance(cycle_resets, list),
    }
    invalid_fields = [name for name, valid in required_shapes.items() if not valid]
    if invalid_fields:
        raise ConceptGraphError(
            f"schema sync: {label} missing or invalid required schema_version={bdoc['schema_version']} fields: "
            + ", ".join(invalid_fields)
            + "; recover or migrate evidence records before validation; do not infer completed evaluations"
        )
    if not migrated:
        gen_ev0 = eval_attempts[0]
        if (
            not isinstance(gen_ev0, dict)
            or gen_ev0.get("cycle_id") != "phase6-2026-11-02"
            or gen_ev0.get("historical_pre_sidecar_unseal") is not True
            or len(gen_ev0.get("unsealed_coverage_unit_ids") or []) != 79
            or len(gen_ev0.get("unsealed_indexing_unit_ids") or []) != 43
        ):
            raise ConceptGraphError(
                f"schema sync: {label} evaluation_attempts[0] must preserve genesis Phase 6 unseal record (cycle_id='phase6-2026-11-02', 79 coverage / 43 indexing units)"
            )
    if active_rc is not None:
        if (
            not isinstance(active_rc, dict)
            or len(str(active_rc.get("cycle_id") or "").strip()) < 8
            or not _HEX64_RE.match(str(active_rc.get("initial_baseline_non_guides_sha256") or ""))
            or len(str(active_rc.get("opened_at") or "").strip()) < 10
        ):
            raise ConceptGraphError(f"schema sync: {label} active_refresh_cycle must be null or a valid cycle record")
    for idx_rs, r_rec in enumerate(cycle_resets):
        if (
            not isinstance(r_rec, dict)
            or len(str(r_rec.get("closed_cycle_id") or "").strip()) < 8
            or r_rec.get("resolution") not in ("completed_structural_apply", "governed_reset")
            or len(str(r_rec.get("rationale") or "").strip()) < 15
            or len(str(r_rec.get("closed_at") or "").strip()) < 10
        ):
            raise ConceptGraphError(f"schema sync: {label} malformed refresh_cycle_resets[{idx_rs}]")

    hist_sha = hashlib.sha256(canonical_taxonomy_json(raw_hist_units).encode("utf-8")).hexdigest()
    if (
        hist_sha != GENESIS_HISTORICAL_PHASE3_UNITS_SHA256
        or bdoc.get("historical_phase3_units_sha256") != GENESIS_HISTORICAL_PHASE3_UNITS_SHA256
    ):
        raise ConceptGraphError(f"schema sync: {label} historical_phase3_units_sha256 mismatch")
    init_counts: collections.Counter[str] = collections.Counter()
    for uid_k, umeta in raw_hist_units.items():
        if not isinstance(umeta, dict):
            raise ConceptGraphError(f"schema sync: {label} malformed historical_phase3_units[{uid_k!r}]")
        isp = umeta.get("initial_phase3_split")
        csp = umeta.get("current_exposure_split")
        if isp not in ("dev", "held_out_test", "reserve") or csp != "dev":
            raise ConceptGraphError(
                f"schema sync: {label} historical_phase3_units[{uid_k!r}] must carry valid initial_phase3_split and current_exposure_split='dev'"
            )
        init_counts[str(isp)] += 1
    if init_counts["dev"] != 200 or init_counts["held_out_test"] != 126 or init_counts["reserve"] != 56:
        raise ConceptGraphError(
            f"schema sync: {label} historical_phase3_units initial_phase3_split counts {dict(init_counts)} disagree with pre-registered 200/126/56"
        )

    burned = {str(x).strip() for x in raw_burned if isinstance(x, str) and x.strip()}
    missing_genesis = sorted(GENESIS_BURNED_CLUSTER_IDS - burned)
    if missing_genesis:
        raise ConceptGraphError(f"schema sync: {label} dropped pinned genesis burned cluster IDs: {missing_genesis[:5]}")

    merges = resolve_cluster_merges(raw_merges_obj, label)
    for gk, gv in GENESIS_CLUSTER_MERGES.items():
        if merges.get(gk) != gv:
            raise ConceptGraphError(f"schema sync: {label} altered or dropped pinned genesis cluster_merge {gk!r} -> {gv!r}")

    internal_set = {str(x).strip() for x in raw_internal if isinstance(x, str) and x.strip()}
    internal_set |= {merges.get(x, x) for x in internal_set}
    missing_int = sorted(GENESIS_INTERNAL_CLUSTERS - internal_set)
    if missing_int:
        raise ConceptGraphError(f"schema sync: {label} dropped pinned genesis internal_clusters: {missing_int[:5]}")

    ed_aff = {str(x).strip() for x in raw_ed_aff if isinstance(x, str) and x.strip()}
    ed_aff |= {merges.get(x, x) for x in ed_aff}
    missing_ed = sorted(GENESIS_EDITOR_AFFILIATED_CLUSTERS - ed_aff)
    if missing_ed:
        raise ConceptGraphError(f"schema sync: {label} dropped pinned genesis editor_affiliated_clusters: {missing_ed[:5]}")

    ed_co = {str(x).strip() for x in raw_ed_co if isinstance(x, str) and x.strip()}
    ed_co |= {merges.get(x, x) for x in ed_co}
    missing_ed_co = sorted(GENESIS_EDITOR_COAUTHOR_5YR_CLUSTERS - ed_co - unknown_coi)
    if missing_ed_co:
        raise ConceptGraphError(f"schema sync: {label} dropped pinned genesis editor_coauthor_5yr_clusters: {missing_ed_co[:5]}")

    burned_canonical = {merges.get(x, x) for x in burned} | burned
    canon_burned_ext = {
        merges.get(x, x)
        for x in burned
        if merges.get(x, x) not in internal_set and not merges.get(x, x).lower().startswith(_INTERNAL_LINEAGE_PREFIXES)
    }

    ledger_work_to_pf: dict[str, str] = {}
    for ext_pf in sorted(canon_burned_ext):
        mrec = raw_exposed_meta.get(ext_pf)
        if not isinstance(mrec, dict):
            raise ConceptGraphError(f"schema sync: {label} missing exposed_cluster_metadata entry for burned external cluster {ext_pf!r}")
        m_auth = mrec.get("authors")
        m_inst = mrec.get("institutional_author")
        m_orgs = mrec.get("institutional_organizations")
        m_ed = mrec.get("editor_affiliated")
        m_ed_co = mrec.get("editor_coauthor_5yr")
        m_wids = mrec.get("work_ids")
        m_hist_multi = mrec.get("historical_multi_work_cluster")
        m_wr = mrec.get("work_rosters")
        if (
            not isinstance(m_auth, list)
            or any(not isinstance(a, str) or not a.strip() for a in m_auth)
            or not isinstance(m_inst, bool)
            or not isinstance(m_orgs, list)
            or any(not isinstance(o, str) or len(o.strip()) < 3 for o in m_orgs)
            or not isinstance(m_ed, bool)
            or not (isinstance(m_ed_co, bool) or (ext_pf in unknown_coi and m_ed_co is None))
            or not isinstance(m_wids, list)
            or not m_wids
            or any(not isinstance(w, str) or len(w.strip()) < 3 for w in m_wids)
            or not isinstance(m_hist_multi, bool)
            or not isinstance(m_wr, dict)
            or not m_wr
        ):
            raise ConceptGraphError(
                f"schema sync: {label} exposed_cluster_metadata[{ext_pf!r}] missing complete authors, institutional_organizations, "
                "work_ids, work_rosters, or boolean COI/multi-work flags"
            )
        if m_inst and not m_orgs:
            raise ConceptGraphError(
                f"schema sync: {label} institutional exposed_cluster_metadata[{ext_pf!r}] requires non-empty institutional_organizations"
            )
        if not m_inst and not m_auth:
            raise ConceptGraphError(
                f"schema sync: {label} non-institutional exposed_cluster_metadata[{ext_pf!r}] requires non-empty authors"
            )
        if ext_pf in ed_aff and not m_ed:
            raise ConceptGraphError(
                f"schema sync: {label} editor-affiliated cluster {ext_pf!r} must set editor_affiliated=true in exposed_cluster_metadata"
            )
        if (ext_pf in ed_co or m_ed) and not m_ed_co and ext_pf not in unknown_coi:
            raise ConceptGraphError(
                f"schema sync: {label} editor-coauthor/affiliated cluster {ext_pf!r} must set editor_coauthor_5yr=true in exposed_cluster_metadata"
            )
        if m_ed_co and ext_pf not in ed_co:
            raise ConceptGraphError(
                f"schema sync: {label} cluster {ext_pf!r} has editor_coauthor_5yr=true but is omitted from editor_coauthor_5yr_clusters"
            )
        if len(m_wids) > 1 and not (m_hist_multi and ext_pf in GENESIS_BURNED_CLUSTER_IDS):
            raise ConceptGraphError(
                f"schema sync: {label} prospective external cluster {ext_pf!r} cannot list multiple work_ids {m_wids}"
            )
        norm_wids_set = {_norm_work_id_token(w) for w in m_wids}
        norm_wr_keys = {_norm_work_id_token(wk) for wk in m_wr}
        if len(norm_wids_set) != len(m_wids) or len(norm_wr_keys) != len(m_wr) or norm_wr_keys != norm_wids_set:
            raise ConceptGraphError(
                f"schema sync: {label} exposed_cluster_metadata[{ext_pf!r}] must declare complete per-work work_rosters matching work_ids"
            )
        union_wr_auth: set[str] = set()
        union_wr_orgs: set[str] = set()
        any_wr_ed_co = False
        for wr_k, wr_v in m_wr.items():
            if (
                not isinstance(wr_k, str)
                or not wr_k.strip()
                or not isinstance(wr_v, dict)
                or not isinstance(wr_v.get("authors"), list)
                or any(not isinstance(wa_s, str) or not wa_s.strip() for wa_s in wr_v["authors"])
                or not isinstance(wr_v.get("institutional_organizations"), list)
                or any(not isinstance(wo_s, str) or len(wo_s.strip()) < 3 for wo_s in wr_v["institutional_organizations"])
                or not (isinstance(wr_v.get("editor_coauthor_5yr"), bool) or (ext_pf in unknown_coi and wr_v.get("editor_coauthor_5yr") is None))
            ):
                raise ConceptGraphError(f"schema sync: {label} malformed work_rosters[{wr_k!r}] on {ext_pf!r}")
            if not m_inst and not wr_v["authors"]:
                raise ConceptGraphError(f"schema sync: {label} non-institutional work_rosters[{wr_k!r}] on {ext_pf!r} requires non-empty authors")
            if m_inst and not wr_v["institutional_organizations"]:
                raise ConceptGraphError(
                    f"schema sync: {label} institutional work_rosters[{wr_k!r}] on {ext_pf!r} requires non-empty institutional_organizations"
                )
            for wa_s in wr_v["authors"]:
                nt_a = norm_person_or_org_token(wa_s, is_org=False)
                if nt_a:
                    union_wr_auth.add(nt_a)
            for wo_s in wr_v["institutional_organizations"]:
                nt_o = norm_person_or_org_token(wo_s, is_org=True)
                if nt_o:
                    union_wr_orgs.add(nt_o)
            if wr_v["editor_coauthor_5yr"]:
                any_wr_ed_co = True
        top_auth_norm = {norm_person_or_org_token(a_s, is_org=False) for a_s in m_auth if norm_person_or_org_token(a_s, is_org=False)}
        top_org_norm = {norm_person_or_org_token(o_s, is_org=True) for o_s in m_orgs if norm_person_or_org_token(o_s, is_org=True)}
        if union_wr_auth != top_auth_norm or union_wr_orgs != top_org_norm or (any_wr_ed_co != m_ed_co and not (ext_pf in unknown_coi and m_ed_co is None and not any_wr_ed_co)):
            raise ConceptGraphError(
                f"schema sync: {label} exposed_cluster_metadata[{ext_pf!r}] top-level authors/orgs/editor_coauthor_5yr must equal the union of work_rosters"
            )
        for nw in sorted(norm_wids_set):
            prev_owner = ledger_work_to_pf.get(nw)
            if prev_owner is not None and prev_owner != ext_pf:
                raise ConceptGraphError(
                    f"schema sync: {label} assigns work_id {nw!r} to multiple canonical clusters ({prev_owner!r} vs {ext_pf!r})"
                )
            ledger_work_to_pf[nw] = ext_pf

    return {
        "burned_canonical": burned_canonical,
        "cluster_merges": merges,
        "internal_clusters": internal_set,
        "editor_affiliated_clusters": ed_aff,
        "editor_coauthor_5yr_clusters": ed_co,
        "exposed_cluster_metadata": raw_exposed_meta,
        "historical_phase3_units": raw_hist_units,
    }


def verify_burned_clusters_append_only(
    cur_doc: dict,
    prior_doc: dict,
    *,
    prior_label: str = "prior",
    cur_label: str = "burned-clusters.json",
    evidence_root: Path | None = None,
) -> None:
    """Enforce strict monotonic append-only invariants from `prior_doc` to `cur_doc`."""
    cur_parsed = validate_burned_clusters_v3_ledger(cur_doc, cur_label, evidence_root=evidence_root)
    if not isinstance(prior_doc, dict):
        raise ConceptGraphError(f"schema sync: {prior_label} burned-clusters ledger must be a JSON object")
    merges: dict[str, str] = cur_parsed["cluster_merges"]  # type: ignore[assignment]
    burned_canonical: set[str] = cur_parsed["burned_canonical"]  # type: ignore[assignment]
    internal_set: set[str] = cur_parsed["internal_clusters"]  # type: ignore[assignment]
    ed_aff: set[str] = cur_parsed["editor_affiliated_clusters"]  # type: ignore[assignment]
    ed_co: set[str] = cur_parsed["editor_coauthor_5yr_clusters"]  # type: ignore[assignment]
    raw_exposed_meta: dict[str, dict] = cur_parsed["exposed_cluster_metadata"]  # type: ignore[assignment]
    raw_hist_units: dict[str, dict] = cur_parsed["historical_phase3_units"]  # type: ignore[assignment]

    prior_burned = {merges.get(str(x).strip(), str(x).strip()) for x in (prior_doc.get("burned_or_exposed_cluster_ids") or []) if str(x).strip()}
    if not prior_burned.issubset(burned_canonical):
        raise ConceptGraphError(
            f"schema sync: {cur_label} dropped {prior_label} burned clusters: {sorted(prior_burned - burned_canonical)[:5]}"
        )
    for pk, pv in (prior_doc.get("cluster_merges") or {}).items():
        spk, spv = str(pk).strip(), str(pv).strip()
        if spk and spv and merges.get(spk) != merges.get(spv, spv):
            raise ConceptGraphError(f"schema sync: {cur_label} altered {prior_label} cluster_merge {spk!r} -> {spv!r}")
    prior_int = {merges.get(str(x).strip(), str(x).strip()) for x in (prior_doc.get("internal_clusters") or []) if str(x).strip()}
    if not prior_int.issubset(internal_set):
        raise ConceptGraphError(f"schema sync: {cur_label} dropped {prior_label} internal_clusters: {sorted(prior_int - internal_set)[:5]}")
    prior_ed = {merges.get(str(x).strip(), str(x).strip()) for x in (prior_doc.get("editor_affiliated_clusters") or []) if str(x).strip()}
    if not prior_ed.issubset(ed_aff):
        raise ConceptGraphError(f"schema sync: {cur_label} dropped {prior_label} editor_affiliated_clusters: {sorted(prior_ed - ed_aff)[:5]}")
    prior_ed_co = {
        merges.get(str(x).strip(), str(x).strip()) for x in (prior_doc.get("editor_coauthor_5yr_clusters") or []) if str(x).strip()
    }
    if not prior_ed_co.issubset(ed_co):
        raise ConceptGraphError(
            f"schema sync: {cur_label} dropped {prior_label} editor_coauthor_5yr_clusters: {sorted(prior_ed_co - ed_co)[:5]}"
        )

    prior_hist = prior_doc.get("historical_phase3_units") or {}
    if isinstance(prior_hist, dict):
        for huid, hmeta in prior_hist.items():
            cur_hu = raw_hist_units.get(huid)
            if not isinstance(cur_hu, dict) or not isinstance(hmeta, dict) or any(cur_hu.get(hk) != hv for hk, hv in hmeta.items()):
                raise ConceptGraphError(f"schema sync: {cur_label} altered or dropped historical_phase3_units[{huid!r}] from {prior_label}")

    prior_exp = prior_doc.get("exposed_cluster_metadata") or {}
    if isinstance(prior_exp, dict):
        for epf_raw, emeta in prior_exp.items():
            epf = merges.get(str(epf_raw).strip(), str(epf_raw).strip())
            cur_em = raw_exposed_meta.get(epf) or raw_exposed_meta.get(str(epf_raw).strip())
            if not isinstance(cur_em, dict) or not isinstance(emeta, dict):
                raise ConceptGraphError(f"schema sync: {cur_label} dropped exposed_cluster_metadata[{epf!r}] present in {prior_label}")
            prev_auth = {norm_person_or_org_token(str(a), is_org=False) for a in (emeta.get("authors") or []) if str(a).strip()}
            cur_auth = {norm_person_or_org_token(str(a), is_org=False) for a in (cur_em.get("authors") or []) if str(a).strip()}
            if not prev_auth.issubset(cur_auth):
                raise ConceptGraphError(f"schema sync: {cur_label} dropped authors from exposed_cluster_metadata[{epf!r}] relative to {prior_label}")
            prev_orgs = {norm_person_or_org_token(str(o), is_org=True) for o in (emeta.get("institutional_organizations") or []) if str(o).strip()}
            cur_orgs = {norm_person_or_org_token(str(o), is_org=True) for o in (cur_em.get("institutional_organizations") or []) if str(o).strip()}
            if not prev_orgs.issubset(cur_orgs):
                raise ConceptGraphError(
                    f"schema sync: {cur_label} dropped institutional_organizations from exposed_cluster_metadata[{epf!r}] relative to {prior_label}"
                )
            prev_wids = {_norm_work_id_token(w) for w in (emeta.get("work_ids") or []) if str(w).strip()}
            cur_wids = {_norm_work_id_token(w) for w in (cur_em.get("work_ids") or []) if str(w).strip()}
            if not prev_wids.issubset(cur_wids):
                raise ConceptGraphError(f"schema sync: {cur_label} dropped work_ids from exposed_cluster_metadata[{epf!r}] relative to {prior_label}")
            if bool(emeta.get("institutional_author", False)) and not bool(cur_em.get("institutional_author", False)):
                raise ConceptGraphError(f"schema sync: {cur_label} cleared institutional_author on exposed_cluster_metadata[{epf!r}]")
            if bool(emeta.get("editor_affiliated", False)) and not bool(cur_em.get("editor_affiliated", False)):
                raise ConceptGraphError(f"schema sync: {cur_label} cleared editor_affiliated on exposed_cluster_metadata[{epf!r}]")
            if bool(emeta.get("editor_coauthor_5yr", False)) and not bool(cur_em.get("editor_coauthor_5yr", False)):
                raise ConceptGraphError(f"schema sync: {cur_label} cleared editor_coauthor_5yr on exposed_cluster_metadata[{epf!r}]")
            if "historical_multi_work_cluster" in emeta and (
                bool(emeta.get("historical_multi_work_cluster", False)) != bool(cur_em.get("historical_multi_work_cluster", False))
            ):
                raise ConceptGraphError(
                    f"schema sync: {cur_label} altered immutable historical_multi_work_cluster on exposed_cluster_metadata[{epf!r}]"
                )
            for marker_k in ("first_burned_by_stage1_registration_id", "first_burned_by_stage2_attempt_id"):
                prior_mk = emeta.get(marker_k)
                if prior_mk is not None and cur_em.get(marker_k) != prior_mk:
                    raise ConceptGraphError(
                        f"schema sync: {cur_label} altered or cleared {marker_k} on exposed_cluster_metadata[{epf!r}]"
                    )
            prior_wr = emeta.get("work_rosters")
            if isinstance(prior_wr, dict) and prior_wr:
                cur_wr = cur_em.get("work_rosters")
                if not isinstance(cur_wr, dict):
                    raise ConceptGraphError(f"schema sync: {cur_label} dropped work_rosters on exposed_cluster_metadata[{epf!r}] relative to {prior_label}")
                cur_wr_by_norm = {_norm_work_id_token(ck): cv for ck, cv in cur_wr.items()}
                for w_k, w_rec in prior_wr.items():
                    c_wrec = cur_wr.get(w_k) or cur_wr_by_norm.get(_norm_work_id_token(w_k))
                    if not isinstance(w_rec, dict) or not isinstance(c_wrec, dict):
                        raise ConceptGraphError(f"schema sync: {cur_label} dropped work_rosters[{w_k!r}] on exposed_cluster_metadata[{epf!r}]")
                    pa_w = {norm_person_or_org_token(str(a), is_org=False) for a in (w_rec.get("authors") or []) if str(a).strip()}
                    ca_w = {norm_person_or_org_token(str(a), is_org=False) for a in (c_wrec.get("authors") or []) if str(a).strip()}
                    po_w = {norm_person_or_org_token(str(o), is_org=True) for o in (w_rec.get("institutional_organizations") or []) if str(o).strip()}
                    co_w = {norm_person_or_org_token(str(o), is_org=True) for o in (c_wrec.get("institutional_organizations") or []) if str(o).strip()}
                    if not pa_w.issubset(ca_w) or not po_w.issubset(co_w):
                        raise ConceptGraphError(f"schema sync: {cur_label} dropped authors/orgs from work_rosters[{w_k!r}] on {epf!r}")
                    if bool(w_rec.get("editor_coauthor_5yr", False)) and not bool(c_wrec.get("editor_coauthor_5yr", False)):
                        raise ConceptGraphError(f"schema sync: {cur_label} cleared editor_coauthor_5yr on work_rosters[{w_k!r}] of {epf!r}")

    for hist_key in ("evaluation_attempts", "stage1_human_registrations", "stage2_human_attempts", "refresh_cycle_resets"):
        prior_list = prior_doc.get(hist_key)
        if isinstance(prior_list, list) and prior_list:
            cur_list = cur_doc.get(hist_key)
            if not isinstance(cur_list, list) or len(cur_list) < len(prior_list) or cur_list[: len(prior_list)] != prior_list:
                raise ConceptGraphError(f"schema sync: {cur_label} altered or truncated append-only {hist_key} relative to {prior_label}")

    prior_arc = prior_doc.get("active_refresh_cycle")
    if isinstance(prior_arc, dict) and str(prior_arc.get("cycle_id") or "").strip():
        p_cid = str(prior_arc["cycle_id"]).strip()
        cur_arc = cur_doc.get("active_refresh_cycle")
        same_active = (
            isinstance(cur_arc, dict)
            and cur_arc.get("cycle_id") == p_cid
            and cur_arc.get("initial_baseline_non_guides_sha256") == prior_arc.get("initial_baseline_non_guides_sha256")
            and cur_arc.get("opened_at") == prior_arc.get("opened_at")
        )
        if not same_active:
            prior_resets = prior_doc.get("refresh_cycle_resets") if isinstance(prior_doc.get("refresh_cycle_resets"), list) else []
            cur_resets = cur_doc.get("refresh_cycle_resets") if isinstance(cur_doc.get("refresh_cycle_resets"), list) else []
            new_resets = cur_resets[len(prior_resets) :]
            if not any(isinstance(r, dict) and r.get("closed_cycle_id") == p_cid for r in new_resets):
                raise ConceptGraphError(
                    f"schema sync: {cur_label} replaced or cleared active_refresh_cycle {p_cid!r} without an explicit refresh_cycle_resets entry"
                )


def compute_lineage_components(
    units: list[dict] | None,
    burned_doc: dict,
    *,
    include_all_exposed_meta: bool = True,
    strict_prospective_work_bijection: bool = False,
) -> dict[str, object]:
    """Unified transitive `<50%` personal-author, `>=1` institutional-organization, and shared-`work_id` closure.

    Evaluates cluster-level rosters, per-work `work_rosters`, and normalized `work_id` identities with
    separated personal-author and institutional-organization denominators so multi-work consortium clusters
    cannot dilute per-paper authorship overlap, institutional strings cannot skew author-count denominators,
    and duplicate `work_id` entries can never masquerade as independent lineage components.
    """
    merges = resolve_cluster_merges(burned_doc.get("cluster_merges") or {})
    internal_set = {merges.get(str(x).strip(), str(x).strip()) for x in (burned_doc.get("internal_clusters") or []) if str(x).strip()}
    internal_set.add("pf_internal_mlsysbook")
    editor_aff_set = {merges.get(str(x).strip(), str(x).strip()) for x in (burned_doc.get("editor_affiliated_clusters") or []) if str(x).strip()}
    editor_co_set = {
        merges.get(str(x).strip(), str(x).strip()) for x in (burned_doc.get("editor_coauthor_5yr_clusters") or []) if str(x).strip()
    } | set(editor_aff_set)
    exposed_meta = burned_doc.get("exposed_cluster_metadata") or {}

    records: list[tuple[str, set[str], set[str]]] = []
    present_ext_pfs: set[str] = set()
    has_internal = False
    ed_pfs: set[str] = set(editor_aff_set)
    ed_co_pfs: set[str] = set(editor_co_set)
    work_id_to_pfs: dict[str, set[str]] = collections.defaultdict(set)
    prospective_pf_work_ids: dict[str, set[str]] = collections.defaultdict(set)
    work_id_signatures: dict[str, tuple[frozenset[str], frozenset[str], bool, bool]] = {}

    def _append_work_rosters(pf_id: str, obj: dict) -> None:
        raw_wr = obj.get("work_rosters")
        wr_iter = raw_wr.values() if isinstance(raw_wr, dict) else (raw_wr if isinstance(raw_wr, list) else ())
        for wr in wr_iter:
            if isinstance(wr, dict):
                wa, wo = _extract_auth_and_org_sets(wr)
                if wa or wo:
                    records.append((pf_id, wa, wo))
                if bool(wr.get("editor_coauthor_5yr", False)):
                    ed_co_pfs.add(pf_id)
            elif isinstance(wr, list):
                wa = {norm_person_or_org_token(str(a), is_org=False) for a in wr}
                wa.discard("")
                if wa:
                    records.append((pf_id, wa, set()))

    for idx_u, u in enumerate(units or []):
        if not isinstance(u, dict):
            if strict_prospective_work_bijection:
                raise ConceptGraphError(f"schema sync: units[{idx_u}] in lineage closure must be a mapping")
            continue
        raw_pf = str(u.get("provenance_family_id") or "").strip()
        if not raw_pf:
            if strict_prospective_work_bijection:
                raise ConceptGraphError(f"schema sync: units[{idx_u}] missing provenance_family_id")
            continue
        pf = merges.get(raw_pf, raw_pf)
        is_int = (
            pf in internal_set
            or u.get("origin") == "internal"
            or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_"))
        )
        if is_int:
            has_internal = True
            internal_set.add(pf)
            continue
        present_ext_pfs.add(pf)
        ed_u = bool(u.get("editor_affiliated", False)) or bool(u.get("_editor_affiliated", False))
        ed_co_u = ed_u or bool(u.get("editor_coauthor_5yr", False)) or bool(u.get("_editor_coauthor_5yr", False))
        if ed_u:
            ed_pfs.add(pf)
        if ed_co_u:
            ed_co_pfs.add(pf)
        ua, uo = _extract_auth_and_org_sets(u)
        u_wids = _extract_work_ids(u)
        if strict_prospective_work_bijection:
            raw_wid = _norm_work_id_token(u.get("work_id"))
            if len(raw_wid) < 3 or u_wids != {raw_wid}:
                raise ConceptGraphError(
                    f"schema sync: external unit {u.get('unit_id') or idx_u!r} ({pf!r}) requires a single explicit work_id (>=3 chars)"
                )
            inst_flag = u.get("institutional_author", False)
            if not isinstance(inst_flag, bool):
                raise ConceptGraphError(
                    f"schema sync: external unit {u.get('unit_id') or idx_u!r} institutional_author must be boolean"
                )
            if inst_flag and not uo:
                raise ConceptGraphError(
                    f"schema sync: institutional external unit {u.get('unit_id') or idx_u!r} requires non-empty institutional_organizations"
                )
            if not inst_flag and not ua:
                raise ConceptGraphError(
                    f"schema sync: external unit {u.get('unit_id') or idx_u!r} requires non-empty personal authors"
                )
            sig = (frozenset(ua), frozenset(uo), ed_u or (pf in ed_pfs), ed_co_u or (pf in ed_co_pfs))
            prev_sig = work_id_signatures.get(raw_wid)
            if prev_sig is not None and prev_sig != sig:
                raise ConceptGraphError(
                    f"schema sync: conflicting authorship/organization/COI metadata across units sharing work_id {raw_wid!r}"
                )
            work_id_signatures[raw_wid] = sig
            prospective_pf_work_ids[pf].add(raw_wid)
        for wid in u_wids:
            work_id_to_pfs[wid].add(pf)
        records.append((pf, ua, uo))
        _append_work_rosters(pf, u)

    for pf_raw, meta in sorted(exposed_meta.items()):
        if not isinstance(meta, dict):
            continue
        pf = merges.get(str(pf_raw).strip(), str(pf_raw).strip())
        if pf in internal_set or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_")):
            continue
        if not include_all_exposed_meta and pf not in present_ext_pfs:
            continue
        present_ext_pfs.add(pf)
        if bool(meta.get("editor_affiliated", False)):
            ed_pfs.add(pf)
            ed_co_pfs.add(pf)
        if bool(meta.get("editor_coauthor_5yr", False)):
            ed_co_pfs.add(pf)
        for wid in _extract_work_ids(meta):
            work_id_to_pfs[wid].add(pf)
        ma, mo = _extract_auth_and_org_sets(meta)
        records.append((pf, ma, mo))
        _append_work_rosters(pf, meta)

    if strict_prospective_work_bijection:
        for wid, pfs_w in sorted(work_id_to_pfs.items()):
            if len(pfs_w) > 1:
                raise ConceptGraphError(
                    f"schema sync: work_id {wid!r} maps to multiple canonical provenance_family_ids {sorted(pfs_w)}"
                )
        for pf_k, wids_p in sorted(prospective_pf_work_ids.items()):
            meta_k = exposed_meta.get(pf_k)
            is_hist_multi = isinstance(meta_k, dict) and meta_k.get("historical_multi_work_cluster") is True
            if not is_hist_multi and len(wids_p) != 1:
                raise ConceptGraphError(
                    f"schema sync: prospective external cluster {pf_k!r} must bind a single work_id, got {sorted(wids_p)}"
                )

    ext_pfs = sorted(present_ext_pfs)
    parent = {p: p for p in ext_pfs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for _wid, pfs_w in sorted(work_id_to_pfs.items()):
        ext_w_pfs = [p for p in sorted(pfs_w) if p in parent]
        for p_other in ext_w_pfs[1:]:
            union(ext_w_pfs[0], p_other)

    for i in range(len(records)):
        pf_i, auth_i, org_i = records[i]
        if not auth_i and not org_i:
            continue
        for j in range(i + 1, len(records)):
            pf_j, auth_j, org_j = records[j]
            if pf_i == pf_j:
                continue
            if org_i and org_j and (org_i & org_j):
                union(pf_i, pf_j)
                continue
            if auth_i and auth_j and (len(auth_i & auth_j) / min(len(auth_i), len(auth_j))) >= 0.5:
                union(pf_i, pf_j)

    pf_to_component: dict[str, str] = {}
    component_members: dict[str, list[str]] = {}
    if has_internal or internal_set:
        for ipf in sorted(internal_set):
            pf_to_component[ipf] = "eff_lineage:internal"
        if has_internal:
            component_members["eff_lineage:internal"] = sorted(internal_set)

    for p in ext_pfs:
        cid = f"eff_lineage:{find(p)}"
        pf_to_component[p] = cid
        component_members.setdefault(cid, []).append(p)

    comp_ed: set[str] = set()
    comp_ed_co: set[str] = set()
    unknown_pfs = set(burned_doc.get("historical_unknown_coauthor_clusters") or [])
    comp_unknown: set[str] = set()
    for p in ext_pfs:
        if p in ed_pfs:
            comp_ed.add(pf_to_component[p])
            comp_ed_co.add(pf_to_component[p])
        if p in ed_co_pfs:
            comp_ed_co.add(pf_to_component[p])
        if p in unknown_pfs:
            comp_unknown.add(pf_to_component[p])

    return {
        "merges": merges,
        "internal_clusters": internal_set,
        "editor_affiliated_clusters": ed_pfs,
        "editor_coauthor_5yr_clusters": ed_co_pfs,
        "pf_to_component": pf_to_component,
        "component_members": component_members,
        "component_editor_affiliated": comp_ed,
        "component_editor_coauthor_5yr": comp_ed_co,
        "independence_excluded_clusters": ed_co_pfs | unknown_pfs,
        "component_independence_excluded": comp_ed_co | comp_unknown,
    }


def _compute_warrant_lineages(warrant_sources: list[dict], burned_doc: dict) -> tuple[set[str], set[str]]:
    """Return `(all_effective_lineages, external_non_editor_effective_lineages)` under shared `<50%` overlap closure."""
    closure = compute_lineage_components(warrant_sources, burned_doc, include_all_exposed_meta=True)
    merges: dict[str, str] = closure["merges"]  # type: ignore[assignment]
    internal_set: set[str] = closure["internal_clusters"]  # type: ignore[assignment]
    pf_to_comp: dict[str, str] = closure["pf_to_component"]  # type: ignore[assignment]
    comp_ed: set[str] = closure["component_editor_affiliated"]  # type: ignore[assignment]
    comp_ed_co: set[str] = closure["component_independence_excluded"]  # type: ignore[assignment]

    all_lineages: set[str] = set()
    ext_non_ed_lineages: set[str] = set()
    for src in warrant_sources:
        raw_pf = str(src["provenance_family_id"]).strip()
        pf = merges.get(raw_pf, raw_pf)
        is_int = (
            pf in internal_set
            or src.get("origin") == "internal"
            or pf.startswith(("pf_mlsysbook_", "pf_staffml_", "pf_internal_"))
        )
        if is_int:
            all_lineages.add("eff_lineage:internal")
        else:
            eff = pf_to_comp[pf]
            all_lineages.add(eff)
            if (
                eff not in comp_ed
                and eff not in comp_ed_co
                and not bool(src.get("editor_affiliated", False))
                and not bool(src.get("editor_coauthor_5yr", False))
            ):
                ext_non_ed_lineages.add(eff)
    return all_lineages, ext_non_ed_lineages


def replay_patch_on_taxonomy_doc(doc: dict, change: dict, mode: str) -> dict:
    """ deterministically apply `change` to `doc` in place and return expected operation metadata."""
    if not isinstance(change, dict):
        raise ConceptGraphError("schema sync: replayed_patch must be a mapping")

    if mode == "guides-apply":
        g = doc.get("guides") or {}
        roles = {r["id"]: r for r in g.get("roles", [])}
        symptoms = {s["id"]: s for s in g.get("symptoms", [])}
        paths = {p["id"]: p for p in g.get("learning_paths", [])}
        expected_ops: list[str] = []

        def _g_remove(items: list[str], gone: list[str] | None, ctx: str) -> None:
            for x in gone or []:
                if x not in items:
                    raise ConceptGraphError(f"schema sync: guides replay {ctx}: cannot remove absent {x!r}")
                items.remove(x)

        def _g_add(items: list[str], new: list[str] | None) -> None:
            for x in new or []:
                if x not in items:
                    items.append(x)

        def _g_place(steps: list[dict], before: str | None, after: str | None) -> int:
            subs = [s["subject"] for s in steps]
            if before:
                if before not in subs:
                    raise ConceptGraphError(f"schema sync: guides replay before={before!r} not in steps")
                return subs.index(before)
            if after:
                if after not in subs:
                    raise ConceptGraphError(f"schema sync: guides replay after={after!r} not in steps")
                return subs.index(after) + 1
            return len(steps)

        for u in change.get("roles_update", []) or []:
            expected_ops.append(f"role_update:{u['id'].strip()}")
            r = roles[u["id"].strip()]
            _g_remove(r["core"], u.get("remove_core"), u["id"])
            _g_remove(r.setdefault("working", []), u.get("remove_working"), u["id"])
            _g_add(r["core"], u.get("add_core"))
            _g_add(r["working"], u.get("add_working"))
            r["working"] = [w for w in r["working"] if w not in r["core"]]
            if not r["working"]:
                r.pop("working", None)
            if u.get("summary"):
                r["summary"] = u["summary"].strip()
        for r in change.get("roles_add", []) or []:
            expected_ops.append(f"role_add:{r['id'].strip()}")
            new_r = deepcopy(r)
            g["roles"].append(new_r)
            roles[new_r["id"]] = new_r

        for u in change.get("symptoms_update", []) or []:
            expected_ops.append(f"symptom_update:{u['id'].strip()}")
            s = symptoms[u["id"].strip()]
            for key in ("causes", "checks", "levers", "quantities"):
                if f"remove_{key}" in u or f"add_{key}" in u:
                    _g_remove(s.setdefault(key, []), u.get(f"remove_{key}"), u["id"])
                    _g_add(s[key], u.get(f"add_{key}"))
                    if not s[key] and key in ("checks", "quantities"):
                        s.pop(key, None)
            for key in ("label", "signal"):
                if u.get(key):
                    s[key] = u[key].strip()
            if u.get("reorder"):
                lists = {k: s.get(k, []) for k in ("causes", "checks", "levers") if k in s}
                head = {k: v for k, v in s.items() if k not in ("causes", "checks", "levers")}
                s.clear()
                s.update(head)
                for key in u["reorder"]:
                    if key in lists:
                        s[key] = lists[key]
        for s in change.get("symptoms_add", []) or []:
            expected_ops.append(f"symptom_add:{s['id'].strip()}")
            new_s = deepcopy(s)
            g["symptoms"].append(new_s)
            symptoms[new_s["id"]] = new_s

        for u in change.get("paths_update", []) or []:
            expected_ops.append(f"path_update:{u['id'].strip()}")
            p = paths[u["id"].strip()]
            for mv in u.get("move", []) or []:
                matches = [st for st in p["steps"] if st["subject"] == mv["subject"]]
                step = matches[0]
                p["steps"].remove(step)
                p["steps"].insert(_g_place(p["steps"], mv.get("before"), mv.get("after")), step)
            for ins in u.get("insert", []) or []:
                p["steps"].insert(
                    _g_place(p["steps"], ins.get("before"), ins.get("after")),
                    {"subject": ins["subject"].strip(), "why": ins["why"].strip()},
                )
            for subject, why in (u.get("why") or {}).items():
                matches = [st for st in p["steps"] if st["subject"] == subject]
                matches[0]["why"] = why.strip()
        for p in change.get("paths_add", []) or []:
            expected_ops.append(f"path_add:{p['id'].strip()}")
            new_p = deepcopy(p)
            g["learning_paths"].append(new_p)
            paths[new_p["id"]] = new_p

        if change.get("version"):
            raise ConceptGraphError("schema sync: guides-apply is forbidden from modifying taxonomy version")
        if not str(doc.get("version") or "").endswith("-draft"):
            raise ConceptGraphError("schema sync: guides-apply may only mutate a -draft taxonomy")

        return {"expected_ops": sorted(expected_ops)}

    if mode == "finalize-structural-evaluation":
        if change != {"finalize_structural_evaluation": True}:
            raise ConceptGraphError(
                "schema sync: finalize-structural-evaluation replayed_patch must be {'finalize_structural_evaluation': True}"
            )
        return {"expected_ops": ["finalize_evaluation"]}

    rows = {row["id"]: row for area in doc["areas"] for row in area["rows"]}
    code2row = {row["code"]: row["id"] for row in rows.values() if "code" in row}
    baseline_row_sec = {rid: list(row.get("secondary_concepts", []) or []) for rid, row in rows.items()}
    concepts = {c["id"]: c for c in doc["concepts"]}
    families = {f["id"]: f for f in doc["families"]}
    relations = {r["id"]: r for r in doc["relations"]}

    def resolve_row_id(home_ref: str) -> str:
        return code2row.get(home_ref, home_ref)

    def family_for(home_ref: str, cid: str) -> str:
        rid = resolve_row_id(home_ref)
        members = [c["family"] for c in doc["concepts"] if c.get("home") == rid and "family" in c]
        if not members:
            raise ConceptGraphError(f"schema sync: replay concept {cid} in empty subject {rid} lacks explicit family")
        return collections.Counter(members).most_common(1)[0][0]

    expected_ops = []
    exp_c_added: list[str] = []
    exp_c_retired: list[str] = []
    exp_defs_updated: list[str] = []
    exp_aliases_added: dict[str, list[str]] = {}
    exp_aliases_attested: dict[str, list[str]] = {}
    exp_aliases_removed: dict[str, list[str]] = {}
    exp_other_c_upd: list[str] = []
    exp_sec_added: dict[str, list[str]] = {}
    exp_sec_removed: dict[str, list[str]] = {}
    exp_other_s_upd: list[str] = []
    adopted_alias_norms_by_cid: dict[str, set[str]] = collections.defaultdict(set)

    for u in change.get("concepts_update", []) or []:
        cid = u["id"].strip()
        expected_ops.append(f"update:{cid}")
        node = concepts[cid]
        for key in ("label", "definition"):
            if u.get(key):
                node[key] = " ".join(str(u[key]).split())
        if u.get("scope"):
            node["scope"] = u["scope"].strip()
        if u.get("definition"):
            exp_defs_updated.append(cid)
        if u.get("label") or u.get("scope") or u.get("removed_aliases_errata"):
            exp_other_c_upd.append(cid)
        rem_list = [str(x).strip() for x in u.get("remove_aliases", []) or []]
        rem_norms = {_norm_prov_tokens(x) for x in rem_list}
        base_aliases = [str(a) for a in node.get("aliases", []) or []]
        if rem_list:
            exp_aliases_removed[cid] = rem_list
        aliases = [a for a in base_aliases if _norm_prov_tokens(a) not in rem_norms]
        added_alist: list[str] = []
        for a in u.get("add_aliases", []) or []:
            astr = str(a).strip()
            aliases.append(astr)
            added_alist.append(astr)
            adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(astr))
        if added_alist:
            exp_aliases_added[cid] = added_alist
        attested_alist: list[str] = []
        for a in u.get("attest_existing_aliases", []) or []:
            astr = str(a).strip()
            if astr not in aliases or astr in added_alist:
                raise ConceptGraphError(
                    f"schema sync: attest_existing_aliases {astr!r} on {cid} must name an already-active surviving alias"
                )
            attested_alist.append(astr)
        if attested_alist:
            exp_aliases_attested[cid] = attested_alist
        if u.get("label"):
            adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(node["label"]))
        saved_added_in = node.pop("added_in", None)
        if aliases:
            node["aliases"] = aliases
        else:
            node.pop("aliases", None)
        if saved_added_in is not None:
            node["added_in"] = saved_added_in

    for mv in change.get("moves", []) or []:
        mcid = mv["id"].strip()
        expected_ops.append(f"move:{mcid}")
        mhome = resolve_row_id(mv["home"].strip())
        concepts[mcid]["home"] = mhome
        dest_row = rows[mhome]
        if "secondary_concepts" in dest_row and mcid in dest_row["secondary_concepts"]:
            cleaned_sec = [s for s in dest_row["secondary_concepts"] if s != mcid]
            if cleaned_sec:
                dest_row["secondary_concepts"] = cleaned_sec
            else:
                dest_row.pop("secondary_concepts", None)
            if mcid not in exp_sec_removed.setdefault(mhome, []):
                exp_sec_removed[mhome].append(mcid)

    target_version = str(change.get("version") or doc["version"]).strip()
    for c in change.get("concepts_add", []) or []:
        cid = c["id"].strip()
        expected_ops.append(f"add:{cid}")
        home_rid = resolve_row_id(c["home"].strip())
        node = {
            "id": cid,
            "label": " ".join(str(c["label"]).split()),
            "home": home_rid,
            "family": (c.get("family") or family_for(c["home"], cid)).strip(),
            "definition": " ".join(str(c["definition"]).split()),
            "status": "draft",
        }
        adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(node["label"]))
        if c.get("aliases"):
            node["aliases"] = [str(x).strip() for x in c["aliases"]]
            exp_aliases_added[cid] = list(node["aliases"])
            for ca in node["aliases"]:
                adopted_alias_norms_by_cid[cid].add(_norm_prov_tokens(ca))
        if c.get("scope"):
            node["scope"] = str(c["scope"]).strip()
        node["added_in"] = target_version
        exp_c_added.append(cid)
        doc["concepts"].append(node)
        concepts[cid] = node

    migrate_hv_map: dict[str, str] = {}
    for r in change.get("concepts_retire", []) or []:
        rcid = r["id"].strip()
        expected_ops.append(f"retire:{rcid}")
        rep = r["replaced_by"].strip()
        former = concepts.pop(rcid)
        exp_c_retired.append(rcid)
        doc["concepts"] = [c for c in doc["concepts"] if c["id"] != rcid]
        doc.setdefault("retired", [])
        former_lbl = str(former.get("label") or "").strip()
        former_als = [str(a) for a in (former.get("aliases") or [])]
        ret_entry: dict[str, object] = {"id": rcid}
        if former_lbl:
            ret_entry["label"] = former_lbl
        if former_als:
            ret_entry["aliases"] = list(former_als)
        ret_entry["replaced_by"] = rep
        ret_entry["version"] = str(r.get("version") or target_version).strip()
        ret_entry["reason"] = " ".join(str(r["reason"]).split())
        doc["retired"].append(ret_entry)
        doc["placement_examples"] = [x for x in doc.get("placement_examples", []) if x["concept"] != rcid]
        for rid_k, row_obj in rows.items():
            if "secondary_concepts" in row_obj and rcid in row_obj["secondary_concepts"]:
                sec = [s for s in row_obj["secondary_concepts"] if s != rcid]
                if sec:
                    row_obj["secondary_concepts"] = sec
                else:
                    row_obj.pop("secondary_concepts", None)
                if rcid not in exp_sec_removed.setdefault(rid_k, []):
                    exp_sec_removed[rid_k].append(rcid)
        if r.get("migrate_hierarchy_and_views") is True:
            migrate_hv_map[rcid] = rep

    if doc.get("retired"):
        rep_map = {rec_r["id"]: rec_r["replaced_by"] for rec_r in doc["retired"]}
        for rec_r in doc["retired"]:
            cur = rec_r["replaced_by"]
            seen_chain = {rec_r["id"]}
            while cur in rep_map:
                if cur in seen_chain:
                    raise ConceptGraphError(f"schema sync: cycle in retired replaced_by chain at {rec_r['id']}")
                seen_chain.add(cur)
                cur = rep_map[cur]
            rec_r["replaced_by"] = cur
            if rec_r["id"] in migrate_hv_map:
                migrate_hv_map[rec_r["id"]] = cur
            active_rep_norms = (
                adopted_alias_norms_by_cid.get(cur, set())
                | {_norm_prov_tokens(str(concepts[cur].get("label") or ""))}
                | {_norm_prov_tokens(str(a)) for a in (concepts[cur].get("aliases") or [])}
            )
            if rec_r.get("label") and _norm_prov_tokens(str(rec_r["label"])) in active_rep_norms:
                rec_r.pop("label", None)
            if rec_r.get("aliases"):
                rem_ret_a = [a for a in rec_r["aliases"] if _norm_prov_tokens(a) not in active_rep_norms]
                if rem_ret_a:
                    rec_r["aliases"] = rem_ret_a
                else:
                    rec_r.pop("aliases", None)

    if migrate_hv_map:
        sh = doc.get("skos_hierarchy")
        if isinstance(sh, dict):
            if isinstance(sh.get("hierarchy"), list):
                new_h = []
                seen_h: set[tuple[str, str]] = set()
                for h_item in sh["hierarchy"]:
                    nw = migrate_hv_map.get(h_item["narrower"], h_item["narrower"])
                    br = migrate_hv_map.get(h_item["broader"], h_item["broader"])
                    if nw == br or (nw, br) in seen_h:
                        continue
                    seen_h.add((nw, br))
                    upd_h = dict(h_item)
                    upd_h["narrower"] = nw
                    upd_h["broader"] = br
                    new_h.append(upd_h)
                sh["hierarchy"] = new_h
            if isinstance(sh.get("top_concepts"), list):
                new_tc = []
                for tc in sh["top_concepts"]:
                    mtc = migrate_hv_map.get(tc, tc)
                    if mtc not in new_tc:
                        new_tc.append(mtc)
                sh["top_concepts"] = new_tc
        for vw in doc.get("views") or []:
            if isinstance(vw, dict) and isinstance(vw.get("concepts"), list):
                new_vc = []
                for vc in vw["concepts"]:
                    mvc = migrate_hv_map.get(vc, vc)
                    if mvc not in new_vc:
                        new_vc.append(mvc)
                vw["concepts"] = new_vc

    removed_edges: set[str] = set()
    for x in change.get("relationships_remove", []) or []:
        sx = x.strip()
        expected_ops.append(f"rel_remove:{sx}")
        removed_edges.add(sx)
    doc["relationships"] = [e for e in doc["relationships"] if e["id"] not in removed_edges]
    edges_by_id = {e["id"]: e for e in doc["relationships"]}
    for u in change.get("relationships_update", []) or []:
        uid = u["id"].strip()
        expected_ops.append(f"rel_update:{uid}")
        edge = edges_by_id[uid]
        edge.update({k: v for k, v in u.items() if k in ("predicate", "conditions", "references", "reference_role")})
        if "relation" in u:
            rel_v = u["relation"].strip()
            edge["relation"] = rel_v
            edge["kind"] = relations[rel_v]["kind"]

    edge_ids = set(edges_by_id)
    for r in change.get("relationships_add", []) or []:
        src, tgt = r["source"].strip(), r["target"].strip()
        expected_ops.append(f"edge:{min(src, tgt)}--{max(src, tgt)}")
        rel_v = r["relation"].strip()
        base = f"rel.{src.removeprefix('mlsys.')}.{tgt.removeprefix('mlsys.')}"
        eid, n = base, 2
        while eid in edge_ids:
            eid, n = f"{base}.{n}", n + 1
        edge_ids.add(eid)
        edge = {
            "id": eid,
            "source": src,
            "target": tgt,
            "predicate": r["predicate"].strip(),
            "conditions": r["conditions"].strip(),
            "references": list(r.get("references") or families[concepts[src]["family"]]["references"]),
            "basis": "editorial-synthesis",
            "status": "draft",
            "kind": relations[rel_v]["kind"],
            "reference_role": "reading-basis-not-entailment-proof",
            "relation": rel_v,
        }
        doc["relationships"].append(edge)

    gaps = {g["id"]: g for g in doc.get("gaps", []) or []}
    for gid in change.get("gaps_remove", []) or []:
        sgid = gid.strip()
        expected_ops.append(f"gap_remove:{sgid}")
        gaps.pop(sgid)
    for g in change.get("gaps_update", []) or []:
        ugid = g["id"].strip()
        expected_ops.append(f"gap_update:{ugid}")
        for k, v in g.items():
            if k == "id":
                continue
            gaps[ugid][k] = resolve_row_id(v.strip()) if k == "subject_row" else v
    for g in change.get("gaps_add", []) or []:
        agid = g["id"].strip()
        expected_ops.append(f"gap_add:{agid}")
        gaps[agid] = {
            "id": agid,
            "subject_row": resolve_row_id(g["subject_row"].strip()),
            "topic": g["topic"].strip(),
            "status": "candidate",
            "reason": g["reason"].strip(),
        }
    doc["gaps"] = list(gaps.values())

    for su in change.get("subject_updates", []) or []:
        rid = resolve_row_id(su["row"].strip())
        expected_ops.append(f"row:{rid}")
        row = rows[rid]
        if "remove_includes" in su or "add_includes" in su:
            exp_other_s_upd.append(rid)
            base_inc = list(row.get("includes", []) or [])
            row["includes"] = [i for i in base_inc if i not in set(su.get("remove_includes", []) or [])] + list(
                su.get("add_includes", []) or []
            )
        if "remove_secondary_concepts" in su or "add_secondary_concepts" in su:
            orig_sec = baseline_row_sec[rid]
            cur_sec = list(row.get("secondary_concepts", []) or [])
            rem_secs = [s.strip() for s in su.get("remove_secondary_concepts", []) or []]
            for rs in rem_secs:
                if rs in orig_sec and rs not in exp_sec_removed.setdefault(rid, []):
                    exp_sec_removed[rid].append(rs)
            sec = [s for s in cur_sec if s not in set(rem_secs)]
            added_secs_here: list[str] = []
            for raw_s in su.get("add_secondary_concepts", []) or []:
                s = raw_s.strip()
                if s not in sec:
                    sec.append(s)
                added_secs_here.append(s)
            if added_secs_here:
                exp_sec_added[rid] = added_secs_here
            if sec:
                row["secondary_concepts"] = sec
            else:
                row.pop("secondary_concepts", None)

    for pe in change.get("placement_examples_add", []) or []:
        pcid = pe["concept"].strip()
        expected_ops.append(f"pe:{pcid}")
        phome = resolve_row_id(pe["home"].strip())
        doc.setdefault("placement_examples", [])
        doc["placement_examples"] = [x for x in doc["placement_examples"] if x["concept"] != pcid]
        doc["placement_examples"].append({"concept": pcid, "home": phome, "why": pe["why"].strip()})

    exp_guides_replaced = False
    if change.get("guides_replace") is not None:
        if not isinstance(change["guides_replace"], dict):
            raise ConceptGraphError("schema sync: guides_replace must be a mapping")
        expected_ops.append("guides:replace")
        doc["guides"] = deepcopy(change["guides_replace"])
        exp_guides_replaced = True

    if change.get("version"):
        expected_ops.append("version")
        new_ver_str = str(change["version"]).strip()
        new_status_str = expected_status_for_version(new_ver_str)
        doc["version"] = new_ver_str
        doc["status"] = new_status_str
        if isinstance(doc.get("skos_hierarchy"), dict) and "status" in doc["skos_hierarchy"]:
            doc["skos_hierarchy"]["status"] = new_status_str

    return {
        "expected_ops": sorted(expected_ops),
        "concepts_added": sorted(exp_c_added),
        "concepts_retired": sorted(exp_c_retired),
        "definitions_updated": sorted(exp_defs_updated),
        "aliases_added": {k: list(v) for k, v in sorted(exp_aliases_added.items())},
        "aliases_attested": {k: list(v) for k, v in sorted(exp_aliases_attested.items())},
        "aliases_removed": {k: list(v) for k, v in sorted(exp_aliases_removed.items())},
        "other_concept_updates": sorted(exp_other_c_upd),
        "secondary_concepts_added": {k: list(v) for k, v in sorted(exp_sec_added.items())},
        "secondary_concepts_removed": {k: list(v) for k, v in sorted(exp_sec_removed.items())},
        "other_subject_updates": sorted(exp_other_s_upd),
        "guides_replaced": exp_guides_replaced,
    }


def init_genesis_provenance_state(genesis_doc: dict) -> dict[str, object]:
    """Initialize top-level term-provenance state at 0.4-draft genesis prior to applied_campaigns[0]."""
    concepts_by_id = {c["id"]: c for c in genesis_doc.get("concepts", [])}
    return {
        "concepts": {},
        "subjects": {},
        "retired_concepts": {},
        "grandfathered_draft_alias_exceptions": {
            gcid: deepcopy(grec)
            for gcid, grec in BOOTSTRAP_GRANDFATHERED_ALIAS_EXCEPTIONS.items()
            if gcid in concepts_by_id
            and set(grec["aliases"]).issubset(set(concepts_by_id[gcid].get("aliases") or []))
        },
        "pending_campaign_ids": [],
        "pending_pre_evaluation_override": False,
        "last_reconcile_receipt": None,
        "release_governance_receipt": None,
    }
