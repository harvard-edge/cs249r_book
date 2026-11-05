"""Historical evaluation estimators; isolated from everyday taxonomy editing."""
from __future__ import annotations

import collections
from functools import lru_cache

from ..concept_graph import ConceptGraphError
from .legacy_policy import (
    _STAGE2_ANCHOR_ROW_QUOTAS,
    _STAGE2_PAIR_ORDER,
)


def expected_stage2_anchor_schedule() -> dict[str, list[tuple[str, str, str]]]:
    """Deterministically generate the 40-anchor 3-receiver secondary-pair schedule from candidate-experts.md §4.2."""
    rank = {p: idx for idx, p in enumerate(_STAGE2_PAIR_ORDER)}
    counts_by_group = {"Q": 12, "P": 10, "S": 8, "F": 6, "T1": 2, "T2": 2}
    schedule: dict[str, list[tuple[str, str, str]]] = {}
    for g, n_anc in counts_by_group.items():
        rem = dict(_STAGE2_ANCHOR_ROW_QUOTAS[g])
        group_rows: list[tuple[str, str, str]] = []
        for _ in range(n_anc):
            cands = sorted(rem.keys(), key=lambda p: (-rem[p], rank[p]))
            picked = sorted(cands[:3], key=lambda p: rank[p])
            for p in picked:
                rem[p] -= 1
            group_rows.append((picked[0], picked[1], picked[2]))
        if any(v != 0 for v in rem.values()):
            raise ConceptGraphError(f"schema sync: internal anchor schedule quota residual for {g}: {rem}")
        schedule[g] = group_rows
    return schedule


def _masi_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    jacc = inter / len(a | b)
    if a == b:
        mono = 1.0
    elif a.issubset(b) or b.issubset(a):
        mono = 2.0 / 3.0
    else:
        mono = 1.0 / 3.0
    return jacc * mono


def _clamp_ieee_signed_unit(val: float) -> float:
    if 1.0 < val <= 1.0 + 1e-12:
        return 1.0
    if -1.0 - 1e-12 <= val < -1.0:
        return -1.0
    return val


def _clamp_ieee_unit(val: float) -> float:
    if 1.0 < val <= 1.0 + 1e-12:
        return 1.0
    if -1e-12 <= val < 0.0:
        return 0.0
    return val


def _precompute_nominal_alpha(
    items_vals: list[list[str]],
) -> tuple[list[list[tuple[int, int]]], list[float], list[int], int]:
    cat_to_idx: dict[str, int] = {}
    item_cat_counts: list[list[tuple[int, int]]] = []
    item_dis: list[float] = []
    item_r: list[int] = []
    for vals in items_vals:
        r_i = len(vals)
        if r_i < 2:
            raise ConceptGraphError("schema sync: Krippendorff alpha requires r_i >= 2 raters per item")
        counts: dict[int, int] = {}
        for v in vals:
            c_idx = cat_to_idx.get(v)
            if c_idx is None:
                c_idx = len(cat_to_idx)
                cat_to_idx[v] = c_idx
            counts[c_idx] = counts.get(c_idx, 0) + 1
        c_pairs = list(counts.items())
        same_pairs = sum(c * (c - 1) for _, c in c_pairs)
        diff_pairs = r_i * (r_i - 1) - same_pairs
        item_cat_counts.append(c_pairs)
        item_dis.append(diff_pairs / (r_i - 1))
        item_r.append(r_i)
    return (item_cat_counts, item_dis, item_r, len(cat_to_idx))


def _fast_nominal_alpha_from_precomputed(
    item_cat_counts: list[list[tuple[int, int]]],
    item_dis: list[float],
    item_r: list[int],
    n_cats: int,
    weights: list[float],
) -> tuple[float, bool]:
    n_val = 0.0
    d_o_num = 0.0
    marginals = [0.0] * n_cats
    for c_pairs, dis_i, r_i, w in zip(item_cat_counts, item_dis, item_r, weights):
        if w <= 0.0:
            continue
        n_val += w * r_i
        d_o_num += w * dis_i
        for c_idx, cnt in c_pairs:
            marginals[c_idx] += w * cnt
    if n_val <= 1.0:
        return (1.0, True)
    d_o = d_o_num / n_val
    d_e_num = n_val * n_val - sum(nc * nc for nc in marginals)
    d_e = d_e_num / (n_val * (n_val - 1.0))
    if d_e <= 1e-15:
        return ((1.0 if d_o <= 1e-15 else 0.0), True)
    return (_clamp_ieee_signed_unit(1.0 - d_o / d_e), False)


def _weighted_krippendorff_nominal_alpha(items_vals: list[list[str]], weights: list[float]) -> tuple[float, bool]:
    item_cat_counts, item_dis, item_r, n_cats = _precompute_nominal_alpha(items_vals)
    return _fast_nominal_alpha_from_precomputed(item_cat_counts, item_dis, item_r, n_cats, weights)


def _precompute_masi_alpha(
    items_sets: list[list[frozenset[str]]],
) -> tuple[list[list[int]], list[float], list[tuple[int, int, float]], int]:
    distinct_sets: list[frozenset[str]] = []
    set_to_idx: dict[frozenset[str], int] = {}
    encoded_items: list[list[int]] = []
    for s_list in items_sets:
        row_ids: list[int] = []
        for fs in s_list:
            idx = set_to_idx.get(fs)
            if idx is None:
                idx = len(distinct_sets)
                set_to_idx[fs] = idx
                distinct_sets.append(fs)
            row_ids.append(idx)
        encoded_items.append(row_ids)
    n_pat = len(distinct_sets)
    sim_lookup: dict[tuple[int, int], float] = {}
    nonzero_pairs: list[tuple[int, int, float]] = []
    for p in range(n_pat):
        sim_lookup[(p, p)] = 1.0
        for q in range(p + 1, n_pat):
            sim = _masi_similarity(distinct_sets[p], distinct_sets[q])
            if sim > 0.0:
                sim_lookup[(p, q)] = sim
                sim_lookup[(q, p)] = sim
                nonzero_pairs.append((p, q, sim))

    item_disagreements: list[float] = []
    for row_ids in encoded_items:
        r_i = len(row_ids)
        if r_i < 2:
            raise ConceptGraphError("schema sync: MASI alpha requires r_i >= 2 raters per item")
        dis_sum = 0.0
        for u in range(r_i):
            pu = row_ids[u]
            for v in range(u + 1, r_i):
                pv = row_ids[v]
                if pu != pv:
                    dis_sum += 2.0 * (1.0 - sim_lookup.get((pu, pv), 0.0))
        item_disagreements.append(dis_sum / (r_i - 1))
    return (encoded_items, item_disagreements, nonzero_pairs, n_pat)


def _weighted_krippendorff_masi_alpha(
    items_sets: list[list[frozenset[str]]],
    weights: list[float],
) -> tuple[float, bool]:
    encoded_items, item_disagreements, nonzero_pairs, n_pat = _precompute_masi_alpha(items_sets)
    return _fast_masi_alpha_from_precomputed(encoded_items, item_disagreements, nonzero_pairs, n_pat, weights)


def _fast_masi_alpha_from_precomputed(
    encoded_items: list[list[int]],
    item_disagreements: list[float],
    nonzero_pairs: list[tuple[int, int, float]],
    n_pat: int,
    weights: list[float],
) -> tuple[float, bool]:
    n_val = 0.0
    d_o_num = 0.0
    n_p = [0.0] * n_pat
    for row_ids, dis_i, w in zip(encoded_items, item_disagreements, weights):
        if w <= 0.0:
            continue
        n_val += w * len(row_ids)
        d_o_num += w * dis_i
        for pid in row_ids:
            n_p[pid] += w
    if n_val <= 1.0:
        return (1.0, True)
    d_o = d_o_num / n_val
    off_diag_mass = n_val * n_val - sum(x * x for x in n_p)
    pos_sim_mass = 0.0
    for p, q, sim in nonzero_pairs:
        np_v = n_p[p]
        if np_v > 0.0:
            nq_v = n_p[q]
            if nq_v > 0.0:
                pos_sim_mass += np_v * nq_v * sim
    d_e_num = off_diag_mass - 2.0 * pos_sim_mass
    if d_e_num < 0.0 and d_e_num > -1e-11:
        d_e_num = 0.0
    d_e = d_e_num / (n_val * (n_val - 1.0))
    if d_e <= 1e-15:
        return ((1.0 if d_o <= 1e-15 else 0.0), True)
    return (_clamp_ieee_signed_unit(1.0 - d_o / d_e), False)


def _precompute_pair_stratified_kappa(
    groups: list[str],
    subj1: list[str],
    subj2: list[str],
) -> list[tuple[str, list[int], int, list[int], list[int], list[bool]]]:
    by_g: dict[str, list[int]] = collections.defaultdict(list)
    for idx, g in enumerate(groups):
        by_g[g].append(idx)
    strata: list[tuple[str, list[int], int, list[int], list[int], list[bool]]] = []
    for g, idxs in sorted(by_g.items()):
        cat_map: dict[str, int] = {}
        s1_int: list[int] = []
        s2_int: list[int] = []
        eq_bool: list[bool] = []
        for i in idxs:
            a, b = subj1[i], subj2[i]
            if a not in cat_map:
                cat_map[a] = len(cat_map)
            if b not in cat_map:
                cat_map[b] = len(cat_map)
            s1_int.append(cat_map[a])
            s2_int.append(cat_map[b])
            eq_bool.append(a == b)
        strata.append((g, idxs, len(cat_map), s1_int, s2_int, eq_bool))
    return strata


def _fast_pair_stratified_kappa_from_precomputed(
    strata: list[tuple[str, list[int], int, list[int], list[int], list[bool]]],
    weights: list[float],
) -> tuple[float, bool, list[str]]:
    total_w = sum(w for w in weights if w > 0.0)
    if total_w <= 0.0:
        return (1.0, True, [])
    weighted_kappa_sum = 0.0
    zv_strata: list[str] = []
    for g, idxs, n_cats_g, s1_int, s2_int, eq_bool in strata:
        n_g = 0.0
        po_num = 0.0
        m1 = [0.0] * n_cats_g
        m2 = [0.0] * n_cats_g
        for loc_i, glob_i in enumerate(idxs):
            w = weights[glob_i]
            if w <= 0.0:
                continue
            n_g += w
            if eq_bool[loc_i]:
                po_num += w
            m1[s1_int[loc_i]] += w
            m2[s2_int[loc_i]] += w
        if n_g <= 0.0:
            continue
        po_g = po_num / n_g
        pe_g = sum((m1[c] / n_g) * (m2[c] / n_g) for c in range(n_cats_g) if m1[c] > 0.0)
        if 1.0 - pe_g <= 1e-15:
            zv_strata.append(g)
            k_g = 1.0 if abs(po_g - 1.0) <= 1e-15 else 0.0
        else:
            k_g = _clamp_ieee_signed_unit((po_g - pe_g) / (1.0 - pe_g))
        weighted_kappa_sum += (n_g / total_w) * k_g
    return (_clamp_ieee_signed_unit(weighted_kappa_sum), bool(zv_strata), zv_strata)


def _weighted_pair_stratified_cohen_kappa_with_strata(
    groups: list[str],
    subj1: list[str],
    subj2: list[str],
    weights: list[float],
) -> tuple[float, bool, list[str]]:
    strata = _precompute_pair_stratified_kappa(groups, subj1, subj2)
    return _fast_pair_stratified_kappa_from_precomputed(strata, weights)


def _weighted_pair_stratified_cohen_kappa(
    groups: list[str],
    subj1: list[str],
    subj2: list[str],
    weights: list[float],
) -> tuple[float, bool]:
    kb, any_zv, _strata = _weighted_pair_stratified_cohen_kappa_with_strata(groups, subj1, subj2, weights)
    return (kb, any_zv)


def _precompute_hooper_rolling(
    sets1: list[frozenset[str]],
    sets2: list[frozenset[str]],
) -> list[tuple[int, float, float]]:
    out: list[tuple[int, float, float]] = []
    for a, b in zip(sets1, sets2):
        if a and b:
            inter = len(a & b)
            out.append((1, inter / len(a | b), (2.0 * inter) / (len(a) + len(b))))
        elif not a and not b:
            out.append((2, 1.0, 1.0))
        else:
            out.append((0, 0.0, 0.0))
    return out


def _fast_hooper_rolling_from_precomputed(
    precomp_hr: list[tuple[int, float, float]],
    weights: list[float],
) -> dict[str, float]:
    tot_w = 0.0
    cond_w = 0.0
    cond_h = 0.0
    cond_r = 0.0
    all_h = 0.0
    all_r = 0.0
    empty_w = 0.0
    for (kind, hi, ri), w in zip(precomp_hr, weights):
        if w <= 0.0:
            continue
        tot_w += w
        if kind == 1:
            cond_w += w
            cond_h += w * hi
            cond_r += w * ri
            all_h += w * hi
            all_r += w * ri
        elif kind == 2:
            empty_w += w
            all_h += w
            all_r += w
        else:
            empty_w += w
    if tot_w <= 0.0 or cond_w <= 0.0:
        raise ConceptGraphError("schema sync: degenerate zero denominator in Hooper/Rolling overlap computation")
    return {
        "cond_hooper": _clamp_ieee_unit(cond_h / cond_w),
        "all_hooper": _clamp_ieee_unit(all_h / tot_w),
        "cond_rolling": _clamp_ieee_unit(cond_r / cond_w),
        "all_rolling": _clamp_ieee_unit(all_r / tot_w),
        "empty_rate": _clamp_ieee_unit(empty_w / tot_w),
    }


def _weighted_hooper_rolling(
    sets1: list[frozenset[str]],
    sets2: list[frozenset[str]],
    weights: list[float],
) -> dict[str, float]:
    return _fast_hooper_rolling_from_precomputed(_precompute_hooper_rolling(sets1, sets2), weights)


def _precompute_fleiss_kappa_8rater(
    anchor_subj_lists: list[list[str]],
) -> tuple[list[tuple[float, list[tuple[int, int]]]], int]:
    cat_map: dict[str, int] = {}
    rows: list[tuple[float, list[tuple[int, int]]]] = []
    for subjs in anchor_subj_lists:
        if len(subjs) != 8:
            raise ConceptGraphError("schema sync: 8-rater Fleiss kappa requires 8 subject assignments per anchor")
        counts: dict[int, int] = {}
        for s in subjs:
            c_idx = cat_map.get(s)
            if c_idx is None:
                c_idx = len(cat_map)
                cat_map[s] = c_idx
            counts[c_idx] = counts.get(c_idx, 0) + 1
        c_pairs = list(counts.items())
        p_i = sum(c * (c - 1) for _, c in c_pairs) / 56.0
        rows.append((p_i, c_pairs))
    return (rows, len(cat_map))


def _fast_fleiss_kappa_8rater_from_precomputed(
    rows: list[tuple[float, list[tuple[int, int]]]],
    n_cats: int,
    weights: list[float],
) -> tuple[float, bool]:
    w_sum = sum(w for w in weights if w > 0.0)
    if w_sum <= 0.0:
        return (1.0, True)
    po_acc = 0.0
    cat_totals = [0.0] * n_cats
    for (p_i, c_pairs), w in zip(rows, weights):
        if w <= 0.0:
            continue
        po_acc += w * p_i
        for c_idx, c in c_pairs:
            cat_totals[c_idx] += w * c
    p_o = po_acc / w_sum
    denom_8n = 8.0 * w_sum
    p_e = sum((tot / denom_8n) ** 2 for tot in cat_totals if tot > 0.0)
    if 1.0 - p_e <= 1e-15:
        return ((1.0 if abs(p_o - 1.0) <= 1e-15 else 0.0), True)
    return (_clamp_ieee_signed_unit((p_o - p_e) / (1.0 - p_e)), False)


def _weighted_fleiss_kappa_8rater(anchor_subj_lists: list[list[str]], weights: list[float]) -> tuple[float, bool]:
    rows, n_cats = _precompute_fleiss_kappa_8rater(anchor_subj_lists)
    return _fast_fleiss_kappa_8rater_from_precomputed(rows, n_cats, weights)


def _bootstrap_percentile_95_ci(draws: list[float]) -> list[float]:
    """Return symmetric two-sided 95% percentile CI [sorted[250], sorted[9749]] over B=10,000 draws."""
    if len(draws) != 10000:
        raise ConceptGraphError(f"schema sync: bootstrap 95% CI requires 10000 draws (got {len(draws)})")
    s_draws = sorted(draws)
    return [s_draws[250], s_draws[9749]]


def _canonicalize_pair_eval_columns(
    canon_raters: tuple[str, str],
    r1: str,
    r2: str,
    sb1: str,
    sb2: str,
    fs1: frozenset[str],
    fs2: frozenset[str],
) -> tuple[str, str, frozenset[str], frozenset[str]]:
    """Order paired annotations deterministically by sorted (rater_1, rater_2) person_ids."""
    if (r1, r2) == canon_raters:
        return (sb1, sb2, fs1, fs2)
    if (r2, r1) == canon_raters:
        return (sb2, sb1, fs2, fs1)
    raise ConceptGraphError(f"schema sync: pair raters {(r1, r2)} do not match canonical pair {canon_raters}")


@lru_cache(maxsize=1)
def run_stage2_estimator_self_tests() -> bool:
    """Execute the 5 mandatory unit-test fixtures from candidate-experts.md §4.3 before Stage 2 evaluation."""
    groups = ["F"] * 6 + ["Q"] * 6
    s1 = ["F1.1", "F1.1", "F1.2", "F2.1", "F2.1", "F3.1", "Q1.1", "Q1.1", "Q2.1", "Q4.1", "Q4.1", "Q4.2"]
    s2 = ["F1.2", "F1.1", "F1.2", "F2.1", "F3.1", "F3.1", "Q1.1", "Q2.1", "Q2.1", "Q4.1", "Q4.2", "Q4.2"]
    c1 = [
        frozenset({"c1", "c2"}),
        frozenset({"c1"}),
        frozenset(),
        frozenset({"c3"}),
        frozenset({"c2", "c3"}),
        frozenset({"c4"}),
        frozenset({"q1", "q2"}),
        frozenset(),
        frozenset({"q2"}),
        frozenset({"q3", "q4"}),
        frozenset({"q3"}),
        frozenset({"q5"}),
    ]
    c2 = [
        frozenset({"c1", "c2"}),
        frozenset({"c1", "c2"}),
        frozenset(),
        frozenset(),
        frozenset({"c3", "c4"}),
        frozenset({"c4"}),
        frozenset({"q1"}),
        frozenset({"q1"}),
        frozenset({"q2"}),
        frozenset({"q3", "q4"}),
        frozenset({"q4"}),
        frozenset({"q5"}),
    ]
    w_uni = [1.0] * 12

    # Fixture 1: Orientation-swap invariance across single-item & subset rater swaps + 8-rater anchor permutations.
    a_sub_base, _ = _weighted_krippendorff_nominal_alpha([[a, b] for a, b in zip(s1, s2)], w_uni)
    a_masi_base, _ = _weighted_krippendorff_masi_alpha([[a, b] for a, b in zip(c1, c2)], w_uni)
    hr_base = _weighted_hooper_rolling(c1, c2, w_uni)
    kb_base, _ = _weighted_pair_stratified_cohen_kappa(groups, s1, s2, w_uni)
    kb_glob_sw, _ = _weighted_pair_stratified_cohen_kappa(groups, s2, s1, w_uni)
    canon_by_grp = {"F": ("person:idx_F_1", "person:idx_F_2"), "Q": ("person:idx_Q_1", "person:idx_Q_2")}
    for swap_set in ({0}, {0, 1, 3, 4, 7, 10}):
        s1_sw = [s2[i] if i in swap_set else s1[i] for i in range(12)]
        s2_sw = [s1[i] if i in swap_set else s2[i] for i in range(12)]
        c1_sw = [c2[i] if i in swap_set else c1[i] for i in range(12)]
        c2_sw = [c1[i] if i in swap_set else c2[i] for i in range(12)]
        s1_can: list[str] = []
        s2_can: list[str] = []
        c1_can: list[frozenset[str]] = []
        c2_can: list[frozenset[str]] = []
        for i in range(12):
            cr = canon_by_grp[groups[i]]
            ri1, ri2 = (cr[1], cr[0]) if i in swap_set else cr
            csb1, csb2, cfs1, cfs2 = _canonicalize_pair_eval_columns(cr, ri1, ri2, s1_sw[i], s2_sw[i], c1_sw[i], c2_sw[i])
            s1_can.append(csb1)
            s2_can.append(csb2)
            c1_can.append(cfs1)
            c2_can.append(cfs2)
        a_sub_sw, _ = _weighted_krippendorff_nominal_alpha([[a, b] for a, b in zip(s1_sw, s2_sw)], w_uni)
        a_masi_sw, _ = _weighted_krippendorff_masi_alpha([[a, b] for a, b in zip(c1_sw, c2_sw)], w_uni)
        hr_sw = _weighted_hooper_rolling(c1_sw, c2_sw, w_uni)
        kb_can_sw, _ = _weighted_pair_stratified_cohen_kappa(groups, s1_can, s2_can, w_uni)
        if (
            abs(a_sub_base - a_sub_sw) >= 1e-12
            or abs(a_masi_base - a_masi_sw) >= 1e-12
            or abs(hr_base["cond_hooper"] - hr_sw["cond_hooper"]) >= 1e-12
            or abs(hr_base["all_hooper"] - hr_sw["all_hooper"]) >= 1e-12
            or abs(hr_base["cond_rolling"] - hr_sw["cond_rolling"]) >= 1e-12
            or abs(hr_base["all_rolling"] - hr_sw["all_rolling"]) >= 1e-12
            or abs(kb_base - kb_can_sw) >= 1e-12
        ):
            raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 1 (orientation-swap invariance) failed")
    anc_8 = [["F1.1"] * 5 + ["F1.2"] * 3, ["Q4.1"] * 6 + ["Q4.2"] * 2, ["S1.1"] * 4 + ["S1.2"] * 4]
    anc_8_perm = [list(reversed(r)) for r in anc_8]
    fk_base, _ = _weighted_fleiss_kappa_8rater(anc_8, [1.0, 1.0, 1.0])
    fk_sw, _ = _weighted_fleiss_kappa_8rater(anc_8_perm, [1.0, 1.0, 1.0])
    if abs(fk_base - fk_sw) >= 1e-12 or abs(kb_base - kb_glob_sw) >= 1e-12:
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 1 (orientation-swap invariance) failed")

    # Fixture 2: Uniform-weight identity against unweighted direct formulas.
    n_items = len(s1)
    n_val_unw = 2 * n_items
    do_sub_unw = sum(1.0 if a != b else 0.0 for a, b in zip(s1, s2)) / n_items
    cnt_sub = collections.Counter(s1 + s2)
    de_sub_unw = sum(cnt_sub[x] * cnt_sub[y] for x in cnt_sub for y in cnt_sub if x != y) / (n_val_unw * (n_val_unw - 1))
    alpha_sub_unw = 1.0 - do_sub_unw / de_sub_unw
    do_masi_unw = sum(1.0 - _masi_similarity(a, b) for a, b in zip(c1, c2)) / n_items
    all_sets_flat = c1 + c2
    de_masi_unw = sum(
        1.0 - _masi_similarity(all_sets_flat[u], all_sets_flat[v])
        for u in range(n_val_unw)
        for v in range(n_val_unw)
        if u != v
    ) / (n_val_unw * (n_val_unw - 1))
    alpha_masi_unw = 1.0 - do_masi_unw / de_masi_unw
    kb_weighted, _ = _weighted_pair_stratified_cohen_kappa(groups, s1, s2, w_uni)
    kb_unw = 0.0
    for g_id in ("F", "Q"):
        g_idx = [i for i, gg in enumerate(groups) if gg == g_id]
        po = sum(1 for i in g_idx if s1[i] == s2[i]) / len(g_idx)
        c_g1 = collections.Counter(s1[i] for i in g_idx)
        c_g2 = collections.Counter(s2[i] for i in g_idx)
        pe = sum((c_g1[k] / len(g_idx)) * (c_g2[k] / len(g_idx)) for k in set(c_g1) | set(c_g2))
        kb_unw += (len(g_idx) / n_items) * ((po - pe) / (1.0 - pe))
    if (
        abs(a_sub_base - alpha_sub_unw) >= 1e-12
        or abs(a_masi_base - alpha_masi_unw) >= 1e-12
        or abs(kb_weighted - kb_unw) >= 1e-12
    ):
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 2 (uniform-weight identity) failed")

    # Fixture 3: Integer cluster-replication equivalence (sum k_i = N).
    int_mults = [2, 0, 3, 1, 2, 0, 1, 0, 1, 1, 1, 0]  # sum == 12
    a_sub_int, _ = _weighted_krippendorff_nominal_alpha(
        [[a, b] for a, b in zip(s1, s2)],
        [float(k) for k in int_mults],
    )
    a_masi_int, _ = _weighted_krippendorff_masi_alpha(
        [[a, b] for a, b in zip(c1, c2)],
        [float(k) for k in int_mults],
    )
    dup_sub = [[s1[i], s2[i]] for i, k in enumerate(int_mults) for _ in range(k)]
    dup_masi = [[c1[i], c2[i]] for i, k in enumerate(int_mults) for _ in range(k)]
    a_sub_dup, _ = _weighted_krippendorff_nominal_alpha(dup_sub, [1.0] * len(dup_sub))
    a_masi_dup, _ = _weighted_krippendorff_masi_alpha(dup_masi, [1.0] * len(dup_masi))
    if abs(a_sub_int - a_sub_dup) >= 1e-12 or abs(a_masi_int - a_masi_dup) >= 1e-12:
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 3 (integer cluster-replication) failed")

    # Fixture 4: Zero-variance boundary (D_e = 0, D_o = 0 => 1.0; p_e = 1, p_o = 1 => 1.0).
    zv_sub, zv_flag1 = _weighted_krippendorff_nominal_alpha([["F1.1", "F1.1"]] * 4, [1.0] * 4)
    zv_masi, zv_flag2 = _weighted_krippendorff_masi_alpha([[frozenset({"c1"}), frozenset({"c1"})]] * 4, [1.0] * 4)
    zv_kap, zv_flag3, zv_strata = _weighted_pair_stratified_cohen_kappa_with_strata(
        ["F"] * 4, ["F1.1"] * 4, ["F1.1"] * 4, [1.0] * 4
    )
    zv_fl, zv_flag4 = _weighted_fleiss_kappa_8rater([["F1.1"] * 8] * 4, [1.0] * 4)
    if (
        zv_sub != 1.0
        or zv_masi != 1.0
        or zv_kap != 1.0
        or zv_fl != 1.0
        or not (zv_flag1 and zv_flag2 and zv_flag3 and zv_flag4)
        or zv_strata != ["F"]
    ):
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 4 (zero-variance boundary) failed")

    # Fixture 5: Fractional-weight precomputed parity, 95% CI order statistics, and M_eff in [120, 240] union bounds.
    frac_w = [1.5, 0.5, 0.0, 2.25, 0.75, 1.0, 1.8, 0.2, 1.1, 0.9, 1.4, 0.6]  # sum == 12.0
    sub_pairs = [[a, b] for a, b in zip(s1, s2)]
    masi_pairs = [[a, b] for a, b in zip(c1, c2)]
    p_nom = _precompute_nominal_alpha(sub_pairs)
    p_masi = _precompute_masi_alpha(masi_pairs)
    p_kap = _precompute_pair_stratified_kappa(groups, s1, s2)
    p_hr = _precompute_hooper_rolling(c1, c2)
    p_fl = _precompute_fleiss_kappa_8rater(anc_8)
    fast_nom, _ = _fast_nominal_alpha_from_precomputed(*p_nom, frac_w)
    fast_masi, _ = _fast_masi_alpha_from_precomputed(*p_masi, frac_w)
    fast_kap, _, _ = _fast_pair_stratified_kappa_from_precomputed(p_kap, frac_w)
    fast_hr = _fast_hooper_rolling_from_precomputed(p_hr, frac_w)
    fast_fl, _ = _fast_fleiss_kappa_8rater_from_precomputed(*p_fl, [1.2, 0.8, 1.0])
    ci_95 = _bootstrap_percentile_95_ci([float(i) for i in range(10000)])
    per_area_zero_120 = (1.0 - 12.0 / 120.0) ** 120
    per_area_zero_240 = (1.0 - 12.0 / 240.0) ** 240
    if (
        abs(fast_nom - _weighted_krippendorff_nominal_alpha(sub_pairs, frac_w)[0]) >= 1e-12
        or abs(fast_masi - _weighted_krippendorff_masi_alpha(masi_pairs, frac_w)[0]) >= 1e-12
        or abs(fast_kap - _weighted_pair_stratified_cohen_kappa(groups, s1, s2, frac_w)[0]) >= 1e-12
        or any(abs(fast_hr[k] - _weighted_hooper_rolling(c1, c2, frac_w)[k]) >= 1e-12 for k in fast_hr)
        or abs(fast_fl - _weighted_fleiss_kappa_8rater(anc_8, [1.2, 0.8, 1.0])[0]) >= 1e-12
        or ci_95 != [250.0, 9749.0]
        or not (per_area_zero_120 < 3.23e-6 and 20.0 * per_area_zero_120 < 6.46e-5)
        or not (per_area_zero_240 < 4.51e-6 and 20.0 * per_area_zero_240 < 9.01e-5)
    ):
        raise ConceptGraphError("schema sync: Stage 2 self-test Fixture 5 (precomputed bootstrap & CI parity) failed")
    return True
