"""Shared presentation and evidence wiring for Physical AI chapter companions.

Curriculum comes from learning_paths; case inputs, traces, and computations come
from MLSysIM v4_studies. This module never sends commands to physical hardware.
"""
from __future__ import annotations

from copy import deepcopy
import json
import hashlib
import math
import re
import textwrap
from typing import Any

import marimo as mo

from .learning_paths import VOLUME_ACCENTS
from .notebook import notebook_part, notebook_shell, notebook_table, notebook_status


def physical_opening(contract, metadata, current_part):
    """Introduce the chapter's decision within the continuing robot project."""
    from .notebook import notebook_header, notebook_learning_map, notebook_scenario
    return notebook_shell([
        notebook_header(metadata, contract["parts"][0]["question"]),
        notebook_scenario(contract, volume=4),
        notebook_learning_map(contract, current_part=current_part),
    ], volume=4)


def _instrument(result):
    """Only render explicitly typed instruments supplied by the study."""
    instrument = result.get("instrument")
    if not instrument or instrument.get("kind") != "bar":
        return None
    import plotly.graph_objects as go
    figure = go.Figure(go.Bar(x=instrument["x"], y=instrument["y"], marker_color=VOLUME_ACCENTS[4]))
    title_lines = textwrap.wrap(instrument["label"], width=34)
    figure.update_layout(title={"text": "<br>".join(title_lines), "font": {"size": 16}, "x": 0, "xanchor": "left"}, xaxis_title="", yaxis_title=instrument["unit"],
                         template="plotly_white", margin=dict(l=55,r=15,t=80 if len(title_lines) > 1 else 55,b=90), height=355 if len(title_lines) > 1 else 330,
                         font=dict(family="Inter, sans-serif"), showlegend=False)
    figure.update_yaxes(rangemode="tozero")
    return mo.vstack([mo.ui.plotly(figure, config={"displayModeBar": False, "responsive": True}), mo.md(instrument.get("scope", result["limitation"]))])


def _display_value(value):
    """Round only the visible representation; evidence snapshots stay exact."""
    from mlsysim.fmt import fmt, fmt_sci, fmt_qty, _is_integer_like
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if value != 0 and abs(value) < 0.001:
            return str(fmt_sci(value, precision=2))
        return str(fmt(value, precision=0 if _is_integer_like(value) else 3, trim_trailing_zeros=True))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(_display_value(item)) for item in value) + "]"
    if isinstance(value, dict):
        return "; ".join(str(key).replace("_", " ") + ": " + str(_display_value(item)) for key, item in value.items())
    if isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+.+", value):
        from mlsysim.core.units import ureg
        try:
            q = ureg.Quantity(value)
            magnitude = float(q.magnitude)
            precision = 0 if _is_integer_like(magnitude) else max(3, 2-math.floor(math.log10(abs(magnitude))))
            return str(fmt_qty(q, q.units, precision=precision, trim_trailing_zeros=True))
        except (ValueError, TypeError, LookupError):
            pass
    return value


def _display_rows(rows):
    return [{str(key).replace("_", " "): _display_value(value) for key, value in row.items()} for row in rows]


def _live_evidence(result):
    """Separate quantitative queue consequences from documentary role checks."""
    rows = result["rows"]
    queue = [row for row in rows if row.get("policy_status", "").startswith("Constructed queue calculation")]
    if not queue:
        return notebook_table(_display_rows(rows))
    selected = {"offered utilization", "stationary queue exists", "p99 wait before assistance",
                "assistance start service target", "probability queue wait exceeds service target",
                "physical response deadline proven"}
    quantities = [{"Quantity": row["requirement"], "Value": row["supplied_evidence"]} for row in queue]
    summary = [row for row in quantities if row["Quantity"] in selected]
    documentary = [row for row in rows if row not in queue]
    return mo.vstack([
        mo.md("**Staffing consequence · declared queue assumptions**"), notebook_table(_display_rows(summary)),
        mo.md("**Human role and stage evidence · separate documentary checks**"), notebook_table(_display_rows(documentary)),
        mo.accordion({"Full queue accounting and assumptions": notebook_table(_display_rows(quantities))}),
    ], gap=0.6)


def _trace_evidence(result):
    """Keep the decision comparison readable and the exact diagnostic trace available."""
    trace = result.get("trace", [])
    if not trace:
        return None
    items = []
    if all("wilson_lower" in row and "reference_clearance_reserve_m" in row for row in trace):
        from mlsysim.core.units import ureg
        from mlsysim.fmt import fmt_percent, fmt_qty, _is_integer_like
        percent = lambda value: str(fmt_percent(value, style="symbol"))
        quantity = lambda value, unit: str(fmt_qty(value * ureg(unit), unit, precision=0 if _is_integer_like(value) else 3, trim_trailing_zeros=True))
        sampling = [{"Group": row["group"], "Detected / encounters": str(row["detected"]) + " / " + str(row["encounters"]),
                     "Detection rate": percent(row["deadline_detection_rate"]),
                     "Wilson interval": percent(row["wilson_lower"]) + " to " + percent(row["wilson_upper"])} for row in trace]
        stopping = [{"Group": row["group"], "Extra delay": quantity(row["separately_assumed_extra_delay_s"], "s"),
                     "Clearance reserve": quantity(row["reference_clearance_reserve_m"], "m"),
                     "Boundary speed": quantity(row["conditional_speed_at_allocated_boundary_m_per_s"], "m/s"),
                     "Boundary energy": quantity(row["conditional_rigid_body_energy_at_allocated_boundary_J"], "J")} for row in trace]
        items += [mo.md("**Sampling evidence · independent constructed encounters**"), notebook_table(sampling),
                  mo.md("**Separate stopping assumption · not derived from detection rates**"), notebook_table(stopping),
                  mo.md("Negative reserve misses the reference budget. Boundary speed and rigid-body energy describe the allocated clearance boundary, not injury risk. Interval width is shown below.")]
    else:
        columns = {key for row in trace for key in row}
        if len(columns) <= 5 and len(trace) <= 12:
            items += [mo.md("**Selected case observations · instructional evidence**"), notebook_table(_display_rows(trace))]
        else:
            items.append(mo.md("**Diagnostic trace available below.** Use the live quantities and instrument for the main comparison."))
    raw = {"trace": trace}
    if result.get("tick_trace"):
        raw["every_permission_tick"] = result["tick_trace"]
    items.append(mo.accordion({"Exact trace · full precision": mo.md("```json\n" + json.dumps(raw, indent=2, ensure_ascii=False) + "\n```")}))
    if result.get("trace_sampling"):
        items.append(mo.md(result["trace_sampling"]))
    return mo.vstack(items, gap=0.6)


def physical_controls(chapter: int):
    """Construct controls in a cell before a dependent cell reads their values."""
    from mlsysim.labs.experiments.v4_studies import choices, study

    investigations = []
    for part in "ABCD":
        cases = choices(chapter, part)
        labels = {item["label"]: item["id"] for item in cases}
        briefing = study(chapter, part, cases[0]["id"])
        investigations.append(mo.ui.dictionary({
            "prediction": mo.ui.radio(briefing.get("prediction_options", [
                "The case supports the stated claim under its assumptions",
                "The case conflicts with a required condition",
                "The available evidence cannot settle the claim",
            ]), label=briefing.get("prediction_prompt", "Before exploring, what consequence do you expect?")),
            "choice": mo.ui.dropdown(labels, value=next(iter(labels)), label="Choose a case to investigate", full_width=True),
            "interpretation": mo.ui.text_area(
                placeholder="Compare the evidence, explain the mechanism, and name a condition that would change your choice.",
                label="Your interpretation of the comparison", full_width=True, debounce=False,
            ),
        }))
    return mo.ui.array(investigations)


def _signature(chapter, part, value, result=None, pinned=None):
    from mlsysim.labs.experiments.v4_studies import study
    result = result if result is not None else study(chapter, part, value["choice"])
    return {key: value[key] for key in ("choice", "prediction", "interpretation")} | {
        "study_inputs": deepcopy(result["inputs"]), "study_provenance": deepcopy(result["provenance"]),
        "study_evidence_sha256": hashlib.sha256(json.dumps({key: result.get(key) for key in ("rows", "conditions", "trace", "tick_trace")}, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "pinned_case": deepcopy(pinned),
    }


def _has_intervention(chapter, pinned, current):
    """A changed condition can yield the same outcome and still test a claim."""
    if pinned is None:
        return False
    if int(chapter) == 0:
        return True  # Orientation rehearses evidence handling, not a mechanism.
    baseline = pinned["result"]
    # Documentary interventions change supplied evidence in rows/trace rather
    # than numeric inputs. Display labels or feedback alone do not count.
    keys = ("inputs", "conditions", "rows", "trace")
    return any(baseline.get(key) != current.get(key) for key in keys)


def physical_actions(contract, controls, get_pinned, set_pinned, get_saved, set_saved):
    """Create stable event sources outside the cell that displays their state."""
    from mlsysim.labs.experiments.v4_studies import study
    actions = []
    for index, part in enumerate(contract["parts"]):
        letter = part["id"]

        def pin(_event, part_id=letter, at=index):
            value = controls.value[at]
            result = study(contract["chapter"], part_id, value["choice"])
            set_pinned({**get_pinned(), part_id: {"case": value["choice"], "result": deepcopy(result)}})

        def save(_event, part_id=letter, at=index):
            value = controls.value[at]
            baseline = get_pinned().get(part_id)
            if baseline is None or not value["interpretation"].strip():
                return
            result = study(contract["chapter"], part_id, value["choice"])
            if value["prediction"] is None or not _has_intervention(contract["chapter"], baseline, result):
                return
            set_saved({**get_saved(), part_id: {
                "signature": _signature(contract["chapter"], part_id, value, result, baseline),
                "pinned": deepcopy(baseline), "current": deepcopy(result),
            }})

        actions.append({"pin": mo.ui.button(label="Pin this case", on_click=pin),
                        "save": mo.ui.button(label="Save comparison and interpretation", on_click=save)})
    return actions


def physical_navigation(contract):
    """A stable tab control, created separately from the reactive part output."""
    labels = {f"Part {part['id']}": mo.md("") for part in contract["parts"]}
    return mo.ui.tabs(labels, value=next(iter(labels)))


def physical_discussion_control():
    return mo.ui.text_area(
        label="What do you recommend, and why?",
        placeholder="Use your comparisons to explain your choice, the alternative you rejected, what is still unknown, and what would change your mind. Explain if more evidence is needed before acting.",
        full_width=True, debounce=False,
    )


def physical_investigations(contract, controls, values, get_pinned, set_pinned, get_saved, set_saved, selected_part, pinned_snapshot, saved_snapshot, actions):
    """Render reactive studies; explicit actions retain immutable evidence."""
    from mlsysim.labs.experiments.v4_studies import study

    # Accept compact navigation labels and legacy caller-supplied part IDs.
    selected_part = selected_part.removeprefix("Part ")
    chapter = contract["chapter"]
    pinned = pinned_snapshot
    saved = saved_snapshot
    tabs: dict[str, Any] = {}
    for index, part in enumerate(contract["parts"]):
        letter = part["id"]
        if letter != selected_part:
            continue
        ui = controls[index]
        value = values[index]
        preview = study(chapter, letter, value["choice"])
        items = [notebook_part(f"Part {letter} · {part['label']}", part["question"])]
        if preview.get("brief"):
            items.append(mo.md(preview["brief"]))
        if preview.get("held_fixed"):
            items.append(mo.md("**Keep fixed:** " + preview["held_fixed"]))
        if preview.get("source_kind"):
            items.append(mo.md("**Evidence type:** " + preview["source_kind"]))
        items.append(ui["choice"])
        if preview.get("conditions"):
            items.extend([mo.md("**Declared starting conditions**"), notebook_table(_display_rows(preview["conditions"]))])
        elif preview["inputs"]:
            items.append(mo.accordion({"Declared case inputs": notebook_table([
                {"condition": str(key).replace("_", " "), "value": str(item)}
                for key, item in preview["inputs"].items()
            ])}))
        items.append(ui["prediction"])
        if value["prediction"] is None:
            items.append(mo.md("**Start with a prediction.** Any answer opens the comparison; an incorrect prediction never blocks feedback."))
        else:
            current = preview
            signature = _signature(chapter, letter, value, current, pinned.get(letter))
            trace_view = _trace_evidence(current)
            if trace_view is not None:
                items.append(trace_view)
            items.extend([mo.md("**Live evidence and feedback · updates immediately when the case changes.**"),
                          _live_evidence(current),
                          mo.md("**What this choice changes.** " + current["feedback"]),
                          mo.md("**What this evidence cannot establish.** " + current["limitation"])])

            instrument = _instrument(current)
            if instrument is not None:
                items.insert(len(items) - 2, instrument)

            items.append(actions[index]["pin"])
            if letter in pinned:
                items.extend([mo.md("**Pinned case · retained while live controls change.**"),
                              _live_evidence(pinned[letter]["result"]), ui["interpretation"]])

                items.append(actions[index]["save"])
                if not value["interpretation"].strip():
                    items.append(mo.md("Add your interpretation before saving the comparison."))
                if not _has_intervention(chapter, pinned[letter], current):
                    items.append(mo.md("Change a case before saving: a baseline alone does not test an intervention. The outcome may remain unchanged; that is useful evidence when the starting conditions differ."))
            else:
                items.append(mo.md("Pin a case, then try another case and explain the comparison. Pinning is a snapshot, not a passing verdict."))
            if letter in saved:
                state = "Saved comparison · current" if saved[letter]["signature"] == signature else "Saved comparison · stale relative to this draft"
                items.append(notebook_status(state, "Retained in this session. Download your discussion to keep a copy."))
            items.append(mo.accordion({"Calculation notes and evidence assumptions": mo.md(
                "```json\n" + json.dumps({"declared_conditions": current["inputs"]}, indent=2, default=str) + "\n```"
            )}))
        items.append(mo.md("**Next:** " + part.get("next", contract["next_part"])))
        tabs[f"Part {letter}"] = mo.vstack(items, gap=0.8)
    return notebook_shell(list(tabs.values()), volume=4)


def _discussion_report(contract, values, saved, pinned, discussion):
    current = {}
    stale = []
    for index, part in enumerate(contract["parts"]):
        letter = part["id"]
        signature = _signature(contract["chapter"], letter, values[index], pinned=pinned.get(letter))
        if letter in saved:
            if saved[letter]["signature"] == signature and _has_intervention(contract["chapter"], pinned.get(letter), saved[letter]["current"]):
                current[letter] = saved[letter]
            else:
                stale.append(letter)
    ready = len(current) == len(contract["parts"]) and bool(discussion.strip())
    report = {
        "volume": 4, "chapter": contract["chapter"], "title": contract["title"],
        "status": "complete discussion draft" if ready else "incomplete discussion draft",
        "decision": discussion, "current_evidence": current,
        "stale_part_ids": stale, "historical_evidence": saved,
        "physical_validation": "Not performed. Reference calculations and constructed traces do not qualify the hardware studio.",
    }
    return report


def _markdown_report(contract, report):
    current = report["current_evidence"]
    stale = report["stale_part_ids"]
    discussion = report["decision"]
    markdown = [f"# {contract['title']}", "", f"Status: {report['status']}", "", "## Team decision", "", discussion or "No decision recorded."]
    for letter, evidence in current.items():
        markdown += ["", f"## Part {letter}", "", "Prediction: " + evidence["signature"]["prediction"],
                     "", "Interpretation: " + evidence["signature"]["interpretation"]]
        for label, result in (("Pinned case", evidence["pinned"]["result"]), ("Current case", evidence["current"])):
            markdown += ["", "### " + label, "", result["feedback"], "", "Evidence limits: " + result["limitation"],
                         "", "```json", json.dumps({"rows": result["rows"], "inputs": result["inputs"], "provenance": result["provenance"]}, indent=2, default=str), "```"]
    markdown += ["", "## Still open", "", report["physical_validation"],
                 "Stale parts requiring review: " + (", ".join(stale) or "None retained."), "",
                 "## What comes next", "", contract["next_lab"]["question"]]
    return "\n".join(markdown)



def physical_discussion_action(contract, controls, discussion_ui, get_saved, get_pinned, set_discussion):
    def save(_event):
        report = _discussion_report(contract, controls.value, get_saved(), get_pinned(), discussion_ui.value)
        if report["status"] == "complete discussion draft":
            set_discussion(json.dumps(report, indent=2, default=str))
    def report():
        return _discussion_report(contract, controls.value, get_saved(), get_pinned(), discussion_ui.value)
    discussion_download = mo.download(data=lambda: _markdown_report(contract, report()).encode("utf-8"),
        filename=f"physical-ai-{contract['chapter']:02d}-discussion.md", mimetype="text/markdown", label="Download discussion")
    evidence_download = mo.download(data=lambda: json.dumps(report(), indent=2, default=str).encode("utf-8"),
        filename=f"physical-ai-{contract['chapter']:02d}-evidence.json", mimetype="application/json", label="Download evidence JSON")
    return mo.ui.button(label="Save discussion", on_click=save), discussion_download, evidence_download


def physical_discussion(contract, values, saved, pinned, discussion_ui, discussion, get_discussion, set_discussion, save_button, discussion_download, evidence_download):
    """Export exact snapshots and student reasoning; retain final-save distinction."""
    from .notebook import notebook_submission
    report = _discussion_report(contract, values, saved, pinned, discussion)
    current = report["current_evidence"]
    stale = report["stale_part_ids"]
    ready = report["status"] == "complete discussion draft"
    payload = json.dumps(report, indent=2, default=str)
    status = "Discussion saved in this session" if get_discussion() == payload else "Discussion has unsaved changes"
    summary = f"{len(current)} of {len(contract['parts'])} parts have current saved evidence."
    if stale:
        summary += " Revisit changed evidence in parts " + ", ".join(stale) + "."
    return notebook_shell([
        notebook_submission(contract, orientation=contract["chapter"] == 0), mo.md(summary), discussion_ui,
        save_button if ready else mo.ui.button(label="Save discussion", disabled=True),
        discussion_download,
        mo.accordion({"Machine-readable evidence": evidence_download}),
        notebook_status(status, "This session is temporary. Download before closing; browser work does not authorize physical motion."),
        mo.md("**Carry forward:** " + contract.get("narrative", {}).get("carry_forward", contract["deliverable"])),
        mo.md("**What comes next:** " + contract["next_lab"]["title"] + ". " + contract["next_lab"]["question"]),
    ], volume=4)
