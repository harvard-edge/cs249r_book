"""Shared Volume III lab presentation; scenario semantics belong to MLSysIM.

The notebook cells own reactive boundaries. This module owns the common visual
and evidence workflow so chapter entrypoints do not fork interaction logic.
"""
from __future__ import annotations

import json
from copy import deepcopy


def opening(mo, contract, metadata, project_catalog, track):
    """Keep the continuing project ahead of the chapter's learning route."""
    from .notebook import notebook_header, notebook_learning_map, notebook_scenario, notebook_shell
    project = project_catalog[track.value]
    return notebook_shell([
        notebook_header(metadata, contract["parts"][0]["question"]),
        notebook_scenario(contract, volume=3),
        track,
        mo.md(f"**{project['label']}** · {project['mission']}\n\n{project['boundary']}"),
        mo.accordion({"How to use these investigations": mo.md(
            "Start each part with a prediction, choose a different policy, and compare what changes. "
            "Explain the benefit, the cost, and when you would choose differently, then save the comparison. "
            "Use those comparisons in your final discussion.\n\n"
            "The repair-agent cases supply the common reference throughout this volume. "
            "The other projects let you ask how the same decision would change in a different setting. "
            "These are supplied cases, not live agent runs; each part explains what its evidence supports.")}),
        notebook_learning_map(contract),
    ], volume=3)


def project_selector(mo, catalog):
    return mo.ui.dropdown(options={p["label"]: key for key, p in catalog.items()},
                          value=next(iter(catalog.values()))["label"], label="Your agent project", full_width=True)


def investigation_controls(mo, chapter, contract, track, choices, definition=None):
    def prediction_options(part):
        info = definition(chapter, part, track=track) if definition else {}
        return info.get("prediction_options", [
            "The proposed change will improve the declared outcome",
            "It will shift the cost or leave an obligation unresolved",
            "I am uncertain; I need the comparison",
        ])
    predictions = mo.ui.dictionary({p["id"]: mo.ui.radio(prediction_options(p["id"]), label=p["question"])
                                 for p in contract["parts"]})
    # The optional definition argument keeps the helper usable by earlier clients.
    trials = mo.ui.dictionary({p["id"]: mo.ui.dropdown(
        options={label: key for key, label in choices(chapter, p["id"], track=track).items()},
        label="Choose one policy", allow_select_none=True, full_width=True,
    ) for p in contract["parts"]})
    interpretations = mo.ui.dictionary({p["id"]: mo.ui.text_area(
        label="Explain the changed consequence, its cost or limit, and when you would choose differently.",
        full_width=True, debounce=False,
    ) for p in contract["parts"]})
    return predictions, trials, interpretations


def evaluate_trials(chapter, contract, track, prediction_values, trial_values, study):
    return {p["id"]: study(chapter, p["id"], trial_values[p["id"]], track=track)
            for p in contract["parts"] if prediction_values[p["id"]] is not None
            and trial_values[p["id"]] is not None}


def has_policy_contrast(study):
    """A starting-policy replay is useful inspection, but is not an intervention."""
    inputs = study.get("inputs", {})
    return (inputs.get("choice") is not None and inputs.get("baseline") is not None
            and inputs["choice"] != inputs["baseline"])


def evidence_actions(mo, chapter, contract, track, live, predictions, interpretations, set_saved):
    def save(part_id):
        record = {"track": track, "chapter": chapter, "part": part_id,
                  "prediction": predictions[part_id], "study": deepcopy(live[part_id]),
                  "interpretation": interpretations[part_id]}
        set_saved(lambda prior: {**prior, (track, part_id): record})
    return {p["id"]: mo.ui.button(label="Save comparison and interpretation",
            disabled=(p["id"] not in live or not has_policy_contrast(live[p["id"]])
                      or not interpretations[p["id"]].strip()),
            on_click=lambda _, key=p["id"]: save(key)) for p in contract["parts"]}


def current_evidence(saved, track, live, interpretations, predictions):
    return {part: record for (identity, part), record in saved.items()
            if identity == track and part in live and has_policy_contrast(live[part])
            and record["study"] == live[part]
            and record["interpretation"] == interpretations[part] and record["prediction"] == predictions[part]}



def primary_comparison_rows(result):
    """Present the canonical teaching measures without exposing every raw field."""
    start, trial = result["rows"]
    return [{"Measure": label, "Start": start[field], "Trial": trial[field]}
            for label, field in result.get("primary_fields", {}).items()]


def part_workspace(mo, contract, track, predictions, trials, interpretations,
                   live, captures, saved, definition):
    from .notebook import notebook_part, notebook_shell, notebook_table
    views = {}
    for part in contract["parts"]:
        key = part["id"]
        info = definition(contract["chapter"], key, track=track)
        items = [notebook_part(f"{key} · {part['label']}", part["question"])]
        for field in ("scene", "briefing"):
            if info.get(field): items.append(mo.md(info[field]))
        if info.get("fixed"):
            fixed = info["fixed"]
            fixed = "; ".join(str(x) for x in fixed) if isinstance(fixed, list) else str(fixed)
            items.append(mo.md("**Keep fixed:** " + fixed))
        items.append(mo.md("**Your goal:** " + part["goal"]))
        items.append(predictions[key])
        if predictions.value[key] is None:
            items.append(mo.md("Make a prediction to begin exploring. A wrong prediction is useful if the comparison helps explain why."))
        else:
            items.append(trials[key])
        if key in live:
            result = live[key]
            items.append(mo.md("**Your prediction:** " + str(predictions.value[key]) +
                               "\n\nCompare it with the two outcomes below. The trace supports a scoped explanation, not an automatic right-or-wrong grade."))
            if not has_policy_contrast(result):
                items.append(mo.md("**Starting-policy inspection.** Both rows use the same policy. Choose a different policy to investigate a tradeoff before saving this part."))
            items.extend([notebook_table([{"Run": "Start" if i == 0 else "Trial", "Policy": row.get("policy", ""), "Outcome": row.get("outcome", "")} for i, row in enumerate(result["rows"])], caption="Supplied baseline and selected policy"),
                          *([notebook_table(primary_comparison_rows(result), caption="What changed and what stayed fixed")]
                            if result.get("primary_fields") else []),
                          mo.md(result["feedback"]),
                          mo.accordion({"Inspect the selected event trace": notebook_table([{str(k).replace("_", " ").capitalize(): v for k, v in row.items()} for row in result["events"]]),
                                        "Full comparison details": notebook_table(result["rows"]),
                                        "Inspect the baseline event trace": notebook_table(result.get("baseline_events", [])),
                                        "Assumptions and limits": mo.md("\n".join("- " + str(x) for x in [
                                            *([info["coverage"].get("scope", "")] if isinstance(info.get("coverage"), dict)
                                              and info["coverage"].get("scope") else []),
                                            *result["limits"]]))}),
                          mo.md("**Build your explanation:**\n\n"
                                "1. Identify the changed policy and a consequence visible in the event trace.\n"
                                "2. Explain the mechanism and the competing cost, retained duty, or uncertainty.\n"
                                "3. State a condition under which you would choose differently."),
                          *([mo.md("**Investigate:** " + part["explanation_prompt"])]
                            if part.get("explanation_prompt") else []),
                          interpretations[key], captures[key]])
            if info.get("reversal_probe"):
                items.append(mo.md("**Discuss a hypothetical changed condition:** " + str(info["reversal_probe"])))
        record = saved.get((track, key))
        if record:
            current = key in live and record["study"] == live[key] and record["interpretation"] == interpretations.value[key] and record["prediction"] == predictions.value[key]
            items.append(mo.md("**Saved evidence: current.**" if current else
                               "**Saved evidence: retained history.** The live policy or interpretation changed; save again before using it for today's decision."))
        items.append(mo.md("**What comes next:** " + str(info.get("next", part["artifact"]))))
        # Full concept names remain in the route and active part heading.
        # Compact tabs stay reachable without moving the notebook sideways.
        views[f"Part {key}"] = notebook_shell(items, volume=3)
    return notebook_shell([mo.ui.tabs(views)], volume=3)


def discussion_record(contract, track, live, interpretations, predictions, saved, text):
    evidence = current_evidence(saved, track, live, interpretations, predictions)
    required = [part["id"] for part in contract["parts"]]
    missing = [part for part in required if part not in evidence]
    ready = not missing and bool(text.strip())
    return {"volume": 3, "chapter": contract["chapter"], "track": track,
            "status": "assembled; not mastery or deployment approval" if ready else "draft; required evidence or decision missing",
            "required_parts": required, "completed_parts": list(evidence), "missing_parts": missing,
            "evidence": evidence, "decision": text}


def discussion_action(mo, record, set_discussion):
    frozen = json.loads(json.dumps(record))
    return mo.ui.button(label="Save discussion in this session",
                        disabled=bool(record["missing_parts"]) or not record["decision"].strip(),
                        on_click=lambda _: set_discussion(frozen))


def discussion_markdown(contract, record):
    """Build a readable submission from the same captured evidence as JSON."""
    def table(rows):
        if not rows:
            return []
        columns = list(dict.fromkeys(key for row in rows for key in row))
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", "<br>")
        return ["| " + " | ".join(cell(key) for key in columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
                *["| " + " | ".join(cell(row.get(key, "")) for key in columns) + " |" for row in rows]]

    lines = [f"# {contract['title']}", "", f"Project: {record['track']}",
             "", "Status: " + ("Discussion draft with all comparisons" if not record["missing_parts"]
                                 else "Draft; some comparisons are still missing"),
             "", "## Your recommendation", "", record["decision"] or "No recommendation written yet."]
    for part in contract["parts"]:
        key = part["id"]
        lines += ["", f"## Part {key}: {part['label']}", "", part["question"]]
        evidence = record["evidence"].get(key)
        if not evidence:
            lines += ["", "No current saved comparison for this part."]
            continue
        result = evidence["study"]
        lines += ["", "### Your prediction", "", evidence["prediction"],
                  "", "### Comparison", "", *table([
                      {"Run": "Start" if i == 0 else "Trial", "Policy": row.get("policy", ""),
                       "Outcome": row.get("outcome", "")} for i, row in enumerate(result["rows"])
                  ]), ""]
        if result.get("primary_fields"):
            lines += table(primary_comparison_rows(result)) + [""]
        lines += [result.get("feedback", ""), "", "### Your explanation", "", evidence["interpretation"],
                  "", "### Assumptions and limits", "",
                  *["- " + str(limit) for limit in result.get("limits", [])]]
    lines += ["", "## Still open", "", "Parts still needing evidence: " +
              (", ".join(record["missing_parts"]) or "None."),
              "", "These supplied cases support a discussion of the selected policies; they are not live-agent validation.",
              "", "## What comes next", "", contract["next_lab"]["title"],
              "", contract["next_lab"]["question"]]
    return "\n".join(lines)


def final_workspace(mo, contract, decision, record, save_button, saved_discussion):
    from .notebook import notebook_shell, notebook_submission
    ready = not record["missing_parts"] and bool(record["decision"].strip())
    current = saved_discussion == record
    state = "Saved discussion: current in this session." if current else (
        "Discussion changed after saving; save again." if saved_discussion and saved_discussion.get("track") == record["track"] else
        "Discussion has not been explicitly saved in this session.")
    following = contract["next_lab"]
    missing = ", ".join(record["missing_parts"]) or "none"
    return notebook_shell([
        notebook_submission(contract, orientation=contract["chapter"] == 0),
        mo.md(f"**Parts still needing evidence: {missing}.**"),
        decision, save_button, mo.md(state),
        mo.download(data=discussion_markdown(contract, record), filename=f"vol3_lab_{record['chapter']:02}_discussion.md",
                    mimetype="text/markdown", label="Download discussion" if ready else "Download discussion draft"),
        mo.accordion({"Full evidence JSON": mo.download(data=json.dumps(record, indent=2),
            filename=f"vol3_lab_{record['chapter']:02}_discussion.json", mimetype="application/json",
            label="Download full evidence JSON")}),
        mo.md(f"**What comes next:** {following['title']} — {following['question']}\n\n"
              "Saving keeps this discussion in the current session only. Download your work to retain it. A track switch never transfers evidence to another project."),
    ], volume=3)
