"""Opt-in notebook layout for the MLSysBook reference labs.

Presentation only: callers supply simulator outcomes and evidence states.
Explicit layout containers keep compound widget cells on the same measure as
the opening, experiments, and report, without inspecting Marimo's DOM.
"""
from __future__ import annotations

from html import escape
from typing import Any, Mapping, Sequence

import marimo as mo

from .schemas import LabMetadata
from .learning_paths import VOLUME_ACCENTS, normalize_volume, validate_learning_path


# Marimo exposes these theme variables on its document root. Keeping the token
# values at the root also styles native Markdown/sidebar content outside shells;
# component layout rules below remain scoped to the notebook helpers.
_CSS = """<style>
:root{--marimo-heading-font:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--marimo-text-font:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;--marimo-monospace-font:"JetBrains Mono",monospace}
@media(max-width:680px){header.nb-header{padding-top:36px}}

.nb-header,.nb-part,.nb-status,.nb-cards,.nb-progress,.nb-table-wrap,.nb-bridge{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;box-sizing:border-box}
.nb-header{border-bottom:1px solid #d9dee8;padding:8px 0 16px;border-top:3px solid var(--nb-accent,#A51C30)}
.nb-header .nb-eyebrow{font-size:.72rem;font-weight:650;color:#667085;letter-spacing:.06em;text-transform:uppercase;margin:10px 0 6px}
.nb-header h1{font-size:clamp(1.45rem,3vw,1.9rem);line-height:1.2;margin:0 0 8px;color:#172033}
.nb-header p{font-size:1rem;line-height:1.5;margin:0;max-width:72ch;color:#475467}
.nb-topics{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:12px;font-size:.78rem;color:#475467}
.nb-topics span{border-bottom:1px solid #d9dee8;padding-bottom:2px}
.nb-bridge{display:grid;grid-template-columns:1fr 1.2fr 1fr;gap:16px;padding:4px 0 8px}.nb-bridge>div{border-left:2px solid #d9dee8;padding-left:12px;min-width:0}.nb-bridge>div:nth-child(2){border-color:var(--nb-accent,#A51C30)}.nb-bridge h2{font-size:.72rem;font-weight:650;line-height:1.3;color:#667085;margin:0 0 5px}.nb-bridge p{font-size:.84rem;line-height:1.45;color:#344054;margin:0;overflow-wrap:anywhere}@media(max-width:680px){.nb-bridge{grid-template-columns:1fr;gap:10px}.nb-bridge>div{display:grid;grid-template-columns:70px 1fr;gap:8px;padding-left:10px}.nb-bridge h2{margin-top:2px}}
.nb-part{margin:8px 0 2px}.nb-part h2{font-size:1.05rem;font-weight:650;line-height:1.35;margin:0 0 6px;color:#172033}.nb-part p{font-size:.9rem;line-height:1.5;margin:0;max-width:76ch;color:#475467}
.nb-status{border-top:1px solid #d9dee8;padding:12px 0;font-size:.8rem;color:#667085;line-height:1.5}
.nb-status strong{color:#344054;font-weight:650;margin-right:10px}
.nb-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(155px,100%),1fr));gap:8px;margin:4px 0}
.nb-card{background:#f8fafc;border:1px solid #d9dee8;border-radius:6px;padding:10px 12px;min-width:0;overflow-wrap:anywhere}
.nb-card-label{font-size:.72rem;color:#667085;margin-bottom:4px;display:flex;flex-wrap:wrap;justify-content:space-between;gap:4px 8px}.nb-card-state{font-weight:650;color:#344054;white-space:nowrap}.nb-card-value{font-size:.85rem;font-weight:650;line-height:1.4}
.nb-card[data-state="pass"]{border-left:3px solid #247a4d}.nb-card[data-state="fail"]{border-left:3px solid #b42318}.nb-card[data-state="unknown"]{border-left:3px solid #9a5b00}
.nb-progress{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.75rem;color:#667085;padding:4px 0 8px}
.nb-progress span{white-space:nowrap}.nb-progress strong{font-weight:650;color:#344054}
.nb-table-wrap{max-width:100%;overflow-x:auto;border:1px solid #e2e8f0;border-radius:6px}.nb-table-wrap:focus-visible{outline:2px solid var(--nb-accent,#A51C30);outline-offset:2px}
.nb-table{width:100%;border-collapse:collapse;font-size:.8rem;line-height:1.45;text-align:left}
.nb-table th,.nb-table td{min-width:95px;padding:9px 11px;vertical-align:top;border-bottom:1px solid #e2e8f0;overflow-wrap:anywhere}
.nb-table th:first-child,.nb-table td:first-child{min-width:95px}
.nb-table th{background:#f8fafc;color:#475467;font-weight:650}.nb-table tr:last-child td{border-bottom:0}
.nb-table caption{caption-side:bottom;text-align:left;color:#667085;padding:9px 11px;font-size:.75rem}

.nb-learning-map{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;line-height:1.5;border-top:1px solid #d9dee8;padding:14px 0;box-sizing:border-box}
.nb-learning-map h2{font-size:1.05rem;line-height:1.35;margin:0 0 8px;font-weight:650}.nb-learning-map>h2:not(:first-child){margin-top:16px}.nb-learning-map h3{font-size:.88rem;line-height:1.4;margin:16px 0 6px;font-weight:650}.nb-learning-map p,.nb-learning-map li{font-size:.88rem;overflow-wrap:anywhere}.nb-learning-map p{margin:4px 0;max-width:78ch}.nb-learning-map ul{padding-left:22px;margin:4px 0}.nb-learning-map .nb-route{list-style:none;padding:0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:8px 0}.nb-learning-map .nb-route>li{min-width:0;border:1px solid #d9dee8;border-radius:6px;padding:12px}.nb-learning-map .nb-route>li[aria-current="step"]{border-left:3px solid var(--nb-accent,#A51C30)}.nb-learning-map .nb-route h3{margin:0 0 6px}.nb-learning-map .nb-route small{display:block;color:#475467;font-size:.78rem;line-height:1.5;margin-top:5px}.nb-learning-map .nb-current{display:block;font-size:.72rem;color:var(--nb-accent,#A51C30);margin-bottom:4px;font-weight:650}.nb-learning-map .nb-next{display:grid;grid-template-columns:1fr 1fr;gap:12px}.nb-learning-map .nb-next>div{min-width:0}.nb-learning-map code{font-family:"JetBrains Mono",monospace}.nb-learning-map .nb-start{color:#475467}@media(max-width:680px){.nb-learning-map .nb-route,.nb-learning-map .nb-next{grid-template-columns:1fr}}

.nb-scenario,.nb-submission{font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;line-height:1.6;box-sizing:border-box}
.nb-scenario{display:grid;grid-template-columns:minmax(0,1fr) minmax(210px,280px);gap:24px;align-items:start;margin:8px 0 12px}
.nb-scenario h2,.nb-submission h2{font-size:1.15rem;font-weight:650;line-height:1.35;margin:0 0 10px}
.nb-scenario p,.nb-submission p{font-size:.94rem;margin:0 0 10px;max-width:76ch;overflow-wrap:anywhere}
.nb-scenario figure{margin:0;min-width:0}.nb-scenario img{display:block;width:100%;height:auto;aspect-ratio:8/5;object-fit:cover;border-radius:6px}
.nb-scenario figcaption{font-size:.72rem;line-height:1.45;color:#667085;margin-top:6px}.nb-scenario details{font-size:.7rem;margin-top:3px}.nb-scenario summary{cursor:pointer}.nb-scenario a{color:inherit;text-decoration:underline}
.nb-submission{border-top:1px solid #d9dee8;padding:16px 0;margin-top:12px}
@media(max-width:680px){.nb-scenario{grid-template-columns:1fr;gap:10px}.nb-scenario figure{max-width:440px}.nb-scenario img{max-height:180px}.nb-scenario figcaption{max-width:70ch}}
.dark :is(.nb-scenario,.nb-submission){color:#e9ecef}.dark .nb-scenario figcaption{color:#adb5bd}.dark .nb-submission{border-color:#495057}
</style>"""


def notebook_shell(items: Sequence[Any], *, gap: float = 0.75, volume: int | str = 1):
    """Use one outer measure for every rendered cell in a reference notebook."""
    return mo.vstack(items, gap=gap, align="stretch").style({
        "width": "min(1040px, calc(100vw - 64px))", "max-width": "100%", "margin-inline": "auto",
        "min-width": "0", "box-sizing": "border-box",
        "--panel-width": "1040px", "--readable-width": "100%",
        "--nb-accent": VOLUME_ACCENTS[normalize_volume(volume)],
        "--code-font": '"JetBrains Mono",monospace',
        "--monospace-font": '"JetBrains Mono",monospace',
        "--marimo-monospace-font": '"JetBrains Mono",monospace',
        "--heading-font": 'Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
        "--text-font": 'Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
        "--marimo-heading-font": 'Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
        "--marimo-text-font": 'Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
    })


def notebook_header(metadata: LabMetadata, question: str, *,
                    topics: Sequence[str] = (), label: str | None = None):
    """Compact identity and learning question; scenario controls follow separately."""
    tags = "".join(f"<span>{escape(topic)}</span>" for topic in topics)
    identity = f"{metadata.volume} · {label or metadata.chapter}"
    return mo.Html(_CSS + f"""<header class="nb-header">
      <div class="nb-eyebrow">{escape(identity)}</div>
      <h1>{escape(metadata.title)}</h1><p>{escape(question)}</p>
      <div class="nb-topics" aria-label="Concepts to explore">{tags}</div>
    </header>""")


def _items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


def notebook_bridge(previous: str, today: str, next_step: str, *,
                    labels: tuple[str, str, str] = ("Previously", "Goal", "Next")):
    """A familiar project connection, learning goal, and next chapter question.

    This describes the curriculum thread; it never asserts that prior evidence
    was loaded, compatible, or passed. Callers state supplied starts separately.
    """
    if len(labels) != 3:
        raise ValueError("A lab bridge needs three labels")
    parts = "".join(f'<div><h2>{escape(label)}</h2><p>{escape(body)}</p></div>'
                    for label, body in zip(labels, (previous, today, next_step)))
    return mo.Html(_CSS + '<section class="nb-bridge" aria-label="Lab connections">' + parts + '</section>')


def notebook_scenario(contract: Mapping[str, Any], *, track: str | None = None,
                      volume: int | str | None = None, scene: str | None = None):
    """Open with the authored story and a locally packaged illustrative photo.

    The photo identifies the project setting, never its measured implementation.
    Simulator-owned project episodes can supply ``scene`` without copying their
    narrative into a second registry. Images work in native and browser Python.
    """
    import base64
    from importlib import resources
    import json

    number = normalize_volume(volume if volume is not None else contract["volume"])
    story = scene if scene is not None else contract.get("narrative", {}).get("scene", "")
    if not isinstance(story, str) or not story.strip():
        raise ValueError("A lab opening needs an authored scenario")
    if number in (1, 2):
        from .contexts import get_context
        key = get_context(track or "tinyml").track_id
    else:
        key = "agentic" if number == 3 else "physical"
    directory = resources.files("mlsysbook_labs").joinpath("data", "photos")
    photo = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))[key]
    payload = base64.b64encode(directory.joinpath(photo["file"]).read_bytes()).decode("ascii")
    text = lambda value: escape(str(value), quote=True)
    paragraphs = "".join(f"<p>{text(paragraph)}</p>" for paragraph in story.split("\n\n") if paragraph.strip())
    return mo.Html(_CSS + f'''<section class="nb-scenario" aria-label="Your scenario">
      <div class="nb-scenario-story"><h2>Your scenario</h2>{paragraphs}</div>
      <figure><img src="data:image/jpeg;base64,{payload}" alt="{text(photo['alt'])}" width="640" height="400">
        <figcaption>{text(photo['caption'])}<details><summary>Photo credit</summary>
          <a href="{text(photo['source'])}" target="_blank" rel="noopener noreferrer">{text(photo['creator'])}</a>
          · <a href="{text(photo['license_url'])}" target="_blank" rel="noopener noreferrer">{text(photo['license'])}</a>
          <p>{text(photo.get('modifications', ''))}</p>
        </details></figcaption></figure>
    </section>''')


def notebook_submission(contract: Mapping[str, Any], *, orientation: bool = False):
    """Explain the existing export workflow in student language, without gates."""
    if orientation:
        prompt = "Explain which project you chose and what changed when you tried the controls. Keep the download as your own copy. You can start the next lab without it."
    else:
        prompt = "Use your comparisons to explain your choice, its competing cost, and what would change your mind. A decision to wait for more evidence can be a sound conclusion."
    printing = '' if orientation else 'If a printed copy is requested, open the downloaded file in a Markdown viewer and print it.'
    return mo.Html(_CSS + '<section class="nb-submission"><h2>Discuss and submit</h2>'
                   f'<p>{escape(prompt)}</p><p>When your discussion is ready, use the download controls to keep your work. '
                   'Submit it through your course’s usual system, following your instructor’s instructions. '
                   f'{printing}</p></section>')


def notebook_part(title: str, instruction: str):
    """A short experiment question and action, with deliberate heading spacing."""
    return mo.Html(_CSS + f'<section class="nb-part"><h2>{escape(title)}</h2><p>{escape(instruction)}</p></section>')


def notebook_workspace(controls: Any, results: Any, *,
                       controls_title: str = "Change", results_title: str = "Observe"):
    """Responsive linked controls/results using Marimo's native flex layouts."""
    left = mo.vstack([mo.md(f"**{controls_title}**"), *_items(controls)], gap=0.75).style({
        "flex": "1 1 245px", "min-width": "min(245px,100%)", "max-width": "100%",
        "padding": "16px", "background": "#f8fafc", "border": "1px solid #e2e8f0",
        "border-radius": "8px", "box-sizing": "border-box",
    })
    right = mo.vstack([mo.md(f"**{results_title}**"), *_items(results)], gap=0.65).style({
        "flex": "2 1 510px", "min-width": "0", "max-width": "100%",
        "box-sizing": "border-box",
    })
    return mo.hstack([left, right], justify="start", align="start", wrap=True, gap=1.25)


def notebook_status(label: str, detail: str = ""):
    return mo.Html(_CSS + f'<footer class="nb-status"><strong>{escape(label)}</strong>{escape(detail)}</footer>')


def notebook_progress(statuses: Mapping[str, str]):
    if statuses and all(status == "Not saved" for status in statuses.values()):
        return mo.Html(_CSS + '<div class="nb-progress"><span><strong>Evidence</strong> · no comparisons saved yet</span></div>')
    parts = "".join(f"<span><strong>{escape(part)}</strong> · {escape(status)}</span>"
                    for part, status in statuses.items())
    return mo.Html(_CSS + f'<div class="nb-progress" aria-label="Evidence progress">{parts}</div>')


def notebook_cards(items: Sequence[tuple[str, str, str]]):
    cards = []
    for label, value, state in items:
        if state not in {"pass", "fail", "unknown", "neutral"}:
            raise ValueError(f"Unknown presentation state: {state}")
        state_label = {"pass": "Pass", "fail": "Fail", "unknown": "Unknown", "neutral": ""}[state]
        cards.append(f'<div class="nb-card" data-state="{state}"><div class="nb-card-label">{escape(label)}<span class="nb-card-state">{state_label}</span></div>'
                     f'<div class="nb-card-value">{escape(value)}</div></div>')
    return mo.Html(_CSS + '<div class="nb-cards">' + "".join(cards) + '</div>')


def notebook_table(rows: Sequence[Mapping[str, Any]], *, caption: str = ""):
    """Quiet, accessible read-only evidence; interactive tables remain opt-in."""
    if not rows:
        return mo.md(caption or "No comparable results under these conditions.")
    # Event traces may introduce fields after a policy-summary row. Preserve
    # every observed field instead of silently hiding later causal evidence.
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    heading = "".join(f'<th scope="col">{escape(str(key))}</th>' for key in columns)
    body = "".join('<tr>' + "".join(f'<td>{escape(str(row.get(key, "")))}</td>' for key in columns) + '</tr>' for row in rows)
    note = f'<caption>{escape(caption)}</caption>' if caption else ""
    # Wide evidence keeps readable labels; the scoped wrapper supplies scrolling.
    minimum_width = max(540, len(columns) * 112)
    measure = f' style="min-width:{minimum_width}px"' if len(columns) > 3 else ""
    access = f' tabindex="0" role="region" aria-label="{escape(caption or "Scrollable evidence table", quote=True)}"' if len(columns) >= 3 else ""
    return mo.Html(_CSS + f'<div class="nb-table-wrap"{access}><table class="nb-table"{measure}>{note}<thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>')


def notebook_learning_map(contract: Mapping[str, Any], *, current_part: str | None = None,
                          show_narrative: bool = False):
    """A concise visible destination and complete route, never a second project brief.

    The canonical project brief belongs outside this map. Narrative can be opted
    into for standalone use. Study scope and limitations belong at the actual
    instrument via ``notebook_learning_part``, not in an authoring preamble.
    Every objective and route question stays visible without completion gates.
    """
    validate_learning_path(contract)
    if current_part is not None:
        current_part = current_part.removeprefix("Part ")
    part_ids = {part["id"] for part in contract["parts"]}
    if current_part is not None and current_part not in part_ids:
        raise ValueError(f"Unknown current part: {current_part}")
    text = lambda value: escape(str(value))
    accent = VOLUME_ACCENTS[normalize_volume(contract["volume"])]
    narrative = contract.get("narrative", {})
    project = ""
    if show_narrative:
        project = "".join(f'<p>{text(narrative[key])}</p>' for key in ("project", "scene") if narrative.get(key))
        project = f'<h2>Your project</h2>{project}' if project else ""
    objectives = "".join(f'<li>{text(goal)}</li>' for goal in contract["learning_objectives"])
    parts = []
    for part in contract["parts"]:
        current = part["id"] == current_part
        marker = '<span class="nb-current">You are here</span>' if current else ""
        aria = ' aria-current="step"' if current else ""
        # Support gaps are student-facing warnings, not author implementation status.
        gap = f'<small><strong>Study gap:</strong> {text(part["support_gap"])}</small>' if part.get("support_gap") else ""
        parts.append(f'<li{aria}>{marker}<h3>Part {text(part["id"])} · {text(part["label"])}</h3>'
                     f'<p>{text(part["question"])}</p>{gap}</li>')
    prerequisites = " · ".join(text(item) for item in contract["prerequisites"])
    before = f'<p class="nb-start"><strong>Before you begin:</strong> {prerequisites}</p>' if prerequisites else ""
    next_lab = contract["next_lab"]
    return mo.Html(_CSS + f'<section class="nb-learning-map" style="--nb-accent:{accent}" aria-label="Learning goals and lab route">'
                   f'{project}<h2>What you will learn</h2><ul>{objectives}</ul>'
                   f'<p class="nb-start"><strong>Starting point:</strong> {text(contract["starting_point"])}</p>{before}'
                   f'<h3>Your route through this lab</h3><ol class="nb-route">{"".join(parts)}</ol>'
                   f'<h3>What you will hand in</h3><p>{text(contract["deliverable"])}</p>'
                   f'<h3>What comes next</h3><p><strong>Next investigation:</strong> {text(contract["next_part"])}</p>'
                   f'<p><strong>{text(next_lab["title"])}:</strong> {text(next_lab["question"])}</p></section>')


def _learning_part(contract: Mapping[str, Any], part_id: str) -> Mapping[str, Any]:
    """Find a chapter-owned part without deriving a new learning claim."""
    for part in contract.get("parts", ()):
        if part.get("id") == part_id:
            return part
    raise ValueError(f"Unknown learning part: {part_id}")


def notebook_part_reflection(contract: Mapping[str, Any], part_id: str):
    """Causal explanation and reversal prompts, shown after an investigation."""
    part = _learning_part(contract, part_id)
    prompts = []
    for key, label in (("explanation_prompt", "Explain the comparison"), ("reversal_probe", "When would you choose differently?")):
        if part.get(key):
            prompts.append(f'<p><strong>{label}</strong><br>{escape(str(part[key]))}</p>')
    return mo.Html(_CSS + '<section class="nb-part nb-reflection" aria-label="Explain and transfer">' + "".join(prompts) + '</section>')


def notebook_learning_part(contract: Mapping[str, Any], part_id: str, content: Any):
    """Frame one investigation with canonical goal, actual study limits and handoff."""
    part = _learning_part(contract, part_id)
    details = []
    for key, label in (("goal", "Goal"), ("artifact", "Retain"), ("scope", "Study scope"), ("limitation", "Limitations")):
        if part.get(key):
            details.append(f'<p><strong>{label}:</strong> {escape(str(part[key]))}</p>')
    header = mo.Html(_CSS + f'<section class="nb-part"><h2>Part {escape(str(part_id))} · {escape(str(part["label"]))}</h2>'
                     f'<p>{escape(str(part["question"]))}</p>{"".join(details)}</section>')
    items = [header, *_items(content), notebook_part_reflection(contract, part_id)]
    if part.get("next"):
        items.append(notebook_status("Next", str(part["next"])))
    return notebook_shell(items, volume=contract["volume"])


_LIVE_PREDICTION = object()


def notebook_prediction_label(widget: Any, *, value: Any = _LIVE_PREDICTION) -> str:
    """Display a public option label while preserving the stored prediction ID."""
    element = getattr(widget, "element", widget)
    value = widget.value if value is _LIVE_PREDICTION else value
    options = getattr(element, "options", {})
    if isinstance(options, Mapping):
        for label, option_value in options.items():
            if option_value == value:
                return str(label)
    return "Not submitted" if value is None else str(value)


def notebook_tab_hint():
    """A small-screen cue beside native, horizontally scrollable part tabs."""
    return mo.Html(_CSS + '<style>.nb-tab-hint{display:none}@media(max-width:680px){.nb-tab-hint{display:block;font-family:Inter,sans-serif;font-size:.75rem;color:#475467;margin:4px 0}}</style><p class="nb-tab-hint">More parts → Swipe the part labels to see the full route.</p>')


def notebook_download_readiness(*, volume: int | str = 1):
    """Keep an unavailable-download explanation on the shared page measure."""
    return notebook_shell([mo.md("Finish the comparisons and Synthesis to make your discussion download available.")], volume=volume)
