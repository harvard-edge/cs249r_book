"""
Dual-format renderer — outputs to both raw markdown and corpus JSON.

Two output paths:
1. MARKDOWN: Append to the existing .md files (human-readable, GitHub browsable)
2. JSON: Rebuild corpus.json (consumed by the StaffML Next.js app)

The markdown format matches the exact <details> template used across all
track files. The JSON format matches build_corpus.py's output schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .schemas import Question, LEVEL_META


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown(question: Question) -> str:
    """Render a Question to the exact markdown format used in track files."""
    meta = LEVEL_META.get(question.level, LEVEL_META["L3"])
    badge_label = meta["label"].value
    badge_color = meta["color"].value
    badge_alt = meta["alt"]

    # Topic tags
    topic_tags = " ".join(f"<code>{t.strip()}</code>" for t in question.topic.split(","))

    # Build the inner content
    inner_parts = []

    inner_parts.append(
        f"  **Common Mistake:** {question.common_mistake}"
    )
    inner_parts.append("")
    inner_parts.append(
        f"  **Realistic Solution:** {question.realistic_solution}"
    )

    if question.napkin_math:
        inner_parts.append("")
        inner_parts.append(f"  > **Napkin Math:** {question.napkin_math}")

    if question.key_equation:
        inner_parts.append("")
        inner_parts.append(f"  > **Key Equation:** {question.key_equation}")

    if question.options:
        inner_parts.append("")
        inner_parts.append("  > **Options:**")
        for opt in question.options:
            marker = "[x]" if opt.is_correct else "[ ]"
            inner_parts.append(f"  > {marker} {opt.text}")

    if question.deep_dive_title and question.deep_dive_url:
        inner_parts.append("")
        inner_parts.append(
            f"  📖 **Deep Dive:** [{question.deep_dive_title}]({question.deep_dive_url})"
        )

    inner_content = "\n".join(inner_parts)

    return f"""<details>
<summary><b><img src="https://img.shields.io/badge/Level-{badge_label}-{badge_color}?style=flat-square" alt="{badge_alt}" align="center"> {question.title}</b> · {topic_tags}</summary>

- **Interviewer:** "{question.scenario}"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

{inner_content}
  </details>
</details>"""


def render_corpus_entry(question: Question) -> dict:
    """Render a Question to the corpus.json format for StaffML."""
    topic_slug = question.topic.split(",")[0].strip()
    title_slug = question.title.lower().replace(" ", "-")[:20]

    entry = {
        "id": f"{question.track}-{topic_slug}-{title_slug}",
        "track": question.track,
        "scope": _track_to_scope(question.track),
        "level": question.level,
        "title": question.title,
        "topic": topic_slug,
        "scenario": question.scenario,
        "details": {
            "common_mistake": question.common_mistake,
            "realistic_solution": question.realistic_solution,
        },
    }

    if question.napkin_math:
        entry["details"]["napkin_math"] = question.napkin_math

    if question.deep_dive_title:
        entry["details"]["deep_dive_title"] = question.deep_dive_title
    if question.deep_dive_url:
        entry["details"]["deep_dive_url"] = question.deep_dive_url

    if question.options:
        entry["details"]["options"] = [opt.text for opt in question.options]
        entry["details"]["correct_index"] = next(
            (i for i, opt in enumerate(question.options) if opt.is_correct), -1
        )

    return entry


def _track_to_scope(track: str) -> str:
    """Map track name to a human-readable scope."""
    return {
        "cloud": "Cloud",
        "edge": "Edge",
        "mobile": "Mobile",
        "tinyml": "TinyML",
        "foundations": "Foundations",
    }.get(track, track.title())


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

# Map track + topic keywords to the appropriate markdown file.
# Each track has multiple scope files organized by system layer.
TRACK_FILE_MAP: dict[str, dict[str, str]] = {
    "cloud": {
        "default": "cloud/01_single_machine.md",
        # Route to specific files by topic keywords
        "parallelism": "cloud/02_distributed_systems.md",
        "distributed": "cloud/02_distributed_systems.md",
        "allreduce": "cloud/02_distributed_systems.md",
        "zero": "cloud/02_distributed_systems.md",
        "tensor-parallel": "cloud/02_distributed_systems.md",
        "pipeline-parallel": "cloud/02_distributed_systems.md",
        "serving": "cloud/03_serving_stack.md",
        "latency": "cloud/03_serving_stack.md",
        "kv-cache": "cloud/03_serving_stack.md",
        "batching": "cloud/03_serving_stack.md",
        "ttft": "cloud/03_serving_stack.md",
        "queueing": "cloud/03_serving_stack.md",
        "drift": "cloud/04_production_ops.md",
        "mlops": "cloud/04_production_ops.md",
        "feature-store": "cloud/04_production_ops.md",
        "canary": "cloud/04_production_ops.md",
        "monitoring": "cloud/04_production_ops.md",
    },
    "edge": {
        "default": "edge/01_hardware_platform.md",
        "pipeline": "edge/02_realtime_pipeline.md",
        "realtime": "edge/02_realtime_pipeline.md",
        "camera": "edge/02_realtime_pipeline.md",
        "lidar": "edge/02_realtime_pipeline.md",
        "sensor": "edge/02_realtime_pipeline.md",
        "fleet": "edge/03_deployed_system.md",
        "ota": "edge/03_deployed_system.md",
        "firmware": "edge/03_deployed_system.md",
        "degradation": "edge/03_deployed_system.md",
        "watchdog": "edge/03_deployed_system.md",
    },
    "mobile": {
        "default": "mobile/01_device_hardware.md",
        "app": "mobile/02_app_experience.md",
        "anr": "mobile/02_app_experience.md",
        "jank": "mobile/02_app_experience.md",
        "ux": "mobile/02_app_experience.md",
        "store": "mobile/03_ship_and_update.md",
        "update": "mobile/03_ship_and_update.md",
        "download": "mobile/03_ship_and_update.md",
        "version": "mobile/03_ship_and_update.md",
        "a-b-test": "mobile/03_ship_and_update.md",
    },
    "tinyml": {
        "default": "tinyml/01_microcontroller.md",
        "sensor": "tinyml/02_sensing_pipeline.md",
        "audio": "tinyml/02_sensing_pipeline.md",
        "mfcc": "tinyml/02_sensing_pipeline.md",
        "fft": "tinyml/02_sensing_pipeline.md",
        "microphone": "tinyml/02_sensing_pipeline.md",
        "deploy": "tinyml/03_deployed_device.md",
        "ota": "tinyml/03_deployed_device.md",
        "field": "tinyml/03_deployed_device.md",
        "battery": "tinyml/03_deployed_device.md",
        "duty": "tinyml/03_deployed_device.md",
    },
    "foundations": {
        "default": "foundations.md",
    },
}

# Level markers used in the markdown section headers (cleaned format)
LEVEL_SECTION_MARKERS: dict[str, str] = {
    "L1": "#### 🟢 L1/L2",
    "L2": "#### 🟢 L1/L2",
    "L3": "#### 🟢 L3",
    "L4": "#### 🔵 L4",
    "L5": "#### 🟡 L5",
    "L6+": "#### 🔴 L6+",
}


def get_target_file(track: str, topic: str) -> str:
    """Determine which markdown file to write to based on track and topic."""
    track_files = TRACK_FILE_MAP.get(track, {"default": "foundations.md"})

    # Check if any topic keyword matches
    for keyword, filepath in track_files.items():
        if keyword != "default" and keyword in topic:
            return filepath

    return track_files["default"]


def _find_insertion_point(content: str, level: str) -> int:
    """Find the correct insertion point for a question at the given level.

    Looks for the level's section header (e.g., "#### 🔵 L4") and finds
    the end of that section (the next section header or file end).
    The question is inserted just before the next section boundary.

    Falls back to EOF if the section isn't found.
    """
    lines = content.split("\n")
    level_marker = LEVEL_SECTION_MARKERS.get(level, "")

    if not level_marker:
        return len(content)

    # Find the section for this level
    section_start = -1
    for i, line in enumerate(lines):
        if level_marker in line:
            section_start = i
            break

    if section_start == -1:
        # Section doesn't exist — append to EOF
        return len(content)

    # Find the end of this section (next #### or ### or --- boundary)
    section_end = len(lines)
    for i in range(section_start + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("####") or line.startswith("###") or line == "---":
            section_end = i
            break

    # Walk backward from section_end to find the last </details> tag
    insert_line = section_end
    for i in range(section_end - 1, section_start, -1):
        if "</details>" in lines[i]:
            insert_line = i + 1
            break

    # Convert line number to character offset
    offset = sum(len(lines[j]) + 1 for j in range(insert_line))
    return offset


def append_to_markdown_file(
    question: Question,
    base_dir: Optional[Path] = None,
) -> Path:
    """Insert a question into the correct section of the target markdown file.

    Finds the right file (by track + topic routing), then the right section
    (by level header), and inserts the question at the end of that section.
    Falls back to EOF append if the section structure isn't found.

    Returns the path of the file that was modified.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    target_rel = get_target_file(question.track, question.topic)
    target_path = base_dir / target_rel

    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    content = target_path.read_text(encoding="utf-8")
    md = render_markdown(question)

    # Try to insert at the correct section
    insert_pos = _find_insertion_point(content, question.level)

    if insert_pos >= len(content):
        # Fallback: append to EOF
        new_content = content.rstrip() + "\n\n" + md + "\n"
    else:
        # Insert within the correct section
        new_content = content[:insert_pos] + "\n" + md + "\n\n" + content[insert_pos:]

    target_path.write_text(new_content, encoding="utf-8")
    return target_path
