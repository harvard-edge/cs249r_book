"""Curriculum schedule and chapter text parser for the Physical AI course."""

import os
import re
from typing import Dict, List, NamedTuple, Optional


class WeeklyAssignment(NamedTuple):
    week_number: int
    title: str
    chapter_dirs: List[str]
    core_qmd_files: List[str]
    focus_concepts: List[str]


CURRICULUM_SCHEDULE: Dict[int, WeeklyAssignment] = {
    1: WeeklyAssignment(
        week_number=1,
        title="Kit Bring-Up & The Causal Boundary",
        chapter_dirs=["01_boundary"],
        core_qmd_files=["books/vol4/01_boundary/01_boundary.qmd"],
        focus_concepts=[
            "Causal boundary", "Irreversibility", "Proposal-permission boundary",
            "S·P·A taxonomy", "Stopping distance napkin math", "Moravec's paradox in physical AI"
        ]
    ),
    2: WeeklyAssignment(
        week_number=2,
        title="Actuator Dynamics & Reflected Inertia",
        chapter_dirs=["02_body"],
        core_qmd_files=["books/vol4/02_body/02_body.qmd"],
        focus_concepts=[
            "Reflected rotor inertia (N^2 J_rotor)", "Quasi-direct drive vs harmonic drives",
            "Thermal impedance networks", "Torque-speed curve", "Back-EMF"
        ]
    ),
    3: WeeklyAssignment(
        week_number=3,
        title="The Two-Speed Brain & Multi-Rate Nervous System",
        chapter_dirs=["03_brain", "04_nervous"],
        core_qmd_files=["books/vol4/03_brain/03_brain.qmd", "books/vol4/04_nervous/04_nervous.qmd"],
        focus_concepts=[
            "Two-speed cognitive architecture", "Proposal-permission privilege split",
            "Lock-free seqlocks", "Clock synchronization", "Deterministic fieldbuses"
        ]
    ),
    4: WeeklyAssignment(
        week_number=4,
        title="Embodied Data Foundations & Teleoperation",
        chapter_dirs=["05_data"],
        core_qmd_files=["books/vol4/05_data/05_data.qmd"],
        focus_concepts=[
            "Compounding covariate shift", "Multimodal time synchronization",
            "Bilateral teleoperation latency", "Action space representations"
        ]
    ),
    5: WeeklyAssignment(
        week_number=5,
        title="Policy Synthesis Regimes & Closed-Loop Survival",
        chapter_dirs=["06_training", "07_evaluation"],
        core_qmd_files=["books/vol4/06_training/06_training.qmd", "books/vol4/07_evaluation/07_evaluation.qmd"],
        focus_concepts=[
            "Imitation learning (ACT/Diffusion Policy)", "Error compounding O(T^2 eps)",
            "Butler & Finelli exposure wall", "Sim-to-real gap"
        ]
    ),
    6: WeeklyAssignment(
        week_number=6,
        title="Perception & 3D Spatial Encoders",
        chapter_dirs=["08_perception"],
        core_qmd_files=["books/vol4/08_perception/08_perception.qmd"],
        focus_concepts=[
            "Photon-to-token latency waterfall", "Rolling shutter distortion",
            "IMU preintegration", "3D scene representations"
        ]
    ),
    7: WeeklyAssignment(
        week_number=7,
        title="Spatial Memory & Intent Grounding",
        chapter_dirs=["09_memory", "10_intent"],
        core_qmd_files=["books/vol4/09_memory/09_memory.qmd", "books/vol4/10_intent/10_intent.qmd"],
        focus_concepts=[
            "SE(3) coordinate frames", "Volumetric occupancy & TSDFs",
            "VLM spatial grounding", "Expiring intent leases"
        ]
    ),
    8: WeeklyAssignment(
        week_number=8,
        title="Trajectory Planning & Seam Continuity",
        chapter_dirs=["11_planning"],
        core_qmd_files=["books/vol4/11_planning/11_planning.qmd"],
        focus_concepts=[
            "Receding horizon control", "C^2 jerk-bounded splines",
            "Chunk seam blending", "Stopping contingency suffixes"
        ]
    ),
    9: WeeklyAssignment(
        week_number=9,
        title="The Signature Enforcer & Heterogeneous Placement",
        chapter_dirs=["12_enforcement", "13_placement"],
        core_qmd_files=["books/vol4/12_enforcement/12_enforcement.qmd", "books/vol4/13_placement/13_placement.qmd"],
        focus_concepts=[
            "Control Barrier Functions (CBF-QP)", "Minimal intervention safety shields",
            "AXI bus contention", "Inductive voltage droop L dI/dt"
        ]
    ),
    10: WeeklyAssignment(
        week_number=10,
        title="Human Intervention & Shared Autonomy",
        chapter_dirs=["14_intervention"],
        core_qmd_files=["books/vol4/14_intervention/14_intervention.qmd"],
        focus_concepts=[
            "Shared control arbitration", "Bumpless control transfer",
            "Takeover latency dynamics", "Authority handshakes"
        ]
    ),
    11: WeeklyAssignment(
        week_number=11,
        title="Verification & Synthetic Falsification",
        chapter_dirs=["15_verification"],
        core_qmd_files=["books/vol4/15_verification/15_verification.qmd"],
        focus_concepts=[
            "4-stage qualification ladder (Sim->PIL->HIL->In-Situ)",
            "Temporal logic falsification", "Fault injection stress testing"
        ]
    ),
    12: WeeklyAssignment(
        week_number=12,
        title="Release Assurance & Safety Cases",
        chapter_dirs=["16_release"],
        core_qmd_files=["books/vol4/16_release/16_release.qmd"],
        focus_concepts=[
            "Claim-Argument-Evidence (CAE)", "Goal Structuring Notation (GSN)",
            "UL 4600 / ISO 26262 alignment", "Cryptographic audit ledgers"
        ]
    ),
    13: WeeklyAssignment(
        week_number=13,
        title="The Physical Intelligence Frontier",
        chapter_dirs=["17_frontier"],
        core_qmd_files=["books/vol4/17_frontier/17_frontier.qmd"],
        focus_concepts=[
            "Epistemic uncertainty in the physical world", "Long-tail generalization",
            "Unsolved challenges in physical AI systems"
        ]
    ),
    14: WeeklyAssignment(
        week_number=14,
        title="Capstone Defense & Fault Defense Jury",
        chapter_dirs=["17_frontier"],
        core_qmd_files=["books/vol4/17_frontier/17_frontier.qmd"],
        focus_concepts=[
            "Live fault injection defense", "Safe holding state transition",
            "Full-stack physical AI verification"
        ]
    ),
}


class TextSection(NamedTuple):
    title: str
    start_line: int
    end_line: int
    content: str
    level: int  # 1 for #, 2 for ##, 3 for ###


def parse_chapter_sections(file_path: str) -> List[TextSection]:
    """Parse a Quarto .qmd file into structured sections with exact line tracking."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Chapter file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections: List[TextSection] = []
    current_title = "Preamble / Frontmatter"
    current_start = 1
    current_lines = []
    in_code_block = False
    header_re = re.compile(r"^(#{1,3})\s+(.+)$")

    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        match = header_re.match(line) if not in_code_block else None
        if match:
            # End previous section if it has content
            if current_lines:
                sections.append(
                    TextSection(
                        title=current_title,
                        start_line=current_start,
                        end_line=i - 1,
                        content="".join(current_lines),
                        level=current_level,
                    )
                )
            hashes, title_text = match.groups()
            current_level = len(hashes)
            # Clean title (remove pandoc labels {#sec-...} and classes {.unnumbered})
            clean_title = re.sub(r"\{[^}]*\}", "", title_text).strip()
            current_title = clean_title
            current_start = i
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            TextSection(
                title=current_title,
                start_line=current_start,
                end_line=len(lines),
                content="".join(current_lines),
                level=current_level,
            )
        )

    return sections


def mask_code_blocks(text: str) -> tuple[str, Dict[str, str]]:
    """Extract all ```...``` code blocks and replace with protected tokens.
    Guarantees that simulation agents NEVER see or edit mlsysim / python code blocks!"""
    blocks: Dict[str, str] = {}
    counter = 0

    def repl(match):
        nonlocal counter
        counter += 1
        placeholder = f"<!-- PROTECTED_CODE_BLOCK_{counter:04d} -->"
        blocks[placeholder] = match.group(0)
        return placeholder

    masked_text = re.sub(r"```.*?```", repl, text, flags=re.DOTALL)
    return masked_text, blocks


def unmask_code_blocks(text: str, blocks: Dict[str, str]) -> str:
    """Restore all original code blocks byte-for-byte into their placeholders."""
    for placeholder, original in blocks.items():
        text = text.replace(placeholder, original)
    return text

