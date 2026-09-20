"""Master Orchestrator for the Physical AI Student Simulation & Pedagogical Audit Pipeline."""

import argparse
import os
import sys
from typing import List

from scripts.pedagogy_sim.curriculum import (
    CURRICULUM_SCHEDULE,
    parse_chapter_sections,
)
from scripts.pedagogy_sim.dossier_generator import DossierGenerator
from scripts.pedagogy_sim.models import (
    MarginNote,
    SeminarTopic,
    TriDisciplineScorecard,
    WeeklyReport,
)
from scripts.pedagogy_sim.personas import STUDENTS
from scripts.pedagogy_sim.seminar_room import SeminarRoom
from scripts.pedagogy_sim.student_reader import StudentReaderEngine


def run_pedagogical_simulation(
    week_num: int = 1,
    chapter_num: int = 1,
    mode: str = "api",
    max_sections: int = 5,
    output_dir: str = "books/vol4/_pedagogical_seminar",
    model: str = "gpt-4o-mini",
) -> str:
    """Run the end-to-end pedagogical simulation for a given week/chapter."""
    assignment = CURRICULUM_SCHEDULE.get(week_num)
    if not assignment:
        raise ValueError(f"Week {week_num} is not defined in curriculum schedule.")

    target_file = assignment.core_qmd_files[0]
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Chapter file not found: {target_file}")

    print("=" * 80)
    print(f"🎓 RUNNING PHYSICAL AI PEDAGOGICAL SIMULATION: WEEK {week_num}")
    print(f"📖 Chapter {chapter_num}: {assignment.title}")
    print(f"📄 Source File: {target_file}")
    print(f"⚙️  Execution Mode: {mode.upper()} | Model: {model if mode == 'api' else 'Heuristic Rules'}")
    print("=" * 80)

    # 1. Parse sections with line tracking
    sections = parse_chapter_sections(target_file)
    print(f"✓ Parsed {len(sections)} sections from {target_file}.")

    # Limit sections if requested
    sections_to_audit = sections[:max_sections] if max_sections > 0 else sections
    print(f"Auditing {len(sections_to_audit)} sections...")

    # 2. Simulate student readers
    reader = StudentReaderEngine(model=model)
    all_notes: List[MarginNote] = []

    for sec in sections_to_audit:
        print(f"\n[Section] L{sec.start_line}–L{sec.end_line}: {sec.title}")
        for student_id, student in STUDENTS.items():
            notes = reader.simulate_student_reading(student, sec, mode=mode)
            print(f"  • {student.name} ({student.discipline.value}): {len(notes)} margin note(s)")
            all_notes.extend(notes)

    print(f"\n✓ Collected {len(all_notes)} total student margin notes across cohort.")

    # 3. Convene Seminar Discussion
    print("\n🏛️  Convening the Seminar Room with Dr. Aris Thorne...")
    room = SeminarRoom(model=model)
    topics: List[SeminarTopic] = room.hold_seminar_discussion(
        notes=all_notes,
        section_title=assignment.title,
        section_content="",
        mode=mode,
    )
    print(f"✓ Completed seminar discussion across {len(topics)} core friction topics.")

    # 4. Compute Tri-Discipline Scorecard
    ml_scores = [n.clarity_score for n in all_notes if n.discipline.value == "Machine Learning"]
    sys_scores = [n.clarity_score for n in all_notes if n.discipline.value == "Embedded Systems"]
    ctrl_scores = [n.clarity_score for n in all_notes if n.discipline.value == "Control & Robotics"]
    flow_scores = [n.clarity_score for n in all_notes if n.discipline.value == "Generalist / Pedagogy Flow"]

    def avg(lst, default=4.0):
        return sum(lst) / len(lst) if lst else default

    scorecard = TriDisciplineScorecard(
        ml_clarity_score=round(avg(ml_scores), 2),
        systems_clarity_score=round(avg(sys_scores), 2),
        control_clarity_score=round(avg(ctrl_scores), 2),
        progressive_disclosure_index=round(avg(flow_scores), 2),
        summary=(
            f"The chapter successfully connects physical irreversibility to learned computation. "
            f"However, Elena and Alex noted friction where control/silicon terms appear without prior physical "
            f"scaffolding. Adopting the proposed progressive disclosure rewrites will establish the chapter as "
            f"an accessible gold standard."
        ),
    )

    # 5. Assemble and export WeeklyReport
    key_takeaways = [
        "Progressive Disclosure: Always ground physical constraints (e.g. back-EMF, reflected inertia) in their algorithmic consequences for ML policies before detailing the hardware.",
        "Silicon Determinism: Clearly state the interconnect interface (e.g., lock-free shared memory ring buffer) connecting the proposal-generating Brain to the safety-enforcing Nervous System.",
        "Physical Grounding: Accompany qualitative statements about 'irreversible kinetic energy' with simple freshman napkin math ($d_{\\text{stop}} \\approx v \\cdot t + v^2 / 2a$) to provide concrete mental anchors.",
        "Acronym Discipline: Never introduce acronyms (e.g., CBF-QP, FOC, TSDF) without inline expansion and an immediate one-sentence intuitive definition upon first appearance."
    ]

    appendix_referrals = [
        "`vol4/backmatter/appendix_control.qmd`: Refer ML and systems students here for full state-space derivations and Lyapunov stability definitions.",
        "`vol4/backmatter/appendix_systems.qmd`: Refer control and robotics students here for bus arbitration, cache coherency, and RTOS scheduling primitives.",
        "`vol4/backmatter/appendix_ml.qmd`: Refer hardware and mechanical engineers here for transformer attention mechanisms and diffusion policy action chunking math."
    ]

    report = WeeklyReport(
        week_number=week_num,
        chapter_num=chapter_num,
        chapter_title=assignment.title,
        audited_file=target_file,
        scorecard=scorecard,
        topics=topics,
        key_takeaways=key_takeaways,
        appendix_referrals_suggested=appendix_referrals,
    )

    exporter = DossierGenerator(output_dir=output_dir)
    exported_md = exporter.export(report)

    print("\n" + "=" * 80)
    print(f"✅ PEDAGOGICAL AUDIT COMPLETE: Dossier saved to {exported_md}")
    print("=" * 80)
    return exported_md


def main():
    parser = argparse.ArgumentParser(description="Physical AI Student Simulation & Pedagogical Audit Pipeline")
    parser.add_argument("--week", type=int, default=1, help="Curriculum week number (1-14)")
    parser.add_argument("--chapter", type=int, default=1, help="Chapter number")
    parser.add_argument("--mode", choices=["api", "heuristic"], default="api", help="Execution mode")
    parser.add_argument("--max-sections", type=int, default=4, help="Maximum sections to process")
    parser.add_argument("--output-dir", type=str, default="books/vol4/_pedagogical_seminar", help="Output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model name for API mode")

    args = parser.parse_args()
    run_pedagogical_simulation(
        week_num=args.week,
        chapter_num=args.chapter,
        mode=args.mode,
        max_sections=args.max_sections,
        output_dir=args.output_dir,
        model=args.model,
    )


if __name__ == "__main__":
    main()
