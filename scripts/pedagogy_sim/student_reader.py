"""Individual student reading simulation engine."""

import json
import os
import re
from typing import List, Optional
from scripts.pedagogy_sim.curriculum import TextSection
from scripts.pedagogy_sim.models import (
    ConfusionCategory,
    Discipline,
    MarginNote,
)
from scripts.pedagogy_sim.personas import STUDENTS, StudentPersona


class StudentReaderEngine:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def simulate_student_reading(
        self,
        student: StudentPersona,
        section: TextSection,
        mode: str = "api",
    ) -> List[MarginNote]:
        """Simulate a student reading a section and annotating line-by-line."""
        if mode == "api" and self.client:
            try:
                return self._read_with_llm(student, section)
            except Exception as e:
                print(f"[Warning] LLM call failed for {student.name}: {e}. Falling back to heuristic reader.")
                return self._read_with_heuristics(student, section)
        else:
            return self._read_with_heuristics(student, section)

    def _read_with_llm(self, student: StudentPersona, section: TextSection) -> List[MarginNote]:
        """Call LLM with student persona system prompt and structured JSON output schema."""
        lines = section.content.splitlines()
        annotated_text = "\n".join([f"L{section.start_line + idx}: {line}" for idx, line in enumerate(lines)])

        user_prompt = (
            f'Read the following section from the textbook "Physical AI: Machine Learning Systems That Sense and Act".\n\n'
            f'Section Title: {section.title}\n'
            f'Lines: {section.start_line} to {section.end_line}\n\n'
            + annotated_text
            + f"\n\n---\nTASK:\nAs {student.name} ({student.discipline.value}), evaluate this passage line-by-line for Progressive Disclosure and Tri-Discipline clarity.\n"
            "Flag 1 to 3 specific places where:\n"
            "1. Jargon or acronyms appear without definition or grounding.\n"
            "2. The text takes an abrupt cognitive leap or skips intuitive stepping stones.\n"
            "3. Concepts from other disciplines (ML, Systems, Control/Robotics) are used without sufficient bridge for a student of your background.\n"
            "4. Physical consequences, silicon limits, or algorithmic implications are hand-waved.\n\n"
            "Format your response strictly as a JSON object matching this schema:\n"
            "{\n"
            '  "notes": [\n'
            "    {\n"
            '      "line_start": <int>,\n'
            '      "line_end": <int>,\n'
            '      "text_snippet": "<exact short quote from text>",\n'
            '      "clarity_score": <int 1-5, where 1=baffled, 3=moderate friction, 5=crystal clear>,\n'
            '      "category": "<UNANNOUNCED_TERM | MISSING_PHYSICAL_GROUNDING | UNJUSTIFIED_SYSTEMS_ASSUMPTION | CONTROL_STABILITY_VOID | COGNITIVE_LEAP | INVERTED_ORDER | EXCESSIVE_DENSITY>",\n'
            '      "note": "<Your personal margin thought in your disciplinary voice>",\n'
            '      "proposed_bridge": "<How you wish the author bridged or explained this for progressive disclosure>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        from scripts.pedagogy_sim.llm_bridge import call_llm_json

        raw_json = call_llm_json(
            system_prompt=student.system_prompt(),
            user_prompt=user_prompt,
            model=self.model,
            temperature=0.4,
        )
        if not raw_json:
            return self._read_with_heuristics(student, section)
        notes = []
        for item in raw_json.get("notes", []):
            try:
                note = MarginNote(
                    line_start=int(item.get("line_start", section.start_line)),
                    line_end=int(item.get("line_end", section.start_line)),
                    text_snippet=item.get("text_snippet", ""),
                    student_id=student.student_id,
                    student_name=student.name,
                    discipline=student.discipline,
                    clarity_score=int(item.get("clarity_score", 3)),
                    category=ConfusionCategory(item.get("category", "COGNITIVE_LEAP")),
                    note=item.get("note", ""),
                    proposed_bridge=item.get("proposed_bridge", ""),
                )
                notes.append(note)
            except Exception as ex:
                continue
        return notes

    def _read_with_heuristics(self, student: StudentPersona, section: TextSection) -> List[MarginNote]:
        """Offline deterministic heuristic reader that detects progressive disclosure friction."""
        lines = section.content.splitlines()
        notes = []

        # Heuristic triggers tailored to each persona
        if student.student_id == "elena":
            # Progressive disclosure & unannounced acronyms
            acronym_patterns = [
                (r"\bCBF-QP\b", "CBF-QP", ConfusionCategory.UNANNOUNCED_TERM, "Control Barrier Function Quadratic Program"),
                (r"\bFOC\b", "FOC", ConfusionCategory.UNANNOUNCED_TERM, "Field-Oriented Control"),
                (r"\bTSDF\b", "TSDF", ConfusionCategory.UNANNOUNCED_TERM, "Truncated Signed Distance Function"),
                (r"\bSE\(3\)\b", "SE(3)", ConfusionCategory.UNANNOUNCED_TERM, "Special Euclidean group in 3D"),
                (r"\bMIPI CSI-2\b", "MIPI CSI-2", ConfusionCategory.UNANNOUNCED_TERM, "Mobile Industry Processor Interface Camera Serial Interface 2"),
                (r"\bQDD\b", "QDD", ConfusionCategory.UNANNOUNCED_TERM, "Quasi-Direct Drive actuator"),
            ]
            for idx, line in enumerate(lines):
                cur_line_num = section.start_line + idx
                for pat, term, cat, expansion in acronym_patterns:
                    if re.search(pat, line) and not (r"\index{" in line or "dfn-" in line):
                        notes.append(
                            MarginNote(
                                line_start=cur_line_num,
                                line_end=cur_line_num,
                                text_snippet=line[:80].strip(),
                                student_id=student.student_id,
                                student_name=student.name,
                                discipline=student.discipline,
                                clarity_score=2,
                                category=cat,
                                note=f"The acronym '{term}' is used here without expansion or prior motivation. As an undergrad, I had to stop and google what this stands for.",
                                proposed_bridge=f"Explicitly write out '{expansion} ({term})' upon first mention, with a 1-sentence intuitive definition of its role.",
                            )
                        )
                        break

        elif student.student_id == "alex":
            # ML lens: hardware or physics dumped without algorithmic intuition
            for idx, line in enumerate(lines):
                cur_line_num = section.start_line + idx
                if any(w in line.lower() for w in ["back-emf", "winding resistance", "reflected rotor inertia", "phase dissipation"]):
                    notes.append(
                        MarginNote(
                            line_start=cur_line_num,
                            line_end=cur_line_num,
                            text_snippet=line[:80].strip(),
                            student_id=student.student_id,
                            student_name=student.name,
                            discipline=student.discipline,
                            clarity_score=2,
                            category=ConfusionCategory.MISSING_PHYSICAL_GROUNDING,
                            note="This physics formula is dropped without explaining the algorithmic consequence. Why does back-EMF matter to my policy output?",
                            proposed_bridge="Connect the physical limit directly to the policy action space: 'Because back-EMF opposes current as speed rises, an ML policy commanding maximum torque at high velocity will clip against actuator limits.'",
                        )
                    )
                    break

        elif student.student_id == "priya":
            # Systems lens: hand-wavy claims about real-time, buses, or memory
            for idx, line in enumerate(lines):
                cur_line_num = section.start_line + idx
                if any(w in line.lower() for w in ["real-time", "proposal", "permission", "lock-free"]):
                    notes.append(
                        MarginNote(
                            line_start=cur_line_num,
                            line_end=cur_line_num,
                            text_snippet=line[:80].strip(),
                            student_id=student.student_id,
                            student_name=student.name,
                            discipline=student.discipline,
                            clarity_score=3,
                            category=ConfusionCategory.UNJUSTIFIED_SYSTEMS_ASSUMPTION,
                            note="The text mentions proposals crossing to the safety supervisor, but doesn't specify the interconnect mechanism or bus arbitration.",
                            proposed_bridge="Specify the IPC bus interface (e.g. shared memory ring buffer with memory barriers, or SPI/EtherCAT) and the timing guarantee.",
                        )
                    )
                    break

        elif student.student_id == "marcus":
            # Robotics/Control lens: irreversible energy, dynamics, stopping bounds
            for idx, line in enumerate(lines):
                cur_line_num = section.start_line + idx
                if any(w in line.lower() for w in ["irreversibility", "stopping distance", "kinetic energy"]):
                    notes.append(
                        MarginNote(
                            line_start=cur_line_num,
                            line_end=cur_line_num,
                            text_snippet=line[:80].strip(),
                            student_id=student.student_id,
                            student_name=student.name,
                            discipline=student.discipline,
                            clarity_score=3,
                            category=ConfusionCategory.MISSING_PHYSICAL_GROUNDING,
                            note="The prose talks qualitatively about irreversible kinetic energy, but doesn't state the concrete braking equation.",
                            proposed_bridge="Provide the freshman napkin-math stopping distance equation: $d_{\\text{stop}} = v \\cdot t_{\\text{react}} + \\frac{v^2}{2 a_{\\max}}$ to ground the intuition.",
                        )
                    )
                    break

        return notes[:2]
