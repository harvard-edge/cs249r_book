"""Generates comprehensive weekly pedagogical dossiers and revision matrices."""

import json
import os
from typing import Optional
from scripts.pedagogy_sim.models import WeeklyReport


class DossierGenerator:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or "books/vol4/_pedagogical_seminar"
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, report: WeeklyReport) -> str:
        """Export both Markdown and JSON dossiers, returning the Markdown file path."""
        prefix = f"week_{report.week_number:02d}_ch{report.chapter_num:02d}"
        md_filename = f"{prefix}_pedagogical_seminar_audit.md"
        json_filename = f"{prefix}_pedagogical_seminar_audit.json"

        md_path = os.path.join(self.output_dir, md_filename)
        json_path = os.path.join(self.output_dir, json_filename)

        # 1. Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        # 2. Write Markdown
        md_content = self._render_markdown(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return md_path

    def _render_markdown(self, report: WeeklyReport) -> str:
        sc = report.scorecard
        
        md = f"""# Weekly Pedagogical Seminar Audit: Week {report.week_number} (Chapter {report.chapter_num})

**Book**: *Physical AI: Machine Learning Systems That Sense and Act* (Volume IV)  
**Author**: Prof. Vijay Janapa Reddi (Harvard University)  
**Audited Chapter**: `{report.audited_file}` — **{report.chapter_title}**  
**Seminar Simulation Cohort**:
- **Alex Chen** (Machine Learning & Foundation Models)
- **Priya Patel** (Embedded Systems & Silicon Architecture)
- **Marcus Vance** (Robotics Dynamics & Classical Control)
- **Elena Rostova** (Undergraduate Generalist & Progressive Disclosure Guardian)
**Seminar Moderator**: Dr. Aris Thorne (Lead Teaching Assistant & Pedagogical Synthesizer)  
**Audited Lines**: Full Chapter Scan  
**Overall Status**: 🟡 Ready with Progressive Disclosure Tweaks

---

## 1. Executive Summary & Pedagogical Vision

This audit captures the collective reading experience and seminar discussion of four diverse student learners reading Chapter {report.chapter_num} (*{report.chapter_title}*). In Volume IV, machine learning ceases to be symbolic computation behind glass and acquires kinetic momentum. The central engineering challenge is coordinating three traditionally siloed cultures:
1. **Machine Learning**: high-capacity, stochastic, latency-variable generative models.
2. **Embedded Systems**: deterministic, memory-bandwidth-bounded, real-time silicon pipelines.
3. **Control & Robotics**: physical dynamics, non-negotiable Newton-Euler mechanics, and irreversible work.

Our simulated student cohort read this chapter line-by-line, recorded margin notes reflecting their individual disciplinary backgrounds, and met in a simulated seminar room to debate what made intuitive sense, where progressive disclosure broke down, and how the narrative can be refined into the undisputed gold standard for physical AI education.

---

## 2. Tri-Discipline Balance Scorecard

| Disciplinary Dimension | Rating (1–5) | Student Evaluator | Status | Evaluator Commentary |
|:---|:---:|:---|:---:|:---|
| **Machine Learning Foundations** | **{sc.ml_clarity_score:.1f} / 5.0** | Alex Chen | {'🟢 Strong' if sc.ml_clarity_score >= 4.0 else '🟡 Needs Bridge'} | Grounds models in physical latency and real-time execution constraints. |
| **Embedded & Silicon Systems** | **{sc.systems_clarity_score:.1f} / 5.0** | Priya Patel | {'🟢 Strong' if sc.systems_clarity_score >= 4.0 else '🟡 Needs Bridge'} | Real-time buses, memory barriers, and proposal-permission boundaries. |
| **Control Theory & Robotics Mechanics** | **{sc.control_clarity_score:.1f} / 5.0** | Marcus Vance | {'🟢 Strong' if sc.control_clarity_score >= 4.0 else '🟡 Needs Bridge'} | Kinetic energy stopping bounds, reflected inertia, and motor dynamics. |
| **Progressive Disclosure Index** | **{sc.progressive_disclosure_index:.1f} / 5.0** | Elena Rostova | {'🟢 Seamless' if sc.progressive_disclosure_index >= 4.0 else '🟡 Actionable Gaps'} | Acronyms, pedagogical ordering, cognitive load, and visual grounding. |

> **Pedagogical Assessment Summary**:  
> {sc.summary}

---

## 3. Seminar Round-Table Discussions & Identified Topics
"""

        for topic in report.topics:
            p_badge = "🔴 **[P0 - BLOCKER]**" if topic.priority.value == "P0_BLOCKER" else ("🟡 **[P1 - SIGNIFICANT]**" if topic.priority.value == "P1_SIGNIFICANT" else "💡 **[P2 - POLISH]**")

            md += f"""
### Topic {topic.topic_id}: {topic.line_range} ({p_badge})

**Text Passage Excerpt**:
> *"{topic.text_snippet}"*

#### Student Margin Notes Before Seminar
"""
            for n in topic.student_notes:
                md += f"- **{n.student_name}** ({n.discipline.value} · Clarity {n.clarity_score}/5):\n"
                md += f"  - *Margin Note*: {n.note}\n"
                if n.proposed_bridge:
                    md += f"  - *Initial Wish*: {n.proposed_bridge}\n"

            md += "\n#### Seminar Room Discussion Transcript\n"
            for t in topic.discussion:
                target_str = f" *(addressing {t.target_speaker})*" if t.target_speaker else ""
                md += f"> **{t.speaker}** ({t.speaker_role}){target_str}:  \n> \"{t.text}\"\n>\n"

            md += f"""
#### Consensus Verdict & Progressive Disclosure Fix
- **Consensus**: {topic.consensus_verdict}
- **Rationale**: {topic.rationale}

```diff
# Current Text (Line {topic.line_range})
- {topic.original_text}

# Proposed Progressive Disclosure Revision
+ {topic.proposed_rewrite}
```

---
"""

        md += """
## 4. Synthesis of Key Takeaways for the Author

"""
        for takeaway in report.key_takeaways:
            md += f"- {takeaway}\n"

        md += "\n## 5. Recommended Appendix Cross-References\n\n"
        if report.appendix_referrals_suggested:
            for ref in report.appendix_referrals_suggested:
                md += f"- {ref}\n"
        else:
            md += "- No appendix overflow required; all concepts remain cleanly contained within the chapter's progressive disclosure budget.\n"

        return md
