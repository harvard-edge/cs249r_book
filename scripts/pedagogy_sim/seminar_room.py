"""Seminar room simulation: student peer discussion and pedagogical consensus."""

import json
import os
from typing import Dict, List, Optional
from scripts.pedagogy_sim.models import (
    DiscussionTurn,
    MarginNote,
    Priority,
    SeminarTopic,
)
from scripts.pedagogy_sim.personas import MODERATOR_PROMPT, STUDENTS


class SeminarRoom:
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

    def hold_seminar_discussion(
        self,
        notes: List[MarginNote],
        section_title: str,
        section_content: str,
        mode: str = "api",
    ) -> List[SeminarTopic]:
        """Run the collective seminar discussion on the aggregated student notes."""
        if not notes:
            return []

        # Group notes by approximate line proximity (within 30 lines)
        clusters = self._cluster_notes(notes)
        topics: List[SeminarTopic] = []

        for idx, cluster in enumerate(clusters):
            topic_id = f"TOPIC_{idx + 1:02d}"
            if mode == "api" and self.client:
                try:
                    topic = self._discuss_cluster_with_llm(topic_id, cluster, section_title, section_content)
                    topics.append(topic)
                    continue
                except Exception as e:
                    print(f"[Warning] LLM seminar debate failed: {e}. Using deterministic synthesis.")
            
            topic = self._synthesize_cluster_deterministic(topic_id, cluster, section_title)
            topics.append(topic)

        return topics

    def _cluster_notes(self, notes: List[MarginNote]) -> List[List[MarginNote]]:
        """Cluster notes that refer to nearby lines into a single discussion topic."""
        sorted_notes = sorted(notes, key=lambda n: n.line_start)
        clusters: List[List[MarginNote]] = []

        current_cluster: List[MarginNote] = []
        for note in sorted_notes:
            if not current_cluster:
                current_cluster.append(note)
            else:
                if abs(note.line_start - current_cluster[-1].line_start) <= 35:
                    current_cluster.append(note)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [note]
        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _discuss_cluster_with_llm(
        self,
        topic_id: str,
        cluster: List[MarginNote],
        section_title: str,
        section_content: str,
    ) -> SeminarTopic:
        """Simulate a multi-agent student discussion and moderator synthesis via LLM."""
        notes_summary = "\n".join(
            [
                f"- {n.student_name} ({n.discipline.value}, Score: {n.clarity_score}/5, Category: {n.category.value if n.category else 'N/A'}):\n"
                f"  Excerpt (L{n.line_start}-{n.line_end}): \"{n.text_snippet}\"\n"
                f"  Note: {n.note}\n"
                f"  Suggested Bridge: {n.proposed_bridge}"
                for n in cluster
            ]
        )

        min_line = min(n.line_start for n in cluster)
        max_line = max(n.line_end for n in cluster)
        line_range = f"L{min_line}–L{max_line}"

        prompt = f"""We are simulating the Harvard CS/EE 288 weekly reading seminar for "Physical AI Systems".
Section: {section_title}
Passage Under Review: Lines {line_range}

Student Margin Notes on this passage:
{notes_summary}

---
TASK:
1. Simulate a realistic, collegial 3 to 4 turn seminar exchange among the students (Alex, Priya, Marcus, Elena) and Dr. Aris:
   - Elena points out the progressive disclosure friction or jargon leap.
   - The relevant specialist (Marcus for physics/control, Priya for silicon/systems, Alex for deep learning) explains why the concept matters or what intuition is missing.
   - The other students react to how this helps their understanding.
   - Dr. Aris summarizes the consensus and the pedagogical rule.
2. Formulate the authoritative Consensus Verdict on whether this passage violates Progressive Disclosure.
3. Determine Priority: P0_BLOCKER, P1_SIGNIFICANT, or P2_POLISH.
4. Provide the exact Original Text snippet and the Proposed Progressive Disclosure Rewrite (surgical, publication-grade text that can be dropped into the book).
5. State the Pedagogical Rationale.

Format strictly as JSON:
{{
  "discussion": [
    {{
      "speaker": "<Name>",
      "speaker_role": "<Role/Discipline>",
      "target_speaker": "<Optional Target>",
      "text": "<What the speaker says in seminar>"
    }}
  ],
  "consensus_verdict": "<1-2 sentence summary of what the class agreed on>",
  "priority": "<P0_BLOCKER | P1_SIGNIFICANT | P2_POLISH>",
  "original_text": "<Original passage snippet>",
  "proposed_rewrite": "<Exact surgical rewrite with progressive disclosure>",
  "rationale": "<Pedagogical reasoning for the change>"
}}
"""

        from scripts.pedagogy_sim.llm_bridge import call_llm_json

        data = call_llm_json(
            system_prompt=MODERATOR_PROMPT,
            user_prompt=prompt,
            model=self.model,
            temperature=0.4,
        )
        if not data:
            return self._synthesize_cluster_deterministic(topic_id, cluster, section_title)
        turns = [
            DiscussionTurn(
                speaker=d.get("speaker", "Student"),
                speaker_role=d.get("speaker_role", "Participant"),
                target_speaker=d.get("target_speaker"),
                text=d.get("text", "")
            )
            for d in data.get("discussion", [])
        ]

        return SeminarTopic(
            topic_id=topic_id,
            line_range=line_range,
            text_snippet=cluster[0].text_snippet,
            student_notes=cluster,
            discussion=turns,
            consensus_verdict=data.get("consensus_verdict", "Class reached consensus on clarifying terminology."),
            priority=Priority(data.get("priority", "P1_SIGNIFICANT")),
            original_text=data.get("original_text", cluster[0].text_snippet),
            proposed_rewrite=data.get("proposed_rewrite", cluster[0].proposed_bridge or ""),
            rationale=data.get("rationale", "Enhance progressive disclosure and cross-disciplinary accessibility."),
        )

    def _synthesize_cluster_deterministic(
        self,
        topic_id: str,
        cluster: List[MarginNote],
        section_title: str,
    ) -> SeminarTopic:
        """Deterministic fallback seminar synthesis."""
        min_line = min(n.line_start for n in cluster)
        max_line = max(n.line_end for n in cluster)
        line_range = f"L{min_line}–L{max_line}"

        turns: List[DiscussionTurn] = []
        for n in cluster:
            turns.append(
                DiscussionTurn(
                    speaker=n.student_name,
                    speaker_role=n.discipline.value,
                    target_speaker="Seminar Cohort",
                    text=f"At line {n.line_start}, I noted: {n.note} We need to bridge this better.",
                )
            )

        turns.append(
            DiscussionTurn(
                speaker="Dr. Aris Thorne",
                speaker_role="Seminar Moderator",
                target_speaker="All",
                text=f"Excellent observation from {cluster[0].student_name}. In Physical AI, we cannot assume prior familiarity across ML, systems, and control simultaneously. We will revise this to introduce the intuition first.",
            )
        )

        proposed_bridge = cluster[0].proposed_bridge or "Introduce foundational motivation before technical terminology."
        
        return SeminarTopic(
            topic_id=topic_id,
            line_range=line_range,
            text_snippet=cluster[0].text_snippet,
            student_notes=cluster,
            discussion=turns,
            consensus_verdict="Consensus that this passage requires progressive disclosure scaffolding to prevent cognitive friction.",
            priority=Priority.P1_SIGNIFICANT,
            original_text=cluster[0].text_snippet,
            proposed_rewrite=proposed_bridge,
            rationale="Respects progressive disclosure by introducing physical motivation and clear definitions before formal abstractions.",
        )
