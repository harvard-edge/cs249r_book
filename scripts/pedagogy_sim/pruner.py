"""Pass 2: Economy of Language, Tightening, and Pruning Engine."""

import json
import os
import re
from typing import List, NamedTuple, Optional
from pydantic import BaseModel, Field

from scripts.pedagogy_sim.curriculum import TextSection


class PruningEdit(BaseModel):
    line_start: int
    line_end: int
    original_text: str
    tightened_text: str
    words_saved: int
    rationale: str
    pruning_type: str = Field(
        description="REDUNDANCY | THROAT_CLEARING | WINDING_SENTENCE | DIGRESSION"
    )


class SectionPruningReport(BaseModel):
    section_title: str
    original_word_count: int
    new_word_count: int
    total_words_saved: int
    edits: List[PruningEdit]
    summary: str


EDITOR_SYSTEM_PROMPT = """You are a world-class senior STEM textbook editor (MIT Press style) working with Elena Rostova (our student guardian of cognitive flow).
Your mission is PASS 2: SURGICAL PRUNING & ECONOMY OF LANGUAGE.

After Pass 1 added essential technical scaffolding, your job is to aggressively tighten the prose:
1. Cut "throat-clearing" academic filler ("It is important to note that", "We must remember that", "As we have seen").
2. Eliminate redundant restatements (if a point about physical irreversibility or digital glass was made twice in the same section, delete or merge the weaker instance).
3. Compress long, winding sentences into direct, punchy, active-voice statements.
4. Prune discursive tangents that belong in an appendix or distraction callouts.
5. NEVER remove technical substance, mathematical napkin math, or essential definitions added in Pass 1. Only trim the connective tissue and verbal fat.
6. STRICT IMMUTABILITY: NEVER touch, edit, or trim executable code blocks (```{python}...```), `mlsysim` scripts, or `#| label: ...` lines. Code blocks are completely immutable.

Format your output strictly as a JSON object:
{
  "edits": [
    {
      "original_text": "<exact verbatim sentence or paragraph to tighten>",
      "tightened_text": "<punchy, trimmed replacement>",
      "rationale": "<why this was trimmed / redundancy eliminated>",
      "pruning_type": "<REDUNDANCY | THROAT_CLEARING | WINDING_SENTENCE | DIGRESSION>"
    }
  ],
  "summary": "<1-2 sentence overview of the pruning pass>"
}
"""


class PrunerEngine:
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

    def prune_section(self, section: TextSection, mode: str = "api") -> SectionPruningReport:
        orig_words = len(section.content.split())

        if mode == "api" and self.client:
            try:
                return self._prune_with_llm(section, orig_words)
            except Exception as e:
                print(f"[Warning] Pruner LLM call failed: {e}. Falling back to rule-based pruner.")

        return self._prune_with_rules(section, orig_words)

    def _prune_with_llm(self, section: TextSection, orig_words: int) -> SectionPruningReport:
        prompt = (
            f"Review this textbook section for PASS 2 (Trimming, Tightening, and Economy of Language):\n\n"
            f"Section Title: {section.title}\n"
            f"Lines: {section.start_line} to {section.end_line}\n\n"
            f"---\nCONTENT:\n"
            + section.content
            + "\n---\n\n"
            "Identify 1 to 3 places where sentences are wordy, repetitive, or carry unnecessary academic filler.\n"
            "Provide exact substring matches in original_text so they can be replaced cleanly.\n"
        )

        from scripts.pedagogy_sim.llm_bridge import call_llm_json

        data = call_llm_json(
            system_prompt=EDITOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            model=self.model,
            temperature=0.3,
        )
        if not data:
            return self._prune_with_rules(section)
        edits = []
        total_saved = 0

        for item in data.get("edits", []):
            orig = item.get("original_text", "").strip()
            tightened = item.get("tightened_text", "").strip()
            if not orig or not tightened or orig == tightened:
                continue

            # Verify substring exists in section content
            if orig not in section.content:
                continue

            w_orig = len(orig.split())
            w_new = len(tightened.split())
            saved = max(0, w_orig - w_new)
            total_saved += saved

            edits.append(
                PruningEdit(
                    line_start=section.start_line,
                    line_end=section.end_line,
                    original_text=orig,
                    tightened_text=tightened,
                    words_saved=saved,
                    rationale=item.get("rationale", "Tightened for economy of language."),
                    pruning_type=item.get("pruning_type", "WINDING_SENTENCE"),
                )
            )

        return SectionPruningReport(
            section_title=section.title,
            original_word_count=orig_words,
            new_word_count=orig_words - total_saved,
            total_words_saved=total_saved,
            edits=edits,
            summary=data.get("summary", f"Pruned {total_saved} redundant words."),
        )

    def _prune_with_rules(self, section: TextSection, orig_words: int) -> SectionPruningReport:
        """Deterministic rule-based trimmer for common academic throat-clearing."""
        content = section.content
        edits = []
        total_saved = 0

        throat_clearing = [
            (r"\bIt is important to note that\b\s*", "", "Removed throat-clearing prefix"),
            (r"\bIt is worth noting that\b\s*", "", "Removed throat-clearing prefix"),
            (r"\bIn order to\b", "To", "Compressed 'In order to' -> 'To'"),
            (r"\bAs previously mentioned\b,?\s*", "", "Removed backward-looking filler"),
        ]

        for pat, repl, rationale in throat_clearing:
            matches = list(re.finditer(pat, content, re.IGNORECASE))
            for m in matches:
                orig_match = m.group(0)
                w_orig = len(orig_match.split())
                w_new = len(repl.split())
                saved = max(0, w_orig - w_new)
                total_saved += saved
                edits.append(
                    PruningEdit(
                        line_start=section.start_line,
                        line_end=section.end_line,
                        original_text=orig_match,
                        tightened_text=repl,
                        words_saved=saved,
                        rationale=rationale,
                        pruning_type="THROAT_CLEARING",
                    )
                )

        return SectionPruningReport(
            section_title=section.title,
            original_word_count=orig_words,
            new_word_count=orig_words - total_saved,
            total_words_saved=total_saved,
            edits=edits,
            summary=f"Rule-based pruning eliminated {total_saved} filler words.",
        )
