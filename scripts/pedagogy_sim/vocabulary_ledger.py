"""Cumulative Vocabulary and Progressive Disclosure Concept Ledger for Volume IV."""

import json
import os
from typing import Dict, List, Optional, Set
from pydantic import BaseModel


class ConceptDefinition(BaseModel):
    term: str
    introduced_in_chapter: int
    introduced_in_section: str
    definition: str
    discipline: str


class VocabularyLedger:
    def __init__(self, ledger_file: str = "books/vol4/_pedagogical_seminar/vocabulary_ledger.json"):
        self.ledger_file = ledger_file
        self.concepts: Dict[str, ConceptDefinition] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.ledger_file):
            try:
                with open(self.ledger_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.concepts[k.lower()] = ConceptDefinition(**v)
            except Exception as e:
                print(f"[Warning] Failed to load vocabulary ledger: {e}")
        else:
            # Seed with baseline prerequisites established in prerequisites.qmd
            self._seed_prerequisites()

    def _seed_prerequisites(self):
        prereqs = [
            ("newtonian mechanics", 0, "Prerequisites", "Classical physics governing F=ma and rotational torque tau=I*alpha", "Control & Robotics"),
            ("kinetic energy", 0, "Prerequisites", "Ek = 0.5 * m * v^2 energy carried by moving mass", "Control & Robotics"),
            ("stopping distance", 0, "Prerequisites", "d_stop = v*t + v^2/(2a) physical distance needed to brake", "Control & Robotics"),
            ("forward pass", 0, "Prerequisites", "Inference computation passing input through model layers", "Machine Learning"),
            ("loss function", 0, "Prerequisites", "Objective function measuring prediction error", "Machine Learning"),
            ("tensor", 0, "Prerequisites", "Multi-dimensional array representing data and activations", "Machine Learning"),
            ("latency", 0, "Prerequisites", "Time elapsed from input arrival to output delivery", "Embedded Systems"),
            ("memory allocation", 0, "Prerequisites", "Reserving physical or virtual memory for program execution", "Embedded Systems"),
        ]
        for term, ch, sec, dfn, disc in prereqs:
            self.concepts[term.lower()] = ConceptDefinition(
                term=term,
                introduced_in_chapter=ch,
                introduced_in_section=sec,
                definition=dfn,
                discipline=disc,
            )

    def save(self):
        os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            json.dump({k: v.model_dump() for k, v in self.concepts.items()}, f, indent=2)

    def register_concept(self, term: str, chapter: int, section: str, definition: str, discipline: str):
        self.concepts[term.lower()] = ConceptDefinition(
            term=term,
            introduced_in_chapter=chapter,
            introduced_in_section=section,
            definition=definition,
            discipline=discipline,
        )
        self.save()

    def is_known_at(self, term: str, current_chapter: int) -> bool:
        """Check if a term was legitimately introduced in a prior or current chapter."""
        item = self.concepts.get(term.lower())
        if not item:
            return False
        return item.introduced_in_chapter <= current_chapter

    def get_known_terms_up_to(self, chapter: int) -> List[str]:
        return [v.term for v in self.concepts.values() if v.introduced_in_chapter <= chapter]
