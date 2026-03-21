import json
import random
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel

class Question(BaseModel):
    id: str
    track: str
    scope: str
    level: str
    title: str
    topic: str
    scenario: str
    details: dict

class InterviewCorpus:
    def __init__(self, corpus_path: str):
        with open(corpus_path, 'r') as f:
            data = json.load(f)
            self.questions = [Question(**q) for q in data]
            
    def get_random(self, track: Optional[str] = None, level: Optional[str] = None) -> Question:
        pool = self.questions
        if track:
            pool = [q for q in pool if q.track == track]
        if level:
            pool = [q for q in pool if q.level == level]
        return random.choice(pool) if pool else None

# --- Mock Evaluator Logic ---
# This is what the FastAPI /api/evaluate endpoint would do
def generate_grading_prompt(question: Question, user_answer: str, numbers_context: str, rubric_context: str):
    prompt = f"""
You are a Principal ML Systems Engineer (L6) at a Tier-1 Tech Lab (Google/Meta/OpenAI).
Your goal is to conduct a high-stakes technical interview for a {question.level} role.

SCENARIO:
{question.scenario}

GROUND TRUTH (Realistic Solution):
{question.details['realistic_solution']}

NAPKIN MATH:
{question.details['napkin_math']}

PHYSICS CONSTRAINTS (from NUMBERS.md):
{numbers_context}

EVALUATION RUBRIC:
{rubric_context}

USER'S ANSWER:
{user_answer}

INSTRUCTIONS:
1. Grade the user's response as L3 (Junior), L4 (Mid), L5 (Senior), or L6 (Staff).
2. If their math is wrong or ignores physical constraints (bandwidth, memory, speed of light), point it out specifically.
3. Be professional but strict. Do not award L6 for hand-wavy answers.
4. If they pass, explain WHY they passed based on the rubric.
5. Provide a 'Physics Lesson' if they missed a core invariant.

FORMAT YOUR RESPONSE AS JSON:
{{
  "grade": "L4",
  "score": 75,
  "feedback": "...",
  "lesson": "..."
}}
"""
    return prompt

if __name__ == "__main__":
    # Test loading
    corpus = InterviewCorpus("/Users/VJ/GitHub/MLSysBook/interviews/corpus.json")
    q = corpus.get_random(track="cloud", level="L4")
    print(f"Random Q: {q.title} ({q.level})")
    print(f"Scenario: {q.scenario}")
