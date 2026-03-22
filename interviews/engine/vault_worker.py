#!/usr/bin/env python3
"""Single vault worker — generates questions for one cube cell to a JSON file."""

import json
import sys
from pathlib import Path

# Ensure the interviews directory is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.generate import generate_questions
from engine.schemas import GenerationRequest


def main():
    track = sys.argv[1]
    concept = sys.argv[2]
    level = sys.argv[3]
    count = int(sys.argv[4])
    out_file = sys.argv[5]

    req = GenerationRequest(
        track=track,
        concept=concept,
        target_level=level,
        competency_area="auto",
        count=count,
    )

    try:
        qs = generate_questions(req)
        data = [q.model_dump() for q in qs]
        Path(out_file).write_text(json.dumps(data, indent=2))
        print(f"{len(data)} questions -> {Path(out_file).name}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        Path(out_file).write_text("[]")


if __name__ == "__main__":
    main()
