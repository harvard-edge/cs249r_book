#!/usr/bin/env python3
"""Build course websites into the unified site's deployment artifact."""

from pathlib import Path
import shutil
import subprocess

SITE = Path(__file__).resolve().parents[1]
COURSES = SITE.parent / "courses"

if __name__ == "__main__":
    subprocess.run(["quarto", "render"], cwd=COURSES, check=True)
    destination = SITE / "_build/courses"
    # This is generated output owned by this build, never a source checkout.
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(COURSES / "_build", destination)
    print("Courses staged in the unified site at /courses/")
