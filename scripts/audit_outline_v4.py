#!/usr/bin/env python3
"""Audit script for MASTER_TEXTBOOK_OUTLINE_V4_MLSYS.md."""

import re
import sys

def audit():
    with open("books/vol3/MASTER_TEXTBOOK_OUTLINE_V4_MLSYS.md") as f:
        text = f.read()

    ch_splits = re.split(r"\n(?=### Chapter \d+:)", text)
    preamble = ch_splits[0]
    chapters_raw = ch_splits[1:]

    print(f"Total Chapters parsed: {len(chapters_raw)}")

    results = []

    for ch_idx, ch_text in enumerate(chapters_raw, 1):
        lines = ch_text.strip().splitlines()
        ch_title_line = lines[0]
        m_ch = re.match(r"### Chapter (\d+):\s*(.*)", ch_title_line)
        ch_num = int(m_ch.group(1)) if m_ch else ch_idx
        ch_title = m_ch.group(2) if m_ch else ch_title_line

        has_compass = "[SYSTEMS STATE AT CHAPTER" in ch_text
        has_forbidden_compass = "Subsystems NOT YET BUILT" in ch_text

        sec_splits = re.split(r"\n(?=#### Section \d+\.\d+:)", ch_text)
        secs_raw = sec_splits[1:]

        ch_info = {
            "chapter_num": ch_num,
            "chapter_title": ch_title,
            "has_compass": has_compass,
            "has_forbidden_compass": has_forbidden_compass,
            "sections_count": len(secs_raw),
            "sections": []
        }

        for s_raw in secs_raw:
            s_lines = s_raw.strip().splitlines()
            s_title_line = s_lines[0]
            m_sec = re.match(r"#### Section (\d+\.\d+):\s*(.*)", s_title_line)
            sec_num = m_sec.group(1) if m_sec else "unknown"
            sec_title = m_sec.group(2) if m_sec else s_title_line

            has_dilemma = ("The Governing Systems Dilemma" in s_raw or 
                           "Governing Systems Dilemma" in s_raw)
            has_intuition = ("The Systems Intuition" in s_raw or 
                             "Systems Intuition" in s_raw)
            has_artifact = ("The Concrete Systems Artifact" in s_raw or 
                            "Concrete Systems Artifact" in s_raw or 
                            "Systems Artifact" in s_raw)
            has_math = ("Grounded Systems Math" in s_raw or 
                        "Grounded Systems Math & Provenance" in s_raw or 
                        "Grounded Math" in s_raw)
            has_worked_example = ("Worked Example" in s_raw or 
                                  "callout-note" in s_raw)
            has_fences = ("Negative Scope" in s_raw or 
                          "DO NOT" in s_raw or 
                          "Strict Negative Scope" in s_raw)
            has_checkpoint = ("Checkpoint" in s_raw or "@chk-" in s_raw)

            has_and_in_title = bool(re.search(r"\b(and|&)\b", sec_title, re.IGNORECASE))

            ch_info["sections"].append({
                "sec_num": sec_num,
                "sec_title": sec_title,
                "has_and_in_title": has_and_in_title,
                "has_dilemma": has_dilemma,
                "has_intuition": has_intuition,
                "has_artifact": has_artifact,
                "has_math": has_math,
                "has_worked_example": has_worked_example,
                "has_fences": has_fences,
                "has_checkpoint": has_checkpoint,
                "raw_text": s_raw
            })

        results.append(ch_info)

    total_sections = sum(c["sections_count"] for c in results)
    print(f"Total sections found: {total_sections}")

    issues = []
    compound_titles = []
    missing_fences = []
    missing_artifacts = []
    missing_math = []
    missing_worked_examples = []
    missing_dilemmas = []
    missing_intuitions = []

    for c in results:
        c_num = c["chapter_num"]
        c_title = c["chapter_title"]
        if bool(re.search(r"\b(and|&)\b", c_title, re.IGNORECASE)):
            compound_titles.append(f"Chapter {c_num} Title: {c_title}")
        if not c["has_compass"]:
            issues.append(f"Chapter {c_num}: Missing Curricular Compass")
        if not c["has_forbidden_compass"]:
            issues.append(f"Chapter {c_num}: Missing Forbidden Subsystems list in Compass")

        for s in c["sections"]:
            s_num = s["sec_num"]
            s_title = s["sec_title"]
            s_id = f"§{s_num} ({s_title})"

            if s["has_and_in_title"]:
                compound_titles.append(f"{s_id}: Compound title with and/&")
            if not s["has_dilemma"]:
                missing_dilemmas.append(s_id)
            if not s["has_intuition"]:
                missing_intuitions.append(s_id)
            if not s["has_artifact"]:
                missing_artifacts.append(s_id)
            if not s["has_math"]:
                missing_math.append(s_id)
            if not s["has_worked_example"]:
                missing_worked_examples.append(s_id)
            if not s["has_fences"]:
                missing_fences.append(s_id)

    print("\n" + "="*70)
    print(f"AUDIT SUMMARY ACROSS {len(results)} CHAPTERS & {total_sections} SECTIONS")
    print("="*70)
    print(f"Compound titles (and/&): {len(compound_titles)}")
    for t in compound_titles:
        print(f"  ❌ {t}")

    print(f"Missing Dilemmas: {len(missing_dilemmas)}")
    for m in missing_dilemmas:
        print(f"  ❌ {m}")

    print(f"Missing Intuitions: {len(missing_intuitions)}")
    for m in missing_intuitions:
        print(f"  ❌ {m}")

    print(f"Missing Artifacts: {len(missing_artifacts)}")
    for m in missing_artifacts:
        print(f"  ❌ {m}")

    print(f"Missing Math: {len(missing_math)}")
    for m in missing_math:
        print(f"  ❌ {m}")

    print(f"Missing Worked Examples: {len(missing_worked_examples)}")
    for m in missing_worked_examples:
        print(f"  ❌ {m}")

    print(f"Missing Fences / Negative Scopes: {len(missing_fences)}")
    for m in missing_fences:
        print(f"  ❌ {m}")

    print(f"General Issues: {len(issues)}")
    for i in issues:
        print(f"  ❌ {i}")

    return results

if __name__ == "__main__":
    audit()
