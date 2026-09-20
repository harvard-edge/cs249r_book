#!/usr/bin/env python3
"""Detailed review of all 18 chapters and 142 sections in MASTER_TEXTBOOK_OUTLINE_V4_MLSYS.md."""

import re

def review():
    with open("books/vol3/MASTER_TEXTBOOK_OUTLINE_V4_MLSYS.md") as f:
        text = f.read()

    ch_splits = re.split(r"\n(?=### Chapter \d+:)", text)
    preamble = ch_splits[0]
    chapters_raw = ch_splits[1:]

    print(f"Total Chapters parsed: {len(chapters_raw)}")

    chapters_summary = []
    compound_titles = []
    missing_negative_scopes = []
    missing_key_points = []
    missing_causal_bridges = []
    missing_anchors = []

    for idx, ch_raw in enumerate(chapters_raw, 1):
        lines = ch_raw.strip().splitlines()
        ch_title_line = lines[0]
        m_ch = re.match(r"### Chapter (\d+):\s*(.*)", ch_title_line)
        ch_num = int(m_ch.group(1)) if m_ch else idx
        ch_title = m_ch.group(2) if m_ch else ch_title_line

        # Check compound in chapter title
        if bool(re.search(r"\b(and|&)\b", ch_title, re.IGNORECASE)):
            compound_titles.append(f"Chapter {ch_num}: {ch_title}")

        # Extract compass
        compass_match = re.search(r"```\n\[SYSTEMS STATE AT CHAPTER \d+\]:(.*?)```", ch_raw, re.DOTALL)
        compass_text = compass_match.group(1) if compass_match else ""

        # Split sections
        sec_splits = re.split(r"\n(?=#### Section \d+\.\d+:)", ch_raw)
        ch_header = sec_splits[0]
        secs_raw = sec_splits[1:]

        ch_data = {
            "num": ch_num,
            "title": ch_title,
            "has_compass": bool(compass_match),
            "sections": []
        }

        for s_raw in secs_raw:
            s_lines = s_raw.strip().splitlines()
            s_title_line = s_lines[0]
            m_sec = re.match(r"#### Section (\d+\.\d+):\s*(.*)", s_title_line)
            sec_num = m_sec.group(1) if m_sec else "unknown"
            sec_title = m_sec.group(2) if m_sec else s_title_line

            if bool(re.search(r"\b(and|&)\b", sec_title, re.IGNORECASE)):
                compound_titles.append(f"Section {sec_num}: {sec_title}")

            has_anchor = ("Heading & Anchor:" in s_raw or "Heading:" in s_raw or "{#sec-" in s_raw)
            if not has_anchor:
                missing_anchors.append(f"Section {sec_num}")

            has_key_point = ("The Single Key Point:" in s_raw or "Key Point:" in s_raw)
            if not has_key_point:
                missing_key_points.append(f"Section {sec_num}")

            has_negative = ("What NOT to Cover" in s_raw or "Negative Scope" in s_raw or "DO NOT" in s_raw)
            if not has_negative:
                missing_negative_scopes.append(f"Section {sec_num}")

            has_bridge = ("Causal Bridge" in s_raw or "Bridge to" in s_raw)
            if not has_bridge:
                missing_causal_bridges.append(f"Section {sec_num}")

            # Count DO NOT fences
            do_not_count = len(re.findall(r"(?:🛑|⛔|🚫)?\s*\*\*DO NOT\*\*", s_raw))

            ch_data["sections"].append({
                "sec_num": sec_num,
                "sec_title": sec_title,
                "do_not_count": do_not_count,
                "length_chars": len(s_raw)
            })

        chapters_summary.append(ch_data)

    print("\n" + "="*80)
    print("CHAPTER-BY-CHAPTER SUMMARY")
    print("="*80)
    total_secs = 0
    for c in chapters_summary:
        s_count = len(c["sections"])
        total_secs += s_count
        min_fences = min([s["do_not_count"] for s in c["sections"]]) if c["sections"] else 0
        max_fences = max([s["do_not_count"] for s in c["sections"]]) if c["sections"] else 0
        print(f"Chapter {c['num']:02d}: {c['title']} ({s_count} sections) | Compass: {'✅' if c['has_compass'] else '❌'} | DO NOT fences per sec: {min_fences}..{max_fences}")

    print("\n" + "="*80)
    print(f"TOTAL: 18 Chapters, {total_secs} Sections")
    print(f"Compound Titles (and/&): {len(compound_titles)}")
    for t in compound_titles:
        print(f"  ❌ {t}")
    print(f"Missing Negative Scopes / Fences: {len(missing_negative_scopes)}")
    for m in missing_negative_scopes:
        print(f"  ❌ {m}")
    print(f"Missing Key Points: {len(missing_key_points)}")
    for m in missing_key_points:
        print(f"  ❌ {m}")
    print(f"Missing Anchors: {len(missing_anchors)}")
    for m in missing_anchors:
        print(f"  ❌ {m}")
    print(f"Missing Causal Bridges: {len(missing_causal_bridges)}")
    for m in missing_causal_bridges:
        print(f"  ❌ {m}")

if __name__ == "__main__":
    review()
