#!/usr/bin/env python3
"""
chapter_section_flow_mapper.py

Extracts the Big-Picture Map and Section-by-Section structural flow of a Quarto (.qmd)
chapter for narrative coherence and transition audits.

Key Capabilities:
1. Extracts Chapter Title, Opening Purpose, and Core Invariants/Questions.
2. Parses all Level 2 (##) and Level 3 (###) sections with exact line numbers and word counts.
3. Captures Section Hooks (opening sentences) and Section Transitions (closing sentences).
4. Inventories all Figures, Tables, Listings, Callouts, and Equations per section.
5. Formats a structured Big-Picture Map and Section Flow trajectory in Markdown or JSON.
"""

import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class SubsectionInfo:
    title: str
    section_id: Optional[str]
    line_number: int


@dataclass
class SectionFlow:
    index: int
    title: str
    section_id: Optional[str]
    start_line: int
    end_line: int
    word_count: int
    subsections: List[SubsectionInfo] = field(default_factory=list)
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    listings: List[str] = field(default_factory=list)
    callouts: List[str] = field(default_factory=list)
    equations: List[str] = field(default_factory=list)
    opening_hook: str = ""
    closing_transition: str = ""
    raw_content: str = field(default="", repr=False)


@dataclass
class ChapterFlowMap:
    file_path: str
    chapter_num: str
    chapter_title: str
    chapter_id: Optional[str]
    purpose_summary: str
    learning_objectives: List[str]
    total_lines: int
    total_words: int
    sections: List[SectionFlow] = field(default_factory=list)


def count_words(text: str) -> int:
    """Count words in text excluding code blocks and raw markup."""
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    return len(text.split())


def clean_prose(text: str) -> str:
    """Clean markdown formatting for scannable summaries."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[*_#`]', '', text)
    return text.strip()


def parse_chapter_flow(file_path: str) -> ChapterFlowMap:
    path = Path(file_path)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    
    # Extract Chapter Title and ID
    chapter_title = ""
    chapter_id = None
    title_line_idx = -1
    for i, line in enumerate(lines[:50]):
        m = re.match(r'^#\s+(.+?)(?:\s*\{#([^}]+)\})?\s*$', line)
        if m:
            chapter_title = m.group(1).strip()
            chapter_id = m.group(2).strip() if m.group(2) else None
            title_line_idx = i
            break

    # Determine Chapter Number from path or title
    m_num = re.search(r'(\d+)_', path.name)
    chapter_num = m_num.group(1) if m_num else "00"

    # Extract Purpose and Learning Objectives
    purpose_summary = ""
    learning_objectives = []
    
    full_text = "".join(lines)
    purpose_m = re.search(r'##\s+Purpose[^{]*?(?:\{[^}]*\})?\s*\n+(.*?)(?=\n##|\Z)', full_text, re.DOTALL)
    if purpose_m:
        purpose_raw = purpose_m.group(1)
        # Extract learning objectives
        obj_matches = re.findall(r'-\s+\*\*([^*]+)\*\*:?\s*([^\n]+)', purpose_raw)
        if obj_matches:
            for verb, obj in obj_matches:
                learning_objectives.append(f"{verb}: {clean_prose(obj)}")
        # Get opening italicized motivation if present
        ital_m = re.search(r'_(Why[^_]+)_', purpose_raw)
        if ital_m:
            purpose_summary = clean_prose(ital_m.group(1))
        else:
            first_p = re.split(r'\n\s*\n', purpose_raw.strip())[0]
            purpose_summary = clean_prose(first_p)[:300]

    # Parse Level 2 Sections
    section_headers = []
    for i, line in enumerate(lines):
        m = re.match(r'^##\s+(.+?)(?:\s*\{#?([^}]+)\})?\s*$', line)
        if m:
            raw_title = m.group(1).strip()
            # Skip Purpose if handled
            if "Purpose" in raw_title:
                continue
            raw_id = m.group(2).strip() if m.group(2) else None
            if raw_id and raw_id.startswith('#'):
                raw_id = raw_id[1:]
            section_headers.append((i + 1, raw_title, raw_id))

    sections: List[SectionFlow] = []
    for s_idx, (start_line, title, sec_id) in enumerate(section_headers):
        end_line = section_headers[s_idx + 1][0] - 1 if s_idx + 1 < len(section_headers) else total_lines
        sec_lines = lines[start_line - 1 : end_line]
        sec_content = "".join(sec_lines)
        w_count = count_words(sec_content)

        # Extract subsections
        subsections = []
        for line_offset, s_line in enumerate(sec_lines):
            sub_m = re.match(r'^###\s+(.+?)(?:\s*\{#?([^}]+)\})?\s*$', s_line)
            if sub_m:
                sub_title = sub_m.group(1).strip()
                sub_id = sub_m.group(2).strip() if sub_m.group(2) else None
                if sub_id and sub_id.startswith('#'):
                    sub_id = sub_id[1:]
                subsections.append(SubsectionInfo(
                    title=sub_title,
                    section_id=sub_id,
                    line_number=start_line + line_offset
                ))

        # Extract labels
        figs = re.findall(r'\{#(fig-[^}\s]+)', sec_content)
        tbls = re.findall(r'\{#(tbl-[^}\s]+)', sec_content)
        lsts = re.findall(r'\{#(lst-[^}\s]+)', sec_content)
        principles = re.findall(r'\{#(pri-[^}\s]+)', sec_content)
        eqs = re.findall(r'\{#(eq-[^}\s]+)', sec_content)

        # Opening hook: first non-empty paragraph after header
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', sec_content) if p.strip()]
        opening_hook = ""
        for p in paragraphs[1:]:
            if p.startswith(':::') or p.startswith('![') or p.startswith('|') or p.startswith('$'):
                continue
            opening_hook = clean_prose(p)[:250] + "..."
            break

        # Closing transition: last meaningful paragraph before next section
        closing_transition = ""
        for p in reversed(paragraphs):
            p_strip = p.strip()
            if (p_strip.startswith(':::') or p_strip.startswith('![') or 
                p_strip.startswith('|') or p_strip.startswith('$') or 
                p_strip.startswith('>') or p_strip.startswith('<!--') or
                p_strip.endswith('-->') or p_strip.startswith('##')):
                continue
            closing_transition = clean_prose(p_strip)[:300] + "..."
            break

        sections.append(SectionFlow(
            index=s_idx + 1,
            title=title,
            section_id=sec_id,
            start_line=start_line,
            end_line=end_line,
            word_count=w_count,
            subsections=subsections,
            figures=figs,
            tables=tbls,
            listings=lsts,
            callouts=principles,
            equations=eqs,
            opening_hook=opening_hook,
            closing_transition=closing_transition,
            raw_content=sec_content
        ))

    total_words = sum(s.word_count for s in sections)

    return ChapterFlowMap(
        file_path=str(path),
        chapter_num=chapter_num,
        chapter_title=chapter_title,
        chapter_id=chapter_id,
        purpose_summary=purpose_summary,
        learning_objectives=learning_objectives,
        total_lines=total_lines,
        total_words=total_words,
        sections=sections
    )


def format_markdown_map(flow: ChapterFlowMap) -> str:
    lines = []
    lines.append(f"# Chapter {flow.chapter_num}: {flow.chapter_title}")
    lines.append(f"**File:** `{flow.file_path}` | **Lines:** {flow.total_lines} | **Words:** {flow.total_words:,}\n")
    
    if flow.purpose_summary:
        lines.append(f"### Core Driving Question & Central Thesis")
        lines.append(f"> {flow.purpose_summary}\n")

    if flow.learning_objectives:
        lines.append("### Learning Objectives")
        for obj in flow.learning_objectives:
            lines.append(f"- {obj}")
        lines.append("")

    lines.append("## Big-Picture Narrative Trajectory")
    lines.append("```mermaid")
    lines.append("flowchart TD")
    for s in flow.sections:
        clean_name = re.sub(r'[^a-zA-Z0-9 ]', '', s.title)[:30]
        lines.append(f'    S{s.index}["S{s.index}: {clean_name}"]')
    for i in range(len(flow.sections) - 1):
        lines.append(f"    S{flow.sections[i].index} -->|leads into| S{flow.sections[i+1].index}")
    lines.append("```\n")

    lines.append("## Section-by-Section Deep Dive\n")
    for s in flow.sections:
        lines.append(f"### Section {s.index}: {s.title} (`#{s.section_id}`)")
        lines.append(f"- **Lines:** {s.start_line}–{s.end_line} ({s.end_line - s.start_line + 1} lines, {s.word_count:,} words)")
        
        if s.subsections:
            sub_strs = [f"`{sub.title}` (L{sub.line_number})" for sub in s.subsections]
            lines.append(f"- **Subsections:** {', '.join(sub_strs)}")
        
        artifacts = []
        if s.figures:
            artifacts.append(f"Figures: {', '.join(s.figures)}")
        if s.tables:
            artifacts.append(f"Tables: {', '.join(s.tables)}")
        if s.listings:
            artifacts.append(f"Listings: {', '.join(s.listings)}")
        if s.callouts:
            artifacts.append(f"Principles: {', '.join(s.callouts)}")
        if s.equations:
            artifacts.append(f"Equations: {len(s.equations)} formal eqns ({', '.join(s.equations[:3])}{'...' if len(s.equations)>3 else ''})")
        
        if artifacts:
            lines.append(f"- **Key Systems Artifacts:** {'; '.join(artifacts)}")
        
        if s.opening_hook:
            lines.append(f"- **Opening Hook:** *\"{s.opening_hook}\"*")
        if s.closing_transition:
            lines.append(f"- **Handoff / Transition to Next:** *\"{s.closing_transition}\"*")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Map big-picture flow and section transitions for Quarto chapters.")
    parser.add_argument("-f", "--file", required=True, help="Path to chapter .qmd file")
    parser.add_argument("--json", action="store_true", help="Output JSON manifest")
    args = parser.parse_args()

    flow = parse_chapter_flow(args.file)

    if args.json:
        # Convert to dict
        data = asdict(flow)
        for s in data['sections']:
            s.pop('raw_content', None)
        print(json.dumps(data, indent=2))
    else:
        print(format_markdown_map(flow))


if __name__ == "__main__":
    main()
