"""Autonomous Section-by-Section Agentic Loop Runner for Progressive Disclosure & Pruning."""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional
from pydantic import BaseModel

from scripts.pedagogy_sim.curriculum import (
    TextSection,
    parse_chapter_sections,
)
from scripts.pedagogy_sim.models import (
    MarginNote,
    SeminarTopic,
)
from scripts.pedagogy_sim.personas import STUDENTS
from scripts.pedagogy_sim.pruner import PrunerEngine, SectionPruningReport
from scripts.pedagogy_sim.seminar_room import SeminarRoom
from scripts.pedagogy_sim.student_reader import StudentReaderEngine


class SectionExecutionRecord(BaseModel):
    section_index: int
    section_title: str
    start_line: int
    end_line: int
    initial_word_count: int
    final_word_count: int
    pass1_scaffolding_edits: int
    pass2_pruning_edits: int
    words_saved_pass2: int
    topics: List[SeminarTopic]
    pruning_report: SectionPruningReport


class AgenticLoopRunner:
    def __init__(
        self,
        file_path: str = "books/vol4/01_boundary/01_boundary.qmd",
        mode: str = "api",
        model: str = "gpt-4o-mini",
        output_dir: str = "books/vol4/_pedagogical_seminar",
    ):
        self.file_path = file_path
        self.mode = mode
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.reader = StudentReaderEngine(model=model)
        self.room = SeminarRoom(model=model)
        self.pruner = PrunerEngine(model=model)

    def run_section_loop(
        self,
        start_section: int = 0,
        num_sections: int = 1,
        apply_live: bool = False,
    ) -> List[SectionExecutionRecord]:
        """Execute the two-pass agentic loop section-by-section."""
        sections = parse_chapter_sections(self.file_path)
        total_sections = len(sections)

        end_section = min(total_sections, start_section + num_sections) if num_sections > 0 else total_sections
        print("=" * 80)
        print("🚀 STARTING AUTONOMOUS TWO-PASS AGENTIC PEDAGOGICAL LOOP")
        print(f"Target File: {self.file_path}")
        print(f"Sections to process: {start_section} to {end_section - 1} (Total available: {total_sections})")
        print(f"Mode: {self.mode.upper()} | Live Apply: {'ENABLED' if apply_live else 'DRY-RUN'}")
        print("=" * 80)

        records: List[SectionExecutionRecord] = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            full_content = f.read()

        for idx in range(start_section, end_section):
            sec = sections[idx]
            initial_w = len(sec.content.split())
            print(f"\n" + "-" * 70)
            print(f"📍 SECTION [{idx}/{total_sections - 1}]: {sec.title} (L{sec.start_line}–L{sec.end_line})")
            print(f"Initial Word Count: {initial_w}")
            print("-" * 70)

            from scripts.pedagogy_sim.curriculum import mask_code_blocks, unmask_code_blocks

            # Mask code blocks so simulation agents NEVER see or edit mlsysim / python blocks
            masked_content, code_blocks = mask_code_blocks(sec.content)
            masked_sec = TextSection(
                title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                content=masked_content,
                level=sec.level,
            )

            # -------------------------------------------------------------
            # PASS 1: Grounding & Progressive Disclosure (Prose Only)
            # -------------------------------------------------------------
            print("\n[Pass 1: Progressive Disclosure Scaffolding]")
            all_notes: List[MarginNote] = []
            for s_id, student in STUDENTS.items():
                notes = self.reader.simulate_student_reading(student, masked_sec, mode=self.mode)
                print(f"  • {student.name}: {len(notes)} margin note(s)")
                all_notes.extend(notes)

            topics = self.room.hold_seminar_discussion(
                notes=all_notes,
                section_title=sec.title,
                section_content=masked_content,
                mode=self.mode,
            )
            print(f"  ✓ Seminar completed: {len(topics)} consensus revision topic(s)")

            # Apply Pass 1 edits to section content in-memory
            working_masked_content = masked_content
            pass1_count = 0
            for t in topics:
                # Strictly reject any edit attempting to touch code blocks or placeholders
                if "PROTECTED_CODE_BLOCK" in t.original_text or "PROTECTED_CODE_BLOCK" in t.proposed_rewrite:
                    continue
                if "```" in t.original_text or "```" in t.proposed_rewrite:
                    continue

                if t.original_text and t.proposed_rewrite and t.original_text in working_masked_content:
                    working_masked_content = working_masked_content.replace(t.original_text, t.proposed_rewrite, 1)
                    pass1_count += 1
                    print(f"    + Applied Pass 1 edit on: \"{t.original_text[:60]}...\"")

            # -------------------------------------------------------------
            # PASS 2: Pruning & Economy of Language (Prose Only)
            # -------------------------------------------------------------
            print("\n[Pass 2: Tightening & Pruning (Economy of Language)]")
            updated_masked_sec = TextSection(
                title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                content=working_masked_content,
                level=sec.level,
            )

            prune_report = self.pruner.prune_section(updated_masked_sec, mode=self.mode)
            pass2_count = 0
            for p_edit in prune_report.edits:
                # Strictly reject any pruning edit attempting to touch code blocks
                if "PROTECTED_CODE_BLOCK" in p_edit.original_text or "PROTECTED_CODE_BLOCK" in p_edit.tightened_text:
                    continue
                if "```" in p_edit.original_text or "```" in p_edit.tightened_text:
                    continue

                if p_edit.original_text in working_masked_content:
                    working_masked_content = working_masked_content.replace(
                        p_edit.original_text, p_edit.tightened_text, 1
                    )
                    pass2_count += 1
                    print(f"    ✂️ Pruned ({p_edit.pruning_type}): -{p_edit.words_saved} words")

            # Restore original code blocks byte-for-byte
            working_section_content = unmask_code_blocks(working_masked_content, code_blocks)

            final_w = len(working_section_content.split())
            delta_w = final_w - initial_w
            sign = "+" if delta_w >= 0 else ""
            print(f"\n📊 Section Summary: {initial_w} -> {final_w} words ({sign}{delta_w} net)")

            # If apply_live is on, update the global content
            if apply_live and working_section_content != sec.content:
                full_content = full_content.replace(sec.content, working_section_content, 1)

            record = SectionExecutionRecord(
                section_index=idx,
                section_title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                initial_word_count=initial_w,
                final_word_count=final_w,
                pass1_scaffolding_edits=pass1_count,
                pass2_pruning_edits=pass2_count,
                words_saved_pass2=prune_report.total_words_saved,
                topics=topics,
                pruning_report=prune_report,
            )
            records.append(record)

        # Write live file if requested
        if apply_live:
            backup_file = f"{self.file_path}.bak"
            if not os.path.exists(backup_file):
                shutil.copyfile(self.file_path, backup_file)
                print(f"\n✓ Created backup file at {backup_file}")
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"✅ Successfully wrote live updates to {self.file_path}!")

        # Save session ledger
        self._export_ledger(records)
        return records

    def _export_ledger(self, records: List[SectionExecutionRecord]):
        json_path = os.path.join(self.output_dir, "agentic_loop_ledger.json")
        md_path = os.path.join(self.output_dir, "agentic_loop_ledger.md")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([r.model_dump() for r in records], f, indent=2)

        md = "# Agentic Progressive Disclosure & Pruning Loop Ledger\n\n"
        md += f"**Target Chapter File**: `{self.file_path}`  \n"
        md += f"**Processed Sections**: {len(records)}  \n\n"
        md += "| Section | Title | Lines | Initial Words | Final Words | Pass 1 Edits | Pass 2 Cuts | Net Words Saved |\n"
        md += "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n"

        for r in records:
            net = r.initial_word_count - r.final_word_count
            md += f"| {r.section_index} | {r.section_title} | L{r.start_line}–L{r.end_line} | {r.initial_word_count} | {r.final_word_count} | {r.pass1_scaffolding_edits} | {r.pass2_pruning_edits} | {net:+d} |\n"

        md += "\n---\n\n"
        for r in records:
            md += f"## Section {r.section_index}: {r.section_title}\n\n"
            md += f"- **Initial Words**: {r.initial_word_count} | **Final Words**: {r.final_word_count}\n"
            md += f"- **Pass 1 Scaffolding Edits Applied**: {r.pass1_scaffolding_edits}\n"
            md += f"- **Pass 2 Pruning Edits Applied**: {r.pass2_pruning_edits} ({r.words_saved_pass2} words trimmed)\n\n"
            if r.topics:
                md += "### Pass 1 Grounding Topics\n"
                for t in r.topics:
                    md += f"- **{t.priority.value}**: {t.consensus_verdict}\n"
                    md += f"  - *Rationale*: {t.rationale}\n"
            if r.pruning_report.edits:
                md += "\n### Pass 2 Pruning Actions\n"
                for p in r.pruning_report.edits:
                    md += f"- **[{p.pruning_type}]** (-{p.words_saved} words): {p.rationale}\n"
            md += "\n---\n\n"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✓ Saved updated session ledger to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Section-by-Section Agentic Loop")
    parser.add_argument("--file", type=str, default="books/vol4/01_boundary/01_boundary.qmd")
    parser.add_argument("--start", type=int, default=0, help="Start section index")
    parser.add_argument("--num", type=int, default=1, help="Number of sections to process")
    parser.add_argument("--apply", action="store_true", help="Apply updates directly to the file")
    parser.add_argument("--mode", choices=["api", "heuristic"], default="api")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=str, default="books/vol4/_pedagogical_seminar")

    args = parser.parse_args()
    runner = AgenticLoopRunner(
        file_path=args.file,
        mode=args.mode,
        model=args.model,
        output_dir=args.output_dir,
    )
    runner.run_section_loop(
        start_section=args.start,
        num_sections=args.num,
        apply_live=args.apply,
    )


if __name__ == "__main__":
    main()
