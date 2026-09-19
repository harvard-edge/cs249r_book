"""Second-Pass Pedagogical Audit & Student Feedback using Gemini 3.1 Pro Preview."""

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional
import threading

from scripts.pedagogy_sim.book_orchestrator import CHAPTER_REGISTRY
from scripts.pedagogy_sim.chapter_signoff import ChapterSignOffEngine
from scripts.pedagogy_sim.curriculum import (
    TextSection,
    mask_code_blocks,
    parse_chapter_sections,
    unmask_code_blocks,
)
from scripts.pedagogy_sim.models import MarginNote
from scripts.pedagogy_sim.personas import STUDENTS
from scripts.pedagogy_sim.pruner import PrunerEngine
from scripts.pedagogy_sim.seminar_room import SeminarRoom
from scripts.pedagogy_sim.student_reader import StudentReaderEngine


class GeminiFeedbackOrchestrator:
    def __init__(
        self,
        output_dir: str = "books/vol4/_pedagogical_seminar",
        model: str = "gemini-3.1-pro-high",
    ):
        self.output_dir = output_dir
        self.model = model
        self.cert_dir = os.path.join(output_dir, "certificates")
        self.log_dir = os.path.join(output_dir, "gemini_feedback")
        os.makedirs(self.cert_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self._lock = threading.Lock()
        self.summary_file = os.path.join(self.log_dir, "GEMINI_AUDIT_SUMMARY.md")

    def audit_chapter(
        self,
        chapter_num: int,
        apply_live: bool = True,
    ) -> Dict:
        """Run full Gemini 3.1 Pro student cohort audit across a chapter."""
        meta = CHAPTER_REGISTRY.get(chapter_num)
        if not meta:
            raise ValueError(f"Unknown chapter {chapter_num}")

        file_path = meta["file"]
        title = meta["title"]
        print(f"\n" + "=" * 80)
        print(f"✨ [Gemini 3.1 Pro Audit] CHAPTER {chapter_num:02d}: {title}")
        print(f"File: {file_path}")
        print("=" * 80)

        reader = StudentReaderEngine(model=self.model)
        room = SeminarRoom(model=self.model)
        pruner = PrunerEngine(model=self.model)

        sections = parse_chapter_sections(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            full_content = f.read()

        pass1_count = 0
        pass2_count = 0
        words_trimmed = 0

        # We audit every section with Gemini
        for idx, sec in enumerate(sections):
            print(f"[Ch {chapter_num:02d} Gemini] Section {idx}/{len(sections)-1}: {sec.title}")
            masked_content, code_blocks = mask_code_blocks(sec.content)
            masked_sec = TextSection(
                title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                content=masked_content,
                level=sec.level,
            )

            # Gemini Student Reader
            notes: List[MarginNote] = []
            for s_id, student in STUDENTS.items():
                s_notes = reader.simulate_student_reading(student, masked_sec, mode="api")
                notes.extend(s_notes)

            working_masked = masked_content
            if notes:
                topics = room.hold_seminar_discussion(
                    notes=notes,
                    section_title=f"{title} - {sec.title}",
                    section_content=masked_content,
                    mode="api",
                )
                for t in topics:
                    if "PROTECTED_CODE_BLOCK" in t.original_text or "PROTECTED_CODE_BLOCK" in t.proposed_rewrite:
                        continue
                    if "```" in t.original_text or "```" in t.proposed_rewrite:
                        continue
                    if t.original_text and t.proposed_rewrite and t.original_text in working_masked:
                        working_masked = working_masked.replace(t.original_text, t.proposed_rewrite, 1)
                        pass1_count += 1

            # Gemini Pruning & Economy of Language
            updated_masked_sec = TextSection(
                title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                content=working_masked,
                level=sec.level,
            )
            report = pruner.prune_section(updated_masked_sec, mode="api")
            for p in report.edits:
                if "PROTECTED_CODE_BLOCK" in p.original_text or "PROTECTED_CODE_BLOCK" in p.tightened_text:
                    continue
                if "```" in p.original_text or "```" in p.tightened_text:
                    continue
                if p.original_text in working_masked:
                    working_masked = working_masked.replace(p.original_text, p.tightened_text, 1)
                    pass2_count += 1
                    words_trimmed += p.words_saved

            working_final = unmask_code_blocks(working_masked, code_blocks)
            if apply_live and working_final != sec.content:
                full_content = full_content.replace(sec.content, working_final, 1)

        if apply_live and (pass1_count > 0 or pass2_count > 0):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"✅ Saved Gemini improvements to {file_path}")

        # Conduct formal Gemini Capstone Sign-Off
        signoff_engine = ChapterSignOffEngine(
            output_dir=self.cert_dir,
            model=self.model,
        )
        cert = signoff_engine.conduct_signoff(
            chapter_num=chapter_num,
            chapter_title=f"{title} (Gemini 3.1 Pro Verification)",
            file_path=file_path,
            summary_stats={
                "total_sections": len(sections),
                "pass1_edits": pass1_count,
                "words_trimmed": words_trimmed,
            },
        )

        cert_md_path = os.path.join(self.cert_dir, f"chapter_{chapter_num:02d}_gemini_signoff.md")
        # Save a dedicated gemini certificate file
        shutil.copyfile(
            os.path.join(self.cert_dir, f"chapter_{chapter_num:02d}_signoff.md"),
            cert_md_path,
        )
        print(f"🎓 Official Gemini Sign-Off Certificate: {cert_md_path}")

        res = {
            "chapter_num": chapter_num,
            "title": title,
            "sections": len(sections),
            "pass1_edits": pass1_count,
            "pass2_cuts": pass2_count,
            "words_trimmed": words_trimmed,
            "clarity_score": cert.overall_clarity_score,
            "verdict": cert.consensus_verdict,
        }
        self._record_summary(res)
        return res

    def _record_summary(self, res: Dict):
        with self._lock:
            log_path = os.path.join(self.log_dir, "gemini_audit_results.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")

    def run_all(self, max_workers: int = 4, apply_live: bool = True):
        """Run Gemini 3.1 Pro verification across all 17 chapters."""
        print("=" * 80)
        print("🌟 INITIATING PASS 2: GEMINI 3.1 PRO PREVIEW WHOLE-BOOK AUDIT")
        print(f"Concurreny: {max_workers} chapter workers | Model: {self.model}")
        print("=" * 80)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ch = {
                executor.submit(self.audit_chapter, ch_num, apply_live): ch_num
                for ch_num in range(1, 18)
            }
            for fut in as_completed(future_to_ch):
                ch_num = future_to_ch[fut]
                try:
                    res = fut.result()
                    print(f"✨ Chapter {ch_num} Gemini Audit Complete! Score: {res['clarity_score']}/5.0")
                except Exception as e:
                    print(f"❌ Error during Chapter {ch_num} Gemini Audit: {e}")

        print("\n🏆 GEMINI 3.1 PRO PREVIEW WHOLE-BOOK AUDIT COMPLETED!")


def main():
    parser = argparse.ArgumentParser(description="Gemini 3.1 Pro Pedagogical Audit")
    parser.add_argument("--chapter", type=int, default=None, help="Specific chapter to audit")
    parser.add_argument("--all", action="store_true", help="Audit all 17 chapters")
    parser.add_argument("--parallel", type=int, default=4, help="Parallel worker count")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro-high")

    args = parser.parse_args()
    orch = GeminiFeedbackOrchestrator(model=args.model)

    if args.all:
        orch.run_all(max_workers=args.parallel, apply_live=not args.dry_run)
    elif args.chapter:
        orch.audit_chapter(chapter_num=args.chapter, apply_live=not args.dry_run)
    else:
        print("Specify --all or --chapter <num>")


if __name__ == "__main__":
    main()
