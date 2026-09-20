"""Full-Book Autonomous Agentic Loop Orchestrator for Volume IV."""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from scripts.pedagogy_sim.curriculum import (
    TextSection,
    parse_chapter_sections,
)
from scripts.pedagogy_sim.models import (
    MarginNote,
    SeminarTopic,
)
from scripts.pedagogy_sim.personas import STUDENTS
from scripts.pedagogy_sim.pruner import PrunerEngine
from scripts.pedagogy_sim.seminar_room import SeminarRoom
from scripts.pedagogy_sim.student_reader import StudentReaderEngine
from scripts.pedagogy_sim.vocabulary_ledger import VocabularyLedger


CHAPTER_REGISTRY: Dict[int, Dict[str, str]] = {
    1: {"title": "The Causal Boundary", "file": "books/vol4/01_boundary/01_boundary.qmd", "part": "Part I: Anatomy"},
    2: {"title": "The Body", "file": "books/vol4/02_body/02_body.qmd", "part": "Part I: Anatomy"},
    3: {"title": "The Brain", "file": "books/vol4/03_brain/03_brain.qmd", "part": "Part I: Anatomy"},
    4: {"title": "The Nervous System", "file": "books/vol4/04_nervous/04_nervous.qmd", "part": "Part I: Anatomy"},
    5: {"title": "Data", "file": "books/vol4/05_data/05_data.qmd", "part": "Part II: Teaching"},
    6: {"title": "Training", "file": "books/vol4/06_training/06_training.qmd", "part": "Part II: Teaching"},
    7: {"title": "Evaluation", "file": "books/vol4/07_evaluation/07_evaluation.qmd", "part": "Part II: Teaching"},
    8: {"title": "Perception", "file": "books/vol4/08_perception/08_perception.qmd", "part": "Part III: Running"},
    9: {"title": "Memory", "file": "books/vol4/09_memory/09_memory.qmd", "part": "Part III: Running"},
    10: {"title": "Intent", "file": "books/vol4/10_intent/10_intent.qmd", "part": "Part III: Running"},
    11: {"title": "Planning", "file": "books/vol4/11_planning/11_planning.qmd", "part": "Part III: Running"},
    12: {"title": "Enforcement", "file": "books/vol4/12_enforcement/12_enforcement.qmd", "part": "Part III: Running"},
    13: {"title": "Placement", "file": "books/vol4/13_placement/13_placement.qmd", "part": "Part III: Running"},
    14: {"title": "Intervention", "file": "books/vol4/14_intervention/14_intervention.qmd", "part": "Part IV: Governing"},
    15: {"title": "Verification", "file": "books/vol4/15_verification/15_verification.qmd", "part": "Part IV: Governing"},
    16: {"title": "Release", "file": "books/vol4/16_release/16_release.qmd", "part": "Part IV: Governing"},
    17: {"title": "The Frontier", "file": "books/vol4/17_frontier/17_frontier.qmd", "part": "Conclusion"},
}


class ChapterProgress(BaseModel):
    chapter_num: int
    title: str
    file_path: str
    part: str
    status: str = "PENDING"  # PENDING | IN_PROGRESS | COMPLETED
    total_sections: int = 0
    completed_sections: List[int] = Field(default_factory=list)
    pass1_edits: int = 0
    pass2_cuts: int = 0
    words_trimmed: int = 0
    initial_words: int = 0
    final_words: int = 0
    last_updated: str = ""


class BookManifest(BaseModel):
    started_at: str
    last_updated: str
    current_chapter: int = 1
    total_chapters: int = 17
    completed_chapters_count: int = 0
    total_words_trimmed: int = 0
    total_scaffolding_edits: int = 0
    chapters: Dict[int, ChapterProgress] = Field(default_factory=dict)


class BookOrchestrator:
    def __init__(
        self,
        output_dir: str = "books/vol4/_pedagogical_seminar",
        mode: str = "api",
        model: str = "gpt-4o-mini",
    ):
        self.output_dir = output_dir
        self.mode = mode
        self.model = model
        self.manifest_file = os.path.join(output_dir, "book_audit_manifest.json")
        self.dashboard_file = os.path.join(output_dir, "VOLUME_4_PEDAGOGICAL_DASHBOARD.md")
        os.makedirs(output_dir, exist_ok=True)

        self.vocab = VocabularyLedger(os.path.join(output_dir, "vocabulary_ledger.json"))
        self.reader = StudentReaderEngine(model=model)
        self.room = SeminarRoom(model=model)
        self.pruner = PrunerEngine(model=model)
        import threading
        self._lock = threading.Lock()

        self.manifest = self._load_or_init_manifest()

    def _load_or_init_manifest(self) -> BookManifest:
        now_str = datetime.now().isoformat()
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert chapter keys back to int
                    ch_dict = {int(k): ChapterProgress(**v) for k, v in data.get("chapters", {}).items()}
                    manifest = BookManifest(
                        started_at=data.get("started_at", now_str),
                        last_updated=data.get("last_updated", now_str),
                        current_chapter=data.get("current_chapter", 1),
                        total_chapters=17,
                        completed_chapters_count=data.get("completed_chapters_count", 0),
                        total_words_trimmed=data.get("total_words_trimmed", 0),
                        total_scaffolding_edits=data.get("total_scaffolding_edits", 0),
                        chapters=ch_dict,
                    )
                    return manifest
            except Exception as e:
                print(f"[Warning] Error loading manifest: {e}. Re-initializing.")

        # Initialize fresh manifest
        chapters = {}
        for num, meta in CHAPTER_REGISTRY.items():
            chapters[num] = ChapterProgress(
                chapter_num=num,
                title=meta["title"],
                file_path=meta["file"],
                part=meta["part"],
                status="PENDING",
            )
        manifest = BookManifest(
            started_at=now_str,
            last_updated=now_str,
            chapters=chapters,
        )
        self._save_manifest(manifest)
        return manifest

    def _save_manifest(self, manifest: Optional[BookManifest] = None):
        with self._lock:
            if manifest:
                self.manifest = manifest
            self.manifest.last_updated = datetime.now().isoformat()
            
            # Calculate aggregates
            comp_count = sum(1 for ch in self.manifest.chapters.values() if ch.status == "COMPLETED")
            self.manifest.completed_chapters_count = comp_count
            self.manifest.total_words_trimmed = sum(ch.words_trimmed for ch in self.manifest.chapters.values())
            self.manifest.total_scaffolding_edits = sum(ch.pass1_edits for ch in self.manifest.chapters.values())

            with open(self.manifest_file, "w", encoding="utf-8") as f:
                f.write(self.manifest.model_dump_json(indent=2))

            self.generate_dashboard()

    def process_chapter(
        self,
        chapter_num: int,
        max_sections: Optional[int] = None,
        apply_live: bool = True,
    ) -> ChapterProgress:
        """Process sections of a chapter starting from its checkpoint."""
        ch = self.manifest.chapters.get(chapter_num)
        if not ch:
            raise ValueError(f"Chapter {chapter_num} not in registry.")

        if not os.path.exists(ch.file_path):
            raise FileNotFoundError(f"Chapter file not found: {ch.file_path}")

        reader = StudentReaderEngine(model=self.model)
        room = SeminarRoom(model=self.model)
        pruner = PrunerEngine(model=self.model)

        sections = parse_chapter_sections(ch.file_path)
        ch.total_sections = len(sections)
        ch.status = "IN_PROGRESS"
        self._save_manifest()

        with open(ch.file_path, "r", encoding="utf-8") as f:
            full_content = f.read()

        print("\n" + "=" * 80)
        print(f"📘 ORCHESTRATING CHAPTER {chapter_num}: {ch.title} ({ch.part})")
        print(f"File: {ch.file_path} | Total Sections: {ch.total_sections}")
        print(f"Already completed sections: {ch.completed_sections}")
        print("=" * 80)

        sections_processed_this_run = 0

        for idx, sec in enumerate(sections):
            if idx in ch.completed_sections:
                continue

            if max_sections and sections_processed_this_run >= max_sections:
                print(f"Reached batch limit of {max_sections} section(s). Checkpointing chapter {chapter_num}.")
                break

            from scripts.pedagogy_sim.curriculum import mask_code_blocks, unmask_code_blocks

            initial_w = len(sec.content.split())
            print(f"\n[Ch {chapter_num:02d}: {ch.title}] Section {idx}/{ch.total_sections - 1}: {sec.title} (L{sec.start_line}–L{sec.end_line})")

            # 1. Mask code blocks so simulation NEVER sees or edits mlsysim / python blocks
            masked_content, code_blocks = mask_code_blocks(sec.content)
            masked_sec = TextSection(
                title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                content=masked_content,
                level=sec.level,
            )

            # PASS 1: Grounding & Progressive Disclosure (Prose Only)
            all_notes: List[MarginNote] = []
            for s_id, student in STUDENTS.items():
                notes = reader.simulate_student_reading(student, masked_sec, mode=self.mode)
                all_notes.extend(notes)

            topics = room.hold_seminar_discussion(
                notes=all_notes,
                section_title=f"{ch.title} - {sec.title}",
                section_content=masked_content,
                mode=self.mode,
            )

            working_masked_content = masked_content
            pass1_applied = 0
            for t in topics:
                # Strictly reject any edit attempting to touch code blocks or placeholders
                if "PROTECTED_CODE_BLOCK" in t.original_text or "PROTECTED_CODE_BLOCK" in t.proposed_rewrite:
                    continue
                if "```" in t.original_text or "```" in t.proposed_rewrite:
                    continue

                if t.original_text and t.proposed_rewrite and t.original_text in working_masked_content:
                    working_masked_content = working_masked_content.replace(t.original_text, t.proposed_rewrite, 1)
                    pass1_applied += 1

            # PASS 2: Pruning & Economy of Language (Prose Only)
            updated_masked_sec = TextSection(
                title=sec.title,
                start_line=sec.start_line,
                end_line=sec.end_line,
                content=working_masked_content,
                level=sec.level,
            )
            prune_report = pruner.prune_section(updated_masked_sec, mode=self.mode)
            pass2_applied = 0
            for p_edit in prune_report.edits:
                # Strictly reject any pruning edit attempting to touch code blocks
                if "PROTECTED_CODE_BLOCK" in p_edit.original_text or "PROTECTED_CODE_BLOCK" in p_edit.tightened_text:
                    continue
                if "```" in p_edit.original_text or "```" in p_edit.tightened_text:
                    continue

                if p_edit.original_text in working_masked_content:
                    working_masked_content = working_masked_content.replace(p_edit.original_text, p_edit.tightened_text, 1)
                    pass2_applied += 1

            # Restore original code blocks byte-for-byte
            working_content = unmask_code_blocks(working_masked_content, code_blocks)

            final_w = len(working_content.split())
            words_saved = prune_report.total_words_saved

            if apply_live and working_content != sec.content:
                full_content = full_content.replace(sec.content, working_content, 1)

            # Update progress
            ch.completed_sections.append(idx)
            ch.pass1_edits += pass1_applied
            ch.pass2_cuts += pass2_applied
            ch.words_trimmed += words_saved
            ch.initial_words += initial_w
            ch.final_words += final_w
            ch.last_updated = datetime.now().isoformat()
            sections_processed_this_run += 1

            print(f"  ✓ Pass 1 Scaffolding: {pass1_applied} edit(s) | Pass 2 Pruning: {pass2_applied} cut(s) (-{words_saved} words)")

            # Checkpoint to disk after each section
            self._save_manifest()

        if apply_live and sections_processed_this_run > 0:
            backup_file = f"{ch.file_path}.bak"
            if not os.path.exists(backup_file):
                shutil.copyfile(ch.file_path, backup_file)
            with open(ch.file_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"✅ Saved live updates to {ch.file_path}")

        if len(ch.completed_sections) >= ch.total_sections and ch.total_sections > 0:
            ch.status = "COMPLETED"
            print(f"🎉 CHAPTER {chapter_num} COMPLETED!")
            try:
                from scripts.pedagogy_sim.chapter_signoff import ChapterSignOffEngine
                signoff_engine = ChapterSignOffEngine(
                    output_dir=os.path.join(self.output_dir, "certificates"),
                    model=self.model,
                )
                cert = signoff_engine.conduct_signoff(
                    chapter_num=chapter_num,
                    chapter_title=ch.title,
                    file_path=ch.file_path,
                    summary_stats={
                        "total_sections": ch.total_sections,
                        "pass1_edits": ch.pass1_edits,
                        "words_trimmed": ch.words_trimmed,
                    }
                )
                print(f"🎓 Official Sign-Off Granted: {cert.consensus_verdict} (Score: {cert.overall_clarity_score}/5.0)")
            except Exception as e:
                print(f"[Warning] Failed to generate sign-off certificate: {e}")
        else:
            ch.status = "IN_PROGRESS"

        self._save_manifest()
        return ch

    def run_whole_book(
        self,
        start_chapter: int = 1,
        max_sections_per_batch: int = 5,
        apply_live: bool = True,
    ):
        """Iterate sequentially across the volume from start_chapter to 17."""
        print("=" * 80)
        print("📚 INITIATING WHOLE-BOOK PEDAGOGICAL AUDIT (VOLUME IV)")
        print(f"Chapters: {start_chapter} through 17 | Batch Size: {max_sections_per_batch} sections/step")
        print("=" * 80)

        for ch_num in range(start_chapter, 18):
            ch = self.manifest.chapters[ch_num]
            if ch.status == "COMPLETED":
                print(f"Skipping completed chapter {ch_num}: {ch.title}")
                continue

            self.manifest.current_chapter = ch_num
            self._save_manifest()

            while ch.status != "COMPLETED":
                self.process_chapter(
                    chapter_num=ch_num,
                    max_sections=max_sections_per_batch,
                    apply_live=apply_live,
                )
                if ch.status != "COMPLETED":
                    print(f"Pausing briefly after batch for Chapter {ch_num}...")
                    time.sleep(1)

        print("\n🏆 WHOLE-BOOK PEDAGOGICAL AUDIT COMPLETE!")

    def run_whole_book_parallel(
        self,
        start_chapter: int = 1,
        max_workers: int = 4,
        max_sections_per_batch: int = 5,
        apply_live: bool = True,
    ):
        """Audit remaining chapters concurrently using a thread pool."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print("=" * 80)
        print("🚀 INITIATING PARALLEL WHOLE-BOOK PEDAGOGICAL AUDIT (VOLUME IV)")
        print(f"Workers: {max_workers} concurrent chapters | Batch Size: {max_sections_per_batch} sections/step")
        print("=" * 80)

        chapters_to_run = [
            num for num in range(start_chapter, 18)
            if self.manifest.chapters[num].status != "COMPLETED"
        ]

        if not chapters_to_run:
            print("🏆 All chapters are already marked COMPLETED!")
            return

        print(f"Queueing {len(chapters_to_run)} chapter(s) for parallel execution: {chapters_to_run}")

        def _worker(ch_num: int):
            ch = self.manifest.chapters[ch_num]
            print(f"\n[Worker Starting] Chapter {ch_num}: {ch.title}")
            while ch.status != "COMPLETED":
                self.process_chapter(
                    chapter_num=ch_num,
                    max_sections=max_sections_per_batch,
                    apply_live=apply_live,
                )
                if ch.status != "COMPLETED":
                    time.sleep(1)
            return ch_num

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ch = {executor.submit(_worker, num): num for num in chapters_to_run}
            for fut in as_completed(future_to_ch):
                ch_num = future_to_ch[fut]
                try:
                    fut.result()
                    print(f"\n🎉 [WORKER COMPLETE] Chapter {ch_num} finished and officially signed off!")
                except Exception as e:
                    print(f"\n❌ [WORKER ERROR] Chapter {ch_num} raised exception: {e}")

        print("\n🏆 WHOLE-BOOK PARALLEL AUDIT COMPLETE!")

    def generate_dashboard(self):
        """Render the comprehensive book-wide dashboard in Markdown."""
        total_ch = len(self.manifest.chapters)
        completed_ch = sum(1 for ch in self.manifest.chapters.values() if ch.status == "COMPLETED")
        in_prog_ch = sum(1 for ch in self.manifest.chapters.values() if ch.status == "IN_PROGRESS")
        
        pct = int((completed_ch / total_ch) * 100) if total_ch else 0
        bar_len = 20
        filled = int((pct / 100) * bar_len)
        bar_str = "█" * filled + "░" * (bar_len - filled)

        md = f"""# Volume IV: Physical AI Pedagogical Dashboard

**Book**: *Physical AI: Machine Learning Systems That Sense and Act*  
**Author**: Prof. Vijay Janapa Reddi (Harvard University)  
**Last Synchronized**: `{self.manifest.last_updated}`  
**Pipeline**: Autonomous Two-Pass Agentic Loop (Pass 1 Grounding + Pass 2 Pruning)

---

## Overall Progress

```text
Progress: {bar_str}  {pct}% ({completed_ch}/{total_ch} chapters completed)
```

- 🟢 **Completed Chapters**: {completed_ch} / {total_ch}
- 🟡 **In-Progress Chapters**: {in_prog_ch}
- ⚪ **Pending Chapters**: {total_ch - completed_ch - in_prog_ch}
- 🛠️ **Total Pass 1 Scaffolding Edits Applied**: **{self.manifest.total_scaffolding_edits}**
- ✂️ **Total Pass 2 Words Pruned**: **{self.manifest.total_words_trimmed} words**

---

## Chapter-by-Chapter Status Table

| Ch | Part | Title | Status | Sections Completed | Pass 1 Edits | Pass 2 Cuts | Words Trimmed |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
"""

        for num in range(1, 18):
            ch = self.manifest.chapters[num]
            if ch.status == "COMPLETED":
                icon = "🟢 Completed"
            elif ch.status == "IN_PROGRESS":
                icon = "🟡 In Progress"
            else:
                icon = "⚪ Pending"

            sec_str = f"{len(ch.completed_sections)}/{ch.total_sections}" if ch.total_sections else "0/--"
            md += f"| **{num:02d}** | {ch.part} | **{ch.title}** | {icon} | {sec_str} | {ch.pass1_edits} | {ch.pass2_cuts} | -{ch.words_trimmed} w |\n"

        md += """
---

## Student Cohort Reviewers

| Evaluator | Persona & Background | Core Oversight Mandate |
|:---|:---|:---|
| **Alex Chen** | MSc in CS / Deep Learning | Grounds foundation models in physical latency and actuation limits. |
| **Priya Patel** | PhD in Computer Systems | Enforces real-time bus determinism, memory bandwidth, and silicon partitioning. |
| **Marcus Vance** | PhD in MechE / Control | Enforces Newton-Euler mechanics, stopping distance bounds, and actuator saturation. |
| **Elena Rostova** | Senior Undergrad in EECS | **Guardian of Progressive Disclosure**: Audits acronyms, cognitive load, and flow. |
| **Dr. Aris Thorne** | Lead TA & Moderator | Convenes weekly seminar discussions and synthesizes surgical rewrites. |

---

## How to Control the Pipeline

```bash
# Resume audit for the current chapter (batch of 5 sections):
.venv/bin/python -m scripts.pedagogy_sim.book_orchestrator --step

# Process next chapter specifically:
.venv/bin/python -m scripts.pedagogy_sim.book_orchestrator --chapter 2 --batch-size 5

# Run whole book in parallel (4 concurrent chapters):
.venv/bin/python -m scripts.pedagogy_sim.book_orchestrator --all --parallel 4
```
"""

        with open(self.dashboard_file, "w", encoding="utf-8") as f:
            f.write(md)


def main():
    parser = argparse.ArgumentParser(description="Full-Book Agentic Loop Orchestrator for Volume IV")
    parser.add_argument("--all", action="store_true", help="Run through all chapters to completion")
    parser.add_argument("--step", action="store_true", help="Process the next batch for current chapter")
    parser.add_argument("--chapter", type=int, default=None, help="Process specific chapter")
    parser.add_argument("--batch-size", type=int, default=5, help="Sections per batch")
    parser.add_argument("--parallel", type=int, default=4, help="Number of concurrent chapter workers (default 4)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without writing edits to book source")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--output-dir", type=str, default="books/vol4/_pedagogical_seminar", help="Output directory")

    args = parser.parse_args()
    orch = BookOrchestrator(output_dir=args.output_dir, model=args.model)

    if args.all:
        if args.parallel > 1:
            orch.run_whole_book_parallel(
                start_chapter=1,
                max_workers=args.parallel,
                max_sections_per_batch=args.batch_size,
                apply_live=not args.dry_run,
            )
        else:
            orch.run_whole_book(
                start_chapter=orch.manifest.current_chapter,
                max_sections_per_batch=args.batch_size,
                apply_live=not args.dry_run,
            )
    elif args.chapter:
        orch.process_chapter(
            chapter_num=args.chapter,
            max_sections=args.batch_size,
            apply_live=not args.dry_run,
        )
    elif args.step:
        orch.process_chapter(
            chapter_num=orch.manifest.current_chapter,
            max_sections=args.batch_size,
            apply_live=not args.dry_run,
        )
    else:
        # Default: generate dashboard and display status
        orch.generate_dashboard()
        print(f"Pedagogical Dashboard generated at: {orch.dashboard_file}")


if __name__ == "__main__":
    main()
