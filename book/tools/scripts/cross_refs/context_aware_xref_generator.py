#!/usr/bin/env python3
"""
Context-aware cross-reference generator that processes entire chapters holistically.
Maintains context across sections to create progressive, diverse connections.
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import yaml

class ContextAwareXRefGenerator:
    def __init__(self, model="gemma2:27b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"

        # Track what we've shown across the chapter
        self.shown_chapters = defaultdict(int)  # chapter -> count
        self.section_connections = {}  # section_id -> list of connected chapters
        self.narrative_stage = "introduction"  # introduction -> development -> advanced

        # Chapter progression categories
        self.chapter_categories = {
            "fundamentals": ["ml_systems", "dl_primer", "workflow"],
            "implementation": ["frameworks", "data_engineering", "training"],
            "optimization": ["efficient_ai", "optimizations", "benchmarking", "hw_acceleration"],
            "operations": ["ops", "ondevice_learning"],
            "advanced": ["privacy_security", "responsible_ai", "robust_ai"],
            "specialized": ["generative_ai", "sustainable_ai", "ai_for_good"],
            "future": ["frontiers", "emerging_topics"]
        }

        # Reverse mapping for quick lookup
        self.chapter_to_category = {}
        for category, chapters in self.chapter_categories.items():
            for chapter in chapters:
                self.chapter_to_category[chapter] = category

    def get_chapter_concepts(self, chapter_path: Path) -> Dict:
        """Load concepts from chapter's concept map"""
        chapter_name = chapter_path.parent.name
        concepts_file = chapter_path.parent / f"{chapter_name}_concepts.yml"

        if concepts_file.exists():
            with open(concepts_file) as f:
                return yaml.safe_load(f)
        return {}

    def get_narrative_stage(self, section_index: int, total_sections: int) -> str:
        """Determine narrative stage based on section position"""
        position = section_index / total_sections

        if position < 0.3:
            return "introduction"  # Focus on fundamentals
        elif position < 0.7:
            return "development"   # Mix of implementation and optimization
        else:
            return "advanced"      # Advanced topics and future directions

    def select_diverse_chapters(self,
                               section_content: str,
                               section_index: int,
                               total_sections: int,
                               max_connections: int = 5) -> List[str]:
        """Select diverse chapters based on context and narrative position"""

        stage = self.get_narrative_stage(section_index, total_sections)
        selected = []

        # Define which categories to prioritize per stage
        stage_priorities = {
            "introduction": ["fundamentals", "implementation"],
            "development": ["implementation", "optimization", "operations"],
            "advanced": ["advanced", "specialized", "future"]
        }

        priority_categories = stage_priorities[stage]

        # Score chapters based on:
        # 1. Relevance to current stage
        # 2. How many times they've been shown (prefer less shown)
        # 3. Category diversity

        chapter_scores = {}
        used_categories = set()

        for category in priority_categories:
            for chapter in self.chapter_categories.get(category, []):
                # Base score from stage relevance
                score = 10 if category in priority_categories[:2] else 5

                # Penalty for overuse
                shown_count = self.shown_chapters[chapter]
                score -= shown_count * 2

                # Bonus for category diversity
                if category not in used_categories:
                    score += 3

                chapter_scores[chapter] = score

        # Sort by score and select top N
        sorted_chapters = sorted(chapter_scores.items(), key=lambda x: x[1], reverse=True)

        for chapter, score in sorted_chapters[:max_connections]:
            selected.append(chapter)
            self.shown_chapters[chapter] += 1
            category = self.chapter_to_category.get(chapter)
            if category:
                used_categories.add(category)

        return selected

    def generate_context_aware_explanation(self,
                                          source_section: str,
                                          target_chapter: str,
                                          section_index: int,
                                          previous_explanations: List[str]) -> str:
        """Generate explanation that avoids repetition and fits narrative flow"""

        stage = self.narrative_stage
        category = self.chapter_to_category.get(target_chapter, "general")

        # Build context for LLM
        context = f"""
        Generate a concise, specific explanation for a cross-reference connection.

        Source section: {source_section}
        Target chapter: {target_chapter}
        Chapter category: {category}
        Narrative stage: {stage}

        Previous explanations used (avoid repetition):
        {json.dumps(previous_explanations[-3:], indent=2) if previous_explanations else "None"}

        Requirements:
        1. 10-15 words maximum
        2. Be specific to the connection context
        3. Avoid generic phrases like "builds on", "extends"
        4. Focus on the specific value this connection provides
        5. Match the narrative stage (early=foundational, middle=practical, late=advanced)

        Generate only the explanation text, nothing else.
        """

        data = {
            "model": self.model,
            "prompt": context,
            "stream": False,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.ollama_url, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()['response'].strip()
        except Exception as e:
            print(f"Error generating explanation: {e}")

        # Fallback explanations based on category
        fallbacks = {
            "fundamentals": f"Core {target_chapter.replace('_', ' ')} concepts",
            "implementation": f"Practical {target_chapter.replace('_', ' ')} patterns",
            "optimization": f"Performance via {target_chapter.replace('_', ' ')}",
            "operations": f"Production {target_chapter.replace('_', ' ')} practices",
            "advanced": f"Critical {target_chapter.replace('_', ' ')} considerations",
            "specialized": f"Specialized {target_chapter.replace('_', ' ')} applications",
            "future": f"Emerging {target_chapter.replace('_', ' ')} directions"
        }

        return fallbacks.get(category, f"See {target_chapter.replace('_', ' ')}")

    def generate_chapter_xrefs(self, chapter_path: Path) -> Dict:
        """Generate cross-references for entire chapter with context awareness"""

        print(f"\n📖 Generating context-aware xrefs for {chapter_path.name}")

        # Reset tracking for new chapter
        self.shown_chapters.clear()
        self.section_connections.clear()

        # Load chapter content and extract sections
        with open(chapter_path) as f:
            content = f.read()

        # Extract sections (simplified - you'd parse properly)
        sections = []
        for line in content.split('\n'):
            if line.startswith('## ') and '{#sec-' in line:
                # Extract section ID
                section_id = line.split('{#')[1].split('}')[0]
                section_title = line.split('## ')[1].split(' {#')[0]
                sections.append((section_id, section_title))

        if not sections:
            print("No sections found")
            return {}

        print(f"Found {len(sections)} sections")

        # Generate xrefs for each section with context
        xrefs_data = {"cross_references": {}}
        previous_explanations = []

        for idx, (section_id, section_title) in enumerate(sections):
            print(f"  Processing section {idx+1}/{len(sections)}: {section_title}")

            # Update narrative stage
            self.narrative_stage = self.get_narrative_stage(idx, len(sections))

            # Select diverse chapters for this section
            target_chapters = self.select_diverse_chapters(
                section_title,
                idx,
                len(sections),
                max_connections=3  # Fewer per section for more diversity
            )

            # Generate connections
            section_refs = []
            for target_chapter in target_chapters:
                # Determine connection type based on stage and category
                connection_type = self.determine_connection_type(
                    self.narrative_stage,
                    self.chapter_to_category.get(target_chapter, "general")
                )

                # Generate unique explanation
                explanation = self.generate_context_aware_explanation(
                    section_title,
                    target_chapter,
                    idx,
                    previous_explanations
                )

                previous_explanations.append(explanation)

                ref = {
                    "target_chapter": target_chapter,
                    "connection_type": connection_type,
                    "explanation": explanation,
                    "priority": self.calculate_priority(idx, len(sections), target_chapter),
                    "strength": 0.3 + (0.4 * (1 - idx/len(sections))),  # Stronger early
                    "quality": 0.9
                }

                section_refs.append(ref)

            xrefs_data["cross_references"][section_id] = section_refs
            self.section_connections[section_id] = target_chapters

        # Summary statistics
        all_chapters = set()
        for refs in xrefs_data["cross_references"].values():
            for ref in refs:
                all_chapters.add(ref["target_chapter"])

        print(f"\n✅ Generated xrefs with {len(all_chapters)} unique chapters")
        print(f"   Chapter distribution: {dict(self.shown_chapters)}")

        return xrefs_data

    def determine_connection_type(self, stage: str, category: str) -> str:
        """Determine connection type based on narrative stage and category"""

        type_mapping = {
            ("introduction", "fundamentals"): "prerequisite",
            ("introduction", "implementation"): "foundation",
            ("development", "implementation"): "extends",
            ("development", "optimization"): "complements",
            ("development", "operations"): "extends",
            ("advanced", "advanced"): "extends",
            ("advanced", "specialized"): "complements",
            ("advanced", "future"): "extends"
        }

        return type_mapping.get((stage, category), "complements")

    def calculate_priority(self, section_index: int, total_sections: int, chapter: str) -> int:
        """Calculate priority based on position and importance"""

        # Fundamentals get priority 1 in early sections
        if section_index < total_sections * 0.3:
            if self.chapter_to_category.get(chapter) == "fundamentals":
                return 1

        # Implementation gets priority 1 in middle sections
        if section_index < total_sections * 0.7:
            if self.chapter_to_category.get(chapter) in ["implementation", "optimization"]:
                return 1

        # Default priority based on position
        if section_index < total_sections * 0.5:
            return 2
        else:
            return 3


def main():
    generator = ContextAwareXRefGenerator()

    # Process introduction chapter as example
    intro_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core/introduction/introduction.qmd")

    if intro_path.exists():
        xrefs = generator.generate_chapter_xrefs(intro_path)

        # Save results
        output_path = intro_path.parent / f"{intro_path.stem}_xrefs_contextual.json"
        with open(output_path, 'w') as f:
            json.dump(xrefs, f, indent=2)

        print(f"\n💾 Saved to {output_path}")
    else:
        print(f"File not found: {intro_path}")


if __name__ == "__main__":
    main()
