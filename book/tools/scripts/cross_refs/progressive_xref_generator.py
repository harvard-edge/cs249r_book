#!/usr/bin/env python3
"""
Progressive cross-reference generator that creates a narrative arc through chapters.
Maintains context to ensure diverse, meaningful connections that evolve with the content.
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import yaml

@dataclass
class ChapterProfile:
    """Profile of a chapter's content and relationships"""
    name: str
    category: str
    complexity_level: int  # 1=basic, 2=intermediate, 3=advanced
    prerequisites: List[str] = field(default_factory=list)
    leads_to: List[str] = field(default_factory=list)
    complements: List[str] = field(default_factory=list)
    concepts: Set[str] = field(default_factory=set)

@dataclass
class SectionContext:
    """Context for a specific section within a chapter"""
    section_id: str
    title: str
    index: int
    total_sections: int
    content_preview: str
    narrative_stage: str
    key_concepts: List[str]

class ProgressiveXRefGenerator:
    def __init__(self, model="gemma2:27b"):
        self.model = model
        self.ollama_url = "http://localhost:11434/api/generate"

        # Initialize chapter profiles with relationships
        self.chapter_profiles = self._initialize_chapter_profiles()

        # Track state across generation
        self.connection_history = []  # All connections made so far
        self.chapter_exposure = defaultdict(lambda: {
            'count': 0,
            'last_section': -1,
            'connection_types': set()
        })

    def _initialize_chapter_profiles(self) -> Dict[str, ChapterProfile]:
        """Initialize detailed profiles for each chapter"""

        profiles = {
            # Fundamentals
            "introduction": ChapterProfile(
                name="introduction", category="foundation", complexity_level=1,
                leads_to=["ml_systems", "dl_primer"],
                complements=["workflow"]
            ),
            "ml_systems": ChapterProfile(
                name="ml_systems", category="foundation", complexity_level=1,
                prerequisites=["introduction"],
                leads_to=["frameworks", "workflow", "data_engineering"],
                complements=["dl_primer"]
            ),
            "dl_primer": ChapterProfile(
                name="dl_primer", category="foundation", complexity_level=1,
                prerequisites=["introduction"],
                leads_to=["frameworks", "training"],
                complements=["ml_systems"]
            ),

            # Implementation
            "workflow": ChapterProfile(
                name="workflow", category="implementation", complexity_level=2,
                prerequisites=["ml_systems"],
                leads_to=["data_engineering", "ops"],
                complements=["frameworks"]
            ),
            "data_engineering": ChapterProfile(
                name="data_engineering", category="implementation", complexity_level=2,
                prerequisites=["workflow", "ml_systems"],
                leads_to=["training", "ops"],
                complements=["frameworks"]
            ),
            "frameworks": ChapterProfile(
                name="frameworks", category="implementation", complexity_level=2,
                prerequisites=["ml_systems", "dl_primer"],
                leads_to=["training", "optimizations"],
                complements=["workflow", "data_engineering"]
            ),
            "training": ChapterProfile(
                name="training", category="implementation", complexity_level=2,
                prerequisites=["dl_primer", "frameworks"],
                leads_to=["optimizations", "efficient_ai"],
                complements=["data_engineering"]
            ),

            # Optimization & Performance
            "efficient_ai": ChapterProfile(
                name="efficient_ai", category="optimization", complexity_level=2,
                prerequisites=["training"],
                leads_to=["optimizations", "hw_acceleration"],
                complements=["benchmarking"]
            ),
            "optimizations": ChapterProfile(
                name="optimizations", category="optimization", complexity_level=2,
                prerequisites=["training", "frameworks"],
                leads_to=["hw_acceleration", "benchmarking"],
                complements=["efficient_ai"]
            ),
            "hw_acceleration": ChapterProfile(
                name="hw_acceleration", category="optimization", complexity_level=3,
                prerequisites=["optimizations"],
                leads_to=["ondevice_learning"],
                complements=["benchmarking", "efficient_ai"]
            ),
            "benchmarking": ChapterProfile(
                name="benchmarking", category="optimization", complexity_level=2,
                prerequisites=["training"],
                complements=["optimizations", "efficient_ai"]
            ),

            # Operations & Deployment
            "ops": ChapterProfile(
                name="ops", category="operations", complexity_level=2,
                prerequisites=["workflow", "data_engineering"],
                leads_to=["ondevice_learning"],
                complements=["benchmarking"]
            ),
            "ondevice_learning": ChapterProfile(
                name="ondevice_learning", category="operations", complexity_level=3,
                prerequisites=["ops", "efficient_ai"],
                complements=["hw_acceleration", "privacy_security"]
            ),

            # Advanced Topics
            "privacy_security": ChapterProfile(
                name="privacy_security", category="advanced", complexity_level=3,
                prerequisites=["ops"],
                complements=["responsible_ai", "robust_ai"]
            ),
            "responsible_ai": ChapterProfile(
                name="responsible_ai", category="advanced", complexity_level=3,
                prerequisites=["ml_systems"],
                complements=["privacy_security", "ai_for_good"]
            ),
            "robust_ai": ChapterProfile(
                name="robust_ai", category="advanced", complexity_level=3,
                prerequisites=["training"],
                complements=["privacy_security", "responsible_ai"]
            ),

            # Specialized
            "generative_ai": ChapterProfile(
                name="generative_ai", category="specialized", complexity_level=3,
                prerequisites=["dl_primer", "training"],
                complements=["responsible_ai"]
            ),
            "sustainable_ai": ChapterProfile(
                name="sustainable_ai", category="specialized", complexity_level=3,
                prerequisites=["efficient_ai"],
                complements=["ai_for_good", "responsible_ai"]
            ),
            "ai_for_good": ChapterProfile(
                name="ai_for_good", category="specialized", complexity_level=3,
                prerequisites=["responsible_ai"],
                complements=["sustainable_ai"]
            ),

            # Future
            "frontiers": ChapterProfile(
                name="frontiers", category="future", complexity_level=3,
                prerequisites=["generative_ai"],
                complements=["emerging_topics"]
            ),
            "emerging_topics": ChapterProfile(
                name="emerging_topics", category="future", complexity_level=3,
                complements=["frontiers"]
            ),
            "conclusion": ChapterProfile(
                name="conclusion", category="summary", complexity_level=1,
                prerequisites=["introduction"],
                complements=["frontiers", "emerging_topics"]
            )
        }

        return profiles

    def extract_sections(self, chapter_path: Path) -> List[SectionContext]:
        """Extract sections with their context from chapter"""

        with open(chapter_path) as f:
            content = f.read()

        sections = []
        lines = content.split('\n')

        for i, line in enumerate(lines):
            if line.startswith('## ') and '{#sec-' in line:
                # Extract section info
                section_id = re.search(r'{#(sec-[^}]+)}', line).group(1)
                title = re.sub(r'\s*{#[^}]+}', '', line.replace('## ', ''))

                # Get content preview (next 100 lines or until next section)
                preview_lines = []
                for j in range(i+1, min(i+101, len(lines))):
                    if lines[j].startswith('## '):
                        break
                    preview_lines.append(lines[j])

                content_preview = '\n'.join(preview_lines[:50])  # First 50 lines

                # Extract key concepts from preview
                key_concepts = self.extract_concepts(content_preview)

                sections.append(SectionContext(
                    section_id=section_id,
                    title=title,
                    index=len(sections),
                    total_sections=0,  # Will update after
                    content_preview=content_preview,
                    narrative_stage="",  # Will calculate
                    key_concepts=key_concepts
                ))

        # Update total sections and narrative stages
        total = len(sections)
        for section in sections:
            section.total_sections = total
            section.narrative_stage = self.determine_narrative_stage(
                section.index, total
            )

        return sections

    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple extraction - could be enhanced with NLP
        concepts = []

        # Look for emphasized terms
        for match in re.finditer(r'\*\*([^*]+)\*\*', text):
            concepts.append(match.group(1).lower())

        for match in re.finditer(r'\*([^*]+)\*', text):
            concepts.append(match.group(1).lower())

        # Look for technical terms
        technical_patterns = [
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
            r'\b(?:learning|training|model|data|system|network|algorithm)\b',
            r'\b(?:performance|optimization|deployment|inference)\b'
        ]

        for pattern in technical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                concepts.append(match.group(0).lower())

        return list(set(concepts))[:10]  # Top 10 unique concepts

    def determine_narrative_stage(self, index: int, total: int) -> str:
        """Determine narrative stage based on position"""
        position = index / max(total - 1, 1)

        if position < 0.25:
            return "foundation"  # Setting the stage
        elif position < 0.5:
            return "development"  # Building concepts
        elif position < 0.75:
            return "application"  # Practical aspects
        else:
            return "synthesis"  # Advanced/future topics

    def find_relevant_section(self, target_chapter: str, key_concepts: List[str]) -> Optional[str]:
        """Find the most relevant section in target chapter based on concepts"""

        # For now, return a generic overview section
        # In production, this would analyze the target chapter's sections
        # and find the best match based on concept overlap

        # Common section patterns
        overview_patterns = [
            f"sec-{target_chapter}-overview",
            f"sec-{target_chapter}-introduction",
            f"sec-{target_chapter}-basics",
            f"sec-{target_chapter}-fundamentals"
        ]

        # For now, return the overview section
        # You could enhance this to actually read the target chapter
        # and find the most relevant section
        return f"sec-{target_chapter}-overview"

    def select_connections(self,
                          section: SectionContext,
                          source_chapter: str,
                          max_refs: int = 4) -> List[Dict]:
        """Select diverse, contextually appropriate connections"""

        connections = []
        source_profile = self.chapter_profiles.get(source_chapter)

        if not source_profile:
            return connections

        # Score each potential target chapter
        scores = {}

        for target_name, target_profile in self.chapter_profiles.items():
            if target_name == source_chapter:
                continue

            score = 0

            # 1. Narrative appropriateness
            stage_bonus = {
                "foundation": {"foundation": 10, "implementation": 5},
                "development": {"implementation": 10, "optimization": 7, "foundation": 3},
                "application": {"optimization": 10, "operations": 8, "advanced": 5},
                "synthesis": {"advanced": 10, "specialized": 8, "future": 10}
            }

            score += stage_bonus.get(section.narrative_stage, {}).get(
                target_profile.category, 0
            )

            # 2. Relationship strength
            if target_name in source_profile.prerequisites:
                score += 8  # Strong prerequisite connection
            if target_name in source_profile.leads_to:
                score += 6  # Natural progression
            if target_name in source_profile.complements:
                score += 4  # Complementary material

            # 3. Complexity appropriateness
            complexity_diff = abs(target_profile.complexity_level -
                                 (section.index / section.total_sections * 3))
            score -= complexity_diff * 2  # Penalty for complexity mismatch

            # 4. Diversity penalty (avoid overexposure)
            exposure = self.chapter_exposure[target_name]
            if exposure['count'] > 0:
                # Stronger penalty if shown recently
                recency_penalty = max(0, 5 - (section.index - exposure['last_section']))
                score -= exposure['count'] * 3 + recency_penalty

            # 5. Concept relevance
            target_concepts = target_profile.concepts
            if target_concepts:
                overlap = len(set(section.key_concepts) & target_concepts)
                score += overlap * 2

            scores[target_name] = max(0, score)

        # Select top scoring chapters
        sorted_targets = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for target_name, score in sorted_targets[:max_refs]:
            if score == 0:
                continue

            target_profile = self.chapter_profiles[target_name]

            # Determine connection type
            connection_type = self.determine_connection_type(
                source_profile, target_profile, section
            )

            # Generate explanation
            explanation = self.generate_progressive_explanation(
                section, source_chapter, target_name, connection_type
            )

            # Calculate priority
            priority = 1 if score > 15 else (2 if score > 8 else 3)

            # Try to find a relevant section in the target chapter
            target_section = self.find_relevant_section(target_name, section.key_concepts)

            connection = {
                "target_chapter": target_name,
                "target_section": target_section,
                "connection_type": connection_type,
                "explanation": explanation,
                "priority": priority,
                "strength": min(0.9, score / 20),  # Normalize to 0-0.9
                "quality": 0.85,
                "narrative_fit": section.narrative_stage
            }

            connections.append(connection)

            # Update exposure tracking
            self.chapter_exposure[target_name]['count'] += 1
            self.chapter_exposure[target_name]['last_section'] = section.index
            self.chapter_exposure[target_name]['connection_types'].add(connection_type)

        return connections

    def determine_connection_type(self,
                                 source: ChapterProfile,
                                 target: ChapterProfile,
                                 section: SectionContext) -> str:
        """Determine appropriate connection type"""

        # Based on relationships
        if target.name in source.prerequisites:
            return "prerequisite"
        elif target.name in source.leads_to:
            return "extends" if section.narrative_stage in ["development", "application"] else "foundation"
        elif target.name in source.complements:
            return "complements"

        # Based on categories and stages
        stage_types = {
            "foundation": {
                "foundation": "prerequisite",
                "implementation": "foundation"
            },
            "development": {
                "implementation": "extends",
                "optimization": "extends",
                "foundation": "builds_on"
            },
            "application": {
                "operations": "applies",
                "optimization": "optimizes",
                "advanced": "considers"
            },
            "synthesis": {
                "advanced": "explores",
                "specialized": "specializes",
                "future": "anticipates"
            }
        }

        return stage_types.get(section.narrative_stage, {}).get(
            target.category, "complements"
        )

    def generate_progressive_explanation(self,
                                        section: SectionContext,
                                        source: str,
                                        target: str,
                                        connection_type: str) -> str:
        """Generate contextually appropriate explanation"""

        # Build context for LLM
        prompt = f"""
        Generate a brief, specific explanation for this cross-reference.

        Context:
        - Section: "{section.title}" (position {section.index+1}/{section.total_sections})
        - Narrative stage: {section.narrative_stage}
        - From: {source} TO: {target}
        - Connection type: {connection_type}
        - Key concepts in section: {', '.join(section.key_concepts[:5])}

        Previous connections to {target}: {self.chapter_exposure[target]['count']}

        Requirements:
        1. Write one clear, concise sentence (not just a fragment)
        2. Be specific to this section's content and how it connects
        3. Avoid generic phrases like "builds on" or "extends"
        4. If {target} was shown before, reference a different aspect
        5. Match narrative stage tone:
           - foundation: "Essential concepts for..."
           - development: "Practical patterns for..."
           - application: "Production considerations for..."
           - synthesis: "Advanced techniques in..."

        Return only the explanation text.
        """

        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.8,
            "max_tokens": 100
        }

        try:
            response = requests.post(self.ollama_url, json=data, timeout=30)
            if response.status_code == 200:
                explanation = response.json()['response'].strip()
                # Remove any trailing periods for consistency
                explanation = explanation.rstrip('.')
                return explanation
        except Exception as e:
            print(f"    Warning: LLM generation failed: {e}")

        # Contextual fallbacks
        fallbacks = {
            ("foundation", "prerequisite"): f"Essential {target.replace('_', ' ')} foundations",
            ("foundation", "foundation"): f"Core {target.replace('_', ' ')} concepts",
            ("development", "extends"): f"Building with {target.replace('_', ' ')}",
            ("development", "builds_on"): f"Applying {target.replace('_', ' ')} principles",
            ("application", "applies"): f"Production {target.replace('_', ' ')} patterns",
            ("application", "optimizes"): f"Optimizing via {target.replace('_', ' ')}",
            ("synthesis", "explores"): f"Advanced {target.replace('_', ' ')} topics",
            ("synthesis", "anticipates"): f"Future of {target.replace('_', ' ')}"
        }

        return fallbacks.get(
            (section.narrative_stage, connection_type),
            f"{target.replace('_', ' ')} insights"
        )

    def generate_chapter_xrefs(self, chapter_path: Path) -> Dict:
        """Generate progressive cross-references for entire chapter"""

        chapter_name = chapter_path.stem
        print(f"\n🚀 Generating progressive xrefs for: {chapter_name}")
        print("=" * 60)

        # Reset state for new chapter
        self.connection_history.clear()
        self.chapter_exposure.clear()

        # Extract sections with context
        sections = self.extract_sections(chapter_path)

        if not sections:
            print("❌ No sections found!")
            return {}

        print(f"📊 Found {len(sections)} sections")
        print(f"   Narrative flow: foundation → development → application → synthesis\n")

        # Generate connections for each section
        xrefs_data = {"cross_references": {}}

        for section in sections:
            print(f"📝 Section {section.index+1}/{len(sections)}: {section.title[:40]}...")
            print(f"   Stage: {section.narrative_stage}")

            # Select contextually appropriate connections
            connections = self.select_connections(section, chapter_name)

            if connections:
                xrefs_data["cross_references"][section.section_id] = connections

                # Log what was connected
                chapters = [c["target_chapter"] for c in connections]
                print(f"   Connected to: {', '.join(chapters)}")

                # Track for context
                self.connection_history.extend(connections)
            else:
                print(f"   No suitable connections found")

        # Summary statistics
        self.print_summary(xrefs_data)

        return xrefs_data

    def print_summary(self, xrefs_data: Dict):
        """Print generation summary"""

        print("\n" + "=" * 60)
        print("📈 GENERATION SUMMARY")
        print("=" * 60)

        # Count unique chapters
        all_chapters = set()
        connection_types = defaultdict(int)
        priorities = defaultdict(int)

        for refs in xrefs_data["cross_references"].values():
            for ref in refs:
                all_chapters.add(ref["target_chapter"])
                connection_types[ref["connection_type"]] += 1
                priorities[ref["priority"]] += 1

        print(f"\n✅ Unique chapters referenced: {len(all_chapters)}")
        print(f"   Chapters: {', '.join(sorted(all_chapters))}")

        print(f"\n📊 Chapter exposure distribution:")
        for chapter, data in sorted(self.chapter_exposure.items(),
                                   key=lambda x: x[1]['count'], reverse=True):
            print(f"   {chapter:20} : {data['count']} references")

        print(f"\n🔗 Connection types:")
        for conn_type, count in sorted(connection_types.items()):
            print(f"   {conn_type:15} : {count}")

        print(f"\n⭐ Priority distribution:")
        for priority in sorted(priorities.keys()):
            print(f"   Priority {priority}: {priorities[priority]} connections")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate progressive cross-references")
    parser.add_argument("chapter", nargs="?", default="introduction",
                       help="Chapter name to process")
    parser.add_argument("--model", default="gemma2:27b",
                       help="Ollama model to use")
    parser.add_argument("--all", action="store_true",
                       help="Process all chapters")

    args = parser.parse_args()

    generator = ProgressiveXRefGenerator(model=args.model)

    base_path = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")

    if args.all:
        # Process all chapters
        for chapter_dir in sorted(base_path.iterdir()):
            if chapter_dir.is_dir():
                chapter_file = chapter_dir / f"{chapter_dir.name}.qmd"
                if chapter_file.exists():
                    xrefs = generator.generate_chapter_xrefs(chapter_file)

                    output_path = chapter_dir / f"{chapter_dir.name}_xrefs.json"
                    with open(output_path, 'w') as f:
                        json.dump(xrefs, f, indent=2)

                    print(f"💾 Saved to {output_path}\n")
    else:
        # Process single chapter
        chapter_path = base_path / args.chapter / f"{args.chapter}.qmd"

        if chapter_path.exists():
            xrefs = generator.generate_chapter_xrefs(chapter_path)

            output_path = chapter_path.parent / f"{args.chapter}_xrefs.json"
            with open(output_path, 'w') as f:
                json.dump(xrefs, f, indent=2)

            print(f"\n💾 Saved to {output_path}")
        else:
            print(f"❌ Chapter not found: {chapter_path}")


if __name__ == "__main__":
    main()
