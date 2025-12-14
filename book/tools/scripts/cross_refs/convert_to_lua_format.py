#!/usr/bin/env python3
"""
Convert Production Cross-Reference Files to Lua Filter Format

This script converts our production cross-reference system files to the format
expected by the existing inject_crossrefs.lua filter.
"""

import os
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

class CrossRefConverter:

    def __init__(self):
        self.base_dir = Path("/Users/VJ/GitHub/MLSysBook/quarto/contents/core")
        self.chapters = [
            'introduction', 'ml_systems', 'dl_primer', 'dnn_architectures', 'workflow',
            'data_engineering', 'frameworks', 'training', 'efficient_ai', 'optimizations',
            'hw_acceleration', 'benchmarking', 'ondevice_learning', 'ops', 'privacy_security',
            'responsible_ai', 'sustainable_ai', 'ai_for_good', 'robust_ai', 'conclusion'
        ]

        # Connection type mapping from our system to Lua filter format
        self.connection_mapping = {
            'foundation': 'Background',
            'prerequisite': 'Background',
            'extends': 'Preview',
            'applies': 'Preview',
            'complements': 'Preview'
        }

    def load_chapter_xrefs(self, chapter: str) -> Dict:
        """Load cross-reference data for a chapter"""
        xref_file = self.base_dir / chapter / f"{chapter}_xrefs.json"

        if not xref_file.exists():
            print(f"Warning: No cross-reference file found for {chapter}")
            return {}

        try:
            with open(xref_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('cross_references', {})
        except Exception as e:
            print(f"Error loading {xref_file}: {e}")
            return {}

    def extract_section_title_from_qmd(self, chapter: str, section_id: str) -> str:
        """Extract section title from QMD file"""
        qmd_file = self.base_dir / chapter / f"{chapter}.qmd"

        if not qmd_file.exists():
            return "Unknown Section"

        try:
            with open(qmd_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Look for section with this ID
                pattern = rf'#{1,6}.*?\{{#{re.escape(section_id)}\}}'
                matches = re.finditer(pattern, content, re.MULTILINE)

                for match in matches:
                    line = match.group(0)
                    # Extract title between # and {
                    title_match = re.search(r'#{1,6}\s*(.+?)\s*\{', line)
                    if title_match:
                        return title_match.group(1).strip()

        except Exception as e:
            print(f"Error reading {qmd_file}: {e}")

        return "Unknown Section"

    def get_target_section_title(self, target_chapter: str, target_section: str) -> str:
        """Get the title of a target section"""
        return self.extract_section_title_from_qmd(target_chapter, target_section)

    def convert_to_lua_format(self) -> Dict:
        """Convert all chapter cross-references to Lua filter format"""

        print("🔄 Converting cross-references to Lua filter format...")

        lua_format = {
            "metadata": {
                "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "model_used": "production_xref_generator.py",
                "model_type": "Production-ready",
                "approach": "concept_driven_production"
            },
            "cross_references": []
        }

        total_connections = 0
        total_sections = 0
        chapters_processed = 0

        for chapter in self.chapters:
            print(f"  Processing {chapter}...")

            chapter_xrefs = self.load_chapter_xrefs(chapter)
            if not chapter_xrefs:
                continue

            # Convert chapter data to Lua format
            chapter_data = {
                "file": f"{chapter}.qmd",
                "sections": []
            }

            for source_section_id, connections in chapter_xrefs.items():
                if not connections:
                    continue

                section_title = self.extract_section_title_from_qmd(chapter, source_section_id)

                section_data = {
                    "section_id": source_section_id,
                    "section_title": section_title,
                    "targets": []
                }

                # Convert each connection to Lua format
                for conn in connections:
                    target_chapter = conn.get('target_chapter')
                    target_section = conn.get('target_section')

                    if not target_chapter or not target_section:
                        continue

                    target_title = self.get_target_section_title(target_chapter, target_section)

                    # Map connection type
                    connection_type = self.connection_mapping.get(
                        conn.get('connection_type', 'extends'),
                        'Preview'
                    )

                    target = {
                        "target_section_id": target_section,
                        "target_section_title": target_title,
                        "connection_type": connection_type,
                        "similarity": conn.get('strength', 0.5),
                        "explanation": conn.get('explanation', '')
                    }

                    section_data["targets"].append(target)
                    total_connections += 1

                if section_data["targets"]:
                    chapter_data["sections"].append(section_data)
                    total_sections += 1

            if chapter_data["sections"]:
                lua_format["cross_references"].append(chapter_data)
                chapters_processed += 1

        # Update metadata
        lua_format["metadata"]["total_sections"] = total_sections
        lua_format["metadata"]["total_cross_references"] = total_connections
        lua_format["metadata"]["chapters_processed"] = chapters_processed

        print(f"✅ Conversion complete:")
        print(f"  📊 {chapters_processed} chapters processed")
        print(f"  📊 {total_sections} sections with connections")
        print(f"  📊 {total_connections} total connections")

        return lua_format

def main():
    converter = CrossRefConverter()

    # Convert to Lua format
    lua_data = converter.convert_to_lua_format()

    # Write to output file
    output_file = Path("/Users/VJ/GitHub/MLSysBook/quarto/data/cross_refs_production.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(lua_data, f, indent=2, ensure_ascii=False)

    print(f"🎯 Production cross-references saved to: {output_file}")
    print(f"📈 Total connections: {lua_data['metadata']['total_cross_references']}")
    print(f"📝 Coverage: {lua_data['metadata']['chapters_processed']}/22 chapters")

if __name__ == "__main__":
    main()
