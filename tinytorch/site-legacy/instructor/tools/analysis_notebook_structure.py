#!/usr/bin/env python3
"""
TinyTorch Module Structure and Educational Scaffolding Analysis

This script analyzes the educational content across all modules to identify:
1. Module length and complexity metrics
2. Cell-by-cell breakdown and learning progression
3. Potential student overwhelm points
4. Test anxiety sources
5. Scaffolding effectiveness

Focus: Machine Learning Systems education with proper learning progression
"""

import os
import re
import ast
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import statistics

@dataclass
class CellAnalysis:
    """Analysis of a single notebook cell"""
    cell_type: str  # markdown, code, export, etc.
    line_count: int
    char_count: int
    complexity_score: int  # 1-5 scale
    educational_type: str  # concept, implementation, test, etc.
    has_todo: bool
    has_hints: bool
    concepts_introduced: List[str]

@dataclass
class ModuleAnalysis:
    """Comprehensive analysis of a module"""
    name: str
    path: str
    total_lines: int
    total_cells: int
    cell_analyses: List[CellAnalysis]
    concepts_covered: List[str]
    learning_progression: List[str]
    test_count: int
    todo_count: int
    hint_count: int
    complexity_distribution: Dict[int, int]
    potential_overwhelm_points: List[str]
    scaffolding_quality: int  # 1-5 scale

class NotebookAnalyzer:
    """Analyzes TinyTorch development notebooks for educational effectiveness"""

    def __init__(self, modules_dir: str = "modules/source"):
        self.modules_dir = Path(modules_dir)
        self.module_analyses: List[ModuleAnalysis] = []

    def analyze_all_modules(self) -> Dict[str, ModuleAnalysis]:
        """Analyze all modules in the source directory"""
        results = {}

        for module_dir in sorted(self.modules_dir.iterdir()):
            if module_dir.is_dir() and module_dir.name.startswith(('00_', '01_', '02_', '03_', '04_', '05_', '06_', '07_')):
                print(f"\n📚 Analyzing {module_dir.name}...")
                analysis = self.analyze_module(module_dir)
                results[module_dir.name] = analysis
                self.module_analyses.append(analysis)

        return results

    def analyze_module(self, module_path: Path) -> ModuleAnalysis:
        """Analyze a single module for educational effectiveness"""
        # Find the main development file
        dev_files = list(module_path.glob("*_dev.py"))
        if not dev_files:
            print(f"⚠️  No _dev.py file found in {module_path}")
            return self._create_empty_analysis(module_path.name, str(module_path))

        dev_file = dev_files[0]

        with open(dev_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the file structure
        cells = self._parse_jupytext_cells(content)
        cell_analyses = [self._analyze_cell(cell) for cell in cells]

        # Count tests
        test_dir = module_path / "tests"
        test_count = len(list(test_dir.glob("test_*.py"))) if test_dir.exists() else 0

        # Analyze overall structure
        concepts = self._extract_concepts(content)
        progression = self._analyze_learning_progression(cell_analyses)
        overwhelm_points = self._identify_overwhelm_points(cell_analyses)
        scaffolding_quality = self._assess_scaffolding_quality(cell_analyses)

        return ModuleAnalysis(
            name=module_path.name,
            path=str(module_path),
            total_lines=len(content.split('\n')),
            total_cells=len(cells),
            cell_analyses=cell_analyses,
            concepts_covered=concepts,
            learning_progression=progression,
            test_count=test_count,
            todo_count=sum(1 for cell in cell_analyses if cell.has_todo),
            hint_count=sum(1 for cell in cell_analyses if cell.has_hints),
            complexity_distribution={i: sum(1 for cell in cell_analyses if cell.complexity_score == i) for i in range(1, 6)},
            potential_overwhelm_points=overwhelm_points,
            scaffolding_quality=scaffolding_quality
        )

    def _parse_jupytext_cells(self, content: str) -> List[Dict]:
        """Parse Jupytext percent format cells"""
        cells = []
        current_cell = {"type": "code", "content": ""}

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.strip() == "# %% [markdown]":
                # Save current cell and start markdown cell
                if current_cell["content"].strip():
                    cells.append(current_cell)
                current_cell = {"type": "markdown", "content": ""}
                i += 1
                continue

            elif line.strip() == "# %%":
                # Save current cell and start code cell
                if current_cell["content"].strip():
                    cells.append(current_cell)
                current_cell = {"type": "code", "content": ""}
                i += 1
                continue

            # Add line to current cell
            current_cell["content"] += line + "\n"
            i += 1

        # Add final cell
        if current_cell["content"].strip():
            cells.append(current_cell)

        return cells

    def _analyze_cell(self, cell: Dict) -> CellAnalysis:
        """Analyze a single cell for educational metrics"""
        content = cell["content"]
        lines = content.split('\n')

        # Basic metrics
        line_count = len([l for l in lines if l.strip()])
        char_count = len(content)

        # Educational analysis
        has_todo = "TODO:" in content or "NotImplementedError" in content
        has_hints = "HINT" in content or "APPROACH:" in content or "EXAMPLE:" in content

        # Complexity scoring (1-5 scale)
        complexity = self._calculate_complexity(content, cell["type"])

        # Educational type classification
        edu_type = self._classify_educational_type(content, cell["type"])

        # Extract concepts
        concepts = self._extract_cell_concepts(content, cell["type"])

        return CellAnalysis(
            cell_type=cell["type"],
            line_count=line_count,
            char_count=char_count,
            complexity_score=complexity,
            educational_type=edu_type,
            has_todo=has_todo,
            has_hints=has_hints,
            concepts_introduced=concepts
        )

    def _calculate_complexity(self, content: str, cell_type: str) -> int:
        """Calculate complexity score 1-5 for a cell"""
        if cell_type == "markdown":
            # Markdown complexity based on mathematical content and length
            math_indicators = content.count('$') + content.count('\\') + content.count('equation')
            length_factor = min(len(content) // 500, 3)  # 0-3 based on length
            return min(1 + math_indicators // 4 + length_factor, 5)

        else:  # code cell
            # Code complexity based on various factors
            complexity = 1

            # AST complexity (if parseable)
            try:
                tree = ast.parse(content)
                complexity += len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]) // 2
                complexity += len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While, ast.If))]) // 3
            except:
                # If not parseable, use simpler heuristics
                complexity += content.count('def ') + content.count('class ')
                complexity += content.count('for ') + content.count('while ') + content.count('if ')

            # Length factor
            complexity += min(len(content.split('\n')) // 20, 2)

            return min(complexity, 5)

    def _classify_educational_type(self, content: str, cell_type: str) -> str:
        """Classify the educational purpose of a cell"""
        if cell_type == "markdown":
            if any(word in content.lower() for word in ["step", "what is", "definition", "concept"]):
                return "concept_introduction"
            elif any(word in content.lower() for word in ["example", "visual", "analogy"]):
                return "example_illustration"
            elif any(word in content.lower() for word in ["summary", "recap", "conclusion"]):
                return "concept_reinforcement"
            else:
                return "explanation"
        else:  # code
            if "TODO:" in content or "NotImplementedError" in content:
                return "student_implementation"
            elif "#| export" in content:
                return "solution_code"
            elif "test" in content.lower() or "assert" in content:
                return "verification"
            elif "import" in content:
                return "setup"
            else:
                return "demonstration"

    def _extract_cell_concepts(self, content: str, cell_type: str) -> List[str]:
        """Extract key concepts introduced in this cell"""
        concepts = []

        if cell_type == "markdown":
            # Look for concept indicators
            lines = content.split('\n')
            for line in lines:
                if line.startswith('#'):
                    # Extract from headers
                    concept = line.strip('#').strip()
                    if concept and len(concept) < 50:
                        concepts.append(concept)
                elif '**' in line:
                    # Extract from bold text
                    bold_matches = re.findall(r'\*\*(.*?)\*\*', line)
                    concepts.extend([match for match in bold_matches if len(match) < 30])

        else:  # code
            # Extract class and function names
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        concepts.append(f"Class: {node.name}")
                    elif isinstance(node, ast.FunctionDef):
                        concepts.append(f"Function: {node.name}")
            except:
                pass

        return concepts[:5]  # Limit to top 5 concepts

    def _extract_concepts(self, content: str) -> List[str]:
        """Extract all major concepts from module content"""
        concepts = set()

        # Extract from headers
        headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        concepts.update([h.strip() for h in headers if len(h.strip()) < 50])

        # Extract from class/function definitions
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    concepts.add(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    concepts.add(node.name)
        except:
            pass

        return sorted(list(concepts))

    def _analyze_learning_progression(self, cell_analyses: List[CellAnalysis]) -> List[str]:
        """Analyze the learning progression through the module"""
        progression = []

        for i, cell in enumerate(cell_analyses):
            if cell.educational_type == "concept_introduction":
                progression.append(f"Step {len(progression)+1}: Concept Introduction")
            elif cell.educational_type == "student_implementation":
                progression.append(f"Step {len(progression)+1}: Hands-on Implementation")
            elif cell.educational_type == "verification":
                progression.append(f"Step {len(progression)+1}: Verification & Testing")

        return progression

    def _identify_overwhelm_points(self, cell_analyses: List[CellAnalysis]) -> List[str]:
        """Identify potential student overwhelm points"""
        overwhelm_points = []

        for i, cell in enumerate(cell_analyses):
            # Long cells without scaffolding
            if cell.line_count > 50 and not cell.has_hints:
                overwhelm_points.append(f"Cell {i+1}: Long implementation without guidance ({cell.line_count} lines)")

            # High complexity without TODO structure
            if cell.complexity_score >= 4 and not cell.has_todo:
                overwhelm_points.append(f"Cell {i+1}: High complexity without student scaffolding")

            # Sudden complexity jumps
            if i > 0 and cell.complexity_score - cell_analyses[i-1].complexity_score >= 3:
                overwhelm_points.append(f"Cell {i+1}: Sudden complexity jump from {cell_analyses[i-1].complexity_score} to {cell.complexity_score}")

        return overwhelm_points

    def _assess_scaffolding_quality(self, cell_analyses: List[CellAnalysis]) -> int:
        """Assess overall scaffolding quality (1-5 scale)"""
        if not cell_analyses:
            return 1

        score = 3  # Start with average

        # Positive factors
        implementation_cells = [c for c in cell_analyses if c.educational_type == "student_implementation"]
        if implementation_cells:
            hint_ratio = sum(1 for c in implementation_cells if c.has_hints) / len(implementation_cells)
            score += hint_ratio * 2  # Up to +2 for good hint coverage

        # Check for good progression
        concept_cells = [c for c in cell_analyses if c.educational_type == "concept_introduction"]
        if len(concept_cells) >= 2:
            score += 0.5  # Good conceptual foundation

        # Negative factors
        overwhelm_ratio = len([c for c in cell_analyses if c.complexity_score >= 4]) / len(cell_analyses)
        if overwhelm_ratio > 0.3:
            score -= 1  # Too many high-complexity cells

        return max(1, min(5, int(score)))

    def _create_empty_analysis(self, name: str, path: str) -> ModuleAnalysis:
        """Create empty analysis for modules without dev files"""
        return ModuleAnalysis(
            name=name,
            path=path,
            total_lines=0,
            total_cells=0,
            cell_analyses=[],
            concepts_covered=[],
            learning_progression=[],
            test_count=0,
            todo_count=0,
            hint_count=0,
            complexity_distribution={i: 0 for i in range(1, 6)},
            potential_overwhelm_points=[],
            scaffolding_quality=1
        )

    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        if not self.module_analyses:
            return "No modules analyzed yet. Run analyze_all_modules() first."

        report = []
        report.append("# TinyTorch Educational Content Analysis Report")
        report.append("=" * 50)

        # Overall statistics
        total_lines = sum(m.total_lines for m in self.module_analyses)
        total_cells = sum(m.total_cells for m in self.module_analyses)
        avg_scaffolding = statistics.mean(m.scaffolding_quality for m in self.module_analyses)

        report.append(f"\n## 📊 Overall Statistics")
        report.append(f"- Total modules analyzed: {len(self.module_analyses)}")
        report.append(f"- Total lines of content: {total_lines:,}")
        report.append(f"- Total cells: {total_cells}")
        report.append(f"- Average scaffolding quality: {avg_scaffolding:.1f}/5.0")

        # Module-by-module breakdown
        report.append(f"\n## 📚 Module-by-Module Analysis")

        for analysis in self.module_analyses:
            report.append(f"\n### {analysis.name}")
            report.append(f"- **Lines**: {analysis.total_lines:,}")
            report.append(f"- **Cells**: {analysis.total_cells}")
            report.append(f"- **Concepts**: {len(analysis.concepts_covered)}")
            report.append(f"- **TODOs**: {analysis.todo_count}")
            report.append(f"- **Hints**: {analysis.hint_count}")
            report.append(f"- **Tests**: {analysis.test_count}")
            report.append(f"- **Scaffolding Quality**: {analysis.scaffolding_quality}/5")

            if analysis.potential_overwhelm_points:
                report.append(f"- **⚠️ Potential Overwhelm Points**:")
                for point in analysis.potential_overwhelm_points[:3]:  # Show top 3
                    report.append(f"  - {point}")

        # Recommendations
        report.append(f"\n## 🎯 Educational Recommendations")

        # Identify modules needing attention
        low_scaffolding = [m for m in self.module_analyses if m.scaffolding_quality <= 2]
        high_complexity = []

        for m in self.module_analyses:
            if m.total_cells > 0:  # Avoid division by zero
                complex_cells = m.complexity_distribution.get(4, 0) + m.complexity_distribution.get(5, 0)
                if complex_cells > m.total_cells * 0.3:
                    high_complexity.append(m)

        if low_scaffolding:
            report.append(f"\n### 🚨 Modules Needing Better Scaffolding:")
            for module in low_scaffolding:
                report.append(f"- **{module.name}**: Quality {module.scaffolding_quality}/5")

        if high_complexity:
            report.append(f"\n### 📈 Modules with High Complexity:")
            for module in high_complexity:
                complex_ratio = (module.complexity_distribution.get(4, 0) + module.complexity_distribution.get(5, 0)) / max(module.total_cells, 1)
                report.append(f"- **{module.name}**: {complex_ratio:.1%} high-complexity cells")

        # Best practices recommendations
        report.append(f"\n### ✅ Recommended Best Practices:")

        if self.module_analyses:
            min_lines = min(m.total_lines for m in self.module_analyses if m.total_lines > 0)
            max_lines = max(m.total_lines for m in self.module_analyses)
            report.append(f"- **Ideal module length**: 200-400 lines (current range: {min_lines}-{max_lines})")
        else:
            report.append(f"- **Ideal module length**: 200-400 lines")

        report.append(f"- **Cell complexity**: Max 30% high-complexity cells")
        report.append(f"- **Scaffolding ratio**: All implementation cells should have hints")
        report.append(f"- **Progression**: Concept → Example → Implementation → Verification")

        return "\n".join(report)

if __name__ == "__main__":
    analyzer = NotebookAnalyzer()
    results = analyzer.analyze_all_modules()

    print("\n" + "="*60)
    print(analyzer.generate_report())

    # Save detailed report
    with open("educational_analysis_report.md", "w") as f:
        f.write(analyzer.generate_report())

    print(f"\n📄 Detailed report saved to: educational_analysis_report.md")
