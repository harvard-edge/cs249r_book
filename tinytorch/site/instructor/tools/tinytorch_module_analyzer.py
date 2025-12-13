#!/usr/bin/env python3
"""
TinyTorch Module Analyzer & Report Card Generator

A comprehensive tool for analyzing educational quality and generating
actionable report cards for TinyTorch modules.

Usage:
    python tinyorch_module_analyzer.py --module 02_activations
    python tinyorch_module_analyzer.py --all
    python tinyorch_module_analyzer.py --compare 01_tensor 02_activations
    python tinyorch_module_analyzer.py --watch modules/source/
"""

import os
import re
import ast
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Union
import statistics
from datetime import datetime
import subprocess

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
    overwhelm_factors: List[str]  # Specific issues that could overwhelm students

@dataclass
class ModuleReportCard:
    """Comprehensive report card for a module"""
    # Basic Info
    module_name: str
    module_path: str
    analysis_date: str

    # Size Metrics
    total_lines: int
    total_cells: int
    avg_cell_length: float

    # Educational Quality
    scaffolding_quality: int  # 1-5 scale
    complexity_distribution: Dict[int, int]
    learning_progression_quality: int  # 1-5 scale

    # Content Analysis
    concepts_covered: List[str]
    todo_count: int
    hint_count: int
    test_count: int

    # Issues and Recommendations
    critical_issues: List[str]
    overwhelm_points: List[str]
    recommendations: List[str]

    # Detailed Breakdown
    cell_analyses: List[CellAnalysis]

    # Grades
    overall_grade: str  # A, B, C, D, F
    category_grades: Dict[str, str]

    # Comparisons
    vs_targets: Dict[str, str]  # How this compares to target metrics
    vs_best_practices: List[str]  # Specific best practice violations

class TinyTorchModuleAnalyzer:
    """Comprehensive analyzer for TinyTorch educational modules"""

    def __init__(self, modules_dir: str = "../../modules/source"):
        self.modules_dir = Path(modules_dir)
        self.target_metrics = {
            'ideal_lines': (200, 400),
            'max_cell_lines': 30,
            'max_complexity_ratio': 0.3,
            'min_scaffolding_quality': 4,
            'max_concepts_per_cell': 3,
            'min_hint_ratio': 0.8  # 80% of implementation cells should have hints
        }

    def analyze_module(self, module_name: str) -> ModuleReportCard:
        """Generate comprehensive report card for a module"""
        module_path = self.modules_dir / module_name

        if not module_path.exists():
            raise FileNotFoundError(f"Module {module_name} not found at {module_path}")

        # Find development file
        dev_files = list(module_path.glob("*_dev.py"))
        if not dev_files:
            return self._create_empty_report_card(module_name, str(module_path))

        dev_file = dev_files[0]

        with open(dev_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse and analyze
        cells = self._parse_jupytext_cells(content)
        cell_analyses = [self._analyze_cell(cell, i) for i, cell in enumerate(cells)]

        # Generate comprehensive metrics
        report_card = self._generate_report_card(
            module_name, str(module_path), content, cells, cell_analyses
        )

        return report_card

    def _parse_jupytext_cells(self, content: str) -> List[Dict]:
        """Parse Jupytext percent format cells with enhanced metadata"""
        cells = []
        current_cell = {"type": "code", "content": "", "directives": []}

        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for NBDev directives
            if line.strip().startswith('#|'):
                current_cell["directives"].append(line.strip())
                current_cell["content"] += line + "\n"
                i += 1
                continue

            if line.strip() == "# %% [markdown]":
                # Save current cell and start markdown cell
                if current_cell["content"].strip():
                    cells.append(current_cell)
                current_cell = {"type": "markdown", "content": "", "directives": []}
                i += 1
                continue

            elif line.strip() == "# %%":
                # Save current cell and start code cell
                if current_cell["content"].strip():
                    cells.append(current_cell)
                current_cell = {"type": "code", "content": "", "directives": []}
                i += 1
                continue

            # Add line to current cell
            current_cell["content"] += line + "\n"
            i += 1

        # Add final cell
        if current_cell["content"].strip():
            cells.append(current_cell)

        return cells

    def _analyze_cell(self, cell: Dict, cell_index: int) -> CellAnalysis:
        """Comprehensive analysis of a single cell"""
        content = cell["content"]
        lines = content.split('\n')

        # Basic metrics
        line_count = len([l for l in lines if l.strip()])
        char_count = len(content)

        # Educational analysis
        has_todo = "TODO:" in content or "NotImplementedError" in content
        has_hints = any(hint in content for hint in ["HINT", "APPROACH:", "EXAMPLE:", "💡"])

        # Complexity scoring with enhanced factors
        complexity = self._calculate_complexity_enhanced(content, cell["type"])

        # Educational type classification
        edu_type = self._classify_educational_type_enhanced(content, cell["type"], cell.get("directives", []))

        # Extract concepts
        concepts = self._extract_cell_concepts_enhanced(content, cell["type"])

        # Identify overwhelm factors
        overwhelm_factors = self._identify_cell_overwhelm_factors(content, line_count, complexity, has_hints)

        return CellAnalysis(
            cell_type=cell["type"],
            line_count=line_count,
            char_count=char_count,
            complexity_score=complexity,
            educational_type=edu_type,
            has_todo=has_todo,
            has_hints=has_hints,
            concepts_introduced=concepts,
            overwhelm_factors=overwhelm_factors
        )

    def _calculate_complexity_enhanced(self, content: str, cell_type: str) -> int:
        """Enhanced complexity calculation with more factors"""
        if cell_type == "markdown":
            complexity = 1

            # Math content
            math_indicators = content.count('$') + content.count('\\') + content.count('equation')
            complexity += min(math_indicators // 4, 2)

            # Length factor
            complexity += min(len(content) // 800, 2)  # Longer markdown is more complex

            # Technical vocabulary
            technical_terms = ['tensor', 'gradient', 'backpropagation', 'convolution', 'optimization']
            tech_count = sum(1 for term in technical_terms if term.lower() in content.lower())
            complexity += min(tech_count // 3, 1)

            return min(complexity, 5)

        else:  # code cell
            complexity = 1

            # AST complexity (if parseable)
            try:
                tree = ast.parse(content)
                # Functions and classes
                complexity += len([node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))]) // 2
                # Control structures
                complexity += len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While, ast.If))]) // 3
                # Advanced features
                complexity += len([node for node in ast.walk(tree) if isinstance(node, (ast.ListComp, ast.Lambda, ast.Try))]) // 2
            except:
                # Fallback to simpler heuristics
                complexity += content.count('def ') + content.count('class ')
                complexity += content.count('for ') + content.count('while ') + content.count('if ')
                complexity += content.count('try:') + content.count('lambda ')

            # Length factor
            complexity += min(len(content.split('\n')) // 25, 2)

            # Import complexity
            import_count = content.count('import ') + content.count('from ')
            complexity += min(import_count // 5, 1)

            # Mathematical operations
            math_ops = ['@', 'np.', 'torch.', 'einsum', 'matmul']
            math_count = sum(content.count(op) for op in math_ops)
            complexity += min(math_count // 3, 1)

            return min(complexity, 5)

    def _classify_educational_type_enhanced(self, content: str, cell_type: str, directives: List[str]) -> str:
        """Enhanced educational type classification"""
        if cell_type == "markdown":
            content_lower = content.lower()

            if any(word in content_lower for word in ["step", "what is", "definition", "understanding"]):
                return "concept_introduction"
            elif any(word in content_lower for word in ["example", "visual", "analogy", "imagine"]):
                return "example_illustration"
            elif any(word in content_lower for word in ["summary", "recap", "conclusion", "review"]):
                return "concept_reinforcement"
            elif any(word in content_lower for word in ["real-world", "production", "industry"]):
                return "practical_connection"
            else:
                return "explanation"
        else:  # code
            # Check NBDev directives
            if any("export" in directive for directive in directives):
                if "hide" in " ".join(directives):
                    return "instructor_solution"
                else:
                    return "student_implementation"

            if "TODO:" in content or "NotImplementedError" in content:
                return "student_implementation"
            elif "test" in content.lower() or "assert" in content:
                return "verification"
            elif "import" in content:
                return "setup"
            elif "print" in content and ("✅" in content or "🎉" in content):
                return "feedback_celebration"
            else:
                return "demonstration"

    def _extract_cell_concepts_enhanced(self, content: str, cell_type: str) -> List[str]:
        """Enhanced concept extraction with better recognition"""
        concepts = []

        if cell_type == "markdown":
            # Headers
            headers = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
            concepts.extend([h.strip() for h in headers if len(h.strip()) < 50])

            # Bold concepts
            bold_matches = re.findall(r'\*\*(.*?)\*\*', content)
            concepts.extend([match for match in bold_matches if len(match) < 30])

            # Definition patterns
            definition_patterns = [
                r'(\w+)\s+is\s+defined\s+as',
                r'(\w+)\s*:\s*[A-Z]',  # Term: Definition
                r'\*\*(\w+)\*\*\s*:',      # **Term**: (fixed escaping)
            ]

            for pattern in definition_patterns:
                try:
                    matches = re.findall(pattern, content)
                    concepts.extend(matches)
                except re.error:
                    continue  # Skip problematic patterns

        else:  # code
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        concepts.append(f"Class: {node.name}")
                    elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                        concepts.append(f"Function: {node.name}")
            except:
                # Fallback to regex
                class_matches = re.findall(r'class\s+(\w+)', content)
                func_matches = re.findall(r'def\s+(\w+)', content)
                concepts.extend([f"Class: {c}" for c in class_matches])
                concepts.extend([f"Function: {f}" for f in func_matches if not f.startswith('_')])

        return list(set(concepts))[:5]  # Unique, limited to top 5

    def _identify_cell_overwhelm_factors(self, content: str, line_count: int, complexity: int, has_hints: bool) -> List[str]:
        """Identify specific factors that could overwhelm students"""
        factors = []

        # Length issues
        if line_count > 50:
            factors.append(f"Very long cell ({line_count} lines)")
        elif line_count > 30:
            factors.append(f"Long cell ({line_count} lines)")

        # Complexity without support
        if complexity >= 4 and not has_hints:
            factors.append("High complexity without guidance")

        # Multiple concepts
        concept_count = len(self._extract_cell_concepts_enhanced(content, "code" if "def " in content else "markdown"))
        if concept_count > 3:
            factors.append(f"Too many concepts ({concept_count})")

        # Mathematical density
        math_indicators = content.count('$') + content.count('\\') + content.count('equation')
        if math_indicators > 10:
            factors.append("Math-heavy without scaffolding")

        # Code density
        if "def " in content:
            func_count = content.count('def ')
            if func_count > 2:
                factors.append(f"Multiple functions in one cell ({func_count})")

        # Missing error handling
        if "TODO:" in content and line_count > 20 and "try:" not in content:
            factors.append("Complex implementation without error handling guidance")

        return factors

    def _generate_report_card(self, module_name: str, module_path: str, content: str,
                            cells: List[Dict], cell_analyses: List[CellAnalysis]) -> ModuleReportCard:
        """Generate comprehensive report card"""

        # Basic metrics
        total_lines = len(content.split('\n'))
        total_cells = len(cells)
        avg_cell_length = statistics.mean([ca.line_count for ca in cell_analyses]) if cell_analyses else 0

        # Educational quality metrics
        scaffolding_quality = self._assess_scaffolding_quality_enhanced(cell_analyses)
        complexity_dist = {i: sum(1 for ca in cell_analyses if ca.complexity_score == i) for i in range(1, 6)}
        learning_progression = self._assess_learning_progression(cell_analyses)

        # Content analysis
        all_concepts = []
        for ca in cell_analyses:
            all_concepts.extend(ca.concepts_introduced)
        concepts_covered = list(set(all_concepts))

        todo_count = sum(1 for ca in cell_analyses if ca.has_todo)
        hint_count = sum(1 for ca in cell_analyses if ca.has_hints)

        # Test count
        test_dir = Path(module_path) / "tests"
        test_count = len(list(test_dir.glob("test_*.py"))) if test_dir.exists() else 0

        # Issues and recommendations
        critical_issues = self._identify_critical_issues(cell_analyses, total_lines, total_cells)
        overwhelm_points = self._compile_overwhelm_points(cell_analyses)
        recommendations = self._generate_recommendations(cell_analyses, total_lines, scaffolding_quality)

        # Grades
        overall_grade, category_grades = self._calculate_grades(
            scaffolding_quality, complexity_dist, total_cells, avg_cell_length
        )

        # Comparisons
        vs_targets = self._compare_to_targets(total_lines, avg_cell_length, complexity_dist, total_cells)
        vs_best_practices = self._check_best_practices(cell_analyses)

        return ModuleReportCard(
            module_name=module_name,
            module_path=module_path,
            analysis_date=datetime.now().isoformat(),
            total_lines=total_lines,
            total_cells=total_cells,
            avg_cell_length=avg_cell_length,
            scaffolding_quality=scaffolding_quality,
            complexity_distribution=complexity_dist,
            learning_progression_quality=learning_progression,
            concepts_covered=concepts_covered,
            todo_count=todo_count,
            hint_count=hint_count,
            test_count=test_count,
            critical_issues=critical_issues,
            overwhelm_points=overwhelm_points,
            recommendations=recommendations,
            cell_analyses=cell_analyses,
            overall_grade=overall_grade,
            category_grades=category_grades,
            vs_targets=vs_targets,
            vs_best_practices=vs_best_practices
        )

    def _assess_scaffolding_quality_enhanced(self, cell_analyses: List[CellAnalysis]) -> int:
        """Enhanced scaffolding quality assessment"""
        if not cell_analyses:
            return 1

        score = 3  # Start with average

        # Implementation scaffolding
        impl_cells = [ca for ca in cell_analyses if ca.educational_type == "student_implementation"]
        if impl_cells:
            hint_ratio = sum(1 for ca in impl_cells if ca.has_hints) / len(impl_cells)
            score += (hint_ratio - 0.5) * 2  # +2 for 100% hints, -1 for 0% hints

        # Concept progression
        concept_cells = [ca for ca in cell_analyses if ca.educational_type == "concept_introduction"]
        if len(concept_cells) >= 2:
            score += 0.5

        # Complexity progression
        complexities = [ca.complexity_score for ca in cell_analyses]
        if len(complexities) > 1:
            max_jump = max(complexities[i] - complexities[i-1] for i in range(1, len(complexities)))
            if max_jump <= 2:
                score += 1  # Good progression
            elif max_jump >= 4:
                score -= 2  # Bad progression

        # Overwhelm factors
        overwhelm_count = sum(len(ca.overwhelm_factors) for ca in cell_analyses)
        if overwhelm_count == 0:
            score += 1
        elif overwhelm_count > len(cell_analyses):  # More than one per cell on average
            score -= 1

        return max(1, min(5, int(score)))

    def _assess_learning_progression(self, cell_analyses: List[CellAnalysis]) -> int:
        """Assess quality of learning progression"""
        if len(cell_analyses) < 3:
            return 3

        # Check for educational flow
        edu_types = [ca.educational_type for ca in cell_analyses]

        # Good patterns
        good_patterns = [
            ["concept_introduction", "example_illustration", "student_implementation"],
            ["concept_introduction", "student_implementation", "verification"],
            ["explanation", "demonstration", "student_implementation"]
        ]

        score = 3
        for pattern in good_patterns:
            if self._contains_pattern(edu_types, pattern):
                score += 1
                break

        # Check complexity progression
        complexities = [ca.complexity_score for ca in cell_analyses]
        if self._is_smooth_progression(complexities):
            score += 1
        elif self._has_complexity_cliffs(complexities):
            score -= 2

        return max(1, min(5, score))

    def _contains_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence contains the pattern"""
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i+len(pattern)] == pattern:
                return True
        return False

    def _is_smooth_progression(self, complexities: List[int]) -> bool:
        """Check if complexity increases smoothly"""
        for i in range(1, len(complexities)):
            if complexities[i] - complexities[i-1] > 2:
                return False
        return True

    def _has_complexity_cliffs(self, complexities: List[int]) -> bool:
        """Check for sudden complexity jumps"""
        for i in range(1, len(complexities)):
            if complexities[i] - complexities[i-1] >= 3:
                return True
        return False

    def _identify_critical_issues(self, cell_analyses: List[CellAnalysis], total_lines: int, total_cells: int) -> List[str]:
        """Identify critical issues that need immediate attention"""
        issues = []

        # Overwhelming length
        if total_lines > 1000:
            issues.append(f"Module too long ({total_lines} lines) - students will be overwhelmed")

        # High complexity ratio
        if total_cells > 0:
            high_complexity_ratio = sum(1 for ca in cell_analyses if ca.complexity_score >= 4) / total_cells
            if high_complexity_ratio > 0.5:
                issues.append(f"Too many high-complexity cells ({high_complexity_ratio:.1%})")

        # Missing scaffolding
        impl_cells = [ca for ca in cell_analyses if ca.educational_type == "student_implementation"]
        if impl_cells:
            no_hints_ratio = sum(1 for ca in impl_cells if not ca.has_hints) / len(impl_cells)
            if no_hints_ratio > 0.5:
                issues.append(f"Implementation cells lack guidance ({no_hints_ratio:.1%} without hints)")

        # Complexity cliffs
        complexities = [ca.complexity_score for ca in cell_analyses]
        if self._has_complexity_cliffs(complexities):
            issues.append("Sudden complexity jumps will overwhelm students")

        # Very long cells
        long_cells = [ca for ca in cell_analyses if ca.line_count > 50]
        if long_cells:
            issues.append(f"{len(long_cells)} cells are too long (>50 lines)")

        return issues

    def _compile_overwhelm_points(self, cell_analyses: List[CellAnalysis]) -> List[str]:
        """Compile all overwhelm points from cells"""
        points = []
        for i, ca in enumerate(cell_analyses):
            for factor in ca.overwhelm_factors:
                points.append(f"Cell {i+1}: {factor}")
        return points

    def _generate_recommendations(self, cell_analyses: List[CellAnalysis], total_lines: int, scaffolding_quality: int) -> List[str]:
        """Generate specific actionable recommendations"""
        recommendations = []

        # Length recommendations
        if total_lines > 800:
            recommendations.append("Break module into smaller sections or multiple modules")

        # Scaffolding recommendations
        if scaffolding_quality <= 2:
            recommendations.append("Add implementation ladders: break complex functions into 3 progressive steps")
            recommendations.append("Add concept bridges: connect new ideas to familiar concepts")
            recommendations.append("Include confidence builders: early wins to build momentum")

        # Complexity recommendations
        high_complexity_cells = [ca for ca in cell_analyses if ca.complexity_score >= 4]
        if len(high_complexity_cells) > len(cell_analyses) * 0.3:
            recommendations.append("Reduce complexity: apply 'Rule of 3s' (max 3 concepts per cell)")
            recommendations.append("Add progressive disclosure: introduce concepts when needed")

        # Hint recommendations
        impl_cells = [ca for ca in cell_analyses if ca.educational_type == "student_implementation"]
        unhinted_cells = [ca for ca in impl_cells if not ca.has_hints]
        if len(unhinted_cells) > 0:
            recommendations.append(f"Add hints to {len(unhinted_cells)} implementation cells")

        # Long cell recommendations
        long_cells = [ca for ca in cell_analyses if ca.line_count > 30]
        if long_cells:
            recommendations.append(f"Split {len(long_cells)} long cells into smaller, focused cells")

        # Testing recommendations
        if not any("verification" in ca.educational_type for ca in cell_analyses):
            recommendations.append("Add immediate feedback tests after implementations")

        return recommendations

    def _calculate_grades(self, scaffolding_quality: int, complexity_dist: Dict[int, int],
                         total_cells: int, avg_cell_length: float) -> Tuple[str, Dict[str, str]]:
        """Calculate letter grades for different aspects"""

        def score_to_grade(score: float) -> str:
            if score >= 4.5: return "A"
            elif score >= 3.5: return "B"
            elif score >= 2.5: return "C"
            elif score >= 1.5: return "D"
            else: return "F"

        # Category scores (1-5 scale)
        scores = {}

        # Scaffolding grade
        scores["Scaffolding"] = scaffolding_quality

        # Complexity grade
        if total_cells > 0:
            high_complexity_ratio = (complexity_dist.get(4, 0) + complexity_dist.get(5, 0)) / total_cells
            complexity_score = 5 - (high_complexity_ratio * 4)  # Penalize high complexity
            scores["Complexity"] = max(1, complexity_score)
        else:
            scores["Complexity"] = 3

        # Length grade
        if avg_cell_length <= 20:
            length_score = 5
        elif avg_cell_length <= 30:
            length_score = 4
        elif avg_cell_length <= 50:
            length_score = 3
        elif avg_cell_length <= 80:
            length_score = 2
        else:
            length_score = 1
        scores["Cell_Length"] = length_score

        # Overall grade
        overall_score = statistics.mean(scores.values())

        # Convert to letter grades
        category_grades = {category: score_to_grade(score) for category, score in scores.items()}
        overall_grade = score_to_grade(overall_score)

        return overall_grade, category_grades

    def _compare_to_targets(self, total_lines: int, avg_cell_length: float,
                          complexity_dist: Dict[int, int], total_cells: int) -> Dict[str, str]:
        """Compare metrics to target values"""
        comparisons = {}

        # Length comparison
        min_lines, max_lines = self.target_metrics['ideal_lines']
        if min_lines <= total_lines <= max_lines:
            comparisons["Length"] = f"✅ Good ({total_lines} lines)"
        elif total_lines < min_lines:
            comparisons["Length"] = f"⚠️ Too short ({total_lines} lines, target: {min_lines}-{max_lines})"
        else:
            comparisons["Length"] = f"❌ Too long ({total_lines} lines, target: {min_lines}-{max_lines})"

        # Cell length comparison
        max_cell_length = self.target_metrics['max_cell_lines']
        if avg_cell_length <= max_cell_length:
            comparisons["Cell_Length"] = f"✅ Good ({avg_cell_length:.1f} avg lines)"
        else:
            comparisons["Cell_Length"] = f"❌ Too long ({avg_cell_length:.1f} avg, target: ≤{max_cell_length})"

        # Complexity comparison
        if total_cells > 0:
            high_complexity_ratio = (complexity_dist.get(4, 0) + complexity_dist.get(5, 0)) / total_cells
            max_complexity_ratio = self.target_metrics['max_complexity_ratio']
            if high_complexity_ratio <= max_complexity_ratio:
                comparisons["Complexity"] = f"✅ Good ({high_complexity_ratio:.1%} high-complexity)"
            else:
                comparisons["Complexity"] = f"❌ Too complex ({high_complexity_ratio:.1%}, target: ≤{max_complexity_ratio:.1%})"

        return comparisons

    def _check_best_practices(self, cell_analyses: List[CellAnalysis]) -> List[str]:
        """Check adherence to best practices"""
        violations = []

        # Rule of 3s violations
        for i, ca in enumerate(cell_analyses):
            if len(ca.concepts_introduced) > 3:
                violations.append(f"Cell {i+1}: Too many concepts ({len(ca.concepts_introduced)})")

            if ca.line_count > 30:
                violations.append(f"Cell {i+1}: Too long ({ca.line_count} lines)")

            if ca.complexity_score >= 4 and not ca.has_hints:
                violations.append(f"Cell {i+1}: High complexity without guidance")

        # Progression violations
        complexities = [ca.complexity_score for ca in cell_analyses]
        for i in range(1, len(complexities)):
            if complexities[i] - complexities[i-1] >= 3:
                violations.append(f"Cells {i}-{i+1}: Complexity cliff ({complexities[i-1]}→{complexities[i]})")

        return violations

    def _create_empty_report_card(self, module_name: str, module_path: str) -> ModuleReportCard:
        """Create empty report card for modules without dev files"""
        return ModuleReportCard(
            module_name=module_name,
            module_path=module_path,
            analysis_date=datetime.now().isoformat(),
            total_lines=0,
            total_cells=0,
            avg_cell_length=0,
            scaffolding_quality=1,
            complexity_distribution={i: 0 for i in range(1, 6)},
            learning_progression_quality=1,
            concepts_covered=[],
            todo_count=0,
            hint_count=0,
            test_count=0,
            critical_issues=["No development file found"],
            overwhelm_points=[],
            recommendations=["Create a development file following TinyTorch conventions"],
            cell_analyses=[],
            overall_grade="F",
            category_grades={"Scaffolding": "F", "Complexity": "F", "Cell_Length": "F"},
            vs_targets={},
            vs_best_practices=[]
        )

    def generate_report_card_html(self, report_card: ModuleReportCard) -> str:
        """Generate beautiful HTML report card"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TinyTorch Module Report Card: {report_card.module_name}</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
                .report-card {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); max-width: 1000px; margin: 0 auto; }}
                .header {{ text-align: center; border-bottom: 3px solid #2196F3; padding-bottom: 20px; margin-bottom: 30px; }}
                .grade-box {{ display: inline-block; margin: 10px; padding: 20px; border-radius: 8px; text-align: center; min-width: 100px; }}
                .grade-A {{ background: #4CAF50; color: white; }}
                .grade-B {{ background: #8BC34A; color: white; }}
                .grade-C {{ background: #FF9800; color: white; }}
                .grade-D {{ background: #FF5722; color: white; }}
                .grade-F {{ background: #F44336; color: white; }}
                .metrics {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
                .metric-box {{ padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .critical {{ background: #ffebee; border-left: 4px solid #f44336; }}
                .good {{ background: #e8f5e8; border-left: 4px solid #4caf50; }}
                .warning {{ background: #fff3e0; border-left: 4px solid #ff9800; }}
                .recommendations {{ background: #e3f2fd; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .cell-analysis {{ margin: 10px 0; padding: 10px; border: 1px solid #eee; border-radius: 3px; }}
                .complexity-1 {{ border-left: 4px solid #4CAF50; }}
                .complexity-2 {{ border-left: 4px solid #8BC34A; }}
                .complexity-3 {{ border-left: 4px solid #FF9800; }}
                .complexity-4 {{ border-left: 4px solid #FF5722; }}
                .complexity-5 {{ border-left: 4px solid #F44336; }}
            </style>
        </head>
        <body>
            <div class="report-card">
                <div class="header">
                    <h1>📊 TinyTorch Module Report Card</h1>
                    <h2>{report_card.module_name}</h2>
                    <p>Analysis Date: {report_card.analysis_date[:10]}</p>
                </div>

                <div class="grades">
                    <h3>📈 Overall Grade</h3>
                    <div class="grade-box grade-{report_card.overall_grade}">
                        <h2>{report_card.overall_grade}</h2>
                        <p>Overall</p>
                    </div>
        """

        # Category grades
        for category, grade in report_card.category_grades.items():
            html += f'<div class="grade-box grade-{grade}"><h3>{grade}</h3><p>{category.replace("_", " ")}</p></div>'

        html += f"""
                </div>

                <div class="metrics">
                    <div class="metric-box">
                        <h4>📏 Size Metrics</h4>
                        <p><strong>Total Lines:</strong> {report_card.total_lines}</p>
                        <p><strong>Total Cells:</strong> {report_card.total_cells}</p>
                        <p><strong>Avg Cell Length:</strong> {report_card.avg_cell_length:.1f} lines</p>
                    </div>

                    <div class="metric-box">
                        <h4>🎯 Quality Metrics</h4>
                        <p><strong>Scaffolding Quality:</strong> {report_card.scaffolding_quality}/5</p>
                        <p><strong>Learning Progression:</strong> {report_card.learning_progression_quality}/5</p>
                        <p><strong>Concepts Covered:</strong> {len(report_card.concepts_covered)}</p>
                    </div>
                </div>
        """

        # Target comparisons
        if report_card.vs_targets:
            html += '<div class="metric-box"><h4>🎯 vs Targets</h4>'
            for metric, comparison in report_card.vs_targets.items():
                html += f'<p>{comparison}</p>'
            html += '</div>'

        # Critical issues
        if report_card.critical_issues:
            html += '<div class="critical"><h4>🚨 Critical Issues</h4><ul>'
            for issue in report_card.critical_issues:
                html += f'<li>{issue}</li>'
            html += '</ul></div>'

        # Recommendations
        if report_card.recommendations:
            html += '<div class="recommendations"><h4>💡 Recommendations</h4><ul>'
            for rec in report_card.recommendations:
                html += f'<li>{rec}</li>'
            html += '</ul></div>'

        # Cell-by-cell analysis
        html += '<div class="cell-analysis-section"><h3>🔍 Cell-by-Cell Analysis</h3>'
        for i, cell in enumerate(report_card.cell_analyses):
            html += f'''
            <div class="cell-analysis complexity-{cell.complexity_score}">
                <h4>Cell {i+1}: {cell.educational_type.replace("_", " ").title()}</h4>
                <p><strong>Type:</strong> {cell.cell_type} | <strong>Lines:</strong> {cell.line_count} |
                   <strong>Complexity:</strong> {cell.complexity_score}/5</p>
                <p><strong>Concepts:</strong> {", ".join(cell.concepts_introduced[:3]) if cell.concepts_introduced else "None"}</p>
                {f'<p class="warning"><strong>⚠️ Issues:</strong> {", ".join(cell.overwhelm_factors)}</p>' if cell.overwhelm_factors else ''}
            </div>
            '''

        html += '</div></div></body></html>'
        return html

    def save_report_card(self, report_card: ModuleReportCard, format: str = "both") -> List[str]:
        """Save report card in various formats"""
        saved_files = []

        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        base_name = f"{report_card.module_name}_report_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if format in ["json", "both"]:
            # JSON format (for programmatic use)
            json_file = reports_dir / f"{base_name}.json"
            with open(json_file, 'w') as f:
                json.dump(asdict(report_card), f, indent=2, default=str)
            saved_files.append(str(json_file))

        if format in ["html", "both"]:
            # HTML format (for human reading)
            html_file = reports_dir / f"{base_name}.html"
            with open(html_file, 'w') as f:
                f.write(self.generate_report_card_html(report_card))
            saved_files.append(str(html_file))

        return saved_files

    def analyze_all_modules(self) -> Dict[str, ModuleReportCard]:
        """Analyze all modules and return report cards"""
        results = {}

        for module_dir in sorted(self.modules_dir.iterdir()):
            if module_dir.is_dir() and module_dir.name.startswith(('00_', '01_', '02_', '03_', '04_', '05_', '06_', '07_')):
                print(f"📚 Analyzing {module_dir.name}...")
                try:
                    report_card = self.analyze_module(module_dir.name)
                    results[module_dir.name] = report_card
                    print(f"   Grade: {report_card.overall_grade} | Scaffolding: {report_card.scaffolding_quality}/5")
                except Exception as e:
                    print(f"   ❌ Error: {e}")

        return results

    def compare_modules(self, module_names: List[str]) -> str:
        """Generate comparison report between modules"""
        report_cards = {}
        for name in module_names:
            try:
                report_cards[name] = self.analyze_module(name)
            except Exception as e:
                print(f"Error analyzing {name}: {e}")
                continue

        if not report_cards:
            return "No modules could be analyzed for comparison."

        # Generate comparison
        comparison = f"# Module Comparison Report\n\n"
        comparison += f"Comparing: {', '.join(report_cards.keys())}\n\n"

        # Summary table
        comparison += "| Module | Grade | Scaffolding | Lines | Cells | Avg Cell Length |\n"
        comparison += "|--------|-------|-------------|-------|-------|----------------|\n"

        for name, rc in report_cards.items():
            comparison += f"| {name} | {rc.overall_grade} | {rc.scaffolding_quality}/5 | {rc.total_lines} | {rc.total_cells} | {rc.avg_cell_length:.1f} |\n"

        # Best and worst
        best_module = max(report_cards.items(), key=lambda x: x[1].scaffolding_quality)
        worst_module = min(report_cards.items(), key=lambda x: x[1].scaffolding_quality)

        comparison += f"\n## 🏆 Best Scaffolding: {best_module[0]} ({best_module[1].scaffolding_quality}/5)\n"
        comparison += f"## 🚨 Needs Improvement: {worst_module[0]} ({worst_module[1].scaffolding_quality}/5)\n"

        return comparison

def main():
    parser = argparse.ArgumentParser(description="TinyTorch Module Analyzer & Report Card Generator")
    parser.add_argument("--module", help="Analyze specific module (e.g., 02_activations)")
    parser.add_argument("--all", action="store_true", help="Analyze all modules")
    parser.add_argument("--compare", nargs="+", help="Compare multiple modules")
    parser.add_argument("--format", choices=["json", "html", "both"], default="both", help="Output format")
    parser.add_argument("--save", action="store_true", help="Save report cards to files")
    parser.add_argument("--modules-dir", default="../../modules/source", help="Path to modules directory")

    args = parser.parse_args()

    analyzer = TinyTorchModuleAnalyzer(args.modules_dir)

    if args.module:
        # Analyze single module
        print(f"🔍 Analyzing module: {args.module}")
        try:
            report_card = analyzer.analyze_module(args.module)
            print(f"\n📊 Report Card for {args.module}:")
            print(f"Overall Grade: {report_card.overall_grade}")
            print(f"Scaffolding Quality: {report_card.scaffolding_quality}/5")
            print(f"Critical Issues: {len(report_card.critical_issues)}")

            if args.save:
                saved_files = analyzer.save_report_card(report_card, args.format)
                print(f"💾 Saved to: {', '.join(saved_files)}")

        except Exception as e:
            print(f"❌ Error: {e}")

    elif args.all:
        # Analyze all modules
        print("🔍 Analyzing all modules...")
        results = analyzer.analyze_all_modules()

        print("\n📊 Summary Report:")
        for name, rc in results.items():
            print(f"{name}: Grade {rc.overall_grade} | Scaffolding {rc.scaffolding_quality}/5")

        if args.save:
            for name, rc in results.items():
                saved_files = analyzer.save_report_card(rc, args.format)
                print(f"💾 {name} saved to: {', '.join(saved_files)}")

    elif args.compare:
        # Compare modules
        print(f"🔍 Comparing modules: {', '.join(args.compare)}")
        comparison = analyzer.compare_modules(args.compare)
        print(f"\n{comparison}")

        if args.save:
            with open(f"reports/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", 'w') as f:
                f.write(comparison)
            print("💾 Comparison saved to reports/")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
