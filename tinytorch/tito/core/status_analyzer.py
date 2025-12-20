"""
Comprehensive status analysis for TinyTorch modules and environment.

This module provides detailed analysis of:
- Environment health
- Module compliance with TinyTorch standards
- Code quality and functionality
- Testing status
"""

import ast
import sys
import subprocess
import importlib.util
import traceback
import tempfile
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box


@dataclass
class ModuleStatus:
    """Complete status information for a TinyTorch module"""
    name: str
    path: Path
    has_dev_file: bool = False
    has_readme: bool = False
    has_module_yaml: bool = False

    # Compliance checks
    has_introduction: bool = False
    has_math_background: bool = False
    has_implementation: bool = False
    has_testing: bool = False
    has_ml_systems_questions: bool = False
    has_summary: bool = False

    # Code analysis
    classes_count: int = 0
    functions_count: int = 0
    imports_successfully: bool = False
    runs_without_errors: bool = False

    # Testing
    has_inline_tests: bool = False
    tests_pass: bool = False

    # Special requirements
    has_required_profiler: bool = False

    # Issues
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    @property
    def compliance_score(self) -> float:
        """Calculate compliance score (0-1)"""
        checks = [
            self.has_introduction,
            self.has_math_background,
            self.has_implementation,
            self.has_testing,
            self.has_ml_systems_questions,
            self.has_summary
        ]
        return sum(checks) / len(checks)

    @property
    def overall_status(self) -> str:
        """Get overall status: EXCELLENT, GOOD, PARTIAL, BROKEN"""
        if not self.has_dev_file:
            return "BROKEN"
        if not self.imports_successfully:
            return "BROKEN"
        if self.compliance_score >= 0.9 and self.runs_without_errors:
            return "EXCELLENT"
        if self.compliance_score >= 0.7 and self.imports_successfully:
            return "GOOD"
        if self.compliance_score >= 0.4:
            return "PARTIAL"
        return "BROKEN"


class TinyTorchStatusAnalyzer:
    """Comprehensive TinyTorch system status analyzer"""

    def __init__(self, repo_path: Optional[Path] = None):
        """Initialize the status analyzer.

        Args:
            repo_path: Path to TinyTorch repository. If None, uses current working directory.
        """
        if repo_path is None:
            repo_path = Path.cwd()
        self.repo_path = Path(repo_path)
        self.modules_path = self.repo_path / "modules" / "source"
        self.modules: Dict[str, ModuleStatus] = {}
        self.environment_status = {}
        self.tito_status = {}

    def check_environment(self) -> Dict[str, Any]:
        """Check Python environment and dependencies"""
        env_status = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'python_path': sys.executable,
            'virtual_env_active': False,
            'dependencies': {},
            'issues': []
        }

        # Check virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            env_status['virtual_env_active'] = True
        else:
            env_status['issues'].append("Virtual environment not activated")

        # Check critical dependencies
        critical_deps = ['numpy', 'matplotlib', 'pytest', 'rich', 'networkx']

        for dep in critical_deps:
            try:
                spec = importlib.util.find_spec(dep)
                if spec is not None:
                    module = importlib.import_module(dep)
                    version = getattr(module, '__version__', 'unknown')
                    env_status['dependencies'][dep] = {'status': 'installed', 'version': version}
                else:
                    env_status['dependencies'][dep] = {'status': 'missing', 'version': None}
                    env_status['issues'].append(f"Missing dependency: {dep}")
            except Exception as e:
                env_status['dependencies'][dep] = {'status': 'error', 'version': None, 'error': str(e)}
                env_status['issues'].append(f"Error importing {dep}: {str(e)}")

        self.environment_status = env_status
        return env_status

    def check_tito_health(self) -> Dict[str, Any]:
        """Check tito CLI system health"""
        tito_status = {
            'tito_available': False,
            'commands_working': {},
            'issues': []
        }

        try:
            # Check if tito is available
            result = subprocess.run(['tito', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                tito_status['tito_available'] = True
            else:
                tito_status['issues'].append("Tito command not available")
        except Exception as e:
            tito_status['issues'].append(f"Tito CLI error: {str(e)}")

        # Test key commands if tito is available
        if tito_status['tito_available']:
            test_commands = ['system info', 'module status']
            for cmd in test_commands:
                try:
                    result = subprocess.run(f'tito {cmd}'.split(), capture_output=True, text=True, timeout=30)
                    tito_status['commands_working'][cmd] = result.returncode == 0
                    if result.returncode != 0:
                        tito_status['issues'].append(f"Tito '{cmd}' command failed")
                except Exception as e:
                    tito_status['commands_working'][cmd] = False
                    tito_status['issues'].append(f"Tito '{cmd}' error: {str(e)}")

        self.tito_status = tito_status
        return tito_status

    def analyze_module(self, module_path: Path) -> ModuleStatus:
        """Comprehensive analysis of a single module"""
        module_name = module_path.name
        status = ModuleStatus(name=module_name, path=module_path)

        # Check basic files - try multiple naming patterns
        possible_dev_files = [
            module_path / f"{module_name}.py",
            module_path / f"{module_name.split('_', 1)[1]}.py" if '_' in module_name else None,
        ]
        dev_file = None
        for possible_file in possible_dev_files:
            if possible_file and possible_file.exists():
                dev_file = possible_file
                break

        if dev_file is None:
            # Check if there's any *.py file
            dev_files = list(module_path.glob("*.py"))
            if dev_files:
                dev_file = dev_files[0]  # Use the first one found

        status.has_dev_file = dev_file is not None and dev_file.exists()
        status.has_readme = (module_path / "README.md").exists()
        status.has_module_yaml = (module_path / "module.yaml").exists()

        if not status.has_dev_file:
            status.issues.append("Missing dev file")
            return status

        # Analyze Python file
        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST for code analysis
            try:
                tree = ast.parse(content)
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

                status.classes_count = len(classes)
                status.functions_count = len(functions)

                # Check for required profiler classes
                required_profilers = {
                    # Foundation tier (01-08)
                    '02_activations': 'ActivationProfiler',
                    '03_layers': 'LayerArchitectureProfiler',
                    '05_dataloader': 'DataPipelineProfiler',
                    '06_autograd': 'AutogradSystemsProfiler',
                    '08_training': 'TrainingPipelineProfiler',
                    # Architecture tier (09-13)
                    '09_convolutions': 'ConvolutionProfiler',
                    '12_attention': 'AttentionEfficiencyProfiler',
                    # Optimization tier (14-20)
                    '16_compression': 'CompressionSystemsProfiler',
                    '19_benchmarking': 'ProductionBenchmarkingProfiler',
                    '20_capstone': 'ProductionMLSystemProfiler',
                }

                if module_name in required_profilers:
                    status.has_required_profiler = required_profilers[module_name] in classes
                else:
                    status.has_required_profiler = True  # Not required for this module

            except SyntaxError as e:
                status.issues.append(f"Syntax error: {str(e)}")

            # Check module structure compliance
            status.has_introduction = "Module Introduction" in content or "# Introduction" in content
            status.has_math_background = "Mathematical Background" in content or "Mathematical Foundation" in content
            status.has_implementation = "Implementation" in content or "Core Implementation" in content
            status.has_testing = "Testing" in content and "test_" in content
            status.has_ml_systems_questions = "ML Systems Thinking" in content or "Systems Thinking" in content
            status.has_summary = "Module Summary" in content or "MODULE SUMMARY" in content

            # Check for inline tests
            status.has_inline_tests = "def test_" in content

            # Test if module can be imported
            try:
                # Create a temporary file to test import
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                    # Write a minimal test to check if the module can be imported
                    test_code = f"""
import sys
sys.path.insert(0, '{module_path}')
try:
    exec(open('{dev_file}').read())
    print("SUCCESS: Module imports and runs")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
                    temp_file.write(test_code)
                    temp_file_path = temp_file.name

                # Run the test
                result = subprocess.run([sys.executable, temp_file_path],
                                     capture_output=True, text=True, timeout=30)

                status.imports_successfully = "SUCCESS" in result.stdout
                status.runs_without_errors = result.returncode == 0 and "SUCCESS" in result.stdout

                if not status.imports_successfully and result.stderr:
                    status.issues.append(f"Import error: {result.stderr.strip()}")

                # Clean up
                os.unlink(temp_file_path)

            except Exception as e:
                status.issues.append(f"Could not test import: {str(e)}")

        except Exception as e:
            status.issues.append(f"File analysis error: {str(e)}")

        return status

    def check_all_modules(self) -> Dict[str, ModuleStatus]:
        """Check all modules in the source directory"""
        if not self.modules_path.exists():
            return {}

        modules = {}
        module_dirs = [d for d in self.modules_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

        for module_dir in sorted(module_dirs):
            modules[module_dir.name] = self.analyze_module(module_dir)

        self.modules = modules
        return modules

    def generate_comprehensive_report(self, console: Console) -> None:
        """Generate comprehensive status report using rich console"""
        console.print("\n" + "="*80, style="bold")
        console.print("🔥 TINYTORCH COMPREHENSIVE STATUS DASHBOARD", style="bold red", justify="center")
        console.print("="*80, style="bold")

        # Environment Status Panel
        self._print_environment_panel(console)

        # Module Compliance Overview
        self._print_module_overview(console)

        # Detailed Module Status
        self._print_detailed_module_status(console)

        # Action Items
        self._print_action_items(console)

    def _print_environment_panel(self, console: Console):
        """Print environment status panel"""
        env = self.environment_status

        # Environment table
        env_table = Table(title="🐍 Environment Health", box=box.ROUNDED)
        env_table.add_column("Component", style="cyan")
        env_table.add_column("Status", justify="center")
        env_table.add_column("Details", style="dim")

        # Python version
        env_table.add_row("Python", "✅", f"{env['python_version']}")

        # Virtual environment
        venv_status = "✅ OK" if env['virtual_env_active'] else "❌ Not Activated"
        env_table.add_row("Virtual Env", venv_status, "Required for development")

        # Dependencies
        for dep, info in env['dependencies'].items():
            if info['status'] == 'installed':
                status = "✅"
                detail = f"v{info['version']}"
            elif info['status'] == 'missing':
                status = "❌"
                detail = "Not installed"
            else:
                status = "⚠️"
                detail = "Import error"
            env_table.add_row(dep.title(), status, detail)

        console.print(env_table)

    def _print_module_overview(self, console: Console):
        """Print module overview statistics"""
        if not self.modules:
            return

        # Calculate statistics
        total_modules = len(self.modules)
        excellent = sum(1 for m in self.modules.values() if m.overall_status == "EXCELLENT")
        good = sum(1 for m in self.modules.values() if m.overall_status == "GOOD")
        partial = sum(1 for m in self.modules.values() if m.overall_status == "PARTIAL")
        broken = sum(1 for m in self.modules.values() if m.overall_status == "BROKEN")

        # Overview table
        overview_table = Table(title="📊 Module Status Overview", box=box.ROUNDED)
        overview_table.add_column("Status Level", style="bold")
        overview_table.add_column("Count", justify="center")
        overview_table.add_column("Percentage", justify="center")
        overview_table.add_column("Description")

        overview_table.add_row("🎉 EXCELLENT", str(excellent), f"{excellent/total_modules*100:.1f}%", "Fully compliant & working")
        overview_table.add_row("✅ GOOD", str(good), f"{good/total_modules*100:.1f}%", "Minor compliance issues")
        overview_table.add_row("⚠️ PARTIAL", str(partial), f"{partial/total_modules*100:.1f}%", "Major compliance issues")
        overview_table.add_row("❌ BROKEN", str(broken), f"{broken/total_modules*100:.1f}%", "Not working")

        console.print(overview_table)

    def _print_detailed_module_status(self, console: Console):
        """Print detailed status for each module"""
        if not self.modules:
            return

        # Detailed module table
        detail_table = Table(title="📋 Detailed Module Analysis", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        detail_table.add_column("Module", style="cyan", width=15)
        detail_table.add_column("Status", justify="center", width=10)
        detail_table.add_column("Compliance", justify="center", width=10)
        detail_table.add_column("Code", justify="center", width=8)
        detail_table.add_column("Tests", justify="center", width=8)
        detail_table.add_column("Issues", style="red", width=25)

        # Sort modules by name
        for name in sorted(self.modules.keys()):
            module = self.modules[name]

            # Status indicator
            status_map = {
                "EXCELLENT": "🎉",
                "GOOD": "✅",
                "PARTIAL": "⚠️",
                "BROKEN": "❌"
            }
            status = status_map.get(module.overall_status, "❓")

            # Compliance score
            compliance = f"{module.compliance_score*100:.0f}%"

            # Code health
            if module.imports_successfully and module.runs_without_errors:
                code_health = "✅"
            elif module.imports_successfully:
                code_health = "⚠️"
            else:
                code_health = "❌"

            # Testing status
            test_status = "✅" if module.has_inline_tests else "❌"

            # Issues summary
            issues_text = "; ".join(module.issues[:2]) if module.issues else "None"
            if len(module.issues) > 2:
                issues_text += f" (+{len(module.issues)-2} more)"

            detail_table.add_row(name, status, compliance, code_health, test_status, issues_text)

        console.print(detail_table)

    def _print_action_items(self, console: Console):
        """Print prioritized action items"""
        if not self.modules:
            return

        actions = []

        # Environment issues
        if not self.environment_status['virtual_env_active']:
            actions.append("🔴 HIGH: Activate virtual environment (.venv)")

        for dep, info in self.environment_status['dependencies'].items():
            if info['status'] != 'installed':
                actions.append(f"🔴 HIGH: Install {dep} dependency")

        # Broken modules
        broken_modules = [name for name, module in self.modules.items() if module.overall_status == "BROKEN"]
        for module_name in broken_modules:
            actions.append(f"🔴 HIGH: Fix broken module: {module_name}")

        # Compliance issues
        partial_modules = [name for name, module in self.modules.items() if module.overall_status == "PARTIAL"]
        for module_name in partial_modules[:3]:  # Show top 3
            actions.append(f"🟡 MED: Improve compliance for: {module_name}")

        # Missing profilers
        missing_profilers = [name for name, module in self.modules.items() if not module.has_required_profiler]
        for module_name in missing_profilers[:2]:  # Show top 2
            actions.append(f"🟡 MED: Add required profiler to: {module_name}")

        if actions:
            console.print("\n🎯 [bold]Priority Action Items:[/bold]")
            for i, action in enumerate(actions[:8], 1):  # Show top 8
                console.print(f"  {i}. {action}")
        else:
            console.print("\n🎉 [bold green]All systems operational! No critical issues found.[/bold green]")

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete TinyTorch system analysis"""
        # Run all checks
        env_status = self.check_environment()
        tito_status = self.check_tito_health()
        modules_status = self.check_all_modules()

        return {
            'environment': env_status,
            'tito': tito_status,
            'modules': {name: {
                'status': module.overall_status,
                'compliance_score': module.compliance_score,
                'issues': module.issues
            } for name, module in modules_status.items()},
            'summary': {
                'total_modules': len(modules_status),
                'working_modules': sum(1 for m in modules_status.values() if m.overall_status in ["EXCELLENT", "GOOD"]),
                'environment_healthy': len(env_status['issues']) == 0,
                'tito_working': tito_status['tito_available']
            }
        }
