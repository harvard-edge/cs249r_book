"""
Shared helpers for TinyTorch export workflows.

These utilities are used by both ExportCommand and SrcCommand to avoid
duplicate logic when converting source files to notebooks, exporting via
nbdev, and protecting generated files.
"""

import json
import re
import stat
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Mapping from generated package paths back to source files
# Keys are (subpackage, module) tuples matching default_exp directives
SOURCE_MAPPINGS = {
    ("core", "tensor"): "src/01_tensor/01_tensor.py",
    ("core", "activations"): "src/02_activations/02_activations.py",
    ("core", "layers"): "src/03_layers/03_layers.py",
    ("core", "losses"): "src/04_losses/04_losses.py",
    ("core", "dataloader"): "src/05_dataloader/05_dataloader.py",
    ("core", "autograd"): "src/06_autograd/06_autograd.py",
    ("core", "optimizers"): "src/07_optimizers/07_optimizers.py",
    ("core", "training"): "src/08_training/08_training.py",
    ("core", "convolutions"): "src/09_convolutions/09_convolutions.py",
    ("core", "tokenization"): "src/10_tokenization/10_tokenization.py",
    ("core", "embeddings"): "src/11_embeddings/11_embeddings.py",
    ("core", "attention"): "src/12_attention/12_attention.py",
    ("core", "transformer"): "src/13_transformers/13_transformers.py",
    ("perf", "profiling"): "src/14_profiling/14_profiling.py",
    ("perf", "quantization"): "src/15_quantization/15_quantization.py",
    ("perf", "compression"): "src/16_compression/16_compression.py",
    ("perf", "acceleration"): "src/17_acceleration/17_acceleration.py",
    ("perf", "memoization"): "src/18_memoization/18_memoization.py",
    ("bench",): "src/19_benchmarking/19_benchmarking.py",
    ("capstone",): "src/20_capstone/20_capstone.py",
}


def get_export_target(module_path: Path) -> str:
    """Read export target from #| default_exp in the source file."""
    module_name = module_path.name
    source_path = Path("src") / module_name if "modules" in str(module_path) else module_path
    dev_file = source_path / f"{module_name}.py"
    if not dev_file.exists():
        return "unknown"

    try:
        content = dev_file.read_text(encoding="utf-8")
        match = re.search(r"#\|\s*default_exp\s+([^\n\r]+)", content)
        if match:
            return match.group(1).strip()
    except Exception:
        return "unknown"

    return "unknown"


def discover_modules(source_dir: Path = Path("src")) -> List[str]:
    """List module directories under src/ excluding common non-module folders."""
    modules = []
    if source_dir.exists():
        exclude_dirs = {".quarto", "__pycache__", ".git", ".pytest_cache"}
        for module_dir in source_dir.iterdir():
            if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                modules.append(module_dir.name)
    return sorted(modules)


def validate_notebook_integrity(notebook_path: Path) -> Dict:
    """Basic validation for generated notebooks."""
    try:
        notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))

        issues = []
        warnings = []

        if "cells" not in notebook_data:
            issues.append("Missing 'cells' field")
        elif not isinstance(notebook_data["cells"], list):
            issues.append("'cells' field is not a list")

        if "metadata" not in notebook_data:
            warnings.append("Missing metadata field")

        if "nbformat" not in notebook_data:
            warnings.append("Missing nbformat field")

        cell_count = 0
        code_cells = 0
        markdown_cells = 0
        if "cells" in notebook_data:
            for i, cell in enumerate(notebook_data["cells"]):
                cell_count += 1
                if "cell_type" not in cell:
                    issues.append(f"Cell {i}: missing cell_type")
                    continue
                cell_type = cell["cell_type"]
                if cell_type == "code":
                    code_cells += 1
                    if "source" not in cell:
                        warnings.append(f"Code cell {i}: missing source")
                elif cell_type == "markdown":
                    markdown_cells += 1
                    if "source" not in cell:
                        warnings.append(f"Markdown cell {i}: missing source")
                else:
                    warnings.append(f"Cell {i}: unusual cell type '{cell_type}'")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "stats": {
                "total_cells": cell_count,
                "code_cells": code_cells,
                "markdown_cells": markdown_cells,
            },
        }

    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "issues": [f"Invalid JSON: {str(e)}"],
            "warnings": [],
            "stats": {},
        }
    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Validation error: {str(e)}"],
            "warnings": [],
            "stats": {},
        }


def convert_py_to_notebook(module_path: Path, venv_path: Path, console) -> bool:
    """Convert src/<module>.py to modules/<module>.ipynb using jupytext."""
    project_root = Path(__file__).resolve().parents[2]  # tinytorch project root
    module_path = module_path if module_path.is_absolute() else project_root / module_path
    module_name = module_path.name
    dev_file = module_path / f"{module_name}.py"
    if not dev_file.exists():
        console.print(f"[red]❌ Python file not found: {dev_file}[/red]")
        return False

    short_name = module_name.split("_", 1)[1] if "_" in module_name else module_name
    modules_dir = project_root / "modules" / module_name
    modules_dir.mkdir(parents=True, exist_ok=True)
    notebook_file = modules_dir / f"{short_name}.ipynb"

    rel_notebook = notebook_file.relative_to(project_root)
    console.print(f"[dim]📄 Source: {dev_file.name} → Target: {rel_notebook}[/dim]")
    console.print("[dim]🔄 Overwriting existing notebook (Python file is source of truth)[/dim]" if notebook_file.exists() else "[dim]✨ Creating new notebook from Python file[/dim]")

    try:
        jupytext_path = "jupytext"
        venv_jupytext = venv_path / "bin" / "jupytext"

        if venv_jupytext.exists():
            test_result = subprocess.run([str(venv_jupytext), "--version"], capture_output=True, text=True)
            if test_result.returncode == 0:
                jupytext_path = str(venv_jupytext)
                console.print(f"[dim]🔧 Using venv jupytext: {venv_jupytext}[/dim]")
            else:
                console.print("[dim]⚠️  Venv jupytext has issues, falling back to system[/dim]")
                console.print(f"[dim]🔧 Using system jupytext: {jupytext_path}[/dim]")
        else:
            console.print(f"[dim]🔧 Using system jupytext: {jupytext_path}[/dim]")

        console.print(f"[dim]⚙️  Running: {jupytext_path} --to ipynb {dev_file.name} --output {notebook_file}[/dim]")
        result = subprocess.run(
            [jupytext_path, "--to", "ipynb", str(dev_file), "--output", str(notebook_file)],
            capture_output=True,
            text=True,
            cwd=project_root,
        )

        if result.returncode != 0:
            console.print(f"[red]❌ Jupytext failed with return code {result.returncode}[/red]")
            if result.stderr:
                console.print(f"[red]Error: {result.stderr.strip()}[/red]")
            return False

        validation = validate_notebook_integrity(notebook_file)
        if not validation["valid"]:
            console.print("[red]❌ Generated notebook has integrity issues:[/red]")
            for issue in validation["issues"]:
                console.print(f"[red]  • {issue}[/red]")
            return False

        if validation["warnings"]:
            console.print("[yellow]⚠️  Notebook warnings:[/yellow]")
            for warning in validation["warnings"]:
                console.print(f"[yellow]  • {warning}[/yellow]")

        stats = validation["stats"]
        console.print(
            f"[dim]📊 Generated notebook: {stats.get('total_cells', 0)} cells "
            f"({stats.get('code_cells', 0)} code, {stats.get('markdown_cells', 0)} markdown)[/dim]"
        )
        return True

    except FileNotFoundError:
        console.print("[red]❌ Jupytext not found. Install with: pip install jupytext[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Conversion error: {e}[/red]")
        return False


def convert_all_modules(venv_path: Path, console) -> List[str]:
    """Convert all src modules to notebooks."""
    converted = []
    for module_name in discover_modules():
        module_path = Path("src") / module_name
        if convert_py_to_notebook(module_path, venv_path, console):
            converted.append(module_name)
    return converted


def find_source_file_for_export(exported_file: Path) -> str:
    """Map an exported package file back to its source file."""
    rel_path = exported_file.relative_to(Path("tinytorch"))
    module_parts = rel_path.with_suffix("").parts
    if module_parts in SOURCE_MAPPINGS:
        return SOURCE_MAPPINGS[module_parts]
    if len(module_parts) >= 2:
        module_name = module_parts[-1]
        return f"src/XX_{module_name}/XX_{module_name}.py"
    return "src/[unknown]/[unknown].py"


def add_autogenerated_warnings(console) -> None:
    """Inject DO NOT EDIT headers into generated package files."""
    console.print("[yellow]🔧 Adding DO NOT EDIT warnings to all exported files...[/yellow]")
    tinytorch_path = Path("tinytorch")
    if not tinytorch_path.exists():
        return

    files_updated = 0
    for py_file in tinytorch_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if "╔═══════════════════════════════════════════════════════════════════════════════╗" in content:
                continue
            if "AUTOGENERATED! DO NOT EDIT! File to edit:" in content:
                lines = content.split("\n")
                if lines and "AUTOGENERATED! DO NOT EDIT! File to edit:" in lines[0]:
                    lines = lines[1:]
                    if lines and lines[0].strip() == "":
                        lines = lines[1:]
                    content = "\n".join(lines)

            source_file = find_source_file_for_export(py_file)
            warning_header = f"""# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║                        🚨 CRITICAL WARNING 🚨                                ║
# ║                     AUTOGENERATED! DO NOT EDIT!                              ║
# ║                                                                               ║
# ║  This file is AUTOMATICALLY GENERATED from source modules.                   ║
# ║  ANY CHANGES MADE HERE WILL BE LOST when modules are re-exported!            ║
# ║                                                                               ║
# ║  ✅ TO EDIT: {source_file:<54} ║
# ║  ✅ TO EXPORT: Run 'tito module complete <module_name>'                      ║
# ║                                                                               ║
# ║  🛡️ STUDENT PROTECTION: This file contains optimized implementations.        ║
# ║     Editing it directly may break module functionality and training.         ║
# ║                                                                               ║
# ║  🎓 LEARNING TIP: Work in src/ (developers) or modules/ (learners)           ║
# ║     The tinytorch/ directory is generated code - edit source files instead!  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝
"""
            lines = content.split("\n")
            insert_index = 0
            if lines and lines[0].startswith("#!"):
                insert_index = 1
            lines.insert(insert_index, warning_header.rstrip())
            py_file.write_text("\n".join(lines), encoding="utf-8")
            files_updated += 1
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not add warning to {py_file}: {e}[/yellow]")

    if files_updated > 0:
        console.print(f"[green]✅ Added auto-generated warnings to {files_updated} files[/green]")


def ensure_writable_target(export_target: str) -> None:
    """Ensure target file is writable before export."""
    if export_target == "unknown":
        return
    target_file = Path("tinytorch") / (export_target.replace(".", "/") + ".py")
    if target_file.exists():
        try:
            target_file.chmod(target_file.stat().st_mode | stat.S_IWUSR)
        except Exception:
            # Best effort; ignore permission errors
            pass
