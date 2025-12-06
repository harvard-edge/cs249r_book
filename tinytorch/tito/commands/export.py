"""
Export command for TinyTorch CLI: exports notebook code to Python package using nbdev.
"""

import subprocess
import sys
import re
import stat
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Dict
from rich.panel import Panel
from rich.text import Text
import logging

logger = logging.getLogger(__name__)

from .base import BaseCommand

class ExportCommand(BaseCommand):

    @property
    def name(self) -> str:
        return "export"

    @property
    def description(self) -> str:
        return "Export notebook code to Python package"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("modules", nargs="*", help="Export specific modules (e.g., 01_tensor 02_activations)")
        parser.add_argument("--all", action="store_true", help="Export all modules")
        parser.add_argument("--from-release", action="store_true", help="Export from release directory (student version) instead of source")
        parser.add_argument("--test-checkpoint", action="store_true", help="Run checkpoint test after successful export")

    def _get_export_target(self, module_path: Path) -> str:
        """
        Read the actual export target from the source file's #| default_exp directive.
        This is the source of truth, not the YAML file.
        """
        # module_path might be pointing to modules/ (notebooks) or src/ (source)
        # Always look in src/ for the source file
        module_name = module_path.name
        
        # Check if we're in modules/ or src/
        if "modules" in str(module_path):
            # Convert modules/XX_name to src/XX_name
            source_path = Path("src") / module_name
        else:
            source_path = module_path
        
        # Use full module name for file (e.g., 01_tensor.py)
        dev_file = source_path / f"{module_name}.py"
        if not dev_file.exists():
            return "unknown"

        try:
            with open(dev_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for #| default_exp directive with more flexible regex
                match = re.search(r'#\|\s*default_exp\s+([^\n\r]+)', content)
                if match:
                    return match.group(1).strip()
        except Exception as e:
            # Debug: print the error for troubleshooting
            print(f"Debug: Error reading {dev_file}: {e}")
        
        return "unknown"

    def _discover_modules(self) -> list:
        """Discover available modules from src directory."""
        source_dir = Path("src")
        modules = []
        
        if source_dir.exists():
            exclude_dirs = {'.quarto', '__pycache__', '.git', '.pytest_cache'}
            for module_dir in source_dir.iterdir():
                if module_dir.is_dir() and module_dir.name not in exclude_dirs:
                    modules.append(module_dir.name)
        
        return sorted(modules)

    def _show_next_steps(self, completed_module: str) -> None:
        """Show next steps after successful module completion."""
        console = self.console
        
        # Get module number for next module suggestion
        if completed_module.startswith(tuple(f"{i:02d}_" for i in range(100))):
            try:
                module_num = int(completed_module[:2])
                next_num = module_num + 1
                
                # Suggest next module (updated for reordered progression)
                next_modules = {
                    1: ("02_tensor", "Tensor operations - the foundation of ML"),
                    2: ("03_activations", "Activation functions - adding intelligence"),
                    3: ("04_layers", "Neural layers - building blocks"),
                    4: ("05_losses", "Loss functions - measuring performance"),
                    5: ("06_optimizers", "Optimization algorithms - systematic weight updates"),
                    6: ("07_autograd", "Automatic differentiation - gradient computation"),
                    7: ("08_training", "Training loops - end-to-end learning"),
                    8: ("09_spatial", "Spatial processing - convolutional operations"),
                    9: ("10_dataloader", "Data loading - efficient training pipelines"),
                    10: ("11_tokenization", "Text preprocessing - sequence understanding"),
                    11: ("12_embeddings", "Vector representations - semantic learning"),
                    12: ("13_attention", "Attention mechanisms - selective focus"),
                    13: ("14_transformers", "Transformer architectures - sequence modeling"),
                    14: ("15_acceleration", "Performance optimization - efficient computation"),
                    19: ("20_capstone", "Capstone project - complete ML systems"),
                }
                
                if next_num in next_modules:
                    next_module, next_desc = next_modules[next_num]
                    console.print(f"\n[bold cyan]🎯 Continue Your Journey[/bold cyan]")
                    console.print(f"[bold]Next Module:[/bold] {next_module}")
                    console.print(f"[dim]{next_desc}[/dim]")
                    console.print(f"\n[green]Ready to continue? Run:[/green]")
                    console.print(f"[dim]  tito module start {next_module}[/dim]")
                elif next_num > 16:
                    console.print(f"\n[bold green]🏆 Congratulations![/bold green]")
                    console.print(f"[green]You've completed all TinyTorch modules![/green]")
                    console.print(f"[dim]Run 'tito checkpoint status' to see your full progress[/dim]")
            except (ValueError, IndexError):
                pass
        
        # General next steps
        console.print(f"\n[bold]Continue your ML systems journey:[/bold]")
        console.print(f"[dim]  tito checkpoint status    - View overall progress[/dim]")
        console.print(f"[dim]  tito checkpoint timeline  - Visual progress timeline[/dim]")

    def _add_autogenerated_warnings(self, console):
        """Add auto-generated warnings to all exported Python files."""
        console.print("[yellow]🔧 Adding DO NOT EDIT warnings to all exported files...[/yellow]")
        
        tinytorch_path = Path("tinytorch")
        if not tinytorch_path.exists():
            return
        
        files_updated = 0
        for py_file in tinytorch_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue  # Skip __init__.py files
                
            try:
                # Read the current content
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if warning already exists (check for the box format specifically)
                if "╔═══════════════════════════════════════════════════════════════════════════════╗" in content:
                    continue  # Already has the new warning format
                
                # Remove old header format if it exists
                if "AUTOGENERATED! DO NOT EDIT! File to edit:" in content:
                    lines = content.split('\n')
                    # Remove the old header line (usually first line)
                    if lines and "AUTOGENERATED! DO NOT EDIT! File to edit:" in lines[0]:
                        lines = lines[1:]  # Remove first line
                        # Also remove empty line after if it exists
                        if lines and lines[0].strip() == "":
                            lines = lines[1:]
                        content = '\n'.join(lines)
                
                # Find the source file for this export
                source_file = self._find_source_file_for_export(py_file)
                
                # Create enhanced auto-generated warning header
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
# ║  🎓 LEARNING TIP: Work in src/ (developers) or modules/ (learners)    ║
# ║     The tinytorch/ directory is generated code - edit source files instead!  ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

"""
                
                # Add warning at the top (after any existing shebang)
                lines = content.split('\n')
                insert_index = 0
                
                # Skip shebang line if present
                if lines and lines[0].startswith('#!'):
                    insert_index = 1
                
                # Insert warning
                lines.insert(insert_index, warning_header.rstrip())
                
                # Write back the modified content
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                files_updated += 1
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Could not add warning to {py_file}: {e}[/yellow]")
        
        if files_updated > 0:
            console.print(f"[green]✅ Added auto-generated warnings to {files_updated} files[/green]")

    def _find_source_file_for_export(self, exported_file: Path) -> str:
        """Find the source dev file that generated this export."""
        # Convert tinytorch/core/something.py back to source path
        rel_path = exported_file.relative_to(Path("tinytorch"))
        
        # Remove .py extension and convert to module path
        module_parts = rel_path.with_suffix('').parts
        
        # Common mappings (source files are now in src/)
        source_mappings = {
            ('core', 'tensor'): 'src/01_tensor/01_tensor.py',
            ('core', 'activations'): 'src/02_activations/02_activations.py', 
            ('core', 'layers'): 'src/03_layers/03_layers.py',
            ('core', 'dense'): 'src/04_losses/04_losses.py',
            ('core', 'spatial'): 'src/09_spatial/09_spatial.py',
            ('core', 'attention'): 'src/12_attention/12_attention.py',
            ('core', 'dataloader'): 'src/08_dataloader/08_dataloader.py',
            ('core', 'autograd'): 'src/05_autograd/05_autograd.py',
            ('core', 'optimizers'): 'src/06_optimizers/06_optimizers.py',
            ('core', 'training'): 'src/07_training/07_training.py',
            ('core', 'compression'): 'src/16_compression/16_compression.py',
            ('core', 'benchmarking'): 'src/19_benchmarking/19_benchmarking.py',
        }
        
        if module_parts in source_mappings:
            return source_mappings[module_parts]
        
        # Fallback: try to guess based on the file name
        if len(module_parts) >= 2:
            module_name = module_parts[-1]  # e.g., 'tensor' from ('core', 'tensor')
            return f"src/XX_{module_name}/XX_{module_name}.py"
        
        return "src/[unknown]/[unknown].py"

    def _show_export_details(self, console, module_name: Optional[str] = None):
        """Show detailed export information including where each module exports to."""
        exports_text = Text()
        exports_text.append("📦 Export Details:\n", style="bold cyan")
        
        if module_name:
            # Single module export
            module_path = Path(f"modules/{module_name}")
            export_target = self._get_export_target(module_path)
            if export_target != "unknown":
                target_file = export_target.replace('.', '/') + '.py'
                exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
                
                # Extract the short name for display
                short_name = module_name[3:] if module_name.startswith(tuple(f"{i:02d}_" for i in range(100))) else module_name
                exports_text.append(f"     Source: modules/{module_name}/{short_name}.py\n", style="dim")
                exports_text.append(f"     Target: tinytorch/{target_file}\n", style="dim")
            else:
                exports_text.append(f"  ❓ {module_name} → export target not found\n", style="yellow")
        else:
            # All modules export
            modules = self._discover_modules()
            for module_name in modules:
                module_path = Path(f"modules/{module_name}")
                export_target = self._get_export_target(module_path)
                if export_target != "unknown":
                    target_file = export_target.replace('.', '/') + '.py'
                    exports_text.append(f"  🔄 {module_name} → tinytorch/{target_file}\n", style="green")
        
        # Show what was actually created
        exports_text.append("\n📁 Generated Files:\n", style="bold cyan")
        tinytorch_path = Path("tinytorch")
        if tinytorch_path.exists():
            for py_file in tinytorch_path.rglob("*.py"):
                if py_file.name != "__init__.py" and py_file.stat().st_size > 100:  # Non-empty files
                    rel_path = py_file.relative_to(tinytorch_path)
                    exports_text.append(f"  ✅ tinytorch/{rel_path}\n", style="green")
        
        exports_text.append("\n💡 Next steps:\n", style="bold yellow")
        exports_text.append("  • Run: tito test --all\n", style="white")
        exports_text.append("  • Or: tito test <module_name>\n", style="white")
        exports_text.append("  • Or: tito export <module> --test-checkpoint\n", style="white")
        
        console.print(Panel(exports_text, title="Export Summary", border_style="bright_green"))

    def _validate_notebook_integrity(self, notebook_path: Path) -> Dict:
        """Validate notebook integrity and structure."""
        import json
        
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook_data = json.load(f)
            
            # Basic structure checks
            issues = []
            warnings = []
            
            # Check required fields
            if 'cells' not in notebook_data:
                issues.append("Missing 'cells' field")
            elif not isinstance(notebook_data['cells'], list):
                issues.append("'cells' field is not a list")
            
            if 'metadata' not in notebook_data:
                warnings.append("Missing metadata field")
            
            if 'nbformat' not in notebook_data:
                warnings.append("Missing nbformat field")
            
            # Check cells for common issues
            cell_count = 0
            code_cells = 0
            markdown_cells = 0
            
            if 'cells' in notebook_data:
                for i, cell in enumerate(notebook_data['cells']):
                    cell_count += 1
                    
                    if 'cell_type' not in cell:
                        issues.append(f"Cell {i}: missing cell_type")
                        continue
                    
                    cell_type = cell['cell_type']
                    if cell_type == 'code':
                        code_cells += 1
                        if 'source' not in cell:
                            warnings.append(f"Code cell {i}: missing source")
                    elif cell_type == 'markdown':
                        markdown_cells += 1
                        if 'source' not in cell:
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
                    "markdown_cells": markdown_cells
                }
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "issues": [f"Invalid JSON: {str(e)}"],
                "warnings": [],
                "stats": {}
            }
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "stats": {}
            }

    def _convert_py_to_notebook(self, module_path: Path) -> bool:
        """Convert .py file from src/ to .ipynb in modules/ using Jupytext."""
        module_name = module_path.name
        
        # Source file is in src/ with full module name (e.g., src/01_tensor/01_tensor.py)
        dev_file = module_path / f"{module_name}.py"
        if not dev_file.exists():
            self.console.print(f"[red]❌ Python file not found: {dev_file}[/red]")
            return False

        # Output notebook goes to modules/ (e.g., modules/01_tensor/01_tensor.ipynb)
        modules_dir = Path("modules") / module_name
        modules_dir.mkdir(parents=True, exist_ok=True)
        notebook_file = modules_dir / f"{module_name}.ipynb"
        
        # Always regenerate notebook from Python file (Python is source of truth)
        self.console.print(f"[dim]📄 Source: {dev_file.name} → Target: {notebook_file.name}[/dim]")
        
        if notebook_file.exists():
            self.console.print(f"[dim]🔄 Overwriting existing notebook (Python file is source of truth)[/dim]")
        else:
            self.console.print(f"[dim]✨ Creating new notebook from Python file[/dim]")
        
        try:
            # Prefer venv jupytext, fallback to system if needed
            jupytext_path = "jupytext"
            
            # Get the project root directory (where .venv should be)
            project_root = Path(__file__).parent.parent.parent
            venv_jupytext = self.venv_path / "bin" / "jupytext"
            
            if venv_jupytext.exists():
                # Test venv jupytext first
                test_result = subprocess.run([str(venv_jupytext), "--version"], 
                                           capture_output=True, text=True)
                if test_result.returncode == 0:
                    jupytext_path = str(venv_jupytext)
                    self.console.print(f"[dim]🔧 Using venv jupytext: {venv_jupytext}[/dim]")
                else:
                    self.console.print(f"[dim]⚠️  Venv jupytext has issues, falling back to system[/dim]")
                    self.console.print(f"[dim]🔧 Using system jupytext: {jupytext_path}[/dim]")
            else:
                self.console.print(f"[dim]🔧 Using system jupytext: {jupytext_path}[/dim]")
            
            self.console.print(f"[dim]⚙️  Running: {jupytext_path} --to ipynb {dev_file.name} --output {notebook_file.name}[/dim]")

            result = subprocess.run([
                jupytext_path, "--to", "ipynb", str(dev_file), "--output", str(notebook_file)
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                self.console.print(f"[dim]✅ Jupytext conversion successful[/dim]")
                
                # Validate the generated notebook
                validation = self._validate_notebook_integrity(notebook_file)
                if not validation["valid"]:
                    self.console.print(f"[red]❌ Generated notebook has integrity issues:[/red]")
                    for issue in validation["issues"]:
                        self.console.print(f"[red]  • {issue}[/red]")
                    return False
                
                if validation["warnings"]:
                    self.console.print("[yellow]⚠️  Notebook warnings:[/yellow]")
                    for warning in validation["warnings"]:
                        self.console.print(f"[yellow]  • {warning}[/yellow]")
                
                # Show notebook stats
                stats = validation["stats"]
                self.console.print(f"[dim]📊 Generated notebook: {stats.get('total_cells', 0)} cells "
                                 f"({stats.get('code_cells', 0)} code, {stats.get('markdown_cells', 0)} markdown)[/dim]")
                
                return True
            else:
                self.console.print(f"[red]❌ Jupytext failed with return code {result.returncode}[/red]")
                if result.stderr:
                    self.console.print(f"[red]Error: {result.stderr.strip()}[/red]")
                return False
                
        except FileNotFoundError:
            self.console.print(f"[red]❌ Jupytext not found. Install with: pip install jupytext[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Conversion error: {e}[/red]")
            return False
    
    def _convert_all_modules(self) -> list:
        """Convert all modules' .py files from src/ to .ipynb files in modules/."""
        modules = self._discover_modules()
        converted = []
        
        for module_name in modules:
            module_path = Path(f"src/{module_name}")
            if self._convert_py_to_notebook(module_path):
                converted.append(module_name)
        
        return converted

    def run(self, args: Namespace) -> int:
        console = self.console
        logger.info("Starting export command")
        
        # Determine what to export
        if hasattr(args, 'modules') and args.modules:
            logger.info(f"Exporting specific modules: {args.modules}")
            # Export multiple specific modules
            modules_to_export = args.modules
            
            console.print(Panel(f"🔄 Exporting Modules: {', '.join(modules_to_export)}", 
                               title="Complete Export Workflow", border_style="bright_cyan"))
            
            exported_notebooks = []
            
            # Process each module
            for module_name in modules_to_export:
                logger.debug(f"Processing module: {module_name}")
                module_path = Path(f"src/{module_name}")
                if not module_path.exists():
                    console.print(Panel(f"[red]❌ Module '{module_name}' not found in src/[/red]", 
                                      title="Module Not Found", border_style="red"))
                    
                    # Show available modules
                    available_modules = self._discover_modules()
                    if available_modules:
                        help_text = Text()
                        help_text.append("Available modules:\n", style="bold yellow")
                        for module in available_modules:
                            help_text.append(f"  • {module}\n", style="white")
                        console.print(Panel(help_text, title="Available Modules", border_style="yellow"))
                    
                    return 1
                
                # Always convert Python file to notebook (Python file is source of truth)
                # Notebook will be created in modules/ directory
                notebook_file = Path("modules") / module_name / f"{module_name}.ipynb"
                
                console.print(f"📝 Converting {module_name} Python file to notebook...")
                if not self._convert_py_to_notebook(module_path):
                    logger.error(f"Failed to convert .py file to notebook for {module_name}")
                    return 1
                exported_notebooks.append(str(notebook_file))
            
            logger.info(f"Exporting {len(exported_notebooks)} notebooks to tinytorch package")
            
            # Export all notebooks
            success_count = 0
            for notebook_path_str in exported_notebooks:
                try:
                    notebook_path = Path(notebook_path_str)
                    notebook_name = notebook_path.name
                    console.print(f"[dim]🔄 Exporting {notebook_name} to tinytorch package...[/dim]")

                    # --- FIX: Ensure target file is writable before exporting ---
                    module_path = notebook_path.parent
                    export_target = self._get_export_target(module_path)
                    if export_target != "unknown":
                        target_file_rel_path = export_target.replace('.', '/') + '.py'
                        target_file = Path("tinytorch") / target_file_rel_path
                        
                        if target_file.exists():
                            try:
                                # Add write permission for the owner to overwrite the file
                                target_file.chmod(target_file.stat().st_mode | stat.S_IWUSR)
                            except Exception as e:
                                console.print(f"[yellow]⚠️  Could not make {target_file} writable: {e}[/yellow]")
                    
                    cmd = ["nbdev_export", "--path", notebook_path_str]
                    console.print(f"[dim]⚙️  Running: nbdev_export --path {notebook_name}[/dim]")
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
                    if result.returncode == 0:
                        success_count += 1
                        console.print(f"✅ Exported: {notebook_name}")
                        if result.stdout.strip():
                            console.print(f"[dim]📝 {result.stdout.strip()}[/dim]")
                    else:
                        console.print(f"❌ Failed to export: {notebook_name}")
                        console.print(f"   Return code: {result.returncode}")
                        if result.stderr.strip():
                            console.print(f"   Error: {result.stderr.strip()}")
                        if result.stdout.strip():
                            console.print(f"   Output: {result.stdout.strip()}")
                except Exception as e:
                    console.print(f"❌ Error exporting {Path(notebook_path).name}: {e}")
            
            if success_count == len(exported_notebooks):
                logger.info("All notebooks exported successfully")
                # ALWAYS add auto-generated warnings immediately after export
                self._add_autogenerated_warnings(console)
                
                # 🛡️ AUTOMATIC PROTECTION: Enable protection after export
                self._auto_enable_protection(console)
                
                console.print(Panel(f"[green]✅ Successfully exported {success_count}/{len(exported_notebooks)} modules to tinytorch package![/green]", 
                              title="Export Success", border_style="green"))
                return 0
            else:
                logger.warning(f"Exported {success_count}/{len(exported_notebooks)} modules. Some exports failed.")
                console.print(Panel(f"[yellow]⚠️ Exported {success_count}/{len(exported_notebooks)} modules. Some exports failed.[/yellow]", 
                              title="Partial Success", border_style="yellow"))
                return 1
        elif hasattr(args, 'all') and args.all:
            logger.info("Exporting all modules")
            console.print(Panel("🔄 Exporting All Modules to Package", 
                               title="Complete Export Workflow", border_style="bright_cyan"))
            
            # Step 1: Convert all .py files to .ipynb
            console.print("📝 Converting all Python files to notebooks...")
            converted = self._convert_all_modules()
            if not converted:
                logger.error("No modules converted. Check if jupytext is installed and .py files exist.")
                console.print(Panel("[red]❌ No modules converted. Check if jupytext is installed and .py files exist.[/red]", 
                                  title="Conversion Error", border_style="red"))
                return 1
            
            console.print(f"✅ Converted {len(converted)} modules: {', '.join(converted)}")
            console.print("🔄 Exporting all notebook code to tinytorch package...")
            
            # Step 2: Use nbdev_export for all modules  
            cmd = ["nbdev_export"]
        else:
            logger.error("Must specify either module names or --all")
            console.print(Panel("[red]❌ Must specify either module names or --all[/red]\n\n"
                              "[dim]Examples:[/dim]\n"
                              "[dim]  tito module export 01_tensor[/dim]\n"
                              "[dim]  tito module export 01_tensor 02_activations[/dim]\n"
                              "[dim]  tito module export --all[/dim]", 
                              title="Missing Arguments", border_style="red"))
            return 1
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                logger.info("Export command completed successfully")
                # ALWAYS add auto-generated warnings immediately after export
                self._add_autogenerated_warnings(console)
                
                # 🛡️ AUTOMATIC PROTECTION: Enable protection after export
                self._auto_enable_protection(console)
                
                console.print(Panel("[green]✅ Successfully exported notebook code to tinytorch package![/green]", 
                                  title="Export Success", border_style="green"))
                
                # Show detailed export information
                module_names = args.modules if hasattr(args, 'modules') and args.modules else None
                if module_names and len(module_names) == 1:
                    self._show_export_details(console, module_names[0])
                else:
                    self._show_export_details(console, None)
                
            else:
                logger.error(f"Export failed with return code {result.returncode}")
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                console.print(Panel(f"[red]❌ Export failed:\n{error_msg}[/red]", 
                                  title="Export Error", border_style="red"))
                
                # Helpful error guidance
                help_text = Text()
                help_text.append("💡 Common issues:\n", style="bold yellow")
                help_text.append("  • Missing #| default_exp directive in notebook\n", style="white")
                help_text.append("  • Syntax errors in exported code\n", style="white")
                help_text.append("  • Missing settings.ini configuration\n", style="white")
                help_text.append("\n🔧 Run 'tito system health' for detailed diagnosis", style="cyan")
                
                console.print(Panel(help_text, title="Troubleshooting", border_style="yellow"))
                
            return result.returncode
            
        except FileNotFoundError:
            logger.exception("nbdev not found. Install with: pip install nbdev")
            return 1
        except Exception as e:
            logger.exception(f"Unexpected error during export: {e}")
            return 1
    
    def _auto_enable_protection(self, console):
        """🛡️ Automatically enable basic file protection after export.

        NOTE: Auto-protection is disabled to prevent permission issues during development.
        Students who want protection can run 'tito protect --enable' manually.
        """
        # Disabled - causes permission errors on subsequent exports
        # Students can manually enable protection with 'tito protect --enable'
        pass