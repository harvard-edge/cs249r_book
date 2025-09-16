"""
Build command implementation for MLSysBook CLI.

Handles building chapters and full books in different formats (HTML, PDF, EPUB).
"""

import subprocess
import signal
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class BuildCommand:
    """Handles build operations for the MLSysBook."""
    
    def __init__(self, config_manager, chapter_discovery):
        """Initialize build command.
        
        Args:
            config_manager: ConfigManager instance
            chapter_discovery: ChapterDiscovery instance
        """
        self.config_manager = config_manager
        self.chapter_discovery = chapter_discovery
        
    def build_full(self, format_type: str = "html") -> bool:
        """Build full book in specified format.
        
        Args:
            format_type: Format to build ('html', 'pdf', 'epub')
            
        Returns:
            True if build succeeded, False otherwise
        """
        console.print(f"[green]🔨 Building full {format_type.upper()} book...[/green]")
        
        # Handle special case for building both HTML and PDF
        if format_type == "both":
            return self._build_both_formats()
            
        # Create build directory
        output_dir = self.config_manager.get_output_dir(format_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup config
        config_name = self.config_manager.setup_symlink(format_type)
        
        # Determine render target
        render_targets = {
            "html": "html",
            "pdf": "titlepage-pdf", 
            "epub": "epub"
        }
        
        if format_type not in render_targets:
            raise ValueError(f"Unknown format type: {format_type}")
            
        render_to = render_targets[format_type]
        render_cmd = ["quarto", "render", "--to", render_to]
        
        # Show the command being executed
        cmd_str = " ".join(render_cmd)
        console.print(f"[blue]💻 Command: {cmd_str}[/blue]")
        
        # Execute build
        success = self._run_command(
            render_cmd,
            cwd=self.config_manager.book_dir,
            description=f"Building full {format_type.upper()} book"
        )
        
        if success:
            console.print(f"[green]✅ {format_type.upper()} build completed: {output_dir}/[/green]")
        else:
            console.print(f"[red]❌ {format_type.upper()} build failed[/red]")
            
        return success
    
    def build_chapters(self, chapter_names: List[str], format_type: str = "html") -> bool:
        """Build specific chapters.
        
        Args:
            chapter_names: List of chapter names to build
            format_type: Format to build ('html', 'pdf', 'epub')
            
        Returns:
            True if build succeeded, False otherwise
        """
        console.print(f"[green]🚀 Building {len(chapter_names)} chapters[/green] [dim]({format_type})[/dim]")
        console.print(f"[dim]📋 Chapters: {', '.join(chapter_names)}[/dim]")
        
        try:
            # Validate chapters exist
            chapter_files = self.chapter_discovery.validate_chapters(chapter_names)
            
            # Setup configuration
            config_file = self.config_manager.get_config_file(format_type)
            format_args = {
                "html": "html",
                "pdf": "titlepage-pdf",
                "epub": "epub"
            }
            
            if format_type not in format_args:
                raise ValueError(f"Unknown format type: {format_type}")
                
            format_arg = format_args[format_type]
            
            # Create build directory
            output_dir = self.config_manager.get_output_dir(format_type)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup correct configuration symlink
            self.config_manager.setup_symlink(format_type)
            
            # Set up fast build mode for the target chapters
            self._setup_fast_build_mode(config_file, chapter_files)
            
            # Track if config has been restored to avoid double restoration
            self._config_restored = False
            
            # Setup signal handler to restore config on Ctrl+C
            def signal_handler(signum, frame):
                if not self._config_restored:
                    console.print("\n[yellow]🛡️ Ctrl+C detected - restoring config...[/yellow]")
                    self._restore_config(config_file)
                    self._config_restored = True
                    console.print("[green]✅ Config restored[/green]")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Build with project.render configuration
            console.print("[yellow]🔨 Building with fast build configuration...[/yellow]")
            
            render_cmd = ["quarto", "render", "--to", format_arg]
            cmd_str = " ".join(render_cmd)
            console.print(f"[blue]💻 Command: {cmd_str}[/blue]")
            
            # Execute build
            success = self._run_command(
                render_cmd,
                cwd=self.config_manager.book_dir,
                description=f"Building {len(chapter_names)} chapters ({format_type})"
            )
            
            if success:
                console.print(f"[green]✅ Build complete: {output_dir}/[/green]")
            else:
                console.print("[red]❌ Build failed[/red]")
                
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Build error: {e}[/red]")
            return False
        finally:
            # Always restore config (unless already restored by signal handler)
            try:
                if not self._config_restored:
                    self._restore_config(config_file)
            except:
                pass
    
    def _build_both_formats(self) -> bool:
        """Build both HTML and PDF formats sequentially."""
        console.print("[blue]📚 Building both HTML and PDF formats...[/blue]")
        
        # Build HTML first
        console.print("[blue]📄 Building HTML version...[/blue]")
        html_success = self.build_full("html")
        if not html_success:
            console.print("[red]❌ HTML build failed![/red]")
            return False
        
        # Build PDF
        console.print("[blue]📄 Building PDF version...[/blue]")
        pdf_success = self.build_full("pdf")
        if not pdf_success:
            console.print("[red]❌ PDF build failed![/red]")
            return False
        
        console.print("[green]✅ Both HTML and PDF builds completed successfully![/green]")
        return True
    
    def _run_command(self, cmd: List[str], cwd: Path, description: str) -> bool:
        """Run a command with progress indication.
        
        Args:
            cmd: Command to run
            cwd: Working directory
            description: Description for progress display
            
        Returns:
            True if command succeeded, False otherwise
        """
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=False
            ) as progress:
                task = progress.add_task(description, total=None)
                
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minute timeout
                )
                
                progress.update(task, completed=True)
                
            if result.returncode == 0:
                return True
            else:
                console.print(f"[red]Command failed with exit code {result.returncode}[/red]")
                if result.stderr:
                    console.print(f"[red]Error: {result.stderr}[/red]")
                return False
                
        except subprocess.TimeoutExpired:
            console.print("[red]❌ Build timed out after 30 minutes[/red]")
            return False
        except Exception as e:
            console.print(f"[red]❌ Command execution error: {e}[/red]")
            return False
    
    def build_html_only(self, chapter_names: List[str] = None) -> bool:
        """Build HTML-only version with index.qmd and specific files of interest.
        
        Args:
            chapter_names: List of chapter names to include (optional)
            
        Returns:
            True if build succeeded, False otherwise
        """
        console.print("[green]🌐 Building HTML-only version...[/green]")
        
        try:
            # Always include index.qmd
            files_to_render = ["index.qmd"]
            
            # Add specified chapters if provided
            if chapter_names:
                console.print(f"[dim]📋 Including chapters: {', '.join(chapter_names)}[/dim]")
                chapter_files = self.chapter_discovery.validate_chapters(chapter_names)
                
                # Convert to relative paths from book directory
                for chapter_file in chapter_files:
                    rel_path = chapter_file.relative_to(self.config_manager.book_dir)
                    files_to_render.append(str(rel_path))
            else:
                console.print("[dim]📋 Building index.qmd only[/dim]")
            
            # Create temporary config for HTML-only build
            config_file = self.config_manager.get_config_file("html")
            temp_config_file = self._create_html_only_config(config_file, files_to_render)
            
            # Setup symlink to temporary config
            if self.config_manager.active_config.exists() or self.config_manager.active_config.is_symlink():
                self.config_manager.active_config.unlink()
            
            relative_temp_config = temp_config_file.relative_to(self.config_manager.book_dir)
            self.config_manager.active_config.symlink_to(relative_temp_config)
            
            # Build HTML
            render_cmd = ["quarto", "render", "--to", "html"]
            cmd_str = " ".join(render_cmd)
            console.print(f"[blue]💻 Command: {cmd_str}[/blue]")
            
            success = self._run_command(
                render_cmd,
                cwd=self.config_manager.book_dir,
                description="Building HTML-only version"
            )
            
            if success:
                output_dir = self.config_manager.get_output_dir("html")
                console.print(f"[green]✅ HTML-only build completed: {output_dir}/[/green]")
            else:
                console.print("[red]❌ HTML-only build failed[/red]")
                
            return success
            
        except Exception as e:
            console.print(f"[red]❌ HTML-only build error: {e}[/red]")
            return False
        finally:
            # Always restore original config
            try:
                self._restore_html_config()
                # Clean up temporary config file
                if 'temp_config_file' in locals() and temp_config_file.exists():
                    temp_config_file.unlink()
            except:
                pass
    
    def _create_html_only_config(self, base_config_file: Path, files_to_render: List[str]) -> Path:
        """Create a temporary Quarto config for HTML-only builds.
        
        Args:
            base_config_file: Base HTML configuration file
            files_to_render: List of files to include in the build
            
        Returns:
            Path to the temporary configuration file
        """
        import yaml
        
        # Read the base configuration
        with open(base_config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Add project.render configuration to limit files
        if 'project' not in config:
            config['project'] = {}
        
        config['project']['render'] = files_to_render
        
        # Create temporary config file
        temp_config_file = self.config_manager.book_dir / "_quarto_html_only.yml"
        
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        console.print(f"[dim]⚡ Created HTML-only config with {len(files_to_render)} files[/dim]")
        return temp_config_file
    
    def _restore_html_config(self) -> None:
        """Restore the original HTML configuration."""
        try:
            # Remove current symlink
            if self.config_manager.active_config.exists() or self.config_manager.active_config.is_symlink():
                self.config_manager.active_config.unlink()
            
            # Restore HTML config symlink
            self.config_manager.setup_symlink("html")
            console.print("[dim]🛡️ Restored original HTML config[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Error restoring config: {e}[/yellow]")

    def _setup_fast_build_mode(self, config_file: Path, chapter_files: List[Path]) -> None:
        """Setup fast build mode by modifying config for selective chapter builds.
        
        For HTML: Uses render field to specify which files to build
        For PDF/EPUB: Comments out chapters not being built
        """
        console.print("[dim]⚡ Setting up fast build mode...[/dim]")
        
        # Create backup of original config
        backup_file = config_file.with_suffix('.backup')
        if backup_file.exists():
            backup_file.unlink()
        
        # Read original config
        with open(config_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Save backup
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Determine if this is HTML or PDF/EPUB config
        is_html = 'html' in str(config_file).lower()
        
        if is_html:
            # For HTML, use render field approach
            self._setup_html_fast_build(config_file, chapter_files, original_content)
        else:
            # For PDF/EPUB, use commenting approach
            self._setup_pdf_epub_fast_build(config_file, chapter_files, original_content)
    
    def _setup_html_fast_build(self, config_file: Path, chapter_files: List[Path], original_content: str) -> None:
        """Setup HTML fast build using render field."""
        # Build list of files to render
        files_to_render = ["index.qmd", "contents/backmatter/glossary/glossary.qmd"]
        
        for chapter_file in chapter_files:
            try:
                rel_path = chapter_file.relative_to(self.config_manager.book_dir)
                files_to_render.append(str(rel_path))
            except ValueError:
                # Try to construct the path
                chapter_name = chapter_file.stem
                files_to_render.append(f"contents/core/{chapter_name}/{chapter_name}.qmd")
        
        console.print(f"[dim]📋 Files to render: {len(files_to_render)} files[/dim]")
        
        # Process config to update/add render field
        lines = original_content.split('\n')
        modified_lines = []
        i = 0
        render_added = False
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this is the render: section
            if line.strip().startswith('render:'):
                # Skip the entire existing render section
                while i < len(lines) and (lines[i].strip().startswith('render:') or 
                                         lines[i].strip().startswith('-') or
                                         lines[i].strip().startswith('#') or
                                         (lines[i].startswith('  ') and lines[i].strip())):
                    i += 1
                
                # Add our new render section
                modified_lines.append('  render:')
                for file in files_to_render:
                    modified_lines.append(f'    - {file}')
                render_added = True
                continue
            
            # If we hit post-render and haven't added render yet, add it before
            if not render_added and line.strip().startswith('post-render:'):
                modified_lines.append('  render:')
                for file in files_to_render:
                    modified_lines.append(f'    - {file}')
                modified_lines.append('')
                render_added = True
            
            modified_lines.append(line)
            i += 1
        
        # Write modified config
        modified_content = '\n'.join(modified_lines)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        console.print("[green]✓[/green] Fast build mode configured (HTML)")
    
    def _setup_pdf_epub_fast_build(self, config_file: Path, chapter_files: List[Path], original_content: str) -> None:
        """Setup PDF/EPUB fast build by commenting out chapters not being built.
        
        Note: render field doesn't work for PDF/EPUB. We preserve the structure
        but comment out files not in the selected list.
        """
        # Get list of chapter names to keep
        keep_chapters = set(['index'])  # Always keep index.qmd
        always_include = {'index.qmd', 'glossary.qmd', 'references.qmd'}  # Always include these
        
        for chapter_file in chapter_files:
            keep_chapters.add(chapter_file.stem)
        
        # Track what we're building
        files_being_built = []
        
        # Process config - comment out chapters not being built
        lines = original_content.split('\n')
        modified_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this is a part declaration
            if stripped.startswith('- part:') or (stripped.startswith('part:') and not '.qmd' in line):
                # This is a part - look ahead to see if any chapters in this part should be included
                part_has_active_chapters = False
                part_lines = [line]  # Start with the part line
                j = i + 1
                
                # Collect all lines that belong to this part
                while j < len(lines):
                    next_line = lines[j]
                    next_stripped = next_line.strip()
                    
                    # Stop if we hit another part or a non-indented line that indicates end of part
                    if ((next_stripped.startswith('- part:') or 
                         (next_stripped.startswith('part:') and not '.qmd' in next_line)) or
                        (next_line and not next_line[0].isspace() and not next_line.startswith('\t') and 
                         not next_stripped.startswith('#'))):
                        break
                    
                    part_lines.append(next_line)
                    
                    # Check if this line has a chapter we want to include
                    if '.qmd' in next_line:
                        for chapter_name in keep_chapters:
                            if f'{chapter_name}/{chapter_name}.qmd' in next_line or f'{chapter_name}.qmd' in next_line:
                                part_has_active_chapters = True
                                break
                        # Also check always_include
                        for always_file in always_include:
                            if always_file in next_line:
                                part_has_active_chapters = True
                                break
                    
                    j += 1
                
                # Process all lines in this part
                for part_line in part_lines:
                    part_stripped = part_line.strip()
                    
                    if '.qmd' in part_line:
                        # This is a chapter file - check if it should be included
                        should_include = False
                        
                        # Check against always_include files
                        for always_file in always_include:
                            if always_file in part_line:
                                should_include = True
                                break
                        
                        # Check against selected chapters
                        if not should_include:
                            for chapter_name in keep_chapters:
                                if f'{chapter_name}/{chapter_name}.qmd' in part_line or f'{chapter_name}.qmd' in part_line:
                                    should_include = True
                                    break
                        
                        if should_include:
                            # Ensure line is not commented
                            if part_stripped.startswith('#'):
                                uncommented = part_line.replace('# ', '', 1).replace('#', '', 1)
                                modified_lines.append(uncommented)
                                files_being_built.append(part_stripped[2:] if part_stripped.startswith('# ') else part_stripped[1:])
                            else:
                                modified_lines.append(part_line)
                                files_being_built.append(part_stripped[2:] if part_stripped.startswith('- ') else part_stripped)
                        else:
                            # Comment out this chapter
                            if not part_stripped.startswith('#'):
                                indent = len(part_line) - len(part_line.lstrip())
                                commented = ' ' * indent + '# ' + part_line.lstrip()
                                modified_lines.append(commented)
                            else:
                                modified_lines.append(part_line)
                    elif part_has_active_chapters:
                        # This part has active chapters, so keep structural lines as-is
                        modified_lines.append(part_line)
                    else:
                        # This part has no active chapters, comment out all lines in it
                        if not part_stripped.startswith('#') and part_stripped:
                            indent = len(part_line) - len(part_line.lstrip())
                            commented = ' ' * indent + '# ' + part_line.lstrip()
                            modified_lines.append(commented)
                        else:
                            modified_lines.append(part_line)
                
                # Skip ahead since we've processed this whole part
                i = j - 1
                
            elif '.qmd' in line:
                # This is a standalone .qmd file (not in a part)
                should_include = False
                
                # Check against always_include files
                for always_file in always_include:
                    if always_file in line:
                        should_include = True
                        break
                
                # Check against selected chapters
                if not should_include:
                    for chapter_name in keep_chapters:
                        if f'{chapter_name}/{chapter_name}.qmd' in line or f'{chapter_name}.qmd' in line:
                            should_include = True
                            break
                
                if should_include:
                    # Ensure line is not commented
                    if stripped.startswith('#'):
                        uncommented = line.replace('# ', '', 1).replace('#', '', 1)
                        modified_lines.append(uncommented)
                        files_being_built.append(stripped[2:] if stripped.startswith('# ') else stripped)
                    else:
                        modified_lines.append(line)
                        files_being_built.append(stripped)
                else:
                    # Comment out the line
                    if not stripped.startswith('#'):
                        indent = len(line) - len(line.lstrip())
                        commented = ' ' * indent + '# ' + line.lstrip()
                        modified_lines.append(commented)
                    else:
                        modified_lines.append(line)
            else:
                # All other lines - copy as-is
                modified_lines.append(line)
            
            i += 1
        
        # Write modified config
        modified_content = '\n'.join(modified_lines)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        console.print(f"[dim]📋 Files to build: {len(files_being_built)} files[/dim]")
        for file in files_being_built:
            console.print(f"[green]✓[/green] {file}")
        
        console.print("[green]✓[/green] Fast build mode configured (PDF/EPUB)")
    
    def _restore_config(self, config_file: Path) -> None:
        """Restore configuration to pristine state."""
        console.print("[dim]🛡️ Restoring config...[/dim]")
        
        backup_file = config_file.with_suffix('.backup')
        
        if backup_file.exists():
            try:
                # Read backup content
                with open(backup_file, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                
                # Restore original config
                with open(config_file, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                
                # Clean up backup file
                backup_file.unlink()
                
                console.print("[green]✅ Configuration restored successfully[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error restoring config: {e}[/red]")
        else:
            console.print("[yellow]⚠️ No backup file found - config may already be restored[/yellow]")
