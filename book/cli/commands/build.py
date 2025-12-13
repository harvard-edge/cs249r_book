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
        console.print("[dim]📄 Building all files (full book mode)[/dim]")

        # Handle special case for building both HTML and PDF
        if format_type == "both":
            return self._build_both_formats()

        # Create build directory
        output_dir = self.config_manager.get_output_dir(format_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup config
        config_name = self.config_manager.setup_symlink(format_type)

        # Get config file
        config_file = self.config_manager.get_config_file(format_type)

        # Uncomment all files for full build (PDF/EPUB only)
        if format_type in ["pdf", "epub"]:
            console.print("[yellow]📝 Uncommenting all chapter files for full book build...[/yellow]")
            self._uncomment_all_chapters(config_file)

        # Track if config has been restored to avoid double restoration
        self._config_restored = False

        # Setup signal handler to restore config on Ctrl+C
        def signal_handler(signum, frame):
            if not self._config_restored and format_type in ["pdf", "epub"]:
                console.print("\n[yellow]🛡️ Ctrl+C detected - restoring config...[/yellow]")
                self._restore_config(config_file)
                self._config_restored = True
                console.print("[green]✅ Config restored[/green]")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Determine render target
            render_targets = {
                "html": "html",
                "pdf": "titlepage-pdf",
                "epub": "epub"
            }

            if format_type not in render_targets:
                raise ValueError(f"Unknown format type: {format_type}")

            render_to = render_targets[format_type]
            render_cmd = ["quarto", "render", f"--to={render_to}"]

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
        finally:
            # Always restore config for PDF/EPUB builds (unless already restored by signal handler)
            if format_type in ["pdf", "epub"] and not self._config_restored:
                self._restore_config(config_file)

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

            # Show files that will be built
            console.print("[dim]📄 Files to be rendered:[/dim]")
            console.print(f"[dim]  • index.qmd[/dim]")
            for chapter_file in chapter_files:
                rel_path = chapter_file.relative_to(self.config_manager.book_dir)
                console.print(f"[dim]  • {rel_path}[/dim]")

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

            render_cmd = ["quarto", "render", f"--to={format_arg}"]
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
            chapter_names: List of chapter names to include (optional, if None builds all)

        Returns:
            True if build succeeded, False otherwise
        """
        console.print("[green]🌐 Building HTML-only version...[/green]")

        try:
            # Always include index.qmd
            files_to_render = ["index.qmd"]

            # Add specified chapters if provided, otherwise add ALL chapters
            if chapter_names:
                console.print(f"[dim]📋 Including chapters: {', '.join(chapter_names)}[/dim]")
                chapter_files = self.chapter_discovery.validate_chapters(chapter_names)

                # Convert to relative paths from book directory
                for chapter_file in chapter_files:
                    rel_path = chapter_file.relative_to(self.config_manager.book_dir)
                    files_to_render.append(str(rel_path))
            else:
                console.print("[yellow]📝 Adding ALL available chapters to render list...[/yellow]")
                # Get all available chapters
                all_chapters = self.chapter_discovery.get_all_chapters()
                console.print(f"[dim]📋 Found {len(all_chapters)} chapters[/dim]")

                # Add all chapter files to render list
                for chapter_name, chapter_file in all_chapters.items():
                    try:
                        rel_path = chapter_file.relative_to(self.config_manager.book_dir)
                        files_to_render.append(str(rel_path))
                    except ValueError:
                        # If relative path fails, try to construct it
                        files_to_render.append(f"contents/core/{chapter_name}/{chapter_name}.qmd")

            # Show files that will be built
            console.print("[dim]📄 Files to be rendered:[/dim]")
            for file_path in files_to_render:
                console.print(f"[dim]  • {file_path}[/dim]")

            # Use surgical approach - modify existing config file directly
            config_file = self.config_manager.get_config_file("html")
            self._add_render_section(config_file, files_to_render)

            # Ensure symlink points to the HTML config
            self.config_manager.setup_symlink("html")

            # Build HTML
            render_cmd = ["quarto", "render", "--to=html"]
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
            # Always remove render section from config
            try:
                config_file = self.config_manager.get_config_file("html")
                self._remove_render_section(config_file)
            except:
                pass

    def _add_render_section(self, config_file: Path, files_to_render: List[str]) -> None:
        """Add render section to existing config file.

        Args:
            config_file: Path to config file to modify
            files_to_render: List of files to include in render section
        """
        # Read current config
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        modified_lines = []
        render_added = False

        for i, line in enumerate(lines):
            # If we find post-render and haven't added render yet, add it before
            if not render_added and line.strip().startswith('post-render:'):
                modified_lines.append('  render:')
                for file in files_to_render:
                    modified_lines.append(f'    - {file}')
                modified_lines.append('')
                render_added = True

            modified_lines.append(line)

        # Write modified config
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_lines))

        console.print(f"[dim]⚡ Added render section with {len(files_to_render)} files[/dim]")

    def _remove_render_section(self, config_file: Path) -> None:
        """Remove render section from config file.

        Args:
            config_file: Path to config file to modify
        """
        try:
            # Read current config
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            modified_lines = []
            i = 0

            while i < len(lines):
                line = lines[i]

                # Skip render section entirely
                if line.strip().startswith('render:'):
                    # Skip this line and all indented lines that follow
                    i += 1
                    while i < len(lines) and (lines[i].startswith('    -') or lines[i].strip() == ''):
                        i += 1
                    continue

                modified_lines.append(line)
                i += 1

            # Write modified config
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(modified_lines))

            console.print("[dim]🛡️ Removed render section[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Error removing render section: {e}[/yellow]")

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

        # Determine format and call appropriate setup function
        config_name = str(config_file).lower()

        if 'html' in config_name:
            self._setup_html_fast_build(config_file, chapter_files, original_content)
        elif 'pdf' in config_name:
            self._setup_pdf_fast_build(config_file, chapter_files, original_content)
        elif 'epub' in config_name:
            self._setup_epub_fast_build(config_file, chapter_files, original_content)
        else:
            # Fallback to PDF/EPUB approach for unknown formats
            console.print(f"[yellow]⚠️ Unknown config format, using PDF approach: {config_file}[/yellow]")
            self._setup_pdf_fast_build(config_file, chapter_files, original_content)

    def _setup_html_fast_build(self, config_file: Path, chapter_files: List[Path], original_content: str) -> None:
        """Setup HTML fast build using render field."""
        # Build list of files to render
        files_to_render = ["index.qmd"]

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

    def _setup_pdf_fast_build(self, config_file: Path, chapter_files: List[Path], original_content: str) -> None:
        """Setup PDF fast build by commenting out chapters not being built.

        Note: render field doesn't work for PDF. We preserve the structure
        but comment out files not in the selected list.
        """
        # Get list of chapter names to keep
        keep_chapters = set(['index'])  # Always keep index.qmd
        always_include = {'index.qmd'}  # Only include index.qmd for selective builds

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

                    # Check structural lines FIRST (before .qmd check) to avoid treating part declarations as chapters
                    if part_has_active_chapters and ('part:' in part_line or part_stripped.startswith('chapters:')):
                        # This part has active chapters, so ensure structural lines are uncommented
                        # Always ensure part and chapters lines are uncommented when part has active chapters
                        if part_stripped.startswith('#'):
                            uncommented = part_line.replace('# ', '', 1).replace('#', '', 1)
                            modified_lines.append(uncommented)
                        else:
                            modified_lines.append(part_line)
                    elif '.qmd' in part_line:
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
                        # Part has active chapters but this line is neither structural nor a chapter
                        # Keep as-is
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

        console.print("[green]✓[/green] Fast build mode configured (PDF)")

    def _setup_epub_fast_build(self, config_file: Path, chapter_files: List[Path], original_content: str) -> None:
        """Setup EPUB fast build by commenting out chapters not being built.

        EPUB has specific requirements:
        - Must preserve part structure (parts cannot be commented out if they contain active chapters)
        - Must uncomment both part and chapters lines when building chapters in that part
        - Uses same commenting approach as PDF but with stricter part preservation
        """
        # Get list of chapter names to keep
        keep_chapters = set(['index'])  # Always keep index.qmd
        always_include = {'index.qmd'}  # Only include index.qmd for selective builds

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

                    # Check structural lines FIRST (before .qmd check) to avoid treating part declarations as chapters
                    if part_has_active_chapters and ('part:' in part_line or part_stripped.startswith('chapters:')):
                        # EPUB CRITICAL: This part has active chapters, so ensure structural lines are uncommented
                        # Always ensure part and chapters lines are uncommented when part has active chapters
                        if part_stripped.startswith('#'):
                            uncommented = part_line.replace('# ', '', 1).replace('#', '', 1)
                            modified_lines.append(uncommented)
                        else:
                            modified_lines.append(part_line)
                    elif '.qmd' in part_line:
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
                        # Part has active chapters but this line is neither structural nor a chapter
                        # Keep as-is
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

        console.print("[green]✓[/green] Fast build mode configured (EPUB)")

    def _uncomment_all_chapters(self, config_file: Path) -> None:
        """Uncomment all chapter files in the config for full book build.

        Args:
            config_file: Path to config file to modify
        """
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

        # Process config - uncomment all lines with .qmd files
        lines = original_content.split('\n')
        modified_lines = []
        uncommented_count = 0

        for line in lines:
            stripped = line.strip()

            # Check if this is a commented line with a .qmd file
            if stripped.startswith('#') and '.qmd' in line:
                # Uncomment the line while preserving indentation
                # Handle both "# - " and "#- " patterns
                if '# -' in line:
                    uncommented = line.replace('# -', '-', 1)
                elif '#-' in line:
                    uncommented = line.replace('#-', '-', 1)
                else:
                    # Just remove the first # and space
                    uncommented = line.replace('# ', '', 1).replace('#', '', 1)

                modified_lines.append(uncommented)
                uncommented_count += 1
            else:
                # Keep line as-is
                modified_lines.append(line)

        # Write modified config
        modified_content = '\n'.join(modified_lines)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        console.print(f"[green]✓[/green] Uncommented {uncommented_count} chapter files")

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
