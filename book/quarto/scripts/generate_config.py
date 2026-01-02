#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quarto Configuration Generator
==============================
Generates Quarto configuration files by merging shared/common configurations
with format-specific overrides. This eliminates duplication and ensures
consistency across HTML, PDF, and EPUB output formats.

Usage:
    python scripts/generate_config.py [format]
    
    format: html, pdf, epub (default: all formats)

Examples:
    python scripts/generate_config.py        # Generate all formats
    python scripts/generate_config.py html   # Generate only HTML config
    python scripts/generate_config.py pdf    # Generate only PDF config
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ConfigGenerator:
    """Generates Quarto configuration files from shared and format-specific configs."""

    def __init__(self, project_root: Path):
        """Initialize the config generator."""
        self.project_root = project_root
        self.config_dir = project_root / "config"
        self.shared_dir = self.config_dir / "shared"
        self.output_dir = self.config_dir

    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML file and return its contents as a dictionary."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}", file=sys.stderr)
            sys.exit(1)

    def deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override values taking precedence.
        
        For nested dictionaries, recursively merge. For lists, replace entirely.
        For other values, override replaces base.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def load_shared_configs(self) -> Dict[str, Any]:
        """Load all shared configuration files and merge them."""
        shared_config = {}
        
        # List of shared config files to load (in order)
        shared_files = [
            "book-metadata.yml",
            "chapters.yml",
            "bibliography.yml",
            "crossref.yml",
            "diagram.yml",
            "filter-metadata.yml",
        ]
        
        for filename in shared_files:
            file_path = self.shared_dir / filename
            if file_path.exists():
                config = self.load_yaml(file_path)
                shared_config = self.deep_merge(shared_config, config)
            else:
                print(f"Warning: Shared config file not found: {file_path}", file=sys.stderr)
        
        return shared_config

    def load_format_overrides(self, format_name: str) -> Dict[str, Any]:
        """Load format-specific override configuration."""
        override_file = self.config_dir / f"_quarto-{format_name}-overrides.yml"
        
        if override_file.exists():
            return self.load_yaml(override_file)
        else:
            # Return empty dict if no override file exists
            return {}

    def apply_format_specifics(self, config: Dict[str, Any], format_name: str) -> Dict[str, Any]:
        """Apply format-specific customizations to the merged configuration."""
        config = config.copy()
        
        if format_name == "html":
            # HTML-specific customizations
            if "filter-metadata" in config:
                fm = config["filter-metadata"]
                if "quiz-config" in fm:
                    fm["quiz-config"]["auto-discover-pdf"] = False
                if "mlsysbook-ext/custom-numbered-blocks" in fm:
                    fm["mlsysbook-ext/custom-numbered-blocks"]["icon-format"] = "png"
                    # Set collapse for quiz-question
                    if "groups" in fm["mlsysbook-ext/custom-numbered-blocks"]:
                        groups = fm["mlsysbook-ext/custom-numbered-blocks"]["groups"]
                        if "quiz-question" in groups:
                            groups["quiz-question"]["collapse"] = True
                        if "quiz-answer" in groups:
                            groups["quiz-answer"]["collapse"] = True
            
            # HTML uses SVG output for TikZ
            if "diagram" in config and "engine" in config["diagram"]:
                if "tikz" in config["diagram"]["engine"]:
                    config["diagram"]["engine"]["tikz"]["output-format"] = "svg"
            
            # HTML cross-references file
            if "filter-metadata" in config and "cross-references" in config["filter-metadata"]:
                config["filter-metadata"]["cross-references"]["file"] = "data/cross_refs_final.json"
        
        elif format_name == "pdf":
            # PDF-specific customizations
            if "filter-metadata" in config:
                fm = config["filter-metadata"]
                if "quiz-config" in fm:
                    fm["quiz-config"]["auto-discover-pdf"] = True
                if "mlsysbook-ext/custom-numbered-blocks" in fm:
                    fm["mlsysbook-ext/custom-numbered-blocks"]["icon-format"] = "pdf"
                    # Set collapse for quiz-question
                    if "groups" in fm["mlsysbook-ext/custom-numbered-blocks"]:
                        groups = fm["mlsysbook-ext/custom-numbered-blocks"]["groups"]
                        if "quiz-question" in groups:
                            groups["quiz-question"]["collapse"] = False
                        if "quiz-answer" in groups:
                            groups["quiz-answer"]["collapse"] = True
            
            # PDF cross-references file
            if "filter-metadata" in config and "cross-references" in config["filter-metadata"]:
                config["filter-metadata"]["cross-references"]["file"] = "data/cross_refs_refined.json"
        
        elif format_name == "epub":
            # EPUB-specific customizations
            if "filter-metadata" in config:
                fm = config["filter-metadata"]
                if "quiz-config" in fm:
                    fm["quiz-config"]["auto-discover-pdf"] = False
                if "mlsysbook-ext/custom-numbered-blocks" in fm:
                    fm["mlsysbook-ext/custom-numbered-blocks"]["icon-format"] = "png"
                    # Set collapse for quiz-question
                    if "groups" in fm["mlsysbook-ext/custom-numbered-blocks"]:
                        groups = fm["mlsysbook-ext/custom-numbered-blocks"]["groups"]
                        if "quiz-question" in groups:
                            groups["quiz-question"]["collapse"] = True
                        if "quiz-answer" in groups:
                            groups["quiz-answer"]["collapse"] = True
            
            # EPUB cross-references file
            if "filter-metadata" in config and "cross-references" in config["filter-metadata"]:
                config["filter-metadata"]["cross-references"]["file"] = "data/cross_refs_final.json"
        
        return config

    def generate_config(self, format_name: str) -> Dict[str, Any]:
        """Generate a complete configuration for a specific format."""
        # Load shared configurations
        shared_config = self.load_shared_configs()
        
        # Load format-specific overrides
        format_overrides = self.load_format_overrides(format_name)
        
        # Merge shared config with format overrides
        merged_config = self.deep_merge(shared_config, format_overrides)
        
        # Apply format-specific customizations
        final_config = self.apply_format_specifics(merged_config, format_name)
        
        return final_config

    def write_config(self, config: Dict[str, Any], format_name: str) -> None:
        """Write the generated configuration to a file."""
        output_file = self.output_dir / f"_quarto-{format_name}.yml"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write(f"# =============================================================================\n")
                f.write(f"# {format_name.upper()} CONFIGURATION (AUTO-GENERATED)\n")
                f.write(f"# =============================================================================\n")
                f.write(f"# This file is automatically generated from shared configurations.\n")
                f.write(f"# DO NOT EDIT THIS FILE DIRECTLY.\n")
                f.write(f"# \n")
                f.write(f"# To modify this configuration:\n")
                f.write(f"# 1. Edit files in config/shared/ for shared settings\n")
                f.write(f"# 2. Edit config/_quarto-{format_name}-overrides.yml for format-specific overrides\n")
                f.write(f"# 3. Run: python scripts/generate_config.py {format_name}\n")
                f.write(f"# =============================================================================\n\n")
                
                # Write YAML content
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, 
                         allow_unicode=True, width=120, indent=2)
            
            print(f"✓ Generated: {output_file}")
        
        except Exception as e:
            print(f"Error writing configuration file {output_file}: {e}", file=sys.stderr)
            sys.exit(1)

    def generate_all(self, formats: Optional[List[str]] = None) -> None:
        """Generate configurations for all specified formats."""
        if formats is None:
            formats = ["html", "pdf", "epub"]
        
        for format_name in formats:
            print(f"Generating {format_name.upper()} configuration...")
            config = self.generate_config(format_name)
            self.write_config(config, format_name)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Quarto configuration files from shared configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'format',
        nargs='?',
        choices=['html', 'pdf', 'epub', 'all'],
        default='all',
        help='Format to generate (default: all)'
    )
    
    args = parser.parse_args()
    
    # Determine project root (assuming script is in book/quarto/scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # book/quarto/
    
    # Create generator
    generator = ConfigGenerator(project_root)
    
    # Generate configurations
    if args.format == 'all':
        generator.generate_all()
    else:
        generator.generate_all([args.format])
    
    print("\n✓ Configuration generation complete!")


if __name__ == '__main__':
    main()

