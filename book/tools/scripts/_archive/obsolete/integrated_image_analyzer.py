#!/usr/bin/env python3
"""
Integrated Image Analyzer for MLSysBook
Simple, size-based image analysis and compression recommendations.
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import shutil
import glob

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich import print as rprint

console = Console()

# Simple size-based guidelines
SIZE_GUIDELINES = {
    'large': {
        'threshold': 1.0,  # MB
        'recommendation': 'Compress - too large for textbook',
        'priority': 'high'
    },
    'medium': {
        'threshold': 0.5,  # MB
        'recommendation': 'Consider compression for better performance',
        'priority': 'medium'
    },
    'small': {
        'threshold': 0.0,  # MB
        'recommendation': 'Size is acceptable',
        'priority': 'low'
    }
}

# Valid extensions
VALID_EXTENSIONS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.gif': 'GIF',
    '.svg': 'SVG',
    '.webp': 'WEBP',
}

class IntegratedImageAnalyzer:
    def __init__(self, repo_path: str = None):
        self.repo_path = Path(repo_path) if repo_path else Path.cwd()
        self.console = Console()

    def run_git_command(self, cmd: List[str], capture_output: bool = True) -> Tuple[bool, str]:
        """Run a git command and return success status and output"""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                check=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr if e.stderr else str(e)

    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        return os.path.getsize(file_path) / (1024 * 1024)

    def get_size_category(self, size_mb: float) -> str:
        """Get size category based on MB."""
        if size_mb > SIZE_GUIDELINES['large']['threshold']:
            return 'large'
        elif size_mb > SIZE_GUIDELINES['medium']['threshold']:
            return 'medium'
        else:
            return 'small'

    def validate_image_format(self, file_path: str) -> Tuple[bool, str]:
        """Validate image format"""
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                actual_format = img.format.upper()
                ext = os.path.splitext(file_path)[1].lower()
                expected_format = VALID_EXTENSIONS.get(ext)
                if expected_format and actual_format == expected_format:
                    return True, actual_format
                else:
                    return False, f"Expected {expected_format}, got {actual_format}"
        except Exception as e:
            return False, f"Unreadable: {e}"

    def find_all_images(self, directory: str = 'book/contents') -> List[str]:
        """Find all image files in the textbook."""
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.svg']
        image_files = []

        for ext in image_extensions:
            pattern = os.path.join(directory, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))

        return sorted(image_files)

    def analyze_images(self, directory: str = 'book/contents',
                      validate_formats: bool = True,
                      show_recommendations: bool = True) -> Dict:
        """Simple image analysis based on size only."""
        console.print('🔍 Analyzing textbook images...\n')

        image_files = self.find_all_images(directory)

        if not image_files:
            console.print('❌ No image files found')
            return {}

        console.print(f'📸 Found {len(image_files)} images\n')

        # Analysis results
        analysis_results = {
            'large': [],
            'medium': [],
            'small': [],
            'total_size': 0,
            'recommendations': [],
            'validation_errors': []
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing images...", total=len(image_files))

            for image_path in image_files:
                progress.update(task, description=f"Processing {os.path.basename(image_path)}")

                # Get basic info
                size_mb = self.get_file_size_mb(image_path)
                analysis_results['total_size'] += size_mb

                filename = os.path.basename(image_path)
                size_category = self.get_size_category(size_mb)

                # Validate format if requested
                is_valid = True
                validation_msg = ""
                if validate_formats:
                    is_valid, validation_msg = self.validate_image_format(image_path)
                    if not is_valid:
                        analysis_results['validation_errors'].append({
                            'path': image_path,
                            'error': validation_msg
                        })

                result = {
                    'path': image_path,
                    'filename': filename,
                    'size_mb': size_mb,
                    'size_category': size_category,
                    'is_valid': is_valid,
                    'validation_msg': validation_msg,
                    'needs_compression': size_mb > SIZE_GUIDELINES['large']['threshold']
                }

                analysis_results[size_category].append(result)

                if result['needs_compression']:
                    analysis_results['recommendations'].append(result)

                progress.advance(task)

        # Display results
        self.display_analysis_results(analysis_results, show_recommendations)

        return analysis_results

    def display_analysis_results(self, results: Dict, show_recommendations: bool = True):
        """Display simple analysis results."""
        console.print('📊 IMAGE ANALYSIS RESULTS')
        console.print('=' * 50)

        # Size categories
        for category in ['large', 'medium', 'small']:
            images = results[category]
            if images:
                console.print(f'\n🔴 {category.upper()} IMAGES ({len(images)} files):')
                for img in images:
                    status = '⚠️ NEEDS COMPRESSION' if img['needs_compression'] else '✅ OK'
                    valid_status = '✅' if img['is_valid'] else '❌'
                    console.print(f'  {status} {valid_status} {img["filename"]} ({img["size_mb"]:.1f}MB)')

        # Validation errors
        if results['validation_errors']:
            console.print(f'\n❌ VALIDATION ERRORS ({len(results["validation_errors"])} files):')
            for error in results['validation_errors']:
                console.print(f'  ❌ {os.path.basename(error["path"])} - {error["error"]}')

        # Summary
        console.print(f'\n📈 SUMMARY:')
        console.print(f'  Total images: {sum(len(results[cat]) for cat in ["large", "medium", "small"])}')
        console.print(f'  Total size: {results["total_size"]:.1f}MB')
        console.print(f'  Large images: {len(results["large"])}')
        console.print(f'  Medium images: {len(results["medium"])}')
        console.print(f'  Small images: {len(results["small"])}')
        console.print(f'  Validation errors: {len(results["validation_errors"])}')

        if show_recommendations and results['recommendations']:
            self.display_compression_recommendations(results['recommendations'])

    def display_compression_recommendations(self, recommendations: List[Dict]):
        """Display simple compression recommendations."""
        console.print(f'\n🎯 COMPRESSION RECOMMENDATIONS ({len(recommendations)} images):')
        console.print('=' * 50)

        # Group by priority
        high_priority = [r for r in recommendations if r['size_mb'] > 1.0]
        medium_priority = [r for r in recommendations if 0.5 < r['size_mb'] <= 1.0]

        if high_priority:
            console.print(f'\n🔴 HIGH PRIORITY ({len(high_priority)} images > 1MB):')
            for img in high_priority[:10]:  # Show top 10
                console.print(f'  📸 {img["filename"]} ({img["size_mb"]:.1f}MB)')
            if len(high_priority) > 10:
                console.print(f'  ... and {len(high_priority) - 10} more')

        if medium_priority:
            console.print(f'\n🟡 MEDIUM PRIORITY ({len(medium_priority)} images 0.5-1MB):')
            for img in medium_priority[:5]:  # Show top 5
                console.print(f'  📸 {img["filename"]} ({img["size_mb"]:.1f}MB)')
            if len(medium_priority) > 5:
                console.print(f'  ... and {len(medium_priority) - 5} more')

        # Action options
        console.print(f'\n🚀 ACTION OPTIONS:')
        console.print('=' * 50)
        console.print('1. Compress high priority images only (>1MB)')
        console.print('2. Compress all recommended images')
        console.print('3. Generate compression commands')
        console.print('4. Run validation fixes only')
        console.print('5. Skip compression')

    def compress_recommended_images(self, recommendations: List[Dict],
                                  priority: str = 'all') -> List[Dict]:
        """Compress recommended images using the existing compress_images.py script."""
        if not recommendations:
            console.print('❌ No images to compress')
            return []

        # Filter by priority
        if priority == 'high':
            images_to_compress = [r for r in recommendations if r['size_mb'] > 1.0]
        else:
            images_to_compress = recommendations

        if not images_to_compress:
            console.print('❌ No images match the selected priority')
            return []

        console.print(f'🔧 Compressing {len(images_to_compress)} images...')

        # Build command for compress_images.py
        image_paths = [img['path'] for img in images_to_compress]
        cmd = ['python3', 'tools/scripts/maintenance/compress_images.py'] + \
              [f'-f {path}' for path in image_paths] + ['--apply']

        console.print(f'Running: {" ".join(cmd)}')

        # Execute compression
        try:
            result = subprocess.run(cmd, cwd=self.repo_path, capture_output=True, text=True)
            if result.returncode == 0:
                console.print('✅ Compression completed successfully')
                return images_to_compress
            else:
                console.print(f'❌ Compression failed: {result.stderr}')
                return []
        except Exception as e:
            console.print(f'❌ Error during compression: {e}')
            return []

    def generate_compression_commands(self, recommendations: List[Dict]) -> str:
        """Generate compression commands for manual execution."""
        if not recommendations:
            return "No images to compress"

        commands = []
        for img in recommendations:
            cmd = f'python3 tools/scripts/maintenance/compress_images.py -f "{img["path"]}" --apply'
            commands.append(cmd)

        return '\n'.join(commands)

    def fix_validation_errors(self, validation_errors: List[Dict]) -> List[str]:
        """Fix image validation errors."""
        if not validation_errors:
            console.print('✅ No validation errors to fix')
            return []

        console.print(f'🔧 Fixing {len(validation_errors)} validation errors...')

        fixed_files = []
        for error in validation_errors:
            console.print(f'Processing: {os.path.basename(error["path"])}')
            console.print(f'  Error: {error["error"]}')

        return fixed_files

def main():
    """Main function with simplified CLI."""
    parser = argparse.ArgumentParser(
        description='Simple Image Analyzer for MLSysBook',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with recommendations
  python3 integrated_image_analyzer.py --analyze

  # Analyze and compress large images (>1MB)
  python3 integrated_image_analyzer.py --analyze --compress --priority high

  # Generate compression commands only
  python3 integrated_image_analyzer.py --analyze --generate-commands

  # Validate image formats only
  python3 integrated_image_analyzer.py --validate
        """
    )

    parser.add_argument('--analyze', action='store_true',
                       help='Perform image analysis')
    parser.add_argument('--validate', action='store_true',
                       help='Validate image formats only')
    parser.add_argument('--compress', action='store_true',
                       help='Compress recommended images')
    parser.add_argument('--priority', choices=['high', 'all'],
                       default='all', help='Compression priority')
    parser.add_argument('--generate-commands', action='store_true',
                       help='Generate compression commands for manual execution')
    parser.add_argument('--directory', default='book/contents',
                       help='Directory to analyze (default: book/contents)')
    parser.add_argument('--no-recommendations', action='store_true',
                       help='Skip compression recommendations')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with prompts')

    args = parser.parse_args()

    if not any([args.analyze, args.validate]):
        parser.print_help()
        return

    analyzer = IntegratedImageAnalyzer()

    if args.validate:
        # Validation only mode
        results = analyzer.analyze_images(args.directory, validate_formats=True, show_recommendations=False)
        if results.get('validation_errors'):
            console.print(f'\n❌ Found {len(results["validation_errors"])} validation errors')
            if Confirm.ask("Fix validation errors?"):
                analyzer.fix_validation_errors(results['validation_errors'])
        else:
            console.print('✅ All images passed validation')

    elif args.analyze:
        # Full analysis mode
        results = analyzer.analyze_images(
            args.directory,
            validate_formats=True,
            show_recommendations=not args.no_recommendations
        )

        if args.interactive and results.get('recommendations'):
            console.print('\n🤔 What would you like to do?')
            action = Prompt.ask(
                "Choose action",
                choices=['1', '2', '3', '4', '5'],
                default='5'
            )

            if action == '1':
                analyzer.compress_recommended_images(results['recommendations'], 'high')
            elif action == '2':
                analyzer.compress_recommended_images(results['recommendations'], 'all')
            elif action == '3':
                commands = analyzer.generate_compression_commands(results['recommendations'])
                console.print('\n📝 Compression commands:')
                console.print(commands)
            elif action == '4':
                if results.get('validation_errors'):
                    analyzer.fix_validation_errors(results['validation_errors'])
                else:
                    console.print('✅ No validation errors to fix')

        elif args.compress:
            analyzer.compress_recommended_images(results['recommendations'], args.priority)

        elif args.generate_commands:
            commands = analyzer.generate_compression_commands(results['recommendations'])
            console.print('\n📝 Compression commands:')
            console.print(commands)

if __name__ == "__main__":
    main()
