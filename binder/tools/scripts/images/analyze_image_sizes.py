#!/usr/bin/env python3
"""
Textbook Image Size Analyzer
Identifies large images and provides textbook-specific optimization recommendations.
"""

import os
import glob
from pathlib import Path

# Textbook image size guidelines
TEXTBOOK_GUIDELINES = {
    'large': {
        'threshold': 2.0,  # MB
        'recommendation': 'Compress immediately - too large for web/PDF',
        'target_size': '800x600',
        'priority': 'high'
    },
    'medium': {
        'threshold': 1.0,  # MB
        'recommendation': 'Consider compression for better performance',
        'target_size': '1000x750',
        'priority': 'medium'
    },
    'small': {
        'threshold': 0.5,  # MB
        'recommendation': 'Size is acceptable for textbook use',
        'target_size': '1200x900',
        'priority': 'low'
    }
}

# Image type categorization for smart sizing
IMAGE_TYPES = {
    'diagrams': {
        'keywords': ['diagram', 'chart', 'graph', 'flow', 'architecture', 'boat'],
        'target_size': '800x600',
        'max_size': 0.5,  # MB
        'description': 'Technical diagrams and charts'
    },
    'screenshots': {
        'keywords': ['screenshot', 'screen', 'ui', 'interface', 'terminal', 'cli'],
        'target_size': '1000x750',
        'max_size': 0.8,  # MB
        'description': 'UI screenshots and terminal outputs'
    },
    'lab_setup': {
        'keywords': ['setup', 'kit', 'board', 'hardware', 'assembled', 'mounted'],
        'target_size': '1200x900',
        'max_size': 1.2,  # MB
        'description': 'Lab setup and hardware photos'
    },
    'results': {
        'keywords': ['result', 'output', 'inference', 'prediction', 'detection'],
        'target_size': '1000x750',
        'max_size': 0.8,  # MB
        'description': 'Model outputs and results'
    },
    'general': {
        'keywords': [],
        'target_size': '1000x750',
        'max_size': 1.0,  # MB
        'description': 'General textbook images'
    }
}

def get_file_size_mb(file_path):
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def categorize_image(filename):
    """Categorize image based on filename keywords."""
    filename_lower = filename.lower()

    for category, config in IMAGE_TYPES.items():
        if category == 'general':
            continue
        for keyword in config['keywords']:
            if keyword in filename_lower:
                return category, config

    return 'general', IMAGE_TYPES['general']

def get_size_category(size_mb):
    """Get size category based on MB."""
    if size_mb > TEXTBOOK_GUIDELINES['large']['threshold']:
        return 'large'
    elif size_mb > TEXTBOOK_GUIDELINES['medium']['threshold']:
        return 'medium'
    else:
        return 'small'

def analyze_images(directory='book/contents'):
    """Analyze all images in the textbook."""
    print('🔍 Analyzing textbook images...\n')

    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.webp']
    image_files = []

    for ext in image_extensions:
        pattern = os.path.join(directory, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))

    if not image_files:
        print('❌ No image files found')
        return

    print(f'📸 Found {len(image_files)} images\n')

    # Analyze each image
    analysis_results = {
        'large': [],
        'medium': [],
        'small': [],
        'total_size': 0,
        'recommendations': []
    }

    for image_path in sorted(image_files):
        size_mb = get_file_size_mb(image_path)
        analysis_results['total_size'] += size_mb

        filename = os.path.basename(image_path)
        category, config = categorize_image(filename)
        size_category = get_size_category(size_mb)

        result = {
            'path': image_path,
            'filename': filename,
            'size_mb': size_mb,
            'category': category,
            'size_category': size_category,
            'target_size': config['target_size'],
            'max_size': config['max_size'],
            'needs_compression': size_mb > config['max_size']
        }

        analysis_results[size_category].append(result)

        if result['needs_compression']:
            analysis_results['recommendations'].append(result)

    # Print analysis
    print('📊 IMAGE ANALYSIS RESULTS')
    print('=' * 50)

    for category in ['large', 'medium', 'small']:
        images = analysis_results[category]
        if images:
            print(f'\n🔴 {category.upper()} IMAGES ({len(images)} files):')
            for img in images:
                status = '⚠️ NEEDS COMPRESSION' if img['needs_compression'] else '✅ OK'
                print(f'  {status} {img["filename"]} ({img["size_mb"]:.1f}MB) - {img["category"]}')

    print(f'\n📈 SUMMARY:')
    print(f'  Total images: {len(image_files)}')
    print(f'  Total size: {analysis_results["total_size"]:.1f}MB')
    print(f'  Large images: {len(analysis_results["large"])}')
    print(f'  Medium images: {len(analysis_results["medium"])}')
    print(f'  Small images: {len(analysis_results["small"])}')

    if analysis_results['recommendations']:
        print(f'\n🎯 COMPRESSION RECOMMENDATIONS:')
        print('=' * 50)
        for img in analysis_results['recommendations']:
            print(f'  📸 {img["filename"]}')
            print(f'     Current: {img["size_mb"]:.1f}MB')
            print(f'     Target: {img["target_size"]} (max {img["max_size"]:.1f}MB)')
            print(f'     Type: {img["category"]}')
            print()

    return analysis_results

def print_guidelines():
    """Print textbook image guidelines."""
    print('\n📚 TEXTBOOK IMAGE GUIDELINES')
    print('=' * 50)
    print('Based on best practices for academic textbooks:')
    print()

    for category, config in IMAGE_TYPES.items():
        print(f'📸 {category.upper()}:')
        print(f'   Target size: {config["target_size"]}')
        print(f'   Max file size: {config["max_size"]:.1f}MB')
        print(f'   Description: {config["description"]}')
        print(f'   Keywords: {", ".join(config["keywords"]) if config["keywords"] else "general"}')
        print()

    print('💡 GENERAL RECOMMENDATIONS:')
    print('  • Use PNG for diagrams and screenshots with text')
    print('  • Use JPEG for photographs and complex images')
    print('  • Strip metadata to reduce file size')
    print('  • Test images on both web and PDF builds')
    print('  • Ensure readability at target sizes')
    print('  • Consider accessibility (alt text, contrast)')

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print('Usage: python3 analyze_image_sizes.py [directory]')
        print('       python3 analyze_image_sizes.py --guidelines')
        return

    if len(sys.argv) > 1 and sys.argv[1] == '--guidelines':
        print_guidelines()
        return

    directory = sys.argv[1] if len(sys.argv) > 1 else 'book/contents'

    try:
        results = analyze_images(directory)
        print_guidelines()

        if results['recommendations']:
            print(f'\n🚀 NEXT STEPS:')
            print('=' * 50)
            print(f'Run compression on {len(results["recommendations"])} images:')
            print()
            for img in results['recommendations']:
                print(f'python3 tools/scripts/maintenance/compress_images.py -f "{img["path"]}" --apply')
            print()
            print('Or compress all at once:')
            files = [f'"{img["path"]}"' for img in results['recommendations']]
            print(f'python3 tools/scripts/maintenance/compress_images.py {" ".join(files)} --apply')

    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    import sys
    main()
