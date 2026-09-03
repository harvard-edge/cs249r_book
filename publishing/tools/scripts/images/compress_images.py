#!/usr/bin/env python3
"""
Flexible Image Compression Tool
Compresses images to textbook-appropriate sizes with automatic backup.
"""

import os
import subprocess
import shutil
import sys
from datetime import datetime

def compress_images(files, quality=85, apply=False, preserve_dimensions=False, smart_compression=False):
    """Compress images to textbook-appropriate sizes."""
    if not files:
        print('❌ No files specified')
        return

    print(f'🔍 Processing {len(files)} images...')
    if smart_compression:
        print(f'📐 Mode: Smart compression (quality first, resize if needed)')
    elif preserve_dimensions:
        print(f'📐 Mode: Preserve dimensions')
    else:
        print(f'📐 Mode: Smart resize')

    # Create backup
    backup_dir = f'image_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(backup_dir, exist_ok=True)
    print(f'💾 Backup dir: {backup_dir}')

    total_original = 0
    total_compressed = 0

    for image_path in files:
        if os.path.exists(image_path):
            print(f'\n📸 Processing: {os.path.basename(image_path)}')

            # Backup
            backup_path = os.path.join(backup_dir, os.path.basename(image_path))
            shutil.copy2(image_path, backup_path)

            # Get original size
            original_size = os.path.getsize(image_path) / (1024 * 1024)
            total_original += original_size
            print(f'📏 Original: {original_size:.1f}MB')

            # Determine compression approach
            if smart_compression:
                # Smart compression: try quality first, resize if still >1MB
                print(f'🎯 Mode: Smart compression (quality first, resize if >1MB)')

                # First attempt: quality-only compression
                output_path = f'{image_path}.compressed'
                cmd = ['magick', image_path, '-quality', str(quality), '-strip', output_path]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    quality_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f'📊 Quality compression: {original_size:.1f}MB → {quality_size:.1f}MB')

                    # If quality compression got us under 1MB, we're done
                    if quality_size <= 1.0:
                        compressed_size = quality_size
                        total_compressed += compressed_size
                        savings = original_size - compressed_size
                        savings_percent = (savings / original_size) * 100
                        print(f'✅ Quality compression sufficient: {original_size:.1f}MB → {compressed_size:.1f}MB')
                        print(f'💰 Savings: {savings:.1f}MB ({savings_percent:.1f}%)')

                        if apply:
                            shutil.move(output_path, image_path)
                            print('✅ Applied quality compression')
                        else:
                            print(f'💾 Compressed file: {output_path}')
                    else:
                        # Quality compression wasn't enough, try resize
                        print(f'⚠️ Quality compression not sufficient ({quality_size:.1f}MB > 1MB), trying resize...')

                        # Determine target size for resize
                        filename = os.path.basename(image_path).lower()
                        if any(keyword in filename for keyword in ['setup', 'kit', 'board', 'hardware', 'assembled']):
                            target_size = '1200x900'
                        elif any(keyword in filename for keyword in ['screenshot', 'screen', 'ui', 'system']):
                            target_size = '1000x750'
                        elif any(keyword in filename for keyword in ['diagram', 'chart', 'graph', 'boat']):
                            target_size = '800x600'
                        else:
                            target_size = '1000x750'

                        # Resize + quality compression
                        resize_output_path = f'{image_path}.resized'
                        resize_cmd = ['magick', image_path, '-resize', f'{target_size}>', '-quality', str(quality), '-strip', resize_output_path]
                        resize_result = subprocess.run(resize_cmd, capture_output=True, text=True)

                        if resize_result.returncode == 0:
                            # Clean up quality-only file
                            os.remove(output_path)

                            compressed_size = os.path.getsize(resize_output_path) / (1024 * 1024)
                            total_compressed += compressed_size
                            savings = original_size - compressed_size
                            savings_percent = (savings / original_size) * 100
                            print(f'✅ Resize + quality compression: {original_size:.1f}MB → {compressed_size:.1f}MB')
                            print(f'💰 Savings: {savings:.1f}MB ({savings_percent:.1f}%)')

                            if apply:
                                shutil.move(resize_output_path, image_path)
                                print('✅ Applied resize + quality compression')
                            else:
                                print(f'💾 Compressed file: {resize_output_path}')
                        else:
                            print(f'❌ Resize failed: {resize_result.stderr}')
                            # Clean up quality-only file
                            os.remove(output_path)
                else:
                    print(f'❌ Quality compression failed: {result.stderr}')

            elif preserve_dimensions:
                # Quality-only compression (preserves dimensions)
                output_path = f'{image_path}.compressed'
                cmd = ['magick', image_path, '-quality', str(quality), '-strip', output_path]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
                    total_compressed += compressed_size
                    savings = original_size - compressed_size
                    savings_percent = (savings / original_size) * 100
                    print(f'✅ Quality compression: {original_size:.1f}MB → {compressed_size:.1f}MB')
                    print(f'💰 Savings: {savings:.1f}MB ({savings_percent:.1f}%)')

                    if apply:
                        shutil.move(output_path, image_path)
                        print('✅ Applied quality compression')
                    else:
                        print(f'💾 Compressed file: {output_path}')
                else:
                    print(f'❌ Failed: {result.stderr}')
            else:
                # Smart resize + quality compression (original behavior)
                filename = os.path.basename(image_path).lower()
                if any(keyword in filename for keyword in ['setup', 'kit', 'board', 'hardware', 'assembled']):
                    target_size = '1200x900'
                elif any(keyword in filename for keyword in ['screenshot', 'screen', 'ui', 'system']):
                    target_size = '1000x750'
                elif any(keyword in filename for keyword in ['diagram', 'chart', 'graph', 'boat']):
                    target_size = '800x600'
                else:
                    target_size = '1000x750'

                print(f'🎯 Mode: Smart resize to {target_size} + quality compression')

                # Compress
                output_path = f'{image_path}.compressed'
                cmd = ['magick', image_path, '-resize', f'{target_size}>', '-quality', str(quality), '-strip', output_path]
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
                    total_compressed += compressed_size
                    savings = original_size - compressed_size
                    savings_percent = (savings / original_size) * 100
                    print(f'✅ Compressed: {original_size:.1f}MB → {compressed_size:.1f}MB')
                    print(f'💰 Savings: {savings:.1f}MB ({savings_percent:.1f}%)')

                    if apply:
                        shutil.move(output_path, image_path)
                        print('✅ Applied compression')
                    else:
                        print(f'💾 Compressed file: {output_path}')
                else:
                    print(f'❌ Failed: {result.stderr}')
        else:
            print(f'⚠️ File not found: {image_path}')

    print(f'\n📊 Summary:')
    print(f'Total original: {total_original:.1f}MB')
    print(f'Total compressed: {total_compressed:.1f}MB')
    print(f'Total savings: {total_original - total_compressed:.1f}MB')
    print(f'Backup location: {backup_dir}')

    if not apply:
        print(f'\n💡 To apply compression, run with --apply flag')

def main():
    """Parse command line arguments and run compression."""
    files = []
    apply = False
    quality = 85
    preserve_dimensions = False
    smart_compression = False

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-f' and i + 1 < len(sys.argv):
            files.append(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--apply':
            apply = True
            i += 1
        elif sys.argv[i] == '--quality' and i + 1 < len(sys.argv):
            quality = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--preserve-dimensions':
            preserve_dimensions = True
            i += 1
        elif sys.argv[i] == '--smart-compression':
            smart_compression = True
            i += 1
        elif sys.argv[i] in ['-h', '--help']:
            print('Usage: python3 compress_images.py -f file1.png -f file2.jpg')
            print('       python3 compress_images.py -f file.png --apply --quality 90')
            print('       python3 compress_images.py -f file.png --preserve-dimensions --apply')
            print('       python3 compress_images.py -f file.png --smart-compression --apply')
            print('')
            print('Options:')
            print('  -f, --file              Image file to compress (can be used multiple times)')
            print('  --apply                 Apply compression (replace original files)')
            print('  --quality N             JPEG quality (1-100, default: 85)')
            print('  --preserve-dimensions   Only compress quality, keep original dimensions')
            print('  --smart-compression     Quality first, resize only if still >1MB (RECOMMENDED)')
            print('  -h, --help              Show this help message')
            print('')
            print('Compression Modes:')
            print('  Default: Smart resize + quality compression (best file size reduction)')
            print('  --preserve-dimensions: Quality-only compression (preserves contributor intent)')
            print('  --smart-compression: Quality first, resize if needed (BALANCED APPROACH)')
            return
        else:
            i += 1

    if not files:
        print('❌ No files specified')
        print('Usage: python3 compress_images.py -f file1.png -f file2.jpg')
        print('       python3 compress_images.py -f file.png --apply --quality 90')
        return

    compress_images(files, quality, apply, preserve_dimensions, smart_compression)

if __name__ == "__main__":
    main()
