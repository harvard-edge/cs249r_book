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

def compress_images(files, quality=85, apply=False):
    """Compress images to textbook-appropriate sizes."""
    if not files:
        print('❌ No files specified')
        return
    
    print(f'🔍 Processing {len(files)} images...')
    
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
            
            # Determine target size
            filename = os.path.basename(image_path).lower()
            if any(keyword in filename for keyword in ['setup', 'kit', 'board', 'hardware', 'assembled']):
                target_size = '1200x900'
            elif any(keyword in filename for keyword in ['screenshot', 'screen', 'ui', 'system']):
                target_size = '1000x750'
            elif any(keyword in filename for keyword in ['diagram', 'chart', 'graph', 'boat']):
                target_size = '800x600'
            else:
                target_size = '1000x750'
            
            print(f'🎯 Target: {target_size}')
            
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
        elif sys.argv[i] in ['-h', '--help']:
            print('Usage: python3 compress_images.py -f file1.png -f file2.jpg')
            print('       python3 compress_images.py -f file.png --apply --quality 90')
            print('')
            print('Options:')
            print('  -f, --file     Image file to compress (can be used multiple times)')
            print('  --apply        Apply compression (replace original files)')
            print('  --quality N    JPEG quality (1-100, default: 85)')
            print('  -h, --help     Show this help message')
            return
        else:
            i += 1
    
    if not files:
        print('❌ No files specified')
        print('Usage: python3 compress_images.py -f file1.png -f file2.jpg')
        print('       python3 compress_images.py -f file.png --apply --quality 90')
        return
    
    compress_images(files, quality, apply)

if __name__ == "__main__":
    main() 