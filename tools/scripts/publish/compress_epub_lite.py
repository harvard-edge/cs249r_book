#!/usr/bin/env python3
import os
import sys
import shutil
import zipfile
import tempfile
from PIL import Image

def compress_image(path, quality=60, max_size=1200):
    """Compress an image in place with Pillow."""
    try:
        original_size = os.path.getsize(path)
        img = Image.open(path)
        img_format = img.format
        
        # Resize if image is too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
        
        if img_format == "JPEG":
            # Convert RGBA to RGB if needed
            if img.mode in ('RGBA', 'LA'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            img.save(path, "JPEG", quality=quality, optimize=True)
            
        elif img_format == "PNG":
            # Always try aggressive palette conversion for maximum compression
            try:
                img = img.convert("P", palette=Image.ADAPTIVE)
                img.save(path, "PNG", optimize=True)
            except:
                # Fallback to original PNG optimization
                img.save(path, "PNG", optimize=True)
        
        new_size = os.path.getsize(path)
        reduction = (1 - new_size/original_size) * 100 if original_size > 0 else 0
        print(f"  📷 {os.path.basename(path)}: {original_size:,} → {new_size:,} bytes ({reduction:.1f}% reduction)")
        
    except Exception as e:
        print(f" [!] Skipping {path}: {e}")

def compress_epub(input_epub, output_epub, quality=60, max_size=1200):
    tmpdir = tempfile.mkdtemp()

    # Extract EPUB
    with zipfile.ZipFile(input_epub, "r") as zin:
        zin.extractall(tmpdir)

    # Walk and compress images
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                fullpath = os.path.join(root, f)
                compress_image(fullpath, quality, max_size)

    # Repack EPUB
    with zipfile.ZipFile(output_epub, "w", zipfile.ZIP_DEFLATED) as zout:
        # First add the mimetype file uncompressed (EPUB spec requirement)
        mimetype_path = os.path.join(tmpdir, "mimetype")
        if os.path.exists(mimetype_path):
            zout.write(mimetype_path, "mimetype", compress_type=zipfile.ZIP_STORED)

        # Add the rest
        for root, _, files in os.walk(tmpdir):
            for f in files:
                fullpath = os.path.join(root, f)
                relpath = os.path.relpath(fullpath, tmpdir)
                if relpath == "mimetype":
                    continue
                zout.write(fullpath, relpath)

    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compress_epub.py input.epub output.epub [quality] [max_size]")
        sys.exit(1)

    input_epub = sys.argv[1]
    output_epub = sys.argv[2]
    quality = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    max_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1200

    print(f"📊 Starting compression with quality={quality}, max_size={max_size}px")
    original_size = os.path.getsize(input_epub)
    print(f"📖 Original size: {original_size:,} bytes ({original_size/1024/1024:.1f} MB)")
    
    compress_epub(input_epub, output_epub, quality, max_size)
    
    final_size = os.path.getsize(output_epub)
    reduction = (1 - final_size/original_size) * 100
    print(f"\n✅ Compression complete!")
    print(f"📦 Final size: {final_size:,} bytes ({final_size/1024/1024:.1f} MB)")
    print(f"💾 Size reduction: {reduction:.1f}% ({original_size - final_size:,} bytes saved)")
    print(f"🎯 Target achieved: {'✅ Under 100MB' if final_size < 100*1024*1024 else '❌ Over 100MB'}")
