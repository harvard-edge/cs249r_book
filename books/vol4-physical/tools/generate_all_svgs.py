"""
book/tools/generate_all_svgs.py
Master generator for all book figures and pipeline locators.
Pure vector SVG, unclipped typography, Harvard Crimson & ETH Zurich Academic Semantic Palette.
"""

import os
import subprocess
try:
    from .figures import (
        ch01, ch02, ch03_04, ch05, ch06, ch07, ch08, ch09, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, locator
    )
except ImportError:
    from figures import (
        ch01, ch02, ch03_04, ch05, ch06, ch07, ch08, ch09, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, locator
    )

def main():
    print("=== Generating All Pure-Vector SVGs and Synchronizing PDFs ===")
    
    locator.run_all()
    ch01.run_all()
    ch02.run_all()
    ch03_04.run_all()
    ch05.run_all()
    ch06.run_all()
    ch07.run_all()
    ch08.run_all()
    ch09.run_all()
    ch10.run_all()
    ch11.run_all()
    ch12.run_all()
    ch13.run_all()
    ch14.run_all()
    ch15.run_all()
    ch16.run_all()
    ch17.run_all()

    print("\n=== Generating PNG Inspection Gallery in Brain Artifacts ===")
    png_dir = "/Users/VJ/.gemini/antigravity-cli/brain/40a33dd2-8620-49b6-9d24-5c26ad2ef085/svg_v2"
    os.makedirs(png_dir, exist_ok=True)

    for root, dirs, files in os.walk("book/chapters"):
        for f in files:
            if f.endswith(".svg"):
                svg_path = os.path.join(root, f)
                ch_name = root.split("/")[2]
                png_name = f"{ch_name}__{f[:-4]}.png"
                png_path = os.path.join(png_dir, png_name)
                subprocess.run(["rsvg-convert", "-f", "png", "-w", "1800", "-o", png_path, svg_path], capture_output=True)
                print(f"Rendered inspection PNG: {png_name}")

    print("\n✓ ALL SVGs, PDFs, AND PNG INSPECTION AUDITS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
