#!/usr/bin/env python3
"""
TinyTorch VHS Tape Recording Pipeline.

Automates running Charmbracelet's VHS on version-controlled .tape files,
extracting real terminal PTY frames via headless Chrome/ttyd, and compiling
them into high-quality, perfectly-paced animated GIFs for the TinyTorch website.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TAPES_DIR = Path(__file__).resolve().parent / "tapes"
DEMOS_DIR = Path(__file__).resolve().parent.parent / "assets" / "images" / "demos"


def record_tape(tape_path: Path, output_gif: Path, fps: int = 16) -> Path:
    """Record a .tape file with VHS and compile the captured frames into an authentic GIF."""
    print(f"\n🎬 Processing VHS Tape: {tape_path.name}")
    content = tape_path.read_text(encoding="utf-8")

    # Determine dimensions or framerate from tape if specified
    framerate_match = re.search(r"Set\s+Framerate\s+(\d+)", content)
    capture_fps = int(framerate_match.group(1)) if framerate_match else 25

    # Create temporary directory for frame capture
    frames_dir = Path(tempfile.mkdtemp(prefix=f"vhs_{tape_path.stem}_"))
    # VHS requires the destination directory to NOT exist when renaming or must be clean
    shutil.rmtree(frames_dir)

    # Replace Output directive to point to the frames folder with trailing slash
    # If no Output directive exists, add it at the top
    output_line = f'Output "{frames_dir}/"'
    if re.search(r'^Output\s+.*$', content, flags=re.MULTILINE):
        tape_content = re.sub(r'^Output\s+.*$', output_line, content, count=1, flags=re.MULTILINE)
    else:
        tape_content = f"{output_line}\n{content}"

    temp_tape = Path(tempfile.gettempdir()) / f"run_{tape_path.name}"
    temp_tape.write_text(tape_content, encoding="utf-8")

    try:
        print(f"  ▶ Executing VHS session (recording frames into {frames_dir.name})...")
        env = os.environ.copy()
        env["PATH"] = f"{REPO_ROOT}/tinytorch/bin:{env.get('PATH', '')}"
        env["TITO_ALLOW_SYSTEM"] = "1"
        env["TINYTORCH_NON_INTERACTIVE"] = "1"
        env["TITO_NON_INTERACTIVE"] = "1"
        env["PYTHONPATH"] = f"{REPO_ROOT}/tinytorch"

        proc = subprocess.run(
            ["vhs", str(temp_tape)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            print(f"  ❌ VHS failed with code {proc.returncode}")
            print(f"  STDERR:\n{proc.stderr}")
            sys.exit(1)

        # Check if frames were generated
        text_frames = sorted(frames_dir.glob("frame-text-*.png"))
        cursor_frames = sorted(frames_dir.glob("frame-cursor-*.png"))

        if not text_frames:
            print(f"  ❌ No frames captured in {frames_dir}!")
            print(f"  STDOUT: {proc.stdout}")
            print(f"  STDERR: {proc.stderr}")
            sys.exit(1)

        total_frames = len(text_frames)
        duration_s = total_frames / capture_fps
        print(f"  ✓ Captured {total_frames} frames ({duration_s:.1f}s at {capture_fps} fps)")

        # Compile frames to GIF using ffmpeg with dual-overlay and optimal color palette
        print(f"  ▶ Compiling GIF with ffmpeg -> {output_gif.name}...")
        output_gif.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(capture_fps),
            "-start_number", "1",
            "-i", str(frames_dir / "frame-text-%05d.png"),
            "-framerate", str(capture_fps),
            "-start_number", "1",
            "-i", str(frames_dir / "frame-cursor-%05d.png"),
            "-filter_complex",
            f"[0][1]overlay=shortest=1[merged];[merged]fps={fps},split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
            str(output_gif),
        ]

        ff_proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if ff_proc.returncode != 0:
            print(f"  ❌ FFmpeg compilation failed!")
            print(f"  STDERR: {ff_proc.stderr}")
            sys.exit(1)

        size_kb = output_gif.stat().st_size / 1024
        print(f"  ✨ Generated {output_gif.name}: {size_kb:.1f} KB ({total_frames} frames)")
        return output_gif

    finally:
        if temp_tape.exists():
            temp_tape.unlink()
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)


def main():
    TAPES_DIR.mkdir(parents=True, exist_ok=True)
    DEMOS_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        selected_names = set(sys.argv[1:])
        tapes = [t for t in TAPES_DIR.glob("*.tape") if t.name in selected_names or t.stem in selected_names]
    else:
        tapes = sorted(TAPES_DIR.glob("*.tape"))

    if not tapes:
        print(f"No matching .tape files found in {TAPES_DIR}")
        sys.exit(1)

    print(f"Recording {len(tapes)} tape(s):")
    for t in tapes:
        print(f"  - {t.name}")

    for tape in tapes:
        out_name = f"tinytorch-{tape.stem.replace('_', '-')}.gif"
        out_path = DEMOS_DIR / out_name
        record_tape(tape, out_path)

    print("\n✅ VHS recording and compilation complete!")


if __name__ == "__main__":
    main()
