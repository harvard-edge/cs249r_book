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

from PIL import Image, ImageDraw, ImageFont, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TAPES_DIR = Path(__file__).resolve().parent / "tapes"
DEMOS_DIR = Path(__file__).resolve().parent.parent / "assets" / "images" / "demos"

TAPE_TITLES = {
    "01_install_and_run": "tinytorch · install & first module",
    "02_modules_mastery": "tinytorch · module status",
    "03_tinygpt_chat": "tinytorch · tinygpt chat",
    "04_tinycopilot": "tinytorch · tinycopilot",
    "05_mlperf_opt": "tinytorch · mlperf optimization",
    "06_tito_companion": "tinytorch · tito companion",
    "07_tito_olympics": "tinytorch · tito olympics",
}


def create_window_template(term_w: int, term_h: int, title: str) -> tuple[Path, int, int]:
    """Generate a Terminalizer-style macOS window frame with traffic lights, border, and drop shadow."""
    pad_x = 16
    pad_y = 12
    bar_h = 36
    radius = 12
    margin = 12

    win_w = term_w + 2 * pad_x
    win_h = term_h + bar_h + 2 * pad_y
    canvas_w = win_w + 2 * margin
    canvas_h = win_h + 2 * margin

    if canvas_w % 2 != 0:
        canvas_w += 1
        win_w += 1
    if canvas_h % 2 != 0:
        canvas_h += 1
        win_h += 1

    # Canvas with dark slate background matching site card aesthetics (#0f172a)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (15, 23, 42, 255))

    # Soft drop shadow
    shadow_mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_mask)
    sdraw.rounded_rectangle(
        [margin, margin + 4, margin + win_w, margin + win_h + 4],
        radius=radius,
        fill=(0, 0, 0, 180),
    )
    shadow = shadow_mask.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(shadow)

    # Window surface
    win = Image.new("RGBA", (win_w, win_h), (0, 0, 0, 0))
    wdraw = ImageDraw.Draw(win)
    wdraw.rounded_rectangle([0, 0, win_w, win_h], radius=radius, fill=(30, 30, 46, 255))

    # Window title bar (Catppuccin Mantle)
    header_mask = Image.new("L", (win_w, win_h), 0)
    hmask_draw = ImageDraw.Draw(header_mask)
    hmask_draw.rounded_rectangle([0, 0, win_w, win_h], radius=radius, fill=255)
    hmask_draw.rectangle([0, bar_h, win_w, win_h], fill=0)

    header = Image.new("RGBA", (win_w, win_h), (24, 24, 37, 255))
    win.paste(header, (0, 0), header_mask)
    wdraw.line([(0, bar_h), (win_w, bar_h)], fill=(49, 50, 68, 255), width=1)

    # macOS traffic lights
    btn_y = (bar_h - 12) // 2
    wdraw.ellipse([16, btn_y, 16 + 12, btn_y + 12], fill=(255, 95, 86, 255), outline=(224, 68, 62, 255))
    wdraw.ellipse([36, btn_y, 36 + 12, btn_y + 12], fill=(255, 189, 46, 255), outline=(222, 161, 35, 255))
    wdraw.ellipse([56, btn_y, 56 + 12, btn_y + 12], fill=(39, 201, 63, 255), outline=(26, 171, 41, 255))

    # Window title with authentic Tiny🔥Torch flame branding
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    # Clean action subtitle (e.g. 'install & run')
    action = title
    if "·" in title:
        action = title.split("·", 1)[1].strip()

    fire_path = REPO_ROOT / "tinytorch" / "guide" / "assets" / "images" / "logos" / "fire-emoji.png"
    if fire_path.exists():
        fire_raw = Image.open(fire_path).convert("RGBA")
        fire_icon = fire_raw.resize((14, 14), Image.Resampling.LANCZOS)

        prefix = "Tiny"
        suffix = f"Torch · {action}"

        p_box = font.getbbox(prefix)
        pw = p_box[2] - p_box[0]
        ph = p_box[3] - p_box[1]

        s_box = font.getbbox(suffix)
        sw = s_box[2] - s_box[0]

        fw, fh = fire_icon.size
        gap = 2
        total_w = pw + gap + fw + gap + sw

        start_x = (win_w - total_w) // 2
        text_y = (bar_h - ph) // 2 - 1
        fire_y = (bar_h - fh) // 2

        # Draw 'Tiny', flame icon, and 'Torch · <action>'
        wdraw.text((start_x, text_y), prefix, fill=(205, 214, 244, 255), font=font)
        win.alpha_composite(fire_icon, (start_x + pw + gap, fire_y))
        wdraw.text((start_x + pw + gap + fw + gap, text_y), suffix, fill=(205, 214, 244, 255), font=font)
    else:
        bbox = font.getbbox(title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (win_w - tw) // 2
        ty = (bar_h - th) // 2 - 1
        wdraw.text((tx, ty), title, fill=(166, 173, 200, 255), font=font)

    # 1px border outline
    wdraw.rounded_rectangle([0, 0, win_w - 1, win_h - 1], radius=radius, outline=(69, 71, 90, 255), width=1)

    canvas.alpha_composite(win, (margin, margin))
    tpl_path = Path(tempfile.gettempdir()) / f"vhs_tpl_{win_w}_{win_h}_{abs(hash(title))}.png"
    canvas.save(tpl_path)

    term_x = margin + pad_x
    term_y = margin + bar_h + pad_y
    return tpl_path, term_x, term_y


# Chapter cards: one label per phase, shown in the top-right pill. Phases come
# from the recording itself: setup runs until the first fast-forwarded
# stretch, the fast stretch is training, then generation/scoring runs until
# the final hold, and the hold (END_HOLD_SECONDS) shows the result. Tapes with
# no entry here (or no fast stretch) get no chapter cards.
TAPE_CHAPTERS = {
    "04_tinycopilot": ["Building YOUR TinyGPT", "Training YOUR transformer",
                       "Completing prompts it never saw", "Result"],
    "03_tinygpt_chat": ["Building YOUR TinyGPT", "Training on 64 Q&A pairs",
                        "Answering + overfitting check", "Result"],
}

# ---------------------------------------------------------------------------
# Playback speed-up for long, quiet stretches (e.g. a progress bar during
# training). The run itself is never shortened: frames are sub-sampled for
# playback only, and a "⏩ N× speed" badge is drawn while it plays fast.
# ---------------------------------------------------------------------------
SPEEDUP_FACTOR = 5          # playback speed inside a quiet stretch
SPEEDUP_MIN_SECONDS = 6.0   # only stretches at least this long are sped up
SPEEDUP_MARGIN_SECONDS = 1.0  # keep this much at real speed on each side
END_HOLD_SECONDS = 6.0      # never speed up the final hold on the result
SCROLL_THRESHOLD = 0.02     # fraction of pixels changed that counts as scrolling


def frame_activity(text_frames, size=(160, 110)) -> list:
    """Fraction of (downscaled) pixels that change from each frame to the next."""
    import numpy as np
    prev, activity = None, []
    for path in text_frames:
        data = np.asarray(Image.open(path).convert("L").resize(size), dtype=np.int16)
        activity.append(0.0 if prev is None else float((np.abs(data - prev) > 24).mean()))
        prev = data
    return activity


def plan_speedup(activity, capture_fps: int, min_seconds: float = SPEEDUP_MIN_SECONDS,
                 margin_seconds: float = SPEEDUP_MARGIN_SECONDS) -> list:
    """
    Return [(start, end)] frame ranges to play at SPEEDUP_FACTOR.

    A quiet stretch is a run of frames where no second contains a large change
    (no scrolling output); a ticking progress bar or an idle wait qualifies.
    """
    n = len(activity)
    window = max(1, capture_fps)
    busy = [False] * n
    for i in range(n):
        lo, hi = max(0, i - window // 2), min(n, i + window // 2 + 1)
        busy[i] = max(activity[lo:hi]) > SCROLL_THRESHOLD
    end_limit = n - int(END_HOLD_SECONDS * capture_fps)
    margin = int(margin_seconds * capture_fps)
    ranges, i = [], 0
    while i < end_limit:
        if busy[i]:
            i += 1
            continue
        j = i
        while j < end_limit and not busy[j]:
            j += 1
        start, stop = i + margin, j - margin
        if (stop - start) >= min_seconds * capture_fps:
            ranges.append((start, stop))
        i = j
    return ranges


PILL_W, PILL_H = 620, 28   # title-bar layer size (right-aligned pill inside)


def render_pill(chapter=None, factor=None) -> Image.Image:
    """
    A transparent PILL_W x PILL_H layer with a right-aligned pill: the chapter
    ("2/4 · Training YOUR transformer") and, while fast-forwarding, drawn
    arrows plus "N×". It is overlaid on the window title bar, so it can
    never cover terminal output. Menlo has no emoji, so the arrows are polygons.
    """
    layer = Image.new("RGBA", (PILL_W, PILL_H), (0, 0, 0, 0))
    if not chapter and not factor:
        return layer
    draw = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    ink = (30, 30, 46, 255)
    text = "   ".join([t for t in (chapter, f"{factor}×" if factor else None) if t])
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    arrows_w = 26 if factor else 0
    right = PILL_W - 2
    left = right - (w + arrows_w + 24)
    fill = (250, 179, 135, 255) if factor else (137, 180, 250, 255)
    draw.rounded_rectangle((left, 1, right, PILL_H - 2), radius=12, fill=fill)
    tx = left + 12
    if factor:
        cy, th = PILL_H / 2, 6
        for dx in (0, 10):
            draw.polygon([(tx + dx, cy - th), (tx + dx + 10, cy), (tx + dx, cy + th)], fill=ink)
        tx += arrows_w
    draw.text((tx, (PILL_H - h) / 2 - bbox[1]), text, fill=ink, font=font)
    return layer


def plan_chapters(labels, activity, capture_fps: int) -> list:
    """
    Map each raw frame index to a chapter label (or None).

    The longest quiet stretch of at least 3 s is the training phase; before it
    is setup, after it until the final hold is generation and scoring, and the
    final hold is the result. Independent of whether that stretch is long
    enough to be fast-forwarded.
    """
    n = len(activity)
    if not labels or len(labels) != 4:
        return [None] * n
    quiet = plan_speedup(activity, capture_fps, min_seconds=3.0, margin_seconds=0.0)
    if not quiet:
        return [None] * n
    first, last = max(quiet, key=lambda r: r[1] - r[0])
    hold = max(last, n - int(END_HOLD_SECONDS * capture_fps))
    bounds = [(0, first), (first, last), (last, hold), (hold, n)]
    out = [None] * n
    for k, ((a, b), label) in enumerate(zip(bounds, labels), start=1):
        for i in range(a, b):
            out[i] = f"{k}/4 · {label}"
    return out


def build_playback_frames(frames_dir: Path, ranges, factor: int, chapters=None) -> tuple:
    """Write the playback sequence (sub-sampled inside `ranges`) to a new folder."""
    text_frames = sorted(frames_dir.glob("frame-text-*.png"))
    cursor_frames = sorted(frames_dir.glob("frame-cursor-*.png"))
    out = Path(tempfile.mkdtemp(prefix="vhs_playback_"))
    in_range = [False] * len(text_frames)
    for a, b in ranges:
        for k in range(a, b):
            in_range[k] = True
    keep, k, pill_cache = [], 0, {}
    while k < len(text_frames):
        keep.append(k)
        k += factor if in_range[k] else 1
    for new_idx, old_idx in enumerate(keep, start=1):
        src_text = text_frames[old_idx]
        dst_text = out / f"frame-text-{new_idx:05d}.png"
        shutil.copyfile(src_text, dst_text)
        chapter = chapters[old_idx] if chapters else None
        fast = factor if in_range[old_idx] else None
        key = (chapter, fast)
        if key not in pill_cache:
            pill_cache[key] = render_pill(chapter, fast)
        pill_cache[key].save(out / f"frame-pill-{new_idx:05d}.png")
        if old_idx < len(cursor_frames):
            shutil.copyfile(cursor_frames[old_idx], out / f"frame-cursor-{new_idx:05d}.png")
    return out, len(keep)


def record_tape(tape_path: Path, output_gif: Path, fps: int = 16) -> Path:
    """Record a .tape file with VHS and compile the captured frames into an authentic GIF."""
    print(f"\n🎬 Processing VHS Tape: {tape_path.name}")
    content = tape_path.read_text(encoding="utf-8")

    # Determine dimensions or framerate from tape if specified
    framerate_match = re.search(r"Set\s+Framerate\s+(\d+)", content)
    capture_fps = int(framerate_match.group(1)) if framerate_match else 25
    if not framerate_match:
        # 2026-09-29: VHS captures at 50 fps unless told otherwise, while this
        # pipeline compiled at 25, so every demo played at about half speed.
        content = re.sub(r"^(Set\s+Shell\s+.*)$", rf"\1\nSet Framerate {capture_fps}",
                         content, count=1, flags=re.MULTILINE)

    # Create temporary directory for frame capture
    frames_dir = Path(tempfile.mkdtemp(prefix=f"vhs_{tape_path.stem}_"))
    # VHS requires the destination directory to NOT exist when renaming or must be clean
    shutil.rmtree(frames_dir)

    # Replace Output directive to point to the frames folder with trailing slash
    output_line = f'Output "{frames_dir}/"'
    if re.search(r'^Output\s+.*$', content, flags=re.MULTILINE):
        tape_content = re.sub(r'^Output\s+.*$', output_line, content, count=1, flags=re.MULTILINE)
    else:
        tape_content = f"{output_line}\n{content}"

    temp_tape = Path(tempfile.gettempdir()) / f"run_{tape_path.name}"
    temp_tape.write_text(tape_content, encoding="utf-8")
    tpl_path = None

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

        keep_dir = os.environ.get("TT_KEEP_FRAMES")
        if keep_dir:
            dest = Path(keep_dir) / tape_path.stem
            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(frames_dir, dest)
            print(f"  ✓ Kept raw frames in {dest}")

        activity = frame_activity(text_frames)
        ranges = plan_speedup(activity, capture_fps)
        chapters = plan_chapters(TAPE_CHAPTERS.get(tape_path.stem), activity, capture_fps)
        has_pills = bool(ranges) or any(chapters)
        if ranges:
            spans = ", ".join(f"{a / capture_fps:.1f}-{b / capture_fps:.1f}s" for a, b in ranges)
            print(f"  ⏩ Playing quiet stretches at {SPEEDUP_FACTOR}x: {spans}")
        if any(chapters):
            print(f"  ✓ Chapter cards: {TAPE_CHAPTERS[tape_path.stem]}")
        if has_pills:
            raw_dir = frames_dir
            frames_dir, total_frames = build_playback_frames(raw_dir, ranges, SPEEDUP_FACTOR, chapters)
            shutil.rmtree(raw_dir, ignore_errors=True)
            text_frames = sorted(frames_dir.glob("frame-text-*.png"))
            print(f"  ✓ Playback: {total_frames} frames ({total_frames / capture_fps:.1f}s)")

        # Prepare Terminalizer-style window frame
        first_img = Image.open(text_frames[0])
        title = TAPE_TITLES.get(tape_path.stem, f"tinytorch · {tape_path.stem}")
        tpl_path, term_x, term_y = create_window_template(first_img.width, first_img.height, title)
        # The pill sits at the right end of the title bar (see create_window_template:
        # 12 px outer margin, 36 px bar), clear of the traffic lights and centered title.
        canvas_w = Image.open(tpl_path).width
        pill_x, pill_y = canvas_w - 12 - 16 - PILL_W, 12 + (36 - PILL_H) // 2
        pill_inputs, pill_step = [], ""
        if has_pills:
            pill_inputs = ["-framerate", str(capture_fps), "-start_number", "1",
                           "-i", str(frames_dir / "frame-pill-%05d.png")]
            pill_step = f"[base][3]overlay=x={pill_x}:y={pill_y}:shortest=1[merged];"
        base_step = ("[1][2]overlay=shortest=1[term];[0][term]overlay="
                     f"x={term_x}:y={term_y}:shortest=1[{'base' if has_pills else 'merged'}];")

        # Compile frames to GIF using ffmpeg with window overlay and optimal color palette
        print(f"  ▶ Compiling high-fidelity GIF with ffmpeg -> {output_gif.name}...")
        output_gif.parent.mkdir(parents=True, exist_ok=True)

        ffmpeg_gif_cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(tpl_path),
            "-framerate", str(capture_fps),
            "-start_number", "1",
            "-i", str(frames_dir / "frame-text-%05d.png"),
            "-framerate", str(capture_fps),
            "-start_number", "1",
            "-i", str(frames_dir / "frame-cursor-%05d.png"),
            *pill_inputs,
            "-filter_complex",
            base_step + pill_step + f"[merged]fps={fps},split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=none",
            str(output_gif),
        ]

        ff_proc = subprocess.run(ffmpeg_gif_cmd, capture_output=True, text=True)
        if ff_proc.returncode != 0:
            print(f"  ❌ FFmpeg GIF compilation failed!")
            print(f"  STDERR: {ff_proc.stderr}")
            sys.exit(1)

        size_kb = output_gif.stat().st_size / 1024
        print(f"  ✨ Generated {output_gif.name}: {size_kb:.1f} KB ({total_frames} frames)")

        # Also compile TrueColor MP4 for modern video player support with zero color quantization
        output_mp4 = output_gif.with_suffix(".mp4")
        print(f"  ▶ Compiling high-definition MP4 -> {output_mp4.name}...")
        ffmpeg_mp4_cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(tpl_path),
            "-framerate", str(capture_fps),
            "-start_number", "1",
            "-i", str(frames_dir / "frame-text-%05d.png"),
            "-framerate", str(capture_fps),
            "-start_number", "1",
            "-i", str(frames_dir / "frame-cursor-%05d.png"),
            *pill_inputs,
            "-filter_complex",
            base_step + pill_step + f"[merged]fps={fps},format=yuv420p",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-movflags", "+faststart",
            str(output_mp4),
        ]

        ff_mp4_proc = subprocess.run(ffmpeg_mp4_cmd, capture_output=True, text=True)
        if ff_mp4_proc.returncode == 0:
            mp4_size_kb = output_mp4.stat().st_size / 1024
            print(f"  ✨ Generated {output_mp4.name}: {mp4_size_kb:.1f} KB (TrueColor H.264)")
        else:
            print(f"  ⚠️ MP4 compilation notice: {ff_mp4_proc.stderr[:120]}")

        return output_gif

    finally:
        if temp_tape.exists():
            temp_tape.unlink()
        if tpl_path and tpl_path.exists():
            tpl_path.unlink()
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
