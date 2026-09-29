#!/usr/bin/env python3
"""
TinyTorch Real Terminal Screenshot Capture Pipeline.

Captures authentic raster screenshots (.png) of real terminal execution using
Charmbracelet's VHS with headless Chromium PTY rasterization and composites
them into publication-grade macOS Terminalizer-style framed windows with
native traffic lights, drop shadows, and authentic branding.

Guarantees 100% solid, gap-free box-drawing characters and true raster
fidelity.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCREENSHOTS_DIR = (
    REPO_ROOT / "tinytorch" / "guide" / "assets" / "images" / "screenshots"
)

COMMANDS_TO_CAPTURE = [
    {
        "filename": "tito_welcome.png",
        "title": "welcome",
        "cmd": "tito",
        "width": 880,
        "height": 440,
        "sleep": 1.5,
    },
    {
        "filename": "tito_system_health.png",
        "title": "system health",
        "cmd": "tito system health",
        "width": 880,
        "height": 860,
        "sleep": 2.0,
    },
    {
        "filename": "tito_milestone_list.png",
        "title": "milestones",
        "cmd": "tito milestone list",
        "width": 880,
        "height": 1040,
        "sleep": 2.0,
    },
    {
        "filename": "tito_module_status.png",
        "title": "module status",
        "cmd": "tito module status",
        "width": 880,
        "height": 760,
        "sleep": 2.0,
    },
    {
        "filename": "tito_help.png",
        "title": "help",
        "cmd": "tito --help",
        "width": 880,
        "height": 960,
        "sleep": 1.5,
    },
    {
        "filename": "tito_module_test.png",
        "title": "module test",
        "cmd": "tito module test 01",
        "width": 880,
        "height": 760,
        "sleep": 3.5,
    },
    {
        "filename": "tito_system_info.png",
        "title": "system info",
        "cmd": "tito system info",
        "width": 880,
        "height": 460,
        "sleep": 1.5,
    },
    {
        "filename": "milestone_01_run.png",
        "title": "milestone 01 · perceptron",
        "cmd": (
            "python3 milestones/01_1958_perceptron/01_rosenblatt_forward.py"
        ),
        "width": 880,
        "height": 920,
        "sleep": 2.5,
    },
]


def create_window_template(
    term_w: int, term_h: int, title: str
) -> tuple[Image.Image, int, int]:
    """Generate macOS window frame with traffic lights and drop shadow."""
    pad_x = 0
    pad_y = 0
    bar_h = 36
    radius = 12
    margin = 18

    win_w = term_w + 2 * pad_x
    win_h = term_h + bar_h + 2 * pad_y
    canvas_w = win_w + 2 * margin
    canvas_h = win_h + 2 * margin

    # Transparent canvas for universal light/dark mode blending
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # Soft drop shadow
    shadow_mask = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_mask)
    sdraw.rounded_rectangle(
        [margin, margin + 4, margin + win_w, margin + win_h + 4],
        radius=radius,
        fill=(0, 0, 0, 140),
    )
    shadow = shadow_mask.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(shadow)

    # Window surface (Catppuccin Base #1e1e2e)
    win = Image.new("RGBA", (win_w, win_h), (0, 0, 0, 0))
    wdraw = ImageDraw.Draw(win)
    wdraw.rounded_rectangle(
        [0, 0, win_w, win_h], radius=radius, fill=(30, 30, 46, 255)
    )

    # Window title bar (Catppuccin Mantle #181825)
    header_mask = Image.new("L", (win_w, win_h), 0)
    hmask_draw = ImageDraw.Draw(header_mask)
    hmask_draw.rounded_rectangle([0, 0, win_w, win_h], radius=radius, fill=255)
    hmask_draw.rectangle([0, bar_h, win_w, win_h], fill=0)

    header = Image.new("RGBA", (win_w, win_h), (24, 24, 37, 255))
    win.paste(header, (0, 0), header_mask)
    wdraw.line([(0, bar_h), (win_w, bar_h)], fill=(49, 50, 68, 255), width=1)

    # macOS traffic lights
    btn_y = (bar_h - 12) // 2
    wdraw.ellipse(
        [16, btn_y, 16 + 12, btn_y + 12],
        fill=(255, 95, 86, 255),
        outline=(224, 68, 62, 255),
    )
    wdraw.ellipse(
        [36, btn_y, 36 + 12, btn_y + 12],
        fill=(255, 189, 46, 255),
        outline=(222, 161, 35, 255),
    )
    wdraw.ellipse(
        [56, btn_y, 56 + 12, btn_y + 12],
        fill=(39, 201, 63, 255),
        outline=(26, 171, 41, 255),
    )

    # Title with authentic Tiny🔥Torch flame branding
    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    fire_path = (
        REPO_ROOT
        / "tinytorch"
        / "guide"
        / "assets"
        / "images"
        / "logos"
        / "fire-emoji.png"
    )
    prefix = "Tiny"
    suffix = f"Torch · {title}"
    p_box = font.getbbox(prefix)
    pw = p_box[2] - p_box[0]
    ph = p_box[3] - p_box[1]
    s_box = font.getbbox(suffix)
    sw = s_box[2] - s_box[0]

    if fire_path.exists():
        fire_raw = Image.open(fire_path).convert("RGBA")
        fire_icon = fire_raw.resize((14, 14), Image.Resampling.LANCZOS)
        fw, fh = fire_icon.size
        gap = 2
        total_w = pw + gap + fw + gap + sw
        start_x = (win_w - total_w) // 2
        text_y = (bar_h - ph) // 2 - 1
        fire_y = (bar_h - fh) // 2

        wdraw.text(
            (start_x, text_y), prefix, fill=(205, 214, 244, 255), font=font
        )
        win.alpha_composite(fire_icon, (start_x + pw + gap, fire_y))
        wdraw.text(
            (start_x + pw + gap + fw + gap, text_y),
            suffix,
            fill=(205, 214, 244, 255),
            font=font,
        )
    else:
        full_title = f"TinyTorch · {title}"
        bbox = font.getbbox(full_title)
        tw = bbox[2] - bbox[0]
        tx = (win_w - tw) // 2
        ty = (bar_h - (bbox[3] - bbox[1])) // 2 - 1
        wdraw.text((tx, ty), full_title, fill=(205, 214, 244, 255), font=font)

    # 1px border outline
    wdraw.rounded_rectangle(
        [0, 0, win_w - 1, win_h - 1],
        radius=radius,
        outline=(69, 71, 90, 255),
        width=1,
    )
    canvas.alpha_composite(win, (margin, margin))

    term_x = margin + pad_x
    term_y = margin + bar_h + pad_y
    return canvas, term_x, term_y


def capture_screenshot(spec: dict) -> Path:
    """Capture a single command via VHS and save the composited PNG."""
    filename = spec["filename"]
    title = spec["title"]
    cmd = spec["cmd"]
    width = spec["width"]
    height = spec["height"]
    sleep_s = spec.get("sleep", 1.5)
    out_path = SCREENSHOTS_DIR / filename

    print(f"\n📸 Capturing: {filename} ({cmd})")
    prefix = f"vhs_snap_{spec['title'].replace(' ', '_')}_"
    frames_dir = Path(tempfile.mkdtemp(prefix=prefix))
    shutil.rmtree(frames_dir)

    setup_cmd = (
        f"source {REPO_ROOT}/tinytorch/guide/tools/tape_env.sh && "
        f"cd {REPO_ROOT}/tinytorch && "
        f"export PYTHONPATH={REPO_ROOT}/tinytorch && "
        f"export VIRTUAL_ENV={REPO_ROOT}/tinytorch/.venv && "
        "export PS1='' && clear"
    )

    tape_content = f"""
Output "{frames_dir}/"
Set Shell "bash"
Set FontSize 14
Set FontFamily "Menlo"
Set Width {width}
Set Height {height}
Set Padding 20
Set Theme "Catppuccin Mocha"

Hide
Type "{setup_cmd}"
Enter
Sleep 400ms
Show

Type "{cmd}"
Enter
Sleep {int(sleep_s * 1000)}ms
"""
    temp_tape = Path(tempfile.gettempdir()) / f"snap_{filename}.tape"
    temp_tape.write_text(tape_content, encoding="utf-8")

    try:
        env = os.environ.copy()
        env["PATH"] = f"{REPO_ROOT}/tinytorch/bin:{env.get('PATH', '')}"
        env["TITO_ALLOW_SYSTEM"] = "1"
        env["TINYTORCH_NON_INTERACTIVE"] = "1"
        env["TITO_NON_INTERACTIVE"] = "1"
        env["PYTHONPATH"] = f"{REPO_ROOT}/tinytorch"
        env["VIRTUAL_ENV"] = f"{REPO_ROOT}/tinytorch/.venv"

        proc = subprocess.run(
            ["vhs", str(temp_tape)],
            cwd=str(REPO_ROOT / "tinytorch"),
            env=env,
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            print(f"  ❌ VHS failed for {filename}: {proc.stderr}")
            sys.exit(1)

        text_frames = sorted(frames_dir.glob("frame-text-*.png"))
        if not text_frames:
            print(f"  ❌ No frames captured for {filename}!")
            sys.exit(1)

        # Select the final stable frame
        last_frame = Image.open(text_frames[-1]).convert("RGBA")
        canvas, term_x, term_y = create_window_template(
            last_frame.width, last_frame.height, title
        )
        canvas.alpha_composite(last_frame, (term_x, term_y))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        size_kb = out_path.stat().st_size / 1024
        print(
            f"  ✅ Saved: {filename} "
            f"({size_kb:.1f} KB, {canvas.width}x{canvas.height})"
        )
        return out_path

    finally:
        temp_tape.unlink(missing_ok=True)
        shutil.rmtree(frames_dir, ignore_errors=True)


def main():
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] if len(sys.argv) > 1 else None

    for spec in COMMANDS_TO_CAPTURE:
        fn = spec["filename"]
        st = spec["title"]
        if targets and fn not in targets and st not in targets:
            continue
        capture_screenshot(spec)

    print("\n🎉 All terminal screenshots successfully captured!")


if __name__ == "__main__":
    main()
