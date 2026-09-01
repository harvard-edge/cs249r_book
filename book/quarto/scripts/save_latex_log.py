#!/usr/bin/env python3
"""Post-render script: preserve authoritative LaTeX build intermediates.

Quarto deletes intermediate files after post-render scripts finish. Preserve
the log for diagnostics and the AUX for trustworthy mapped chapter builds.
"""
import re
import shutil
import subprocess
from pathlib import Path

OVERFULL_RE = re.compile(
    r"Overfull \\[hv]box \((\d+(?:\.\d+)?)pt too (?:wide|high)\).*?lines (\d+)"
)

VOLUME_OUTPUTS = {
    "vol1": "Machine-Learning-Systems-Vol1",
    "vol2": "Machine-Learning-Systems-Vol2",
}


def _active_volume(script_dir: Path) -> str | None:
    """Identify the active PDF volume from Binder's temporary config link."""
    config = script_dir / "_quarto.yml"
    if not config.exists():
        return None
    resolved_name = config.resolve().name.lower()
    return next((volume for volume in VOLUME_OUTPUTS if volume in resolved_name), None)


def _find_intermediate(script_dir: Path, stem: str, suffix: str) -> Path | None:
    """Find a Quarto/LaTeX intermediate before post-render cleanup."""
    for candidate in (script_dir / f"{stem}{suffix}", script_dir / f"index{suffix}"):
        if candidate.is_file():
            return candidate
    return next(iter(script_dir.glob(f".quarto/**/{stem}{suffix}")), None) or next(
        iter(script_dir.glob(f".quarto/**/index{suffix}")), None
    )


def _regenerate_auxiliary_files(
    script_dir: Path, stem: str, *, passes: int = 3
) -> bool:
    """Recreate converged AUX/log metadata from retained full-volume TeX.

    A single pass is not authoritative for a book: the first run creates the
    TOC, and only later runs paginate the body with that TOC present and settle
    cross-references. Draft mode avoids replacing the production PDF while
    preserving the same pagination inputs.
    """
    tex_path = script_dir / f"{stem}.tex"
    if not tex_path.is_file():
        return False
    command = [
        "lualatex",
        "-draftmode",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    for _ in range(passes):
        completed = subprocess.run(
            command,
            cwd=script_dir,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        if completed.returncode != 0:
            return False
    return True

def main():
    script_dir = Path(__file__).resolve().parent.parent  # quarto/
    volume = _active_volume(script_dir)
    if volume is None:
        print("[latex-build] No active Volume I/II PDF config found")
        return

    stem = VOLUME_OUTPUTS[volume]
    out_dir = script_dir / "_build" / f"pdf-{volume}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quarto 1.10 removes AUX/log files before project post-render hooks even
    # when keep-tex is enabled. A draft-mode pass over the exact retained TeX
    # recreates numbering metadata without writing or replacing the book PDF.
    if _find_intermediate(script_dir, stem, ".aux") is None:
        if _regenerate_auxiliary_files(script_dir, stem):
            print(
                "[latex-build] Regenerated converged AUX/log from retained TeX "
                "(3 draft-mode passes)"
            )
        else:
            print("[latex-build] WARNING: draft-mode AUX regeneration failed")

    log_src = _find_intermediate(script_dir, stem, ".log")
    if log_src is None:
        print("[latex-build] No LaTeX log found (non-PDF build or already cleaned)")
    else:
        log_dst = out_dir / "latex-build.log"
        shutil.copy2(log_src, log_dst)
        print(f"[latex-build] Saved {log_src.name} -> {log_dst.relative_to(script_dir)}")

        text = log_src.read_text(errors="replace")
        severe = [
            (float(match.group(1)), int(match.group(2)))
            for match in OVERFULL_RE.finditer(text)
            if float(match.group(1)) >= 20.0
        ]
        if severe:
            print(f"[latex-build] WARNING: {len(severe)} severe layout overflows (>= 20pt)")
            for points, line in sorted(severe, key=lambda item: -item[0])[:5]:
                print(f"[latex-build]   {points:.1f}pt overflow at .tex line {line}")
        else:
            print("[latex-build] No severe layout overflows detected")

    aux_src = _find_intermediate(script_dir, stem, ".aux")
    if aux_src is None:
        print("[latex-build] WARNING: authoritative full-build AUX not found")
    else:
        aux_dst = out_dir / f"{stem}.aux"
        shutil.copy2(aux_src, aux_dst)
        print(f"[latex-build] Saved {aux_src.name} -> {aux_dst.relative_to(script_dir)}")

if __name__ == "__main__":
    main()
