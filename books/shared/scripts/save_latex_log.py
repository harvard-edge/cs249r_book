#!/usr/bin/env python3
"""Post-render script: preserve authoritative LaTeX build intermediates.

Quarto deletes intermediate files after post-render scripts finish. Preserve
the log for diagnostics and the AUX for trustworthy mapped chapter builds.
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

OVERFULL_RE = re.compile(
    r"Overfull \\[hv]box \((\d+(?:\.\d+)?)pt too (?:wide|high)\).*?lines (\d+)"
)

VOLUME_OUTPUTS = {
    "vol1": "Machine-Learning-Systems-Vol1",
    "vol2": "Machine-Learning-Systems-Vol2",
    "vol3": "Machine-Learning-Systems-Vol3",
    "vol4": "Machine-Learning-Systems-Vol4",
}


def _active_volume(script_dir: Path) -> str | None:
    """Identify the active PDF volume from the source Binder recorded in _quarto.yml."""
    config = script_dir / "_quarto.yml"
    if not config.is_file():
        return None
    text = config.read_text(encoding="utf-8").lower()
    source = text.splitlines()[0] if text else ""
    recorded = re.search(r"_quarto-pdf-(vol[1-4])\.yml", source)
    if recorded:
        return recorded.group(1)
    # Container builds copy the canonical YAML directly, without Binder's
    # provenance comment. Its output directory still identifies the PDF volume.
    direct = re.search(r"^\s*output-dir:\s*['\"]?_build/pdf-(vol[1-4])['\"]?\s*$", text, re.MULTILINE)
    return direct.group(1) if direct else None


def _active_target(script_dir: Path, volume: str) -> tuple[str, Path]:
    """Honor Binder's selective-build stem and output directory."""
    config = yaml.safe_load((script_dir / "_quarto.yml").read_text(encoding="utf-8")) or {}
    stem = str(config.get("book", {}).get("output-file", VOLUME_OUTPUTS[volume]))
    output = (
        os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
        or config.get("project", {}).get("output-dir")
        or f"_build/pdf-{volume}"
    )
    return Path(stem).stem, script_dir / output


def _find_intermediate(script_dir: Path, stem: str, suffix: str) -> Path | None:
    """Find a Quarto/LaTeX intermediate before post-render cleanup."""
    candidates = [script_dir / f"{stem}{suffix}", script_dir / f"index{suffix}"]
    candidates.extend(script_dir.glob(f".quarto/**/{stem}{suffix}"))
    candidates.extend(script_dir.glob(f".quarto/**/index{suffix}"))
    return max((path for path in candidates if path.is_file()),
               key=lambda path: path.stat().st_mtime_ns, default=None)


def _is_current_for_tex(intermediate: Path | None, tex_path: Path) -> bool:
    """Return whether an intermediate was produced from the retained TeX."""
    return bool(
        intermediate
        and intermediate.is_file()
        and intermediate.with_suffix(".tex") == tex_path
        and tex_path.is_file()
        and intermediate.stat().st_mtime_ns >= tex_path.stat().st_mtime_ns
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
    script_dir = Path(__file__).resolve().parents[2]  # books/shared/scripts -> books/
    volume = _active_volume(script_dir)
    if volume is None:
        print("[latex-build] No active volume PDF config found")
        return

    stem, out_dir = _active_target(script_dir, volume)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quarto 1.10 can remove current AUX/log files before project post-render
    # hooks even when keep-tex is enabled. A stale file from an earlier build
    # is equally unsafe. Regenerate unless the discovered AUX is at least as
    # new as the exact retained TeX.
    tex_path = _find_intermediate(script_dir, stem, ".tex") or script_dir / f"{stem}.tex"
    aux_before = tex_path.with_suffix(".aux")
    regenerated = True
    if not _is_current_for_tex(aux_before, tex_path):
        if _regenerate_auxiliary_files(tex_path.parent, tex_path.stem):
            print(
                "[latex-build] Regenerated converged AUX/log from retained TeX "
                "(3 draft-mode passes)"
            )
        else:
            regenerated = False
            print("[latex-build] WARNING: draft-mode AUX regeneration failed")

    log_src = tex_path.with_suffix(".log")
    if not log_src.is_file():
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

    if not regenerated:
        raise RuntimeError("Could not regenerate current PDF AUX from retained TeX")

    aux_src = tex_path.with_suffix(".aux")
    if not _is_current_for_tex(aux_src, tex_path):
        print("[latex-build] WARNING: current authoritative AUX not found")
    else:
        aux_dst = out_dir / f"{stem}.aux"
        shutil.copy2(aux_src, aux_dst)
        print(f"[latex-build] Saved {aux_src.name} -> {aux_dst.relative_to(script_dir)}")

if __name__ == "__main__":
    main()
