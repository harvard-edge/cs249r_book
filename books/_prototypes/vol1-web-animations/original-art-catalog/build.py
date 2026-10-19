#!/usr/bin/env python3
"""Build a source-controlled rollout catalog and a disposable thumbnail preview."""
import argparse
from collections import Counter
from hashlib import sha1
import json
from pathlib import Path
import re
import shutil
import subprocess
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BOOKS = ROOT / "books"
STATUSES = {
    "accepted": "Accepted prototype",
    "reviewed": "Agent-reviewed implementation",
    "implementing": "Original-art work · awaiting rendered review",
    "adapt": "Selected mechanism · needs original-art adaptation",
    "static": "Static by default",
}


def implementation_review_current(key, record, registry, implementation, art, build_root=None):
    """Review evidence expires whenever source, runtime, artwork, or build changes."""
    if not record or record.get("human_student_testing") is not False:
        return False
    entry = registry.get(key, {})
    fields = ("runtime_sha1", "source_sha1", "art_sha1", "art_structural_sha1")
    if any(record.get(field) != entry.get(field) for field in fields) or not all(record.get(field) for field in fields[:3]):
        return False
    reviewed_build = ROOT / record.get("build_root", "")
    if not reviewed_build.is_dir() or (build_root and reviewed_build.resolve() != build_root.resolve()):
        return False
    source = ROOT / entry["source"]
    if not source.is_file():
        return False
    normalized = re.sub(r'\s+data-explore(?:-key)?="[^"]*"', '', source.read_text())
    if sha1(normalized.encode()).hexdigest() != entry["source_sha1"]:
        return False
    extension = BOOKS / "_extensions/mlsysbook-ext/figure-explore"
    paths = [extension / implementation[field] for field in ("script", "stylesheet", "bundle") if implementation.get(field)]
    if implementation.get("shared_player"):
        paths.extend(extension / name for name in ("original-player.js", "original-player.css"))
    if not all(path.is_file() for path in paths):
        return False
    assets = [path.read_bytes() for path in paths]
    if sha1(b'\x00'.join(assets)).hexdigest() != entry["runtime_sha1"]:
        return False
    asset = Path(art.get(key, {}).get("asset", ""))
    if not asset.is_file() or sha1(asset.read_bytes()).hexdigest() != entry["art_sha1"]:
        return False
    if not (reviewed_build / "summary.json").is_file():
        return False
    summary = json.loads((reviewed_build / "summary.json").read_text())
    jobs = [j for j in summary.get("jobs", []) if j.get("volume") == f"vol{entry['volume']}" and j.get("chapter") == source.parent.name and j.get("format") == "html"]
    return any(j.get("ok") is True and j.get("returncode") == 0 for j in jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--art-index", type=Path, default=Path("/tmp/figures-animation-art-current/index.json"))
    parser.add_argument("--manifest", type=Path, default=BOOKS / "_extensions/mlsysbook-ext/figure-explore/mechanisms/implementations.json")
    parser.add_argument("--build-root", type=Path, help="Optional successful chapter-build run used only by the disposable preview")
    args = parser.parse_args()
    catalog = json.loads((HERE.parent / "animation-catalog/catalog.json").read_text())
    selected = {x["key"]: x for x in json.loads((BOOKS / "_extensions/mlsysbook-ext/figure-explore/mechanisms/selection.json").read_text())}
    review = json.loads((HERE / "review-data.json").read_text())
    current = set(review.get("originals_in_progress", []))
    manifest = {}
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        items = manifest if isinstance(manifest, list) else manifest.get("figures", manifest.get("originals", list(manifest)))
        current.update(x if isinstance(x, str) else x["key"] for x in items)
    art = json.loads(args.art_index.read_text()) if args.art_index.exists() else {}
    registry = json.loads((BOOKS / "_extensions/mlsysbook-ext/figure-explore/figure-registry.json").read_text())
    reviews = review.get("implementation_reviews", {})
    reviewed = {key for key in current if key in manifest and implementation_review_current(key, reviews.get(key), registry, manifest[key], art, args.build_root)}
    output = BOOKS / "_build/original-art-catalog"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    figures = []
    ready = {}
    if args.build_root:
        build_root = args.build_root.resolve()
        build_root.relative_to(BOOKS.resolve())
        summary = json.loads((build_root / "summary.json").read_text())
        for job in summary.get("jobs", []):
            if job.get("ok") is not True or job.get("returncode") != 0 or job.get("format") != "html":
                continue
            for original in catalog["figures"]:
                if original["key"] not in manifest:
                    continue
                source = Path(original["source"])
                if job.get("volume") != f"vol{original['volume']}" or job.get("chapter") != original["chapter"]:
                    continue
                chapter = build_root / job["job"] / "output" / source.relative_to("books").with_suffix(".html")
                if not chapter.is_file():
                    continue
                soup = BeautifulSoup(chapter.read_text(), "html.parser")
                figure = soup.find(id=original["figure_id"])
                expected = manifest[original["key"]]
                if figure is None or figure.get("data-mlsys-explore") != expected["name"] or figure.get("data-mlsys-explore-key") != original["key"]:
                    continue
                if not any(expected["script"] in x.get("src", "") for x in soup.find_all("script", src=True)):
                    continue
                ready[original["key"]] = "/" + chapter.relative_to(BOOKS).as_posix() + "?view=explore#" + original["figure_id"]
    for original in catalog["figures"]:
        key = original["key"]
        status = "accepted" if key == "vol1:fig-cnn" else "reviewed" if key in reviewed else "implementing" if key in current else "adapt" if key in selected else "static"
        image = art.get(key, {}).get("asset") or art.get(key, {}).get("image")
        # Prefer the indexed canonical render; source SVG/PNG is a fallback only.
        if not image or not Path(image).is_file():
            image = next((str((ROOT / original["source"]).parent / x) for x in original.get("assets", []) if ((ROOT / original["source"]).parent / x).suffix.lower() in {".svg", ".png", ".jpg", ".jpeg", ".webp"} and ((ROOT / original["source"]).parent / x).is_file()), None)
        thumbnail = ""
        if image and Path(image).is_file():
            dest = assets / (key.replace(":", "--") + Path(image).suffix.lower())
            shutil.copy2(image, dest)
            thumbnail = "/_build/original-art-catalog/assets/" + dest.name
        recommendation = review.get("recommendations", {}).get(key, {})
        selection = selected.get(key, {})
        figures.append({
            "key": key, "volume": original["volume"], "chapter": original["chapter"],
            "chapter_order": original["chapter_order"], "chapter_title": original["chapter_title"].split(" {#")[0],
            "id": original["figure_id"], "title": original["title"], "caption": original["caption"],
            "source": original["source"], "line": original["line"], "status": status,
            "purpose": recommendation.get("purpose") or selection.get("question") or "Retain the complete original figure. Add interaction only when a specific learning question justifies it.",
            "storyboard": recommendation.get("storyboard", []), "guard": recommendation.get("guard", ""),
            "thumbnail": thumbnail, "art_status": "Canonical asset available; not newly visually approved" if thumbnail else "Canonical asset not available in this local build",
            "chapter_animation": ready.get(key, ""),
        })
    figures.sort(key=lambda x: (x["volume"], x["chapter_order"], x["line"]))
    counts = dict(Counter(x["status"] for x in figures))
    payload = {"figures": figures, "counts": counts, "statuses": STATUSES, "volumes": dict(Counter(str(x["volume"]) for x in figures)), "revision": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip(), "inventory_revision": catalog["metadata"]["source_head"], "selected": len(selected), "images": sum(bool(x["thumbnail"]) for x in figures)}
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    template = (HERE / "template.html").read_text()
    page = template.replace("__CATALOG_DATA__", data)
    if not args.build_root:
        (HERE / "index.html").write_text(page)
    (output / "index.html").write_text(page)
    print(json.dumps({"figures": len(figures), "status_counts": counts, "volumes": payload["volumes"], "thumbnails": payload["images"], "verified_chapter_links": len(ready), "output": str(output / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
