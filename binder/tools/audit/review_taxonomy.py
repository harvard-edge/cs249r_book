"""Preview or apply a source-backed correction review to a draft taxonomy.

Usage: python3 binder/tools/audit/review_taxonomy.py REVIEW.json OUTPUT_DIR [--apply]
The output retains the original and candidate files. Application validates first
and restores the source files on an ordinary write/validation exception. It is not
a crash-atomic multi-file transaction; retained originals permit manual recovery.
"""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "binder/cli"))
from core import taxonomy_schema as schema
from core.concept_graph import load_graph
from core.field_graph import load_field, load_taxonomy_yaml
from core.taxonomy.editorial import apply_review, advance_provenance
from core.taxonomy.records import compute_implementation_sha256

FILES = ("taxonomy.yml", "provenance.yml")


class LiteralDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def dump(doc):
    return yaml.dump(doc, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True, width=120).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(root, review):
    schema.validate_term_provenance_sidecar(root)
    taxonomy = root / "taxonomy"
    before = {name: (taxonomy / name).read_bytes() for name in FILES}
    doc = load_taxonomy_yaml(taxonomy / FILES[0])
    provenance = load_taxonomy_yaml(taxonomy / FILES[1])
    candidate = apply_review(doc, review)
    # Preserve the source's introductory comments without reformatting their text.
    header = before[FILES[0]].split(b"schema_version:", 1)[0]
    raw = header + dump(candidate)
    entry = {
        "mode": "editorial-review",
        "campaign_id": "editorial-" + digest(schema.canonical_taxonomy_json(review).encode()),
        "baseline_taxonomy_sha256": digest(before[FILES[0]]),
        "post_apply_taxonomy_sha256": digest(raw),
        "baseline_non_guides_taxonomy_sha256": schema.compute_non_guides_taxonomy_sha256(doc),
        "post_non_guides_taxonomy_sha256": schema.compute_non_guides_taxonomy_sha256(candidate),
        "baseline_term_provenance_sha256": digest(before[FILES[1]]),
        "prior_sidecar_content_sha256": provenance["sidecar_content_sha256"],
        "previous_campaign_entry_sha256": provenance["applied_campaigns"][-1]["campaign_entry_sha256"],
        "taxonomy_schema_script_sha256": digest(Path(schema.__file__).read_bytes()),
        "taxonomy_implementation_sha256": compute_implementation_sha256(root),
        "replayed_patch": deepcopy(review),
    }
    entry["campaign_entry_sha256"] = schema.compute_campaign_entry_sha256(entry)
    replay = schema.load_genesis_baseline_taxonomy(root)
    state = schema.init_genesis_provenance_state(replay)
    burned = schema._load_burned_clusters(root)
    for old in provenance["applied_campaigns"]:
        previous = deepcopy(replay)
        if old["mode"] == "editorial-review":
            replay = apply_review(replay, old["replayed_patch"])
        else:
            schema.replay_patch_on_taxonomy_doc(replay, old["replayed_patch"], old["mode"])
        schema.replay_provenance_step(state, previous, replay, old, burned_doc=burned)
    advance_provenance(state, entry)
    after_prov = schema.build_provenance_doc_from_state(
        state, taxonomy_version=candidate["version"], taxonomy_sha256=digest(raw),
        non_guides_taxonomy_sha256=entry["post_non_guides_taxonomy_sha256"],
        applied_campaigns=provenance["applied_campaigns"] + [entry],
    )
    after = {FILES[0]: raw, FILES[1]: dump(after_prov)}
    return before, after


def validate_candidate(root, after):
    with tempfile.TemporaryDirectory(prefix="taxonomy-review-") as temporary:
        stage = Path(temporary)
        shutil.copytree(root / "taxonomy", stage / "taxonomy", ignore=shutil.ignore_patterns(
            "downloads", "field-focus", "*.svg", "*.html", "*_files", "__pycache__"))
        (stage / "books").symlink_to(root / "books", target_is_directory=True)
        for name, raw in after.items():
            (stage / "taxonomy" / name).write_bytes(raw)
        schema.validate_term_provenance_sidecar(stage)
        # Validate source paths in their real checkout, then adapt display paths
        # for the candidate root. Companion-resource safety remains enforced.
        graph = load_graph(root)
        for chapter in graph["chapters"].values():
            chapter["source"] = stage / chapter["source"].relative_to(root)
        load_field(stage, graph)


def install(root, before, after):
    paths = {name: root / "taxonomy" / name for name in FILES}
    if any(paths[name].read_bytes() != before[name] for name in FILES):
        raise ValueError("Source changed after preview; prepare a fresh review")
    try:
        for name in FILES:
            with tempfile.NamedTemporaryFile(dir=paths[name].parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(after[name])
            try:
                os.replace(temporary, paths[name])
            finally:
                temporary.unlink(missing_ok=True)
        schema.validate_term_provenance_sidecar(root)
    except Exception:
        for name in FILES:
            paths[name].write_bytes(before[name])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    review = schema.json_loads_strict(args.review.read_text(), str(args.review))
    before, after = prepare(ROOT, review)
    validate_candidate(ROOT, after)
    args.output.mkdir(parents=True, exist_ok=False)
    for name in FILES:
        (args.output / ("original-" + name)).write_bytes(before[name])
        (args.output / name).write_bytes(after[name])
    if args.apply:
        install(ROOT, before, after)
    print("Applied validated review." if args.apply else "Validated preview; canonical files unchanged.")
    print(f"Candidate and originals: {args.output}")


if __name__ == "__main__":
    main()
