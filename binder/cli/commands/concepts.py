"""Repeatable chapter concept-map builds and local discovery inventories."""

import argparse
import yaml
from pathlib import Path

from cli.core.concept_graph import ConceptGraphError, build, refresh


def run(root: Path, args: list[str]) -> bool:
    parser = argparse.ArgumentParser(prog="binder concepts", description=__doc__)
    parser.add_argument("action", choices=("build", "check", "refresh", "audit"))
    group = parser.add_mutually_exclusive_group()
    for number in range(1, 5):
        group.add_argument(
            f"--vol{number}", action="store_const", const=f"vol{number}", dest="volume"
        )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail for stale generated assets (build), or unresolved primary declarations (audit)",
    )
    try:
        options = parser.parse_args(args)
        if options.check and options.action not in {"build", "audit"}:
            parser.error("--check is only supported with build and audit")
    except SystemExit as exc:
        return exc.code == 0
    volumes = (options.volume,) if options.volume else ()
    try:
        if options.action == "audit":
            from cli.core.concept_coverage import write_coverage_report

            report = write_coverage_report(root, volumes)
            for volume, counts in report["by_volume"].items():
                print(f"{volume}: {counts['declarations']} declarations; {counts['unresolved_primary']} unresolved primary; statuses={counts['statuses']}")
            print(f"Orphaned decisions: {len(report['orphaned_decisions'])}")
            print("Human reconciliation gate: " + ("passed" if report["gate_passed"] else "not passed"))
            print(report["gate_meaning"])
            print("Reports: books/_build/concepts/coverage-audit*.json and .md")
            return not options.check or report["gate_passed"]
        if options.action == "refresh":
            if options.check:
                parser.error("--check is only supported with build and audit")
            report = refresh(root, volumes)
            print(
                f"Concept review: {len(report['candidates'])} concept candidates, "
                f"{len(report['changed_evidence'])} changed evidence sections, {len(report['invalid'])} invalid evidence links"
            )
            print(
                "Report: books/_build/concepts/review.json (candidate inventory; authored YAML is unchanged)"
            )
            return not report["invalid"]
        if options.action == "check":
            from cli.core.concept_graph import load_graph

            graph = load_graph(root, volumes)
            from cli.core.field_graph import load_field
            field = load_field(root, load_graph(root) if volumes else graph)
            print(
                f"Concept graph valid: {len(graph['nodes'])} concepts, {len(graph['edges'])} relationships, {len(graph['maps'])} maps"
            )
            if field:
                summary = field["summary"]
                print(f"Field reference draft valid: {summary['concepts']} concepts, {summary['relationships']} relationships, {summary['book_links']} candidate book links")
            return True
        result = build(root, volumes, check=options.check)
        print(
            f"Concept maps: {result['maps']} maps, {result['concepts']} concepts, "
            f"{result['relationships']} relationships; catalog covers {result['chapters']} chapters"
        )
        if result.get("field"):
            summary = result["field"]
            print(f"Field reference draft: {summary['concepts']} concepts, {summary['relationships']} relationships, {summary['book_links']} candidate book links")
        if options.check and result["stale"]:
            print("Generated assets are stale:\n" + "\n".join(result["stale"]))
            return False
        return True
    except (ConceptGraphError, OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"Concept maps: {exc}")
        return False
