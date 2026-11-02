"""CLI entrypoint for `binder concepts` (`build`, `check`, `refresh`, `audit`, `coverage`, `agreement`).

Compiles and validates the ML Systems Taxonomy (`taxonomy/ml-systems-taxonomy.yml`) and
chapter concept maps against the LinkML metamodel (`taxonomy/schema/mlsys-taxonomy.linkml.yml`).
"""

import argparse
import yaml
from pathlib import Path

from cli.core.concept_graph import ConceptGraphError, build, refresh


def run(root: Path, args: list[str]) -> bool:
    """Execute a `binder concepts` subcommand and return `True` if all checks pass."""
    parser = argparse.ArgumentParser(prog="binder concepts", description=__doc__)
    parser.add_argument("action", choices=("build", "check", "refresh", "audit", "coverage", "agreement"))
    group = parser.add_mutually_exclusive_group()
    for number in range(1, 5):
        group.add_argument(
            f"--vol{number}", action="store_const", const=f"vol{number}", dest="volume"
        )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail for stale generated assets (build), unresolved primary declarations (audit), invalid checklist judgments (coverage), or tags citing retired identities (agreement)",
    )
    try:
        options = parser.parse_args(args)
        if options.check and options.action not in {"build", "audit", "coverage", "agreement"}:
            parser.error("--check is only supported with build, audit, coverage, and agreement")
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
        if options.action == "coverage":
            from cli.core.taxonomy_coverage import write_report

            report = write_report(root)
            for c in report["checklists"]:
                s = c["summary"]
                tag = "held out" if c["held_out"] else "design input"
                print(f"{c['title']} [{tag}]: {s['covered_pct']}% covered, {s['homed_pct']}% homed "
                      f"({s['counts']['covered']} covered, {s['counts']['partial']} partial, {s['counts']['missing']} missing of {s['in_scope']} in scope)")
            for error in report["errors"]:
                print(f"  invalid: {error}")
            print("Report: books/_build/concepts/taxonomy-coverage.md")
            return not (options.check and report["errors"])
        if options.action == "agreement":
            from cli.core.taxonomy_agreement import write_report as write_agreement

            report = write_agreement(root)
            for record in report["records"]:
                for run in record["runs"]:
                    s = run["scores"]
                    tag = "held out" if run["held_out"] else "rules revised on this sample"
                    print(f"{record['title']} {run['id']} [{tag}]: subject {s['subject_pct']}% (kappa {s['subject_kappa']}), "
                          f"area {s['area_pct']}% (kappa {s['area_kappa']}), primary concept {s['primary_concept_pct']}%, "
                          f"same concept set {s['same_concept_set_pct']}%, Hooper {s['hooper_consistency_pct']}%, Rolling {s['rolling_consistency_pct']}%, compatible {s['compatible_pct']}% of {s['items']}")
            for error in report["errors"]:
                print(f"  invalid: {error}")
            print("Report: books/_build/concepts/taxonomy-agreement.json")
            return not (options.check and report["errors"])
        if options.action == "refresh":
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
                if field.get("framework") and field["framework"].get("version") not in (None, "0.1-draft"):
                    from cli.core.taxonomy_release import release_artifacts
                    rel_artifacts = release_artifacts(root, field["framework"], field)
                    staffml_json = root / "staffml/app/src/data/mlsys-taxonomy.json"
                    if staffml_json in rel_artifacts:
                        if not staffml_json.exists() or staffml_json.read_bytes() != rel_artifacts[staffml_json]:
                            raise ConceptGraphError(
                                "staffml/app/src/data/mlsys-taxonomy.json is missing or out of sync with taxonomy/ml-systems-taxonomy.yml; run './binder/binder concepts build'"
                            )
            if (root / "taxonomy/coverage-ledger.yml").exists() and not volumes:
                from cli.core.concept_coverage import coverage_report as ledger_coverage_report
                ledger_rep = ledger_coverage_report(root, graph=graph, field=field)
                if ledger_rep.get("orphaned_decisions"):
                    raise ConceptGraphError(
                        f"coverage: orphaned decisions in coverage-ledger.yml: {', '.join(ledger_rep['orphaned_decisions'])}"
                    )
                broken_ledger = [
                    item["inventory_id"]
                    for item in ledger_rep.get("inventory", [])
                    if any(
                        r in ("field-target-removed", "declaration-mismatch")
                        or r.startswith("manuscript-evidence-changed:")
                        for r in item.get("stale_reasons", [])
                    )
                ]
                if broken_ledger:
                    raise ConceptGraphError(
                        f"coverage: broken references in coverage-ledger.yml: {', '.join(broken_ledger)}"
                    )
            if (root / "taxonomy/coverage").is_dir():
                from cli.core.taxonomy_coverage import coverage_report
                cov = coverage_report(root)
                if cov["errors"]:
                    for err in cov["errors"]:
                        print(f"  invalid coverage: {err}")
                    return False
            if (root / "taxonomy/agreement").is_dir():
                from cli.core.taxonomy_agreement import agreement_report
                agr = agreement_report(root)
                if agr["errors"]:
                    for err in agr["errors"]:
                        print(f"  invalid agreement: {err}")
                    return False
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
