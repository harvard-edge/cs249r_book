"""Keep ownership boundaries and compatibility explicit during future changes."""

import ast
from pathlib import Path

from core import taxonomy_schema as compatibility
from core.taxonomy import schema
from core.taxonomy.records import compute_implementation_sha256

PACKAGE = Path(schema.__file__).parent


def test_existing_schema_imports_keep_the_same_function_objects():
    assert compatibility.validate_linkml is schema.validate_linkml
    assert compatibility.compile_json_schema is schema.compile_json_schema
    assert (
        compatibility.validate_term_provenance_sidecar
        is schema.validate_term_provenance_sidecar
    )


def test_package_imports_are_acyclic_and_do_not_depend_on_compatibility_shims():
    names = {p.stem for p in PACKAGE.glob("*.py")}
    graph = {}
    for path in PACKAGE.glob("*.py"):
        graph[path.stem] = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            assert not (
                node.level == 2
                and node.module
                in {"taxonomy_schema", "taxonomy_editorial", "taxonomy_release"}
            )
            if node.level == 1 and node.module in names:
                graph[path.stem].add(node.module)

    def visit(name, ancestors):
        assert name not in ancestors, f"Import cycle: {ancestors} -> {name}"
        for dependency in graph[name]:
            visit(dependency, ancestors + [name])

    for name in graph:
        visit(name, [])


def test_implementation_fingerprint_covers_package_content_and_paths(tmp_path):
    core = tmp_path / "binder/cli/core"
    package = core / "taxonomy"
    package.mkdir(parents=True)
    (core / "taxonomy_schema.py").write_text("# compatibility\n")
    module = package / "schema.py"
    module.write_text("rule = 1\n")
    original = compute_implementation_sha256(tmp_path)
    module.write_text("rule = 2\n")
    edited = compute_implementation_sha256(tmp_path)
    assert original != edited
    module.rename(package / "rules.py")
    assert edited != compute_implementation_sha256(tmp_path)


def test_historical_estimators_keep_their_metamorphic_checks():
    from core.taxonomy.metrics import run_stage2_estimator_self_tests

    run_stage2_estimator_self_tests.cache_clear()
    assert run_stage2_estimator_self_tests()


def test_package_split_does_not_relax_reviewed_release_gate():
    import pytest
    from core.concept_graph import ConceptGraphError
    from core.taxonomy.governance import verify_release_governance_receipt

    with pytest.raises(ConceptGraphError, match="release_governance_receipt"):
        verify_release_governance_receipt(None, "1.0", "0" * 64)
    with pytest.raises(ConceptGraphError, match="must carry"):
        verify_release_governance_receipt(None, "0.5", "0" * 64)
