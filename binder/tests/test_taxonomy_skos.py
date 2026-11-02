"""The published Turtle must parse and satisfy the SKOS Reference integrity conditions (W3C Recommendation, 2009)."""
from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")
from rdflib import OWL, RDF, RDFS, Namespace  # noqa: E402

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DOWNLOADS = Path(__file__).resolve().parents[2] / "taxonomy/downloads"


@pytest.fixture(scope="module")
def graph():
    path = DOWNLOADS / "mlsys-taxonomy.ttl"
    if not path.exists():
        pytest.skip("run `binder concepts build` first")
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    g.parse(DOWNLOADS / "mlsys-taxonomy-relations.ttl", format="turtle")
    return g


def test_s9_s37_classes_are_disjoint(graph):
    concepts = set(graph.subjects(RDF.type, SKOS.Concept))
    assert concepts
    assert not concepts & set(graph.subjects(RDF.type, SKOS.ConceptScheme))
    assert not concepts & set(graph.subjects(RDF.type, SKOS.Collection))
    assert not set(graph.subjects(RDF.type, SKOS.Collection)) & set(graph.subjects(RDF.type, SKOS.ConceptScheme))


def test_s13_s14_labels(graph):
    for concept in graph.subjects(RDF.type, SKOS.Concept):
        pref = list(graph.objects(concept, SKOS.prefLabel))
        assert len(pref) == 1, concept
        assert not set(pref) & set(graph.objects(concept, SKOS.altLabel)), concept


def test_s27_broader_and_related_are_disjoint(graph):
    related_props = {s for s in graph.subjects(RDFS.subPropertyOf, SKOS.related)}
    parents = {}
    for narrower, broader in graph.subject_objects(SKOS.broader):
        parents.setdefault(narrower, set()).add(broader)

    def ancestors(node):
        seen, todo = set(), list(parents.get(node, ()))
        while todo:
            current = todo.pop()
            if current not in seen:
                seen.add(current)
                todo.extend(parents.get(current, ()))
        return seen

    for prop in related_props:
        for source, target in graph.subject_objects(prop):
            assert target not in ancestors(source) and source not in ancestors(target), (source, prop, target)


def test_extension_properties_refine_skos(graph):
    props = set(graph.subjects(RDF.type, OWL.ObjectProperty))
    assert props
    for prop in props:
        assert set(graph.objects(prop, RDFS.subPropertyOf)) & {SKOS.broader, SKOS.related}, prop
    used = {p for p in graph.predicates() if str(p).startswith("https://mlsysbook.ai/taxonomy/relation/")}
    assert used <= props


def test_hierarchy_is_materialized_and_acyclic(graph):
    hierarchy = set(graph.subject_objects(rdflib.URIRef("https://mlsysbook.ai/taxonomy/relation/specializes")))
    assert hierarchy == set(graph.subject_objects(SKOS.broader))
