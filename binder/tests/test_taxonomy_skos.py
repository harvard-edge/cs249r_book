"""The published Turtle must parse and satisfy the SKOS Reference integrity conditions (W3C Recommendation, 2009)."""
from pathlib import Path

import pytest

rdflib = pytest.importorskip("rdflib")
from rdflib import OWL, RDF, RDFS, Namespace  # noqa: E402

SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DOWNLOADS = Path(__file__).resolve().parents[2] / "taxonomy/downloads"


@pytest.fixture(scope="module")
def release():
    from core.field_graph import load_unified_taxonomy
    from core.framework import load_framework
    from core.taxonomy_release import release_artifacts

    root = DOWNLOADS.parents[1]
    bundle = load_unified_taxonomy(root)
    field = bundle["field"]
    framework = load_framework(root, field)
    artifacts = release_artifacts(root, framework, field)
    return field, artifacts


@pytest.fixture(scope="module")
def graph(release):
    _, artifacts = release
    g = rdflib.Graph()
    for name in ('mlsys-taxonomy.ttl', 'mlsys-taxonomy-relations.ttl'):
        g.parse(data=artifacts[DOWNLOADS / name].decode(), format="turtle")
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
    related_props = {SKOS.related} | {s for s in graph.subjects(RDFS.subPropertyOf, SKOS.related)}
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
    used |= {p for p in graph.objects(None, RDF.predicate) if str(p).startswith("https://mlsysbook.ai/taxonomy/relation/")}
    assert used <= props


def test_hierarchy_is_materialized_and_acyclic(graph):
    hierarchy = set(graph.subject_objects(rdflib.URIRef("https://mlsysbook.ai/taxonomy/relation/specializes")))
    assert hierarchy == set(graph.subject_objects(SKOS.broader))


def test_qualified_relationships_preserve_all_source_conditions(graph, release):
    field, _ = release
    dct = Namespace("http://purl.org/dc/terms/")
    meta = Namespace("https://mlsysbook.ai/taxonomy/scheme#")
    base = "https://mlsysbook.ai/taxonomy/"
    statements = list(graph.subjects(RDF.type, RDF.Statement))
    by_id = {str(graph.value(node, dct.identifier)): node for node in statements}
    assert len(by_id) == len(statements) == len(field['relationships'])
    for edge in field['relationships']:
        node = by_id[edge['id']]
        source = rdflib.URIRef(base + 'concept/' + edge['source'])
        target = rdflib.URIRef(base + 'concept/' + edge['target'])
        predicate = rdflib.URIRef(base + 'relation/' + edge['relation'])
        assert list(graph.objects(node, RDF.subject)) == [source]
        assert list(graph.objects(node, RDF.predicate)) == [predicate]
        assert list(graph.objects(node, RDF.object)) == [target]
        assert list(graph.objects(node, meta.condition)) == [rdflib.Literal(edge['conditions'], lang='en')]
        assert graph.value(node, RDFS.label) == rdflib.Literal(edge['predicate'], lang='en')
        if edge['relation'] != 'specializes':
            assert (source, predicate, target) not in graph
            assert (source, SKOS.related, target) in graph
        else:
            assert (source, predicate, target) in graph
            assert (source, SKOS.broader, target) in graph


def test_qualified_statements_are_queryable_without_asserting_causality(graph):
    rows = list(graph.query("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX meta: <https://mlsysbook.ai/taxonomy/scheme#>
        SELECT ?source ?target ?condition WHERE {
          ?statement a rdf:Statement ; rdf:subject ?source ;
            rdf:predicate <https://mlsysbook.ai/taxonomy/relation/reduces> ;
            rdf:object ?target ; meta:condition ?condition .
        }
    """))
    assert rows
    reduces = rdflib.URIRef('https://mlsysbook.ai/taxonomy/relation/reduces')
    assert not list(graph.triples((None, reduces, None)))


def test_entry_terms_do_not_automatically_assert_synonymy(graph):
    from core.concept_graph import PAGE_URL
    meta = Namespace(PAGE_URL + 'scheme#')
    for concept, term in graph.subject_objects(SKOS.altLabel):
        assert (concept, meta.entryTerm, term) in graph
    # A named sampling rule is narrower than the umbrella selection concept.
    sampling = rdflib.URIRef(PAGE_URL + 'concept/mlsys.sampling-controls')
    term = rdflib.Literal('top-k sampling', lang='en')
    assert (sampling, meta.entryTerm, term) in graph
    assert (sampling, SKOS.altLabel, term) not in graph
