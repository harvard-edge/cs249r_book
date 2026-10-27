"""Learner-map contracts: provenance, aggregation, and repeatable generation."""

import json
from pathlib import Path
import shutil

import pytest
import yaml

from core.concept_graph import ConceptGraphError, build, load_graph, refresh


@pytest.fixture
def curriculum(tmp_path):
    root = tmp_path
    for volume in ("vol1", "vol2"):
        folder = root / "books" / volume / "01_intro"
        folder.mkdir(parents=True)
        (folder / "01_intro.qmd").write_text(
            '---\n---\n# Introduction {#sec-intro}\n\n## Queueing {#sec-queue}\n\nWaiting increases as utilization rises.\n\n::: {#dfn-admission .callout-definition title="Admission control"}\nAdmission limits arrivals.\n:::\n'
        )
        graph = dict(
            schema_version=1,
            concepts=[
                dict(
                    id=f"{volume}.queue",
                    label="Queueing delay",
                    teaches="Queueing",
                    summary="Waiting for service.",
                    section="sec-queue",
                ),
                dict(
                    id=f"{volume}.utilization",
                    label="Utilization",
                    teaches="Queueing",
                    summary="Busy fraction.",
                    section="sec-queue",
                ),
            ],
            relationships=[
                dict(
                    id=f"{volume}.relation",
                    source=f"{volume}.queue",
                    target=f"{volume}.utilization",
                    predicate="increases with",
                    explanation="Near saturation, waits increase.",
                    conditions="Fixed service process.",
                    status="draft",
                    evidence=dict(
                        section="sec-queue",
                        quote="Waiting increases as utilization rises.",
                    ),
                )
            ],
            maps=[
                dict(
                    id=f"{volume}.queue-map",
                    focus_question="Why do requests wait?",
                    description="Queueing map.",
                    concepts=[f"{volume}.queue", f"{volume}.utilization"],
                    relationships=[f"{volume}.relation"],
                )
            ],
        )
        contract = dict(
            schema_version=2,
            source="01_intro.qmd",
            primary_concepts=["Queueing"],
            secondary_concepts=[],
            claims={"Queueing": "Capacity and arrivals govern waiting."},
            learner_graph=graph,
        )
        (folder / "intro_concepts.yml").write_text(
            yaml.safe_dump({"concept_map": contract})
        )
    site = root / "site/taxonomy"
    (site / "vendor").mkdir(parents=True)
    (site / "index.qmd").write_text("---\n---\n```{=html}\n<div>Viewer</div>\n```\n")
    for name in (
        "explorer.js",
        "explorer.css",
        "atlas.js",
        "vendor/cytoscape.min.js",
        "vendor/LICENSE.cytoscape",
    ):
        (site / name).write_text("fixture")
    return root


def edit_contract(root, change):
    path = root / "books/vol1/01_intro/intro_concepts.yml"
    value = yaml.safe_load(path.read_text())
    change(value["concept_map"])
    path.write_text(yaml.safe_dump(value))


def test_chapter_viewers_use_volume_template_when_explorer_has_its_own_page(curriculum):
    site = curriculum / "site/taxonomy"
    (site / "index.qmd").write_text('```{=html}\n<div id="study-root">Field explorer</div>\n```\n')
    (site / "volume-maps.qmd").write_text('```{=html}\n<div id="concept-map">Volume viewer</div>\n```\n')
    build(curriculum)
    atlas = (site / "atlas.html").read_text()
    assert 'id="concept-map"' in atlas
    assert 'id="study-root"' not in atlas
    embedded = (curriculum / "books/vol1/01_intro/concepts-generated/index.html").read_text()
    assert 'id="concept-map"' in embedded


def test_field_explorer_cannot_silently_become_chapter_viewer(curriculum):
    (curriculum / "site/taxonomy/index.qmd").write_text('```{=html}\n<div id="study-root">Field explorer</div>\n```\n')
    with pytest.raises(ConceptGraphError, match="volume-map template"):
        build(curriculum)


def test_section_provenance_cannot_be_missing_or_replaced(curriculum):
    source = curriculum / "books/vol1/01_intro/01_intro.qmd"
    source.write_text(
        source.read_text().replace(
            "Waiting increases as utilization rises.", "A different statement."
        )
    )
    with pytest.raises(ConceptGraphError, match="evidence quote changed"):
        load_graph(curriculum)


def test_learner_concept_must_belong_to_teaching_owner(curriculum):
    edit_contract(
        curriculum,
        lambda c: c["learner_graph"]["concepts"][0].update(teaches="Not owned"),
    )
    with pytest.raises(ConceptGraphError, match="owned concept"):
        load_graph(curriculum)


def test_map_cannot_publish_an_edge_with_hidden_endpoint(curriculum):
    edit_contract(curriculum, lambda c: c["learner_graph"]["maps"][0]["concepts"].pop())
    with pytest.raises(ConceptGraphError, match="endpoint outside map"):
        load_graph(curriculum)


def test_cross_chapter_references_resolve_without_duplicate_definitions(curriculum):
    path = curriculum / "books/vol1/02_reuse"
    path.mkdir()
    (path / "02_reuse.qmd").write_text("# Reuse {#sec-reuse}\n")
    contract = dict(
        schema_version=2,
        source="02_reuse.qmd",
        primary_concepts=[],
        secondary_concepts=[],
        claims={},
        learner_graph=dict(
            schema_version=1,
            maps=[
                dict(
                    id="vol1.reuse",
                    focus_question="Where is queueing reused?",
                    description="Reuse.",
                    concepts=["vol1.queue", "vol1.utilization"],
                    relationships=["vol1.relation"],
                )
            ],
        ),
    )
    (path / "reuse_concepts.yml").write_text(yaml.safe_dump({"concept_map": contract}))
    graph = load_graph(curriculum)
    assert len(graph["nodes"]) == 4
    assert graph["maps"]["vol1.reuse"]["concepts"][0] == "vol1.queue"


def test_refresh_only_proposes_definitions_and_tracks_source_changes(curriculum):
    contract = curriculum / "books/vol1/01_intro/intro_concepts.yml"
    authored = contract.read_bytes()
    initial = refresh(curriculum, ("vol1",))
    assert initial["candidates"][0]["label"] == "Admission control"
    assert not initial["changed_evidence"]
    source = curriculum / "books/vol1/01_intro/01_intro.qmd"
    source.write_text(source.read_text() + "\nAdditional explanation.\n")
    changed = refresh(curriculum, ("vol1",))
    assert changed["changed_evidence"] == ["vol1.relation"]
    assert refresh(curriculum, ("vol1",))["changed_evidence"] == ["vol1.relation"]
    assert contract.read_bytes() == authored


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz is needed for rendering")
def test_rerun_is_stable_and_authoring_propagates_to_all_views(curriculum):
    build(curriculum)
    assert build(curriculum, check=True)["stale"] == []
    catalog = json.loads((curriculum / "site/taxonomy/catalog.json").read_text())
    assert [v["id"] for v in catalog["volumes"]] == ["vol1", "vol2"]
    edit_contract(
        curriculum,
        lambda c: c["learner_graph"]["concepts"][0].update(
            label="Request waiting time"
        ),
    )
    assert build(curriculum, check=True)["stale"]
    build(curriculum, ("vol1",))
    website = json.loads(
        (curriculum / "site/taxonomy/maps/vol1.queue-map/graph.json").read_text()
    )
    chapter = json.loads(
        (
            curriculum
            / "books/vol1/01_intro/concepts-generated/maps/vol1.queue-map/graph.json"
        ).read_text()
    )
    assert website == chapter
    assert website["nodes"][0]["label"] == "Request waiting time"
    # A scoped chapter build must retain the full series catalog.
    assert (
        len(
            json.loads((curriculum / "site/taxonomy/catalog.json").read_text())[
                "volumes"
            ]
        )
        == 2
    )


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz is needed for rendering")
def test_bad_authoring_does_not_overwrite_published_assets(curriculum):
    build(curriculum)
    asset = curriculum / "site/taxonomy/maps/vol1.queue-map/graph.json"
    original = asset.read_bytes()
    edit_contract(
        curriculum,
        lambda c: c["learner_graph"]["relationships"][0].update(target="vol1.missing"),
    )
    with pytest.raises(ConceptGraphError, match="missing concept"):
        build(curriculum)
    assert asset.read_bytes() == original


def test_map_ids_cannot_escape_generated_directory(curriculum):
    edit_contract(curriculum, lambda c: c["learner_graph"]["maps"][0].update(id=".."))
    with pytest.raises(ConceptGraphError, match="invalid map ID"):
        load_graph(curriculum)


def test_existing_concept_map_gate_catches_bad_learner_evidence(curriculum):
    from checks.concept_maps import check_concept_maps

    edit_contract(
        curriculum,
        lambda c: c["learner_graph"]["relationships"][0]["evidence"].update(
            quote="Unsupported claim"
        ),
    )
    findings = check_concept_maps(curriculum / "books", curriculum)
    assert any(
        "learner_graph" in f.message and "evidence quote changed" in f.message
        for f in findings
    )


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz is needed for rendering")
def test_chapter_viewer_embeds_graph_data_for_offline_use(curriculum):
    import re

    build(curriculum)
    viewer = (
        curriculum / "books/vol1/01_intro/concepts-generated/index.html"
    ).read_text()
    payload = re.search(
        r'<script type="application/json" id="concept-data">(.*?)</script>',
        viewer,
        re.S,
    )[1]
    embedded = json.loads(payload)
    assert embedded["maps"]["vol1.queue-map"]["graph"]["nodes"][0]["id"] == "vol1.queue"
    assert "<svg" in embedded["maps"]["vol1.queue-map"]["svg"]
    assert "<" not in payload


def test_catalog_resolves_declared_reuse_without_merging_volumes(curriculum):
    from core.concept_graph import curriculum_catalog

    edit_contract(curriculum, lambda c: c.update(touches=[]))
    source = curriculum / "books/vol1/02_reuse"
    source.mkdir()
    (source / "02_reuse.qmd").write_text("# Reuse {#sec-reuse}\n")
    (source / "reuse_concepts.yml").write_text(
        yaml.safe_dump(
            {
                "concept_map": dict(
                    source="02_reuse.qmd",
                    primary_concepts=["Admission"],
                    secondary_concepts=[],
                    claims={"Admission": "Limit incoming work."},
                    touches=[dict(concept="Queueing", owner="01_intro")],
                )
            }
        )
    )
    graph = load_graph(curriculum)
    catalog = curriculum_catalog(graph)
    one = catalog["volumes"][0]["chapters"][0]["concepts"][0]
    two = catalog["volumes"][1]["chapters"][0]["concepts"][0]
    assert one["label"] == two["label"]
    assert one["id"] != two["id"]
    assert one["uses"] == [dict(chapter="vol1/02_reuse", relation="revisits")]
    assert two["uses"] == []
    assert catalog["counts"] == dict(concepts=3, chapters=3, references=1)
    reference = catalog["volumes"][0]["chapters"][1]["references"][0]
    assert reference["concept"] == one["id"]
    # Identity is independent of concept order and the build can be repeated.
    assert curriculum_catalog(graph) == catalog


def test_atlas_lenses_cannot_reference_missing_chapters(curriculum):
    from core.concept_graph import curriculum_catalog

    navigation = dict(
        themes=[
            dict(
                id="memory",
                label="Memory",
                question="What is retained?",
                chapters=["vol1/does-not-exist"],
            )
        ]
    )
    with pytest.raises(ConceptGraphError, match="unknown chapter"):
        curriculum_catalog(load_graph(curriculum), navigation)


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz is needed for rendering")
def test_series_preview_embeds_all_concepts_and_matches_site_catalog(curriculum):
    import re

    build(curriculum)
    page = (curriculum / "site/taxonomy/atlas.html").read_text()
    payload = re.search(
        r'<script type="application/json" id="concept-data">(.*?)</script>', page
    ).group(1)
    embedded = json.loads(payload)
    assert embedded["catalog"] == json.loads(
        (curriculum / "site/taxonomy/catalog.json").read_text()
    )
    assert embedded["catalog"]["scope"] == "series"
    assert len(embedded["maps"]) == 2
    chapter_page = (
        curriculum / "books/vol1/01_intro/concepts-generated/index.html"
    ).read_text()
    local = json.loads(
        re.search(
            r'<script type="application/json" id="concept-data">(.*?)</script>',
            chapter_page,
        ).group(1)
    )
    assert local["catalog"]["scope"] == "chapter"
    assert local["catalog"]["counts"] == embedded["catalog"]["counts"]
    assert set(local["maps"]) == set(embedded["maps"])


@pytest.fixture
def volume_connections(curriculum):
    folder = curriculum / "books/vol1/02_capacity"
    folder.mkdir()
    (folder / "02_capacity.qmd").write_text("# Capacity {#sec-capacity}\n")
    (folder / "capacity_concepts.yml").write_text(
        yaml.safe_dump(
            {
                "concept_map": dict(
                    source="02_capacity.qmd",
                    primary_concepts=["Capacity"],
                    secondary_concepts=[],
                    claims={"Capacity": "Service capacity bounds throughput."},
                )
            }
        )
    )
    edit_contract(
        curriculum,
        lambda c: c.update(
            explorer_graph=dict(
                schema_version=1,
                relationships=[
                    dict(
                        source="Queueing",
                        predicate="depends on",
                        target=dict(chapter="02_capacity", concept="Capacity"),
                        explanation="Queues depend on arrivals and service capacity.",
                        conditions="Fixed workload.",
                        status="draft",
                        evidence=[
                            dict(
                                chapter="01_intro",
                                concept="Queueing",
                                quote="Capacity and arrivals govern waiting.",
                            ),
                            dict(
                                chapter="02_capacity",
                                concept="Capacity",
                                quote="Service capacity bounds throughput.",
                            ),
                        ],
                    )
                ],
            )
        ),
    )
    return curriculum


def test_volume_map_joins_cross_chapter_concepts_and_remains_volume_scoped(
    volume_connections,
):
    graph = load_graph(volume_connections)
    overview = graph["maps"]["vol1.overview"]
    nodes = [graph["nodes"][i] for i in overview["concepts"]]
    assert {n["chapter"] for n in nodes} == {"vol1/01_intro", "vol1/02_capacity"}
    assert {n["concept"] for n in nodes} == {"Queueing", "Capacity"}
    assert all(n["volume"] == 1 for n in nodes)
    assert graph["chapters"]["vol1/02_capacity"]["maps"] == ["vol1.overview"]
    assert "vol1.overview" not in graph["chapters"]["vol2/01_intro"]["maps"]
    edge = graph["edges"][overview["relationships"][0]]
    assert edge["label"] == "depends on"
    assert edge["evidence_kind"] == "chapter-claim"
    assert edge["status"] == "draft"


def test_volume_propositions_fail_when_claim_evidence_changes(volume_connections):
    edit_contract(
        volume_connections, lambda c: c["claims"].update(Queueing="An unrelated claim.")
    )
    with pytest.raises(ConceptGraphError, match="evidence claim changed"):
        load_graph(volume_connections)
    assert refresh(volume_connections)["invalid"]


def test_refresh_tracks_pending_changes_to_volume_claim_evidence(volume_connections):
    refresh(volume_connections)
    edit_contract(
        volume_connections,
        lambda c: c["claims"].update(
            Queueing="Capacity and arrivals govern waiting. Additional assumption."
        ),
    )
    changed = refresh(volume_connections)["changed_evidence"]
    assert len(changed) == 1
    assert ".connection." in changed[0]
    assert refresh(volume_connections)["changed_evidence"] == changed


@pytest.mark.skipif(not shutil.which("dot"), reason="Graphviz is needed for rendering")
def test_chapter_viewer_uses_the_same_volume_graph(volume_connections):
    build(volume_connections)
    site = volume_connections / "site/taxonomy/maps/vol1.overview/graph.json"
    chapter = (
        volume_connections
        / "books/vol1/02_capacity/concepts-generated/maps/vol1.overview/graph.json"
    )
    assert site.read_bytes() == chapter.read_bytes()
    data = json.loads(site.read_text())
    assert data["scope"] == "volume"
    assert len(data["nodes"]) == 2


def test_companion_resources_preserve_readiness_and_reject_missing_or_escaped_sources(volume_connections):
    path = volume_connections / 'labs/example.py'
    path.parent.mkdir()
    path.write_text('# Activity source')
    resource = dict(title='Queueing experiment', path='labs/example.py', kind='lab', status='skeleton')
    edit_contract(volume_connections, lambda c: c['explorer_graph'].update(resources={'Queueing': [resource]}))
    graph = load_graph(volume_connections)
    node = next(n for n in graph['nodes'].values() if n.get('kind') == 'teaching-concept' and n['concept'] == 'Queueing')
    assert node['resources'][0]['status'] == 'skeleton'
    assert node['resources'][0]['url'].endswith('/blob/dev/labs/example.py')
    for invalid in ('labs/absent.py', '../outside.py'):
        resource['path'] = invalid
        edit_contract(volume_connections, lambda c: c['explorer_graph'].update(resources={'Queueing': [resource]}))
        with pytest.raises(ConceptGraphError, match='missing or unsafe companion'):
            load_graph(volume_connections)
    edit_contract(volume_connections, lambda c: c['explorer_graph'].update(resources={'Unowned': []}))
    with pytest.raises(ConceptGraphError, match='resources must reference owned'):
        load_graph(volume_connections)


def test_companion_resource_shapes_and_url_encoding(volume_connections):
    for invalid in (None, [], 'not a mapping'):
        edit_contract(volume_connections, lambda c: c['explorer_graph'].update(resources=invalid))
        with pytest.raises(ConceptGraphError, match='resources must reference owned'):
            load_graph(volume_connections)
    for invalid in (None, {}, ''):
        edit_contract(volume_connections, lambda c: c['explorer_graph'].update(resources={'Queueing': invalid}))
        with pytest.raises(ConceptGraphError, match='must be a list'):
            load_graph(volume_connections)
    path = volume_connections / 'lab #1?.py'
    path.write_text('# source')
    record = dict(title='Experiment', path=path.name, kind='lab', status='early-release')
    edit_contract(volume_connections, lambda c: c['explorer_graph'].update(resources={'Queueing': [record]}))
    graph = load_graph(volume_connections)
    node = next(n for n in graph['nodes'].values() if n.get('kind') == 'teaching-concept' and n['concept'] == 'Queueing')
    assert node['resources'][0]['url'].endswith('/lab%20%231%3F.py')


def test_overview_drilldown_uses_exact_chapter_and_teaching_identity(volume_connections):
    from core.concept_graph import map_data
    data = map_data(load_graph(volume_connections), 'vol1.overview')
    queue = next(n for n in data['nodes'] if n['concept'] == 'Queueing')
    capacity = next(n for n in data['nodes'] if n['concept'] == 'Capacity')
    assert [m['id'] for m in queue['detail_maps']] == ['vol1.queue-map']
    assert capacity['detail_maps'] == []


def test_atomic_volume_map_is_primary_and_reports_unmapped_goals(volume_connections):
    from core.concept_graph import curriculum_catalog, map_data

    def add_view(c):
        c['learner_graph']['maps'][0].update(scope='volume', volume='vol1')
        c['secondary_concepts'].append('Unmapped policy')
        c['claims']['Unmapped policy'] = 'An authored goal outside this particular view.'
    edit_contract(volume_connections, add_view)
    graph = load_graph(volume_connections)
    assert graph['chapters']['vol1/02_capacity']['maps'][0] == 'vol1.queue-map'
    assert graph['maps']['vol1.overview']['scope'] == 'argument'
    assert 'vol1.overview' not in {m['id'] for m in curriculum_catalog(graph)['maps']}
    data = map_data(graph, 'vol1.queue-map')
    assert data['coverage']['represented_goals'] == 1
    assert data['coverage']['teaching_goals'] == 3
    assert data['coverage']['chapters'][0]['outside_view'] == ['Unmapped policy']
    assert all(n['teaching_goal'].startswith(n['chapter'] + ':') for n in data['nodes'])


def test_authored_volume_map_rejects_cross_volume_membership(curriculum):
    def change(c):
        view = c['learner_graph']['maps'][0]
        view.update(scope='volume', volume='vol1')
        view['concepts'].append('vol2.queue')
    edit_contract(curriculum, change)
    with pytest.raises(ConceptGraphError, match='stay inside its volume'):
        load_graph(curriculum)


def test_aliases_must_be_explicit_names(curriculum):
    edit_contract(curriculum, lambda c: c['learner_graph']['concepts'][0].update(aliases='queue'))
    with pytest.raises(ConceptGraphError, match='aliases must be nonempty strings'):
        load_graph(curriculum)


def test_refresh_inventories_index_and_glossary_without_inventing_edges(curriculum):
    source = curriculum / 'books/vol1/01_intro/01_intro.qmd'
    source.write_text(source.read_text() + '\nAdmission pressure\\index{Arrival rate!definition}.\n')
    glossary = curriculum / 'books/vol1/backmatter/glossary/glossary.qmd'
    glossary.parent.mkdir(parents=True)
    glossary.write_text('**Utilization**\n: Fraction of available service capacity in use.\n\n**New quantity**\n: A candidate term.\n')
    edit_contract(curriculum, lambda c: c['learner_graph']['concepts'][1].update(aliases=['Busy fraction']))
    before = load_graph(curriculum)
    report = refresh(curriculum, ('vol1',))
    labels = {c['label']: c for c in report['candidates']}
    assert labels['Arrival rate']['kind'] == 'index-definition'
    assert labels['New quantity']['kind'] == 'glossary'
    assert 'Utilization' not in labels
    assert load_graph(curriculum)['edges'] == before['edges']
