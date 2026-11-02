"""Narrative continuity does not certify target-specific numerical behavior."""
import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mlsysim.engine.deployment_journeys import load_deployment_journeys
from mlsysim.engine.scenarios import Scenarios


CATALOG_PATH = Path(__file__).parents[1] / 'mlsysim/engine/data/deployment_journeys.json'
EXPECTED_IDENTITIES = {
    'tinyml': 'wearable-ring.v1',
    'mobile': 'personal-media-assistant.v1',
    'edge': 'autonomous-perception.v1',
    'cloud': 'recommendation-endpoint.v1',
}


@pytest.mark.parametrize('track', EXPECTED_IDENTITIES)
def test_project_identity_and_complete_authored_episodes(track):
    journey = Scenarios.DeploymentJourneys.get(track)
    assert journey.track_id == track
    assert journey.scenario_id == EXPECTED_IDENTITIES[track]
    assert journey.version == Scenarios.DeploymentJourneys.version == 'deployment-journeys.v4'
    assert journey.success and journey.unknowns
    assert 'generic illustrative fixtures' in journey.study_boundary
    assert tuple(episode.chapter for episode in journey.episodes) == tuple(range(17))
    assert all(all(getattr(episode, field) for field in (
        'question', 'task', 'handoff', 'scene', 'choice', 'consequence'
    )) for episode in journey.episodes)
    assert tuple(act.id for act in journey.acts) == ('frame', 'build', 'qualify', 'operate')
    assert tuple(chapter for act in journey.acts for chapter in act.chapters) == tuple(range(1, 17))
    assert all(act.arc for act in journey.acts)
    with pytest.raises(KeyError):
        journey.episode(99)
    with pytest.raises(FrozenInstanceError):
        journey.title = 'An unrelated project'
    with pytest.raises(FrozenInstanceError):
        journey.episode(1).scene = 'An unrelated event'
    with pytest.raises(FrozenInstanceError):
        journey.acts[0].arc = 'An unrelated arc'


@pytest.mark.parametrize('alias, canonical', [
    ('Oura Ring', 'tinyml'), ('wearable', 'tinyml'), ('TinyML', 'tinyml'),
    ('iPhone', 'mobile'), ('MobileML', 'mobile'), ('robotaxi', 'edge'),
    ('EdgeML', 'edge'), ('Cloud/Fleet', 'cloud'), ('CloudML', 'cloud'),
])
def test_saved_profile_aliases_resolve_to_one_narrative(alias, canonical):
    catalog = Scenarios.DeploymentJourneys
    assert catalog.get(alias) is catalog.get(canonical)


def test_aliases_and_catalog_are_immutable_and_unknown_track_is_rejected():
    catalog = Scenarios.DeploymentJourneys
    with pytest.raises(TypeError):
        catalog.aliases['wearable'] = 'cloud'
    with pytest.raises(TypeError):
        catalog.provenance['role'] = 'measured deployment'
    with pytest.raises(KeyError):
        catalog.get('not a deployment context')
    assert len({item.scenario_id for item in catalog.list()}) == len(catalog.list())
    assert catalog.provenance['role'] == 'authored narrative'
    assert 'numerical' in catalog.provenance['coverage']
    # Narrative briefs are deliberately excluded from executable scenario enumeration.
    assert Scenarios.DeploymentJourneys not in Scenarios.list()


def test_placement_narrative_names_change_without_reinterpreting_study_inputs():
    raw = json.loads(CATALOG_PATH.with_name('placement_study.json').read_text())
    for track, fixture in Scenarios.PlacementStudy.fixtures['tracks'].items():
        journey = Scenarios.DeploymentJourneys.get(track)
        assert raw['tracks'][track]['scenario'] == journey.scenario_id
        assert fixture['scenario'] == journey.title
        assert {key: value for key, value in fixture.items() if key != 'scenario'} == {
            key: value for key, value in raw['tracks'][track].items() if key != 'scenario'
        }


@pytest.mark.parametrize('mutation, message', [
    (lambda data: data['scenarios'][0]['episodes'].pop(), 'ordered episodes'),
    (lambda data: data['scenarios'][0]['episodes'].append(copy.deepcopy(data['scenarios'][0]['episodes'][0])), 'ordered episodes'),
    (lambda data: data['scenarios'][0]['episodes'][4].update(choice=''), 'nonempty choice'),
    (lambda data: data['scenarios'][0]['acts'][0]['chapters'].append(5), 'partition'),
    (lambda data: data['scenarios'][0].update(acts=[]), 'authored acts'),
    (lambda data: data['aliases'].update(broken='unknown'), None),
    (lambda data: data['provenance'].update(role='measured deployment'), 'authored narrative'),
])
def test_invalid_narrative_catalog_rejected(tmp_path, mutation, message):
    data = json.loads(CATALOG_PATH.read_text())
    mutation(data)
    source = tmp_path / 'journeys.json'
    source.write_text(json.dumps(data))
    with pytest.raises((ValueError, KeyError), match=message):
        load_deployment_journeys(source)


def test_authored_focus_matches_each_chapters_primary_concept_map():
    """Keep the narrative tied to chapter ownership, not merely a recurring product."""
    import yaml

    source_root = Path(__file__).parents[2] / 'books/vol1'
    for chapter in range(1, 16):
        maps = list(source_root.glob(f'{chapter:02d}_*/*_concepts.yml'))
        assert len(maps) == 1
        expected = tuple(yaml.safe_load(maps[0].read_text())['concept_map']['primary_concepts'])
        for journey in Scenarios.DeploymentJourneys.list():
            assert journey.episode(chapter).core_concepts == expected
    for journey in Scenarios.DeploymentJourneys.list():
        # Orientation teaches the interface; conclusion integrates earlier ownership.
        assert journey.episode(0).core_concepts == ()
        assert journey.episode(16).core_concepts == ()


def test_cloud_recommendation_contract_does_not_relabel_old_task_evidence():
    journey = Scenarios.DeploymentJourneys.get('cloud')
    assert journey.scenario_id == 'recommendation-endpoint.v1'
    assert 'newly listed' in journey.mission
    assert 'Candidate generation is a declared upstream input' in journey.workload
    assert 'one cloud endpoint' in journey.boundary
    assert 'separate development host' in journey.episode(8).task
    assert 'embedding capacity' in journey.episode(6).task
    assert 'language reference' in journey.episode(6).task
    assert 'separately named generative reference task' in journey.episode(13).task
    assert 'not a KV cache' in journey.episode(13).consequence
    assert 'Old classification-service records retain their original identity' in journey.study_boundary
    assert 'exposure-conditioned feedback' in journey.episode(15).task


def test_volume_storyboard_has_complete_immutable_learning_sequence():
    volume = Scenarios.DeploymentJourneys.volume_storyboard
    assert tuple(item.chapter for item in volume.chapters) == tuple(range(17))
    assert volume.chapter(0).chapter == 0
    assert volume.chapter(16).chapter == 16
    assert all(item.investigations and item.qmd_bridge and item.dossier_entry
               and item.transition for item in volume.chapters)
    assert all(story.storyline for story in Scenarios.DeploymentJourneys.list())
    with pytest.raises(FrozenInstanceError):
        volume.chapter(2).investigations[0].question = 'A changed question'
    with pytest.raises(KeyError):
        volume.chapter(99)


@pytest.mark.parametrize('mutation, message', [
    (lambda data: data['volume_storyboard']['chapters'].pop(), 'ordered chapters'),
    (lambda data: data['volume_storyboard']['chapters'][3].update(transition=''), 'nonempty transition'),
    (lambda data: data['volume_storyboard']['chapters'][2].update(investigations=[]), 'requires investigations'),
    (lambda data: data['volume_storyboard']['chapters'][4]['investigations'][0].update(question=''), 'nonempty question'),
    (lambda data: data['scenarios'][0].update(storyline=''), 'continuous storyline'),
])
def test_incomplete_storyboard_is_rejected(tmp_path, mutation, message):
    data = json.loads(CATALOG_PATH.read_text())
    mutation(data)
    source = tmp_path / 'journeys.json'
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        load_deployment_journeys(source)
