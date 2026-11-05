"""Student openings keep scenario text, project imagery, and credit together."""
import json
from importlib import resources

import pytest

from mlsysbook_labs.learning_paths import get_learning_path
from mlsysbook_labs.notebook import notebook_scenario, notebook_submission


@pytest.mark.parametrize(('volume', 'track', 'key'), [
    (1, 'tinyml', 'tinyml'), (1, 'mobile', 'mobile'),
    (1, 'edge', 'edge'), (2, 'cloud', 'cloud'),
    (3, None, 'agentic'), (4, None, 'physical'),
])
def test_project_photo_and_credit_are_packaged_with_scenario(volume, track, key):
    contract = get_learning_path(volume, 0)
    rendered = notebook_scenario(contract, track=track).text
    manifest = json.loads(resources.files('mlsysbook_labs').joinpath('data/photos/manifest.json').read_text())
    photo = manifest[key]
    assert '<h2>Your scenario</h2>' in rendered
    assert 'src="data:image/jpeg;base64,' in rendered
    assert photo['source'] in rendered
    assert photo['license_url'] in rendered
    assert photo['modifications'] in rendered
    assert '<summary>Photo credit</summary>' in rendered
    # The core story is visible before any optional attribution details.
    assert rendered.index('nb-scenario-story') < rendered.index('<details>')


def test_episode_override_uses_actual_selected_project_and_escapes_authored_text():
    rendered = notebook_scenario(get_learning_path(1, 1), track='mobile', scene='The phone <needs> a decision.\n\nWhat changes?').text
    assert '<p>The phone &lt;needs&gt; a decision.</p><p>What changes?</p>' in rendered
    with pytest.raises(ValueError, match='authored scenario'):
        notebook_scenario(get_learning_path(1, 1), scene='')


def test_submission_uses_existing_download_workflow_without_claiming_automatic_submission():
    contract = get_learning_path(2, 0)
    opening = notebook_submission(contract, orientation=True).text
    chapter = notebook_submission(contract).text
    assert '<h2>Discuss and submit</h2>' in opening
    assert 'download controls' in chapter
    assert 'your course’s usual system' in chapter
    assert 'setup record' not in opening
    assert 'printed copy' not in opening
    assert 'open the downloaded file in a Markdown viewer and print it' in chapter
