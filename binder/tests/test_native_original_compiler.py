"""Native original-art factories enroll without placeholder mechanism lessons."""
from argparse import Namespace
import copy
import importlib.util
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
REAL_EXT = ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
spec = importlib.util.spec_from_file_location('native_original_compiler', ROOT/'binder/tools/scripts/compile_figure_explore.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False)+'\n')


@pytest.fixture
def native(tmp_path, monkeypatch):
    ext = tmp_path/'books/_extensions/mlsysbook-ext/figure-explore'
    mechanisms = ext/'mechanisms'; mechanisms.mkdir(parents=True)
    catalog = tmp_path/'catalog.json'
    monkeypatch.setattr(compiler, 'ROOT', tmp_path)
    monkeypatch.setattr(compiler, 'EXT', ext)
    monkeypatch.setattr(compiler, 'CATALOG', catalog)
    for name in ['original-demo.js','original-demo.css','original-player.js','original-player.css',
                 'mechanism-scene.js','mechanism-scene.css']:
        (ext/name).write_text('// Synthetic asset fixture: '+name)
    source = tmp_path/'books/vol1/chapter.qmd'; source.parent.mkdir(parents=True)
    static = '::: {#fig-native fig-cap="Unchanged static caption."}\n\nOriginal diagram.\n\n:::\n'
    source.write_text(static)
    row = dict(key='vol1:fig-native', source='books/vol1/chapter.qmd', figure_id='fig-native', volume=1)
    write_json(catalog, {'metadata':{'source_head':'synthetic-fixture'}, 'figures':[row]})
    write_json(mechanisms/'selection.json', [row])
    implementation = dict(name='original-demo', script='original-demo.js', stylesheet='original-demo.css',
                          shared_player=True, question='Which object moves?', scope='Synthetic compiler fixture only.')
    write_json(mechanisms/'implementations.json', {row['key']:implementation})
    svg = tmp_path/'canonical.svg'
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 8"><rect width="16" height="8"/></svg>')
    image = tmp_path/'canonical.png'; Image.new('RGB',(16,8),'white').save(image)
    art = dict(source=row['source'], asset=str(svg), asset_sha1=compiler.sha1(svg.read_bytes()), image=str(image))
    index = tmp_path/'art.json'; write_json(index, {row['key']:art})
    args = Namespace(selection=str(mechanisms/'selection.json'), art_index=str(index), require_complete=True,
                     enroll=True, check=False, build_root=None, review_record=None, report=str(tmp_path/'report.json'))
    return dict(root=tmp_path,ext=ext,mechanisms=mechanisms,catalog=catalog,source=source,static=static,
                row=row,implementation=implementation,art=art,index=index,args=args)


def compile_fixture(fixture):
    compiler.compile_mechanisms(fixture['args'])
    return json.loads((fixture['ext']/'figure-registry.json').read_text())


def test_native_selected_original_needs_no_generic_chunk_or_generic_runtime(native):
    (native['ext']/'mechanism-scene.js').unlink()
    (native['ext']/'mechanism-scene.css').unlink()
    registry = compile_fixture(native)
    entry = registry[native['row']['key']]
    assert entry['name'] == 'original-demo'
    assert entry['runtime_sha1'] and 'bundle' not in entry and 'lesson_sha1' not in entry
    assert all(native['row']['key'] not in (native['ext']/f'mechanism-lessons-vol{volume}.js').read_text() for volume in range(1,5))
    assert compiler.normalized_source(native['source'].read_text()) == native['static']
    assert 'data-explore="original-demo"' in native['source'].read_text()
    data = json.loads(native['catalog'].read_text())
    row = data['figures'][0]
    assert row['actual_treatment'] == 'original-demo' and row['actual_status'] == 'implemented-awaiting-review'
    assert row['actual_question'] == native['implementation']['question']
    assert row['actual_scope'] == native['implementation']['scope'] and row['actual_steps'] == []
    assert 'without a generic mechanism lesson' in row['actual_review_basis']
    assert json.loads(Path(native['args'].report).read_text())['missing_selected'] == []
    native['args'].check = True
    compiler.compile_mechanisms(native['args'])


@pytest.mark.parametrize('case', ['unselected','unknown'])
def test_manifest_entries_must_be_explicitly_selected(native, case):
    manifest = {native['row']['key']:native['implementation']}
    if case == 'unselected':
        write_json(native['mechanisms']/'selection.json', [])
    else:
        manifest['vol4:fig-unknown'] = native['implementation']
    write_json(native['mechanisms']/'implementations.json', manifest)
    with pytest.raises(ValueError, match='implementation outside selected roster'):
        compile_fixture(native)
    assert not (native['ext']/'figure-registry.json').exists()
    assert native['source'].read_text() == native['static']


@pytest.mark.parametrize('field', ['source','figure_id','volume'])
def test_native_selection_must_match_catalog(native, field):
    selected = dict(native['row']); selected[field] = 'different'
    write_json(native['mechanisms']/'selection.json', [selected])
    with pytest.raises(ValueError, match='selected '+field+' differs'):
        compile_fixture(native)


@pytest.mark.parametrize('case', ['script','stylesheet','shared_player','art','art_hash','malformed_svg'])
def test_native_assets_and_artwork_fail_before_any_output_or_enrollment(native, case):
    if case in ['script','stylesheet']:
        (native['ext']/native['implementation'][case]).unlink()
    elif case == 'shared_player':
        (native['ext']/'original-player.js').unlink()
    elif case == 'art':
        write_json(native['index'], {})
    elif case == 'art_hash':
        art = dict(native['art']); art['asset_sha1'] = '0'*40
        write_json(native['index'], {native['row']['key']:art})
    else:
        Path(native['art']['asset']).write_text('<svg><broken')
        art = dict(native['art']); art['asset_sha1'] = compiler.sha1(Path(art['asset']).read_bytes())
        write_json(native['index'], {native['row']['key']:art})
    with pytest.raises(ValueError):
        compile_fixture(native)
    assert not (native['ext']/'figure-registry.json').exists()
    assert native['source'].read_text() == native['static']


def test_selected_missing_implementation_is_not_complete(native):
    write_json(native['mechanisms']/'implementations.json', {})
    with pytest.raises(ValueError, match='missing selected mechanisms'):
        compile_fixture(native)


def test_native_bundle_is_hashed_without_a_mechanism_lesson(native):
    implementation = dict(native['implementation'],bundle='native-values.js')
    (native['ext']/'native-values.js').write_text('window.nativeValues=[1];')
    write_json(native['mechanisms']/'implementations.json', {native['row']['key']:implementation})
    entry = compile_fixture(native)[native['row']['key']]
    assert entry['bundle'] == 'native-values.js'
    assert entry['bundle_sha1'] == compiler.sha1((native['ext']/'native-values.js').read_bytes())
    assert entry['lesson_sha1'] == entry['bundle_sha1']


def test_original_and_generic_coexist_without_changing_existing_bundles(native):
    # Use a complete, real validated chunk and its real calculation companion.
    for name in ['vol1-a.json','vol1-a.calc.py']:
        shutil.copyfile(REAL_EXT/'mechanisms'/name, native['mechanisms']/name)
    computation = Path('books/vol1/05_nn_computation/05_nn_computation.qmd')
    (native['root']/computation).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/computation, native['root']/computation)
    lessons = json.loads((native['mechanisms']/'vol1-a.json').read_text())
    generic_rows = [dict(key=lesson['key'],source=native['row']['source'],figure_id=lesson['key'].split(':',1)[1],volume=1) for lesson in lessons]
    static = native['static'] + ''.join('::: {#'+r['figure_id']+'}\n\nOriginal.\n\n:::\n' for r in generic_rows)
    native['source'].write_text(static)
    write_json(native['catalog'], {'metadata':{'source_head':'synthetic-fixture'},'figures':generic_rows})
    write_json(native['mechanisms']/'selection.json', generic_rows)
    write_json(native['mechanisms']/'implementations.json', {})
    art = dict(native['art'])
    write_json(native['index'], {r['key']:art for r in generic_rows})
    before = compile_fixture(native)
    bundles = {volume:(native['ext']/f'mechanism-lessons-vol{volume}.js').read_bytes() for volume in range(1,5)}
    write_json(native['catalog'], {'metadata':{'source_head':'synthetic-fixture'},'figures':generic_rows+[native['row']]})
    write_json(native['mechanisms']/'selection.json', generic_rows+[native['row']])
    write_json(native['mechanisms']/'implementations.json', {native['row']['key']:native['implementation']})
    write_json(native['index'], {r['key']:art for r in generic_rows+[native['row']]})
    after = compile_fixture(native)
    assert {key:after[key] for key in before} == before
    assert all((native['ext']/f'mechanism-lessons-vol{volume}.js').read_bytes()==data for volume,data in bundles.items())


def test_current_mechanism_roster_runtime_identities_survive_native_addition():
    keys = {row['key'] for path in (REAL_EXT/'mechanisms').glob('vol*-*.json') for row in json.loads(path.read_text())}
    actual = json.loads((REAL_EXT/'figure-registry.json').read_text())
    manifest = json.loads((REAL_EXT/'mechanisms/implementations.json').read_text())
    registry = {key:copy.deepcopy(actual[key]) for key in keys}
    before = copy.deepcopy(registry)
    registry['vol4:fig-native-test'] = dict(source_sha1='source',art_sha1='art')
    selected = {key:manifest[key] for key in keys}
    selected['vol4:fig-native-test'] = dict(name='original-native-test',script='original-player.js',stylesheet='original-player.css',shared_player=True)
    compiler.apply_implementations(registry, selected)
    assert {key:registry[key] for key in keys} == before
