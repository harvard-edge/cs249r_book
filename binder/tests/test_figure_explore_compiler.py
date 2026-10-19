"""Enrollment must never interpret braces inside chapter-owned attributes."""
import importlib.util
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('figure_compiler', ROOT / 'binder/tools/scripts/compile_figure_explore.py')
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)

@pytest.mark.parametrize('caption', [
    r'A $224{\times}224$ image',
    r'Cost $10^{22}$ and $\mathcal{O}(n)$',
    'Computed `{python} tile_count` values',
    'An ordinary caption',
])
def test_enrollment_leaves_every_original_byte_untouched(caption):
    source = '::: {#fig-demo fig-cap="' + caption + '" fig-alt="Original"}\n\nbody {x}\n:::\n'
    match = compiler.figure_match(source, 'fig-demo')
    marker = ' data-explore="guided-figure" data-explore-key="vol1:fig-demo"'
    enrolled = source[:match.end()] + marker + source[match.end():]
    assert enrolled.startswith('::: {#fig-demo' + marker + ' fig-cap=')
    assert compiler.normalized_source(enrolled) == source
    assert compiler.enrollment_attributes(enrolled, 'fig-demo') == marker


def test_structural_hash_ignores_generated_ids_but_preserves_geometry():
    original = b'<svg><metadata>date</metadata><path id="p123" d="M1 2"/><use href="#p123"/></svg>'
    changed_ids = b'<svg><metadata>other</metadata><path id="p456" d="M1 2"/><use href="#p456"/></svg>'
    assert compiler.structural_svg(original) == compiler.structural_svg(changed_ids)
    assert compiler.structural_svg(original) != compiler.structural_svg(original.replace(b'M1 2', b'M1 3'))


def test_plain_compile_clears_historical_build_evidence():
    import sys
    sys.path.insert(0,str(ROOT/'binder/tools/scripts'))
    data={'metadata':{'build_root':'old','build_route_count':1},'figures':[{'chapter_url':'old.html','built_explore_present':True}]}
    compiler.bind_current_build(data,{}, {}, None)
    assert data['metadata']['build_root'] is None
    assert data['metadata']['build_route_count']==0
    assert data['figures'][0]=={'chapter_url':'','built_explore_present':False}


def test_build_binding_requires_complete_success(tmp_path):
    import json
    import sys
    sys.path.insert(0,str(ROOT/'binder/tools/scripts'))
    (tmp_path/'summary.json').write_text(json.dumps({'jobs':[]}))
    with pytest.raises(ValueError,match='missing successful full vol1 HTML'):
        compiler.bind_current_build({'metadata':{},'figures':[]},{},{},tmp_path)


def test_build_binding_rejects_stale_emitted_runtime(tmp_path):
    import json
    import sys
    sys.path.insert(0,str(ROOT/'binder/tools/scripts'))
    (tmp_path/'summary.json').write_text(json.dumps({'jobs':[dict(volume=f'vol{v}',format='html',chapter=None,ok=True) for v in range(1,5)]}))
    assets=tmp_path/'vol1-html/output/site_libs/runtime'
    assets.mkdir(parents=True)
    for name in ('figure-explore.js','figure-views.js','figure-views.css','mechanism-scene.css'):
        (assets/name).write_bytes((compiler.EXT/name).read_bytes())
    (assets/'mechanism-scene.js').write_text('stale runtime')
    with pytest.raises(ValueError,match='stale emitted mechanism-scene.js'):
        compiler.bind_current_build({'metadata':{},'figures':[]},{'vol1:fig-demo':{'volume':1,'script':'mechanism-scene.js','stylesheet':'mechanism-scene.css'}},{},tmp_path)


def test_build_binding_accepts_original_art_without_generic_lesson_bundle(tmp_path, monkeypatch):
    import json
    import sys
    sys.path.insert(0,str(ROOT/'binder/tools/scripts'))
    monkeypatch.setattr(compiler,'ROOT',tmp_path)
    monkeypatch.setattr(compiler,'CATALOG',tmp_path/'catalog.json')
    (tmp_path/'summary.json').write_text(json.dumps({'jobs':[dict(volume=f'vol{v}',format='html',chapter=None,ok=True) for v in range(1,5)]}))
    key='vol1:fig-original'
    entry={'volume':1,'name':'original-example','script':'original-player.js','stylesheet':'original-player.css','shared_player':True,'art_sha1':'art','art_kind':'svg'}
    for volume in range(1,5):
        output=tmp_path/f'vol{volume}-html/output'
        assets=output/'site_libs/runtime';assets.mkdir(parents=True)
        names=['figure-explore.js','figure-views.js','figure-views.css']
        if volume==1:names+=['original-player.js','original-player.css']
        for name in names:(assets/name).write_bytes((compiler.EXT/name).read_bytes())
    chapter=tmp_path/'vol1-html/output/vol1/chapter.html';chapter.parent.mkdir(parents=True)
    chapter.write_text('<div id="fig-original" data-mlsys-explore="original-example" data-mlsys-explore-key="vol1:fig-original" data-mlsys-art-sha1="art" data-mlsys-art-kind="svg"></div>')
    data={'metadata':{},'figures':[{'key':key,'volume':1,'source':'books/vol1/chapter.qmd','figure_id':'fig-original'}]}
    compiler.bind_current_build(data,{key:entry},{},tmp_path)
    assert data['figures'][0]['built_explore_present'] is True
    assert data['metadata']['build_markers_verified']==1


@pytest.mark.parametrize('tamper',[None,'registry','summary','evidence'])
def test_review_record_is_explicit_and_hash_bound(tmp_path,monkeypatch,tamper):
    import copy
    import hashlib
    import json
    import sys
    sys.path.insert(0,str(ROOT/'binder/tools/scripts'))
    monkeypatch.setattr(compiler,'ROOT',tmp_path)
    ext=tmp_path/'books/_extensions/mlsysbook-ext/figure-explore'
    monkeypatch.setattr(compiler,'EXT',ext)
    catalog=tmp_path/'books/_prototypes/vol1-web-animations/animation-catalog/catalog.json'
    monkeypatch.setattr(compiler,'CATALOG',catalog)
    report=catalog.parent/'evidence/test-report.md';report.parent.mkdir(parents=True)
    report.write_text('Synthetic hash-binding fixture; no actual review is claimed.\n')
    build=tmp_path/'books/_build/parallel/test';build.mkdir(parents=True)
    summary=build/'summary.json';summary.write_text('{"jobs":[]}\n')
    sha=lambda data:hashlib.sha256(data).hexdigest()
    registry_content='{"vol1:fig-demo":{}}\n'
    key='vol1:fig-demo'
    review=dict(verdict='accepted',coverage=dict(cases=['base'],profiles=['desktop-light'],endpoints='Synthetic fixture only.',motion='Synthetic fixture only.',controls='Synthetic fixture only.'),reviewer_type='Synthetic test fixture',limitations='No actual review or students.',evidence=[dict(path=str(report.relative_to(tmp_path)),sha256=sha(report.read_bytes()))])
    record=dict(schema=1,build_root=str(build.relative_to(tmp_path)),registry_sha256=sha(registry_content.encode()),build_summary_sha256=sha(summary.read_bytes()),lessons={key:review})
    if tamper=='registry':record['registry_sha256']='0'*64
    if tamper=='summary':summary.write_text('changed')
    if tamper=='evidence':report.write_text('changed')
    path=tmp_path/'review.json';path.write_text(json.dumps(record))
    data={'metadata':{},'figures':[dict(key=key,status='implemented-awaiting-review',actual_status='implemented-awaiting-review')]}
    outputs={ext/'figure-registry.json':registry_content}
    if tamper:
        with pytest.raises(ValueError,match='hash mismatch'):
            compiler.apply_review_record(data,{key:{}},{key:{}},outputs,build,path)
        assert data['figures'][0]['actual_status']=='implemented-awaiting-review'
    else:
        compiler.apply_review_record(data,{key:{}},{key:{}},outputs,build,path)
        assert data['figures'][0]['actual_status']=='implementation-review-passed'
        assert data['figures'][0]['review_scope']['coverage']['profiles']==['desktop-light']
        compiler.apply_review_record(data,{key:{}},{key:{}},outputs,build,None)
        assert 'review_scope' not in data['figures'][0]
        assert 'review_record' not in data['metadata']


def test_catalog_provenance_does_not_follow_new_commits(monkeypatch):
    monkeypatch.setattr(compiler.subprocess,'check_output',lambda *args,**kwargs:'new-commit\n')
    assert compiler.source_revision({'source_head':'reviewed-baseline'})=='reviewed-baseline'
    assert compiler.source_revision({'source_head':''})=='new-commit'


def test_specialized_implementation_preserves_other_registry_records(tmp_path, monkeypatch):
    monkeypatch.setattr(compiler,'EXT',tmp_path)
    for name in ('convolution.js','convolution.css','convolution-values.js'):
        (tmp_path/name).write_text(name)
    registry={'vol1:fig-cnn':dict(source_sha1='source',art_structural_sha1='art',name='mechanism-scene',bundle='old.js',bundle_sha1='old',lesson_sha1='old'),
              'vol2:fig-other':dict(name='mechanism-scene',source_sha1='other')}
    untouched=dict(registry['vol2:fig-other'])
    manifest={'vol1:fig-cnn':dict(name='convolution',script='convolution.js',stylesheet='convolution.css',bundle='convolution-values.js')}
    compiler.apply_implementations(registry,manifest)
    assert len(registry)==2 and registry['vol2:fig-other']==untouched
    entry=registry['vol1:fig-cnn']
    assert entry['name']=='convolution' and entry['source_sha1']=='source' and entry['art_structural_sha1']=='art'
    assert entry['bundle_sha1']==compiler.sha1(b'convolution-values.js')
    assert entry['lesson_sha1']==entry['bundle_sha1']
    original=entry['runtime_sha1']
    (tmp_path/'convolution.js').write_text('changed')
    compiler.apply_implementations(registry,manifest)
    assert registry['vol1:fig-cnn']['runtime_sha1']!=original


def test_specialized_implementation_rejects_unselected_and_missing_assets(tmp_path,monkeypatch):
    monkeypatch.setattr(compiler,'EXT',tmp_path)
    entry=dict(name='demo',script='demo.js',stylesheet='demo.css')
    with pytest.raises(ValueError,match='selected compiled mechanism'):
        compiler.apply_implementations({}, {'missing':entry})
    with pytest.raises(ValueError,match='missing/invalid implementation'):
        compiler.apply_implementations({'selected':{}},{'selected':entry})


def test_specialized_shared_player_is_part_of_runtime_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(compiler,'EXT',tmp_path)
    for name in ('demo.js','demo.css','original-player.js','original-player.css'):
        (tmp_path/name).write_text(name)
    manifest={'selected':dict(name='demo',script='demo.js',stylesheet='demo.css',shared_player=True)}
    registry={'selected':{}}
    compiler.apply_implementations(registry,manifest)
    original=registry['selected']['runtime_sha1']
    assert registry['selected']['shared_player'] is True
    (tmp_path/'original-player.js').write_text('new shared runtime')
    compiler.apply_implementations(registry,manifest)
    assert registry['selected']['runtime_sha1']!=original
