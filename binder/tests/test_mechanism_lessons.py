"""Fail-closed authoring checks across complete snapshots and variants."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('mechanism_validator',ROOT/'binder/tools/scripts/validate_mechanism_lessons.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

@pytest.fixture
def lesson():
    data=json.loads((ROOT/'books/_extensions/mlsysbook-ext/figure-explore/mechanisms/vol1-a.json').read_text())
    return copy.deepcopy(next(row for row in data if row['key']=='vol1:fig-cnn'))


def test_authored_convolution_passes(lesson):
    assert v.validate_lesson(lesson) is lesson


def test_unknown_fields_fail_closed(lesson):
    lesson['steps'][1]['snapshot']['objects']['unexpected']={}
    with pytest.raises(ValueError,match='object set mismatch'):v.validate_lesson(lesson)


def test_narration_is_not_a_mechanism(lesson):
    lesson['steps'][1]['snapshot']=copy.deepcopy(lesson['steps'][0]['snapshot'])
    lesson['steps'][1]['snapshot']['result']='A different narration does not compute an output.'
    with pytest.raises(ValueError,match='only narration'):v.validate_lesson(lesson)


def test_every_variant_is_validated(lesson):
    lesson['variants']=[dict(id='broken',label='Broken case',scope='Deliberately invalid case for validation.',steps=copy.deepcopy(lesson['steps']))]
    lesson['variants'][0]['steps'][-1]['snapshot']['tokens']=[{'id':'bad'}]
    with pytest.raises(ValueError,match='missing fields'):v.validate_lesson(lesson)


def test_nonfinite_and_duplicate_json_fail(tmp_path):
    path=tmp_path/'bad.json'
    for content in ('{"a":NaN}','{"a":1,"a":2}'):
        path.write_text(content)
        with pytest.raises(ValueError):v.strict_json(path)


def test_offsurface_anchor_fails(lesson):
    lesson['steps'][1]['transition']['to']={'object':lesson['objects'][0]['id'],'index':9999}
    with pytest.raises(ValueError,match='above range'):v.validate_lesson(lesson)


def test_mechanism_bundle_freshness_at_render(tmp_path):
    """Missing or stale lesson bundles leave the complete static figure usable."""
    import hashlib
    import shutil
    import subprocess
    if not shutil.which('quarto'):pytest.skip('Quarto required')
    ext=tmp_path/'books/_extensions/mlsysbook-ext/figure-explore'
    ext.mkdir(parents=True)
    real=ROOT/'books/_extensions/mlsysbook-ext/figure-explore'
    for filename in ('figure-explore.lua','figure-explore.js','figure-views.js','figure-views.css','mechanism-scene.js','mechanism-scene.css'):
        shutil.copy(real/filename,ext/filename)
    source=tmp_path/'books/test.qmd'
    static=f'''---
filters:
  - {ext/'figure-explore.lua'}
---

::: {{#fig-demo fig-cap="**Original caption**: Retained static meaning."}}

Static data and complete diagram.

:::
'''
    enrolled=static.replace('#fig-demo','#fig-demo data-explore="mechanism-scene" data-explore-key="vol1:fig-demo"')
    source.write_text(enrolled)
    bundle=ext/'mechanism-lessons-vol1.js'
    bundle.write_text('window.MLSysMechanismLessons = {};\n')
    registry={'vol1:fig-demo':dict(name='mechanism-scene',figure_id='fig-demo',source='books/test.qmd',source_sha1=hashlib.sha1(static.encode()).hexdigest(),volume=1,script='mechanism-scene.js',stylesheet='mechanism-scene.css',lesson_sha1='0'*40,bundle=bundle.name,bundle_sha1=hashlib.sha1(bundle.read_bytes()).hexdigest(),art_sha1='0'*40,art_kind='svg')}
    (ext/'figure-registry.json').write_text(json.dumps(registry))
    def render():
        result=subprocess.run(['quarto','render',str(source),'--to','html'],cwd=tmp_path,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        return (source.with_suffix('.html')).read_text(),result.stderr
    html,_=render()
    assert 'data-mlsys-lesson-sha1=' in html
    assert 'mechanism-lessons-vol1.js' in html
    assert 'Retained static meaning.' in html
    bundle.write_text('window.MLSysMechanismLessons = {stale:true};\n')
    html,stderr=render()
    assert 'stale mechanism bundle' in stderr
    assert 'data-mlsys-explore=' not in html
    assert 'Retained static meaning.' in html and 'Static data and complete diagram.' in html
    bundle.unlink()
    html,stderr=render()
    assert 'stale mechanism bundle' in stderr
    assert 'data-mlsys-explore=' not in html


def test_two_column_narrow_requires_explicit_opt_in(lesson):
    # Reuse the valid wide grid to create an expressly declared paired layout.
    for obj in lesson['objects']:
        obj['layout']['narrow']=copy.deepcopy(obj['layout']['wide'])
    lesson['narrowColumns']=2
    v.validate_lesson(lesson)
    del lesson['narrowColumns']
    with pytest.raises(ValueError,match='above range'):v.validate_lesson(lesson)


def test_two_column_narrow_still_rejects_overlap(lesson):
    lesson['narrowColumns']=2
    lesson['objects'][1]['layout']['narrow']=copy.deepcopy(lesson['objects'][0]['layout']['narrow'])
    with pytest.raises(ValueError,match='layout overlap'):v.validate_lesson(lesson)


@pytest.mark.parametrize('equal_axes',[False,True])
def test_plot_equal_axes_boolean_opt_in(equal_axes):
    obj=dict(id='position',kind='plot',label='Planar position',role='output',config=dict(x=dict(min=0,max=3,label='x (m)'),y=dict(min=0,max=2,label='y (m)'),equalAxes=equal_axes),layout=dict(wide=dict(row=0,column=0),narrow=dict(row=0,column=0)))
    v.object_config(obj)
    obj['config']['equalAxes']=1
    with pytest.raises(ValueError,match='equalAxes must be boolean'):v.object_config(obj)


def test_empty_or_missing_mechanism_directory_fails(tmp_path):
    with pytest.raises(ValueError,match='does not exist'):v.load_chunks(tmp_path/'missing')
    with pytest.raises(ValueError,match='no mechanism chunks'):v.load_chunks(tmp_path)


def test_validator_help_does_not_validate_empty_input():
    import subprocess
    import sys
    result=subprocess.run([sys.executable,str(ROOT/'binder/tools/scripts/validate_mechanism_lessons.py'),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and 'usage:' in result.stdout
    assert '"validated"' not in result.stdout
