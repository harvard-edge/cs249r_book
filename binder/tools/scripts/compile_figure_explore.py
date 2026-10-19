#!/usr/bin/env python3
"""Validate reviewed figure lessons and compile deterministic web registries.

The default mechanism mode uses mechanisms/selection.json and strict author chunks.
No manuscript changes unless --enroll is explicitly supplied. Unselected figures
remain static. Run --require-complete for release coverage checks. The explicit
--legacy mode is archival and uses the pre-mechanism lesson inventory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
EXT = ROOT / 'books/_extensions/mlsysbook-ext/figure-explore'
CATALOG = ROOT / 'books/_prototypes/vol1-web-animations/animation-catalog/catalog.json'
ATTR = re.compile(r'\s+data-explore(?:-key)?="[^"]*"')


def normalized_source(text):
    return ATTR.sub('', text)


def sha1(data):
    return hashlib.sha1(data).hexdigest()


def structural_svg(data):
    """Ignore metadata and generated identifiers, retain geometry and labels."""
    text = data.decode('utf-8')
    text = re.sub(r'<metadata\b[^>]*>[\s\S]*?</metadata\s*>', '', text)
    identifiers = {}
    def replace_id(match):
        original = match.group(2)
        identifiers.setdefault(original, f'id{len(identifiers)}')
        return 'id=' + match.group(1) + identifiers[original] + match.group(1)
    text = re.sub(r'\bid=(["\'])([^"\']+)\1', replace_id, text)
    text = re.sub(r'#([A-Za-z_][\w:.-]*)', lambda match: '#'+identifiers.get(match.group(1), match.group(1)), text)
    return sha1(text.encode('utf-8'))



def validate(row, original):
    key = row['key']
    for field in ('figure_id', 'source'):
        if row.get(field) != original[field]:
            raise ValueError(f'{key}: {field} does not match catalog')
    if row.get('status') not in ('ready', 'static', 'specialized', 'blocked'):
        raise ValueError(f'{key}: invalid status')
    if row['status'] != 'ready':
        return
    if row.get('artwork_reviewed') is not True:
        raise ValueError(f'{key}: ready without artwork review')
    if not row.get('script'):
        raise ValueError(f'{key}: highlight-only tours are archived; ready lessons require a stateful implementation')
    for field in ('name', 'script', 'stylesheet'):
        if not row.get(field):
            raise ValueError(f'{key}: missing {field}')


def figure_match(text, figure):
    # Match only the wrapper's opening identifier. Captions may contain quoted
    # braces (LaTeX, inline Python); a closing-brace regex is not an attribute parser.
    matches = list(re.finditer(r'^:{3,}\s*\{#' + re.escape(figure) + r'(?=[\s}])', text, re.M))
    if len(matches) != 1:
        raise ValueError(f'{figure}: expected one exact figure wrapper, found {len(matches)}')
    return matches[0]


def enrollment_attributes(text, figure):
    match = figure_match(text, figure)
    # Enrollment always sits immediately after the ID, before original attributes.
    return re.match(r'(?:\s+data-explore(?:-key)?="[^"]*")*', text[match.end():]).group()


def compile_registry(args):
    catalog = {r['key']: r for r in json.loads(CATALOG.read_text())['figures']}
    artwork = json.loads(Path(args.art_index).read_text())
    rows = {}
    invalid = []
    for path in sorted((EXT / 'lessons').glob('vol*-*.json')):
        for row in json.loads(path.read_text()):
            key = row['key']
            if key not in catalog or key in rows:
                raise ValueError(f'{path.name}: unknown or duplicate key {key}')
            try:
                validate(row, catalog[key])
            except ValueError as error:
                if args.require_complete:
                    raise
                invalid.append({'key': key, 'file': path.name, 'error': str(error)})
                continue
            rows[key] = row
    for path in sorted((EXT / 'lessons').glob('specialized-vol*.json')):
        for row in json.loads(path.read_text()):
            key = row['key']
            if key not in catalog:
                raise ValueError(f'{path.name}: unknown key {key}')
            if key in rows and rows[key]['status'] != 'specialized':
                raise ValueError(f'{key}: specialized replacement not reserved')
            validate(row, catalog[key])
            rows[key] = row
    for key, name, script in [('vol1:fig-tensor-memory-layout', 'tensor-memory-layout', 'tensor-memory-layout'), ('vol1:fig-tiling-diagram', 'matrix-tiling', 'matrix-tiling')]:
        rows[key] = {**catalog[key], 'status': 'ready', 'artwork_reviewed': True, 'name': name, 'script': script+'.js', 'stylesheet': script+'.css'}
    missing = sorted(set(catalog) - set(rows))
    unresolved = sorted(k for k, r in rows.items() if r['status'] == 'specialized')
    if args.require_complete and (missing or unresolved):
        raise ValueError(f'Incomplete coverage: missing={missing}, unresolved specialized={unresolved}')
    registry, volumes, source_cache = {}, {n: {} for n in range(1,5)}, {}
    for key, row in sorted(rows.items()):
        if row['status'] != 'ready':
            continue
        original = catalog[key]
        source = original['source']
        path = ROOT / source
        text = source_cache.setdefault(source, path.read_text())
        figure_match(text, original['figure_id'])
        art = artwork.get(key)
        if not art or not Path(art.get('asset', '')).is_file():
            raise ValueError(f'{key}: canonical art missing')
        asset = Path(art['asset'])
        raw = asset.read_bytes()
        from PIL import Image
        with Image.open(art['image']) as im:
            width, height = im.size
        specialized = bool(row.get('script'))
        entry = dict(name=row['name'] if specialized else 'guided-figure', script=row['script'] if specialized else 'guided-figure.js', stylesheet=row['stylesheet'] if specialized else 'guided-figure.css', source=source, source_sha1=sha1(normalized_source(text).encode()), figure_id=original['figure_id'], volume=original['volume'], art_sha1=sha1(raw), art_kind=asset.suffix.lstrip('.').lower(), art_width=width, art_height=height)
        if asset.suffix.lower() == '.svg':
            entry['art_structural_sha1'] = structural_svg(raw)
        for name in ('script', 'stylesheet'):
            if specialized and not (EXT / entry[name]).is_file():
                raise ValueError(f'{key}: missing {entry[name]}')
        registry[key] = entry
        if not specialized:
            lesson = {k: v for k, v in row.items() if k not in ('review_notes',)}
            lesson.update({k: v for k,v in entry.items() if k.startswith('art_')})
            volumes[original['volume']][key] = lesson
    outputs = {EXT/'figure-registry.json': json.dumps(registry, indent=2, ensure_ascii=False)+'\n'}
    for volume, lessons in volumes.items():
        outputs[EXT/f'guided-lessons-vol{volume}.js'] = '// Generated by compile_figure_explore.py; edit lessons/*.json.\nwindow.MLSysGuidedLessons = window.MLSysGuidedLessons || {};\nObject.assign(window.MLSysGuidedLessons, '+json.dumps(lessons, ensure_ascii=False, separators=(',', ':'))+');\n'
    if args.enroll or args.check:
        for original in catalog.values():
            source = original['source']
            source_cache.setdefault(source, (ROOT/source).read_text())
    if args.enroll:
        for source, text in source_cache.items():
            updated = normalized_source(text)
            for key, original in catalog.items():
                if original['source'] != source:
                    continue
                match = figure_match(updated, original['figure_id'])
                wrapper = match.group()
                if key in registry:
                    entry = registry[key]
                    wrapper += f' data-explore="{entry["name"]}" data-explore-key="{key}"'
                updated = updated[:match.start()]+wrapper+updated[match.end():]
            if normalized_source(updated) != normalized_source(text):
                raise AssertionError(f'{source}: enrollment changed static content')
            if updated != text:
                (ROOT/source).write_text(updated)
    for path, content in outputs.items():
        if args.check:
            if not path.is_file() or path.read_text() != content:
                raise ValueError(f'Outdated generated output: {path.relative_to(ROOT)}')
        else:
            path.write_text(content)
    if args.check:
        for key, original in catalog.items():
            wrapper = enrollment_attributes((ROOT/original['source']).read_text(), original['figure_id'])
            if key in registry:
                entry = registry[key]
                if f'data-explore="{entry["name"]}"' not in wrapper or f'data-explore-key="{key}"' not in wrapper:
                    raise ValueError(f'{key}: missing or wrong enrollment')
            elif ATTR.search(wrapper):
                raise ValueError(f'{key}: unready figure has stale enrollment')
    report = {'invalid': invalid, 'catalog_total': len(catalog), 'registered': len(registry), 'status_counts': dict(Counter(r['status'] for r in rows.values())), 'missing': missing, 'unresolved_specialized': unresolved, 'blocked': [k for k,r in rows.items() if r['status']=='blocked'], 'static': [k for k,r in rows.items() if r['status']=='static'], 'by_volume': dict(Counter(str(v['volume']) for v in registry.values()))}
    Path(args.report).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))





def source_revision(metadata):
    """Preserve recorded provenance across commits; content hashes govern freshness."""
    recorded=metadata.get('source_head')
    if isinstance(recorded,str) and recorded.strip():
        return recorded
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def bind_current_build(catalog_data, registry, outputs, build_root):
    """Bind only exact, complete clean-build evidence; never confer visual approval."""
    import os
    from bs4 import BeautifulSoup
    from validate_mechanism_lessons import strict_json, require
    for row in catalog_data['figures']:
        row['chapter_url']=''
        row['built_explore_present']=False
    metadata=catalog_data['metadata']
    metadata['build_root']=None
    metadata['build_markers_verified']=0
    metadata['build_route_count']=0
    if not build_root:
        return
    build=Path(build_root)
    if not build.is_absolute(): build=ROOT/build
    build=build.resolve()
    summary=strict_json(build/'summary.json')
    for volume in range(1,5):
        require(any(j.get('volume')==f'vol{volume}' and j.get('format')=='html' and j.get('chapter') is None and j.get('ok') is True for j in summary.get('jobs',[])),f'{build}: missing successful full vol{volume} HTML job')
        output=build/f'vol{volume}-html/output'
        assets = {'figure-explore.js', 'figure-views.js', 'figure-views.css'}
        for entry in registry.values():
            if entry['volume'] != volume:
                continue
            assets.update(entry[field] for field in ('script', 'stylesheet', 'bundle') if entry.get(field))
            if entry.get('shared_player'):
                assets.update(('original-player.js', 'original-player.css'))
        for name in sorted(assets):
            matches=list(output.rglob(name))
            require(len(matches)==1,f'{build}: expected exactly one emitted {name}')
            expected=outputs[EXT/name].encode('utf-8') if EXT/name in outputs else (EXT/name).read_bytes()
            require(matches[0].read_bytes()==expected,f'{build}: stale emitted {name}')
    chapters={}
    for row in catalog_data['figures']:
        chapter=build/f"vol{row['volume']}-html/output"/Path(row['source']).relative_to('books').with_suffix('.html')
        if chapter not in chapters:
            chapters[chapter]=BeautifulSoup(chapter.read_text(),'html.parser') if chapter.is_file() else None
        soup=chapters[chapter]
        node=soup.find(id=row['figure_id']) if soup else None
        entry=registry.get(row['key'])
        if entry:
            require(node is not None,f"{row['key']}: figure absent from configured build")
            expected={'data-mlsys-explore':entry['name'],'data-mlsys-explore-key':row['key'],'data-mlsys-art-sha1':entry.get('art_structural_sha1',entry['art_sha1']),'data-mlsys-art-kind':entry['art_kind']}
            if entry.get('lesson_sha1'):
                expected['data-mlsys-lesson-sha1']=entry['lesson_sha1']
            require(all(node.get(k)==v for k,v in expected.items()),f"{row['key']}: missing/stale built mechanism freshness marker")
            row['built_explore_present']=True
            row['runtime_verification']='Schema, declared invariants, companion calculations, clean four-volume HTML assets and freshness markers verified; actual themed chapter behavior review remains separate.'
        if node is not None:
            row['chapter_url']=os.path.relpath(chapter,CATALOG.parent)+'#'+row['figure_id']
    metadata['build_root']=os.path.relpath(build,ROOT)
    metadata['build_markers_verified']=sum(r['built_explore_present'] for r in catalog_data['figures'])
    metadata['build_route_count']=sum(bool(r['chapter_url']) for r in catalog_data['figures'])



def apply_review_record(catalog_data, registry, lessons, outputs, build_root, review_path):
    """Apply explicit, hash-bound implementation acceptance; never infer approval."""
    from validate_mechanism_lessons import strict_json, require, fields, array, string
    for row in catalog_data['figures']:
        row.pop('review_scope',None)
    metadata=catalog_data['metadata']
    metadata.pop('review_record',None)
    if not review_path:
        return
    require(build_root,'--review-record requires --build-root')
    build=Path(build_root)
    if not build.is_absolute(): build=ROOT/build
    build=build.resolve()
    path=Path(review_path)
    if not path.is_absolute(): path=ROOT/path
    record=strict_json(path)
    fields(record,'schema build_root registry_sha256 build_summary_sha256 lessons')
    for name in ('build_root','registry_sha256','build_summary_sha256'): string(record[name],True)
    require(type(record['schema']) is int and record['schema']==1,'unsupported review-record schema')
    bound=Path(record['build_root'])
    if not bound.is_absolute(): bound=ROOT/bound
    require(bound.resolve()==build,'review record build root differs from configured build')
    digest=lambda data:hashlib.sha256(data).hexdigest()
    require(record['registry_sha256']==digest(outputs[EXT/'figure-registry.json'].encode('utf-8')),'review registry hash mismatch')
    require(record['build_summary_sha256']==digest((build/'summary.json').read_bytes()),'review build summary hash mismatch')
    require(isinstance(record['lessons'],dict) and record['lessons'],'review record needs explicit lesson acceptance')
    by_key={row['key']:row for row in catalog_data['figures']}
    evidence_dir=(CATALOG.parent/'evidence').resolve()
    for key,review in record['lessons'].items():
        require(key in registry and key in lessons,f'{key}: review names unregistered lesson')
        fields(review,'verdict coverage evidence reviewer_type limitations')
        require(review['verdict']=='accepted',f'{key}: unsupported review verdict')
        string(review['reviewer_type'],True); string(review['limitations'],True)
        coverage=review['coverage']; fields(coverage,'cases profiles endpoints motion controls')
        array(coverage['cases'],1); array(coverage['profiles'],1)
        case_ids={'base'}|{v['id'] for v in lessons[key].get('variants',[])}
        require(len(set(coverage['cases']))==len(coverage['cases']) and set(coverage['cases'])<=case_ids,f'{key}: unknown/duplicate reviewed case')
        profile_ids={'desktop-light','desktop-dark','narrow-light','narrow-dark'}
        require(len(set(coverage['profiles']))==len(coverage['profiles']) and set(coverage['profiles'])<=profile_ids,f'{key}: unknown/duplicate reviewed profile')
        for name in ('endpoints','motion','controls'): string(coverage[name],True)
        array(review['evidence'],1)
        for evidence in review['evidence']:
            fields(evidence,'path sha256'); string(evidence['path'],True); string(evidence['sha256'],True)
            require(not Path(evidence['path']).is_absolute(),f'{key}: evidence path must be repository-relative')
            report=(ROOT/evidence['path']).resolve()
            require(report.is_relative_to(evidence_dir) and report.is_file(),f'{key}: evidence must be an existing public catalog/evidence report')
            require(evidence['sha256']==digest(report.read_bytes()),f'{key}: review evidence hash mismatch')
        row=by_key[key]
        row['status']=row['actual_status']='implementation-review-passed'
        row['review_scope']=review
        row['actual_review_basis']='Explicit hash-bound implementation acceptance, limited to the recorded cases, profiles, evidence, and reviewer limitations.'
        row['visual_verification']='Hash-bound implementation acceptance for cases '+', '.join(coverage['cases'])+'; profiles '+', '.join(coverage['profiles'])+'. Endpoints: '+coverage['endpoints']+' Motion: '+coverage['motion']+' Controls: '+coverage['controls']+' Reviewer: '+review['reviewer_type']+'. Limits: '+review['limitations']
    metadata['review_record']={'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,'sha256':digest(path.read_bytes()),'accepted_keys':sorted(record['lessons'])}
    metadata['actual_statuses']=dict(Counter(row['actual_status'] for row in catalog_data['figures']))
    metadata['review_limit']='Recorded implementation acceptance is limited to the declared cases, profiles, and evidence. Selected entries without acceptance remain pending; static figures retain their static treatment. Simulated reviews are not student testing and establish no learning gains.'


def apply_implementations(registry, manifests=None):
    """Replace selected implementations, preserving every other compiled record.

    Explicit production factories retain the same source/artwork provenance as
    their selected mechanism. Their own runtime/data hashes replace the generic
    scene's bundle identity; an unknown or stale manifest fails the whole build.
    """
    path = EXT/'mechanisms/implementations.json'
    implementations = json.loads(path.read_text()) if manifests is None and path.exists() else (manifests or {})
    for key, implementation in implementations.items():
        if key not in registry:
            raise ValueError(f'{key}: implementation must replace a selected compiled mechanism')
        if set(implementation)-{'name','script','stylesheet','bundle','scope','question','shared_player'}:
            raise ValueError(f'{key}: unknown implementation manifest field')
        for field in ('name','script','stylesheet'):
            value=implementation.get(field)
            if not isinstance(value,str) or not value or '/' in value or '\\' in value:
                raise ValueError(f'{key}: invalid implementation {field}')
        assets=[]
        for field in ('script','stylesheet','bundle'):
            value=implementation.get(field)
            if value is None: continue
            if not isinstance(value,str) or '/' in value or '\\' in value or not (EXT/value).is_file():
                raise ValueError(f'{key}: missing/invalid implementation {field}')
            assets.append((EXT/value).read_bytes())
        if 'shared_player' in implementation and type(implementation['shared_player']) is not bool:
            raise ValueError(f'{key}: shared_player must be boolean')
        if implementation.get('shared_player'):
            for filename in ('original-player.js','original-player.css'):
                if not (EXT/filename).is_file(): raise ValueError(f'{key}: missing shared player asset {filename}')
                assets.append((EXT/filename).read_bytes())
        entry=dict(registry[key])
        for field in ('bundle','bundle_sha1','lesson_sha1','runtime_sha1'): entry.pop(field,None)
        entry.update({field:implementation[field] for field in ('name','script','stylesheet')})
        entry['runtime_sha1']=sha1(b'\x00'.join(assets))
        if implementation.get('shared_player'): entry['shared_player']=True
        else: entry.pop('shared_player',None)
        if implementation.get('bundle'):
            entry['bundle']=implementation['bundle']
            entry['bundle_sha1']=sha1((EXT/entry['bundle']).read_bytes())
            entry['lesson_sha1']=entry['bundle_sha1']
        registry[key]=entry
    return implementations


def compile_mechanisms(args):
    """Enroll only explicitly selected, validated authored mechanisms."""
    import sys
    import subprocess
    sys.path.insert(0, str(Path(__file__).parent))
    from validate_mechanism_lessons import load_chunks, strict_json, require
    catalog_data = strict_json(CATALOG)
    catalog = {r['key']: r for r in catalog_data['figures']}
    require(len(catalog) == len(catalog_data['figures']), 'duplicate catalog key')
    artwork = strict_json(args.art_index)
    selection = strict_json(args.selection)
    selected = {r['key']:r for r in selection}
    require(len(selected) == len(selection), 'duplicate selection key')
    require(set(selected) <= set(catalog), 'selection missing from current catalog')
    lessons = load_chunks(EXT/'mechanisms')
    require(set(lessons) <= set(selected), 'authored lesson outside selected roster')
    missing = sorted(set(selected)-set(lessons))
    require(not args.require_complete or not missing, f'missing selected mechanisms: {missing}')
    volumes = {v:{} for v in range(1,5)}
    lesson_hashes = {}
    for key, lesson in sorted(lessons.items()):
        canonical = json.dumps(lesson, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        digest = sha1(canonical.encode('utf-8'))
        lesson_hashes[key] = digest
        volumes[catalog[key]['volume']][key] = {'canonical':canonical, 'sha1':digest}
    outputs = {}
    bundle_hashes = {}
    for volume, data in volumes.items():
        filename = f'mechanism-lessons-vol{volume}.js'
        content = '// Generated by compile_figure_explore.py; edit mechanisms/*.json.\nwindow.MLSysMechanismLessons = window.MLSysMechanismLessons || {};\nObject.assign(window.MLSysMechanismLessons, '+json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')+');\n'
        outputs[EXT/filename] = content
        bundle_hashes[volume] = sha1(content.encode('utf-8'))
    registry, source_cache = {}, {}
    for key in sorted(lessons):
        original = catalog[key]; source = original['source']; volume = original['volume']
        require(selected[key]['source'] == source, f'{key}: selected source differs from catalog')
        text = source_cache.setdefault(source,(ROOT/source).read_text())
        figure_match(text,original['figure_id'])
        art = artwork.get(key)
        require(art and art.get('source') == source, f'{key}: missing/mismatched canonical artwork')
        asset = Path(art.get('asset',''))
        require(asset.is_file(), f'{key}: canonical asset missing')
        raw = asset.read_bytes()
        require(art.get('asset_sha1') == sha1(raw), f'{key}: canonical art index stale')
        from PIL import Image
        with Image.open(art['image']) as im: width,height = im.size
        entry = dict(name='mechanism-scene',script='mechanism-scene.js',stylesheet='mechanism-scene.css',source=source,source_sha1=sha1(normalized_source(text).encode()),figure_id=original['figure_id'],volume=volume,art_sha1=sha1(raw),art_kind=asset.suffix.lstrip('.').lower(),art_width=width,art_height=height,lesson_sha1=lesson_hashes[key],bundle=f'mechanism-lessons-vol{volume}.js',bundle_sha1=bundle_hashes[volume])
        if asset.suffix.lower()=='.svg': entry['art_structural_sha1']=structural_svg(raw)
        for field in ('script','stylesheet'): require((EXT/entry[field]).is_file(),f'{key}: runtime asset missing {entry[field]}')
        entry['runtime_sha1']=sha1(((EXT/entry['script']).read_bytes()+b'\x00'+(EXT/entry['stylesheet']).read_bytes()))
        registry[key]=entry
    implementations = apply_implementations(registry)
    outputs[EXT/'figure-registry.json'] = json.dumps(registry,indent=2,ensure_ascii=False)+'\n'
    # Reconcile implementation metadata without presenting validation as visual approval.
    for row in catalog_data['figures']:
        key = row['key']; lesson = lessons.get(key)
        row['actual_treatment'] = registry[key]['name'] if lesson else 'static'
        row['status'] = row['actual_status'] = 'implemented-awaiting-review' if lesson else 'static'
        row['first_wave'] = key in registry
        row['built_explore_present'] = False
        row['runtime_verification'] = 'Schema, declared invariant, and companion calculation validation passed; actual themed chapter behavior review pending.' if lesson else 'Static figure retained; no mechanism enrolled.'
        row['visual_verification'] = 'Actual themed chapter state/case inspection pending; validation is not visual approval.' if lesson else 'Static figure retained.'
        row['actual_review_basis'] = 'Authored mechanism passes strict schema, declared invariants, and companion calculations; current implementation inspection remains pending.' if lesson else 'Static figure retained; no mechanism is enrolled.'
        row['actual_question'] = implementations.get(key,{}).get('question',lesson['question']) if lesson else ''
        row['actual_scope'] = implementations.get(key,{}).get('scope',lesson['scope']) if lesson else ''
        row['actual_steps'] = [{'id':step['id'],'title':step['title'],'explanation':step['explanation'],'result':step['snapshot']['result']} for step in lesson['steps']] if lesson and key not in implementations else []
    metadata = catalog_data['metadata']
    metadata.setdefault('proposal_source_head',metadata.get('source_head',''))
    metadata['source_head']=source_revision(metadata)
    metadata['source_head_role']='Recorded source baseline; registry source/art/runtime/lesson/bundle hashes bind the actual delivered contents.'
    metadata['count'] = len(catalog)
    metadata['counts_by_volume'] = dict(Counter(str(r['volume']) for r in catalog.values()))
    metadata['stateful_by_volume'] = dict(Counter(str(r['volume']) for r in registry.values()))
    metadata['first_wave'] = sorted(registry)
    metadata['actual_treatments'] = dict(Counter(r['actual_treatment'] for r in catalog_data['figures']))
    metadata['actual_statuses'] = dict(Counter(r['actual_status'] for r in catalog_data['figures']))
    metadata['review_limit'] = 'Current selected mechanisms pass strict authoring/calculation validation. Actual chapter visual and behavioral review remains pending; no student testing or learning-gain claim.'
    bind_current_build(catalog_data,registry,outputs,args.build_root)
    apply_review_record(catalog_data,registry,lessons,outputs,args.build_root,args.review_record)
    outputs[CATALOG] = json.dumps(catalog_data,indent=2,ensure_ascii=False)+'\n'
    if args.enroll or args.check:
        for original in catalog.values(): source_cache.setdefault(original['source'],(ROOT/original['source']).read_text())
    updated_sources = {}
    if args.enroll:
        for source,text in source_cache.items():
            updated=normalized_source(text)
            for key,original in catalog.items():
                if original['source'] != source: continue
                match=figure_match(updated,original['figure_id'])
                addition=f' data-explore="{registry[key]["name"]}" data-explore-key="{key}"' if key in registry else ''
                updated=updated[:match.end()]+addition+updated[match.end():]
            require(normalized_source(updated)==normalized_source(text),f'{source}: static bytes changed')
            if updated!=text: updated_sources[ROOT/source]=updated
    if args.check:
        for path,content in outputs.items(): require(path.is_file() and path.read_text()==content,f'Outdated generated output: {path.relative_to(ROOT)}')
        for key,original in catalog.items():
            attrs=enrollment_attributes((ROOT/original['source']).read_text(),original['figure_id'])
            expected=f' data-explore="{registry[key]["name"]}" data-explore-key="{key}"' if key in registry else ''
            require(attrs==expected,f'{key}: stale/missing enrollment')
    else:
        # No source or output mutation until every lesson/art/calculation validates.
        for path,content in {**outputs,**updated_sources}.items():
            if not path.is_file() or path.read_text()!=content: path.write_text(content)
    report={'catalog_total':len(catalog),'selected':len(selected),'registered':len(registry),'static':len(catalog)-len(registry),'missing_selected':missing,'by_volume':dict(Counter(str(v['volume']) for v in registry.values())),'source_files_updated':len(updated_sources),'review_status':'Validated schema/calculations; actual chapter visual and behavior review required.'}
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--art-index', default='/tmp/figures-animation-art-current/index.json')
    parser.add_argument('--review-record', help='Explicit public hash-bound implementation acceptance; requires --build-root.')
    parser.add_argument('--build-root', help='Bind exact successful clean four-volume HTML build; does not grant visual approval.')
    parser.add_argument('--report', default='/tmp/figure-explore-compile-report.json')
    parser.add_argument('--enroll', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--mechanisms', action='store_true', help='Compile only selected strict authored mechanisms.')
    parser.add_argument('--selection', default=str(EXT/'mechanisms/selection.json'))
    parser.add_argument('--legacy', action='store_true', help='Explicit archival compiler mode for pre-mechanism lessons.')
    parser.add_argument('--partial', action='store_true', help='Compile available reviewed lessons (the default).')
    args = parser.parse_args()
    try:
        if args.legacy and args.mechanisms:
            raise ValueError('--legacy and --mechanisms are mutually exclusive')
        compile_registry(args) if args.legacy else compile_mechanisms(args)
    except (ValueError, KeyError) as error:
        parser.exit(1, str(error)+'\n')

if __name__ == '__main__':
    main()
