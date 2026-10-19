#!/usr/bin/env python3
"""Strict, dependency-free validation of authored mechanism snapshots.

This validates structural and declared numerical invariants. It does not replace
technical review of the model or visual review in the actual chapter.
"""
from __future__ import annotations
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROLES = 'input output compute memory communication control reference'.split()
KINDS = 'sequence matrix pool lanes plot numberLine register chamber'.split()
FAMILIES = 'sequence cache spatial reduction schedule ownership curve control communication representation'.split()
TRANSITIONS = 'move copy write transform release window execute reveal'.split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, required, optional=''):
    require(isinstance(value, dict), 'expected object')
    required, optional = set(required.split()), set(optional.split())
    require(required <= value.keys(), f'missing fields: {required-value.keys()}')
    require(value.keys() <= required | optional, f'unknown fields: {value.keys()-required-optional}')


def string(value, nonempty=False):
    require(isinstance(value, str) and (not nonempty or value.strip()), 'expected text')
    require(not re.search(r'<[/!A-Za-z]|```|\\(?:begin|frac|times)\b', value), 'markup/executable content is unsupported')


def number(value, low=None, high=None, integer=False):
    require(type(value) in (int, float) and math.isfinite(value), 'expected finite number')
    require(not integer or type(value) is int, 'expected integer')
    require(low is None or value >= low, 'number below range')
    require(high is None or value <= high, 'number above range')


def array(value, low=0, high=None):
    require(isinstance(value, list), 'expected array')
    require(len(value) >= low and (high is None or len(value) <= high), 'array length outside limits')


def enum(value, choices):
    require(value in choices, f'unsupported value {value!r}')


def identifier(value):
    string(value)
    require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', value), f'invalid ID {value!r}')


def ids(values):
    result = []
    for value in values:
        identifier(value['id'])
        result.append(value['id'])
    require(len(result) == len(set(result)), 'duplicate IDs')
    return set(result)


def textvalue(value):
    if value is None:
        return
    if isinstance(value, str):
        string(value)
    else:
        number(value)


def axis(value, label='label'):
    fields(value, f'min max {label}', 'ticks')
    number(value['min']); number(value['max'])
    require(value['min'] < value['max'], 'axis must increase')
    string(value[label], True)
    if 'ticks' in value:
        array(value['ticks'])
        for tick in value['ticks']:
            number(tick, value['min'], value['max'])


def capacity(obj):
    c = obj['config']
    return c['rows']*c['columns'] if obj['kind'] == 'matrix' else len(c['lanes']) if obj['kind'] == 'lanes' else c.get('slots')


def anchor(value, objects):
    fields(value, 'object', 'index x y')
    require(value['object'] in objects, 'anchor object missing')
    obj = objects[value['object']]; kind = obj['kind']; c = obj['config']
    require(not ('index' in value and ('x' in value or 'y' in value)), 'mixed anchor coordinates')
    if 'index' in value:
        require(capacity(obj) is not None, 'index unsupported for object')
        number(value['index'], 0, capacity(obj)-1, True)
    elif 'x' in value or 'y' in value:
        require(kind in ('plot', 'numberLine') and 'x' in value, 'coordinates unsupported')
        if kind == 'plot':
            require('y' in value, 'plot anchor needs y')
            number(value['x'], c['x']['min'], c['x']['max']); number(value['y'], c['y']['min'], c['y']['max'])
        else:
            require('y' not in value, 'number line has no y')
            number(value['x'], c['min'], c['max'])


def object_config(obj, narrow_columns=1):
    fields(obj, 'id kind label role config layout')
    enum(obj['kind'], KINDS); enum(obj['role'], ROLES); string(obj['label'], True)
    c = obj['config']; k = obj['kind']
    required, optional = {
        'sequence': ('slots', 'slotLabels'), 'matrix': ('rows columns', 'rowLabels columnLabels'),
        'pool': ('slots', 'slotLabels columns'), 'lanes': ('lanes start end axisLabel', 'ticks'),
        'plot': ('x y', 'equalAxes'), 'numberLine': ('min max axisLabel', 'ticks'),
        'register': ('type', 'unit'), 'chamber': ('slots', '')}[k]
    fields(c, required, optional)
    if k in ('sequence', 'pool', 'chamber'):
        number(c['slots'], 1, 12 if k == 'chamber' else 24, True)
    if k == 'matrix':
        number(c['rows'], 1, 8, True); number(c['columns'], 1, 8, True)
    if k == 'pool' and 'columns' in c:
        number(c['columns'], 1, c['slots'], True)
    for name, size in [('slotLabels', c.get('slots')), ('rowLabels', c.get('rows')), ('columnLabels', c.get('columns'))]:
        if name in c:
            array(c[name]); require(len(c[name]) == size, 'label count mismatch')
            for label in c[name]: string(label)
    if k == 'lanes':
        array(c['lanes'], 1, 6)
        for label in c['lanes']: string(label, True)
        axis({'min':c['start'], 'max':c['end'], 'label':c['axisLabel'], **({'ticks':c['ticks']} if 'ticks' in c else {})})
    if k == 'numberLine': axis(c, 'axisLabel')
    if k == 'plot':
        axis(c['x']); axis(c['y'])
        if 'equalAxes' in c: require(type(c['equalAxes']) is bool,'equalAxes must be boolean')
    if k == 'register':
        string(c['type'], True)
        if 'unit' in c: string(c['unit'])
    fields(obj['layout'], 'wide narrow')
    for mode, columns in [('wide', 2), ('narrow', narrow_columns)]:
        pos = obj['layout'][mode]; fields(pos, 'row column', 'columnSpan')
        number(pos['row'], 0, 9, True); number(pos['column'], 0, columns-1, True)
        number(pos.get('columnSpan', 1), 1, columns-pos['column'], True)


def state(value, obj):
    k = obj['kind']; c = obj['config']
    required, optional = {'sequence':('values status',''), 'matrix':('values status','window'), 'pool':('slots status',''), 'lanes':('segments status','cursor'), 'plot':('series markers status',''), 'numberLine':('levels markers status',''), 'register':('value status','previous expression'), 'chamber':('values status','')}[k]
    fields(value, required, optional); string(value['status'])
    if k in ('sequence','matrix','chamber'):
        array(value['values']); require(len(value['values']) == capacity(obj), 'cell count mismatch')
        for cell in value['values']: textvalue(cell)
    if k == 'matrix' and value.get('window') is not None:
        w = value['window']; fields(w, 'row column rows columns')
        for name in ('row','column'): number(w[name], 0, None, True)
        for name in ('rows','columns'): number(w[name], 1, None, True)
        require(w['row']+w['rows'] <= c['rows'] and w['column']+w['columns'] <= c['columns'], 'window outside matrix')
    if k == 'pool':
        array(value['slots']); require(len(value['slots']) == c['slots'], 'pool length mismatch')
        for slot in value['slots']:
            if slot is not None:
                fields(slot, 'value owner status'); textvalue(slot['value']); require(slot['value'] is not None, 'resident slot has null value')
                string(slot['owner'], True); enum(slot['status'], 'allocated retained ready inflight shared'.split())
    if k == 'register':
        textvalue(value['value'])
        if 'previous' in value: textvalue(value['previous'])
        if 'expression' in value: string(value['expression'])
    if k == 'lanes':
        array(value['segments']); ids(value['segments'])
        for s in value['segments']:
            fields(s, 'id lane start end label state role'); number(s['lane'], 0, len(c['lanes'])-1, True)
            number(s['start'], c['start'], c['end']); number(s['end'], c['start'], c['end']); require(s['start'] < s['end'], 'empty task')
            string(s['label']); enum(s['state'], 'waiting running done blocked'.split()); enum(s['role'], ROLES)
        if value.get('cursor') is not None: number(value['cursor'], c['start'], c['end'])
    if k == 'plot':
        array(value['series']); array(value['markers']); ids(value['series']+value['markers'])
        for s in value['series']:
            fields(s, 'id label role points'); string(s['label']); enum(s['role'], ROLES); array(s['points'], 1)
            for p in s['points']:
                array(p); require(len(p)==2, 'point must have two coordinates'); number(p[0], c['x']['min'], c['x']['max']); number(p[1], c['y']['min'], c['y']['max'])
        for m in value['markers']:
            fields(m, 'id x y label'); string(m['label']); number(m['x'], c['x']['min'], c['x']['max']); number(m['y'], c['y']['min'], c['y']['max'])
    if k == 'numberLine':
        array(value['levels']); require(value['levels'] == sorted(set(value['levels'])), 'levels unsorted or duplicated')
        for n in value['levels']: number(n, c['min'], c['max'])
        array(value['markers']); ids(value['markers'])
        for m in value['markers']:
            fields(m, 'id value label role'); number(m['value'], c['min'], c['max']); string(m['label']); enum(m['role'], ROLES)


def semantic(snapshot):
    """Exclude narration, styling, metrics and relationship highlights as proof."""
    def physical(value):
        if isinstance(value,dict): return {k:physical(v) for k,v in value.items() if k not in ('status','label','role','expression','previous','cursor')}
        if isinstance(value,list): return [physical(v) for v in value]
        return value
    values = physical(snapshot['objects'])
    tokens = [{k:v for k,v in t.items() if k not in ('role','state','label')} for t in snapshot['tokens']]
    return values, tokens


def validate_lesson(lesson):
    fields(lesson, 'schema key title question scope family objects steps invariants', 'connections checks variants details prediction narrowColumns')
    require(type(lesson['schema']) is int and lesson['schema']==1, 'unsupported schema')
    require(re.fullmatch(r'vol[1-4]:fig-[A-Za-z0-9_-]+', lesson['key']), 'invalid lesson key')
    for name in ('title','question','scope'): string(lesson[name], True)
    enum(lesson['family'], FAMILIES); array(lesson['objects'],2,12); ids(lesson['objects'])
    objects = {o['id']:o for o in lesson['objects']}
    number(lesson.get('narrowColumns',1),1,2,True)
    for obj in objects.values(): object_config(obj,lesson.get('narrowColumns',1))
    for mode in ('wide','narrow'):
        occupied = set()
        for obj in objects.values():
            p=obj['layout'][mode]
            cells={(p['row'], col) for col in range(p['column'],p['column']+p.get('columnSpan',1))}
            require(not cells & occupied, 'layout overlap'); occupied |= cells
    array(lesson['invariants'],1)
    for inv in lesson['invariants']: string(inv, True)
    connections=lesson.get('connections',[]); array(connections,0,8); connection_ids=ids(connections)
    for conn in connections:
        fields(conn,'id from to kind label'); anchor(conn['from'],objects); anchor(conn['to'],objects); string(conn['label']); enum(conn['kind'],'read write copy move dependency feedback association'.split())
    checks=lesson.get('checks',[]); array(checks)
    for check in checks:
        kind=check.get('kind')
        if kind in ('metricEqualsOccupied','metricEqualsNonNull'): fields(check,'kind metric object'); require(check['object'] in objects, 'check object missing')
        elif kind=='constantValues': fields(check,'kind object'); require(check['object'] in objects, 'check object missing')
        elif kind=='nondecreasingMetric': fields(check,'kind metric')
        elif kind=='sum':
            fields(check,'kind sources target'); array(check['sources'],1)
            require(all(x in objects and objects[x]['kind']=='register' for x in check['sources']+[check['target']]), 'sum requires registers')
        else: raise ValueError('unsupported check')
    variants=lesson.get('variants',[]); array(variants,0,4); ids(variants)
    require(all(v['id']!='base' for v in variants), 'reserved variant ID')
    for v in variants:
        fields(v,'id label scope steps','invariants'); string(v['label'],True); string(v['scope'],True)
        if 'invariants' in v:
            array(v['invariants'])
            for inv in v['invariants']: string(inv,True)
    metric_signature=None
    case_ids=[]
    for case in [lesson]+variants:
        steps=case['steps']; array(steps,4,9); step_ids=ids(steps); case_ids.append(step_ids)
        previous=None; first=None
        for index,step in enumerate(steps):
            try:
                fields(step,'id title explanation snapshot transition'); string(step['title'],True); string(step['explanation'],True); require(len(step['explanation'])<=280,'explanation too long')
                snap=step['snapshot']; fields(snap,'objects tokens metrics activeConnections result'); string(snap['result'],True)
                require(isinstance(snap['objects'],dict) and snap['objects'].keys()==objects.keys(),'snapshot object set mismatch')
                for key,obj in objects.items(): state(snap['objects'][key],obj)
                array(snap['tokens'],0,48); token_ids=ids(snap['tokens'])
                for token in snap['tokens']:
                    fields(token,'id label value role owner at state','sourceId sourceAt'); string(token['label']); textvalue(token['value']); enum(token['role'],ROLES); string(token['owner']); anchor(token['at'],objects); enum(token['state'],'resident copy inflight consumed'.split())
                    require(not (token['role']=='memory' or objects[token['at']['object']]['role']=='memory') or token['owner'].strip(),'memory token without owner')
                    if token['state']=='copy': require(('sourceId' in token) != ('sourceAt' in token),'copy needs exactly one provenance')
                    if 'sourceId' in token: require(token['sourceId'] in token_ids and token['sourceId']!=token['id'],'copy source missing/self')
                    if 'sourceAt' in token: anchor(token['sourceAt'],objects)
                array(snap['metrics'],0,6); ids(snap['metrics'])
                for metric in snap['metrics']:
                    fields(metric,'id label value','unit'); string(metric['label'],True); textvalue(metric['value']); require(metric['value'] is not None,'null metric')
                    if 'unit' in metric: string(metric['unit'])
                signature=sorted((m['id'],m.get('unit','')) for m in snap['metrics'])
                if metric_signature is None: metric_signature=signature
                require(signature==metric_signature,'metric IDs/units changed')
                array(snap['activeConnections']); require(set(snap['activeConnections'])<=connection_ids,'unknown active connection')
                transition=step['transition']
                if index==0: require(transition is None,'first transition must be null'); first=snap
                else:
                    fields(transition,'kind subjects meaning','from to duration'); enum(transition['kind'],TRANSITIONS); string(transition['meaning'],True); array(transition['subjects'])
                    for name in ('from','to'):
                        if name in transition: anchor(transition[name],objects)
                    if 'duration' in transition: number(transition['duration'],300,1600)
                    known=set(objects)|connection_ids
                    for endpoint in (previous,snap):
                        known |= {t['id'] for t in endpoint['tokens']}
                        for s in endpoint['objects'].values():
                            for name in ('segments','series','markers'): known |= {x['id'] for x in s.get(name,[])}
                    require(set(transition['subjects'])<=known,'unresolved transition subject')
                    require(semantic(previous)!=semantic(snap),'action changes only narration/style/metrics')
                    old={t['id']:t for t in previous['tokens']}
                    if transition['kind']=='copy':
                        for token in snap['tokens']:
                            if token['id'] in old: require(token['owner']==old[token['id']]['owner'],'copy silently changes ownership')
                            if token['state']=='copy' and 'sourceId' in token:
                                source=token['sourceId']; current={t['id']:t for t in snap['tokens']}
                                require(source in old,'copy source was not present before read')
                                require(all(old[source][f]==current[source][f] for f in ('value','owner','at')),'copy removes or modifies source residency')
                            if token['state']=='copy' and 'sourceAt' in token:
                                location=token['sourceAt']; objid=location['object']; kind=objects[objid]['kind']
                                if 'index' in location and kind in ('pool','sequence','matrix','chamber'):
                                    cells='slots' if kind=='pool' else 'values'; position=location['index']
                                    before=previous['objects'][objid][cells][position]; after=snap['objects'][objid][cells][position]
                                    require(before is not None and before==after,'copy modifies or clears source cell')

                metrics={m['id']:m['value'] for m in snap['metrics']}
                for check in checks:
                    kind=check['kind']; obj=check.get('object'); s=snap['objects'].get(obj)
                    if 'metric' in check: require(check['metric'] in metrics,'check metric missing')
                    if kind=='metricEqualsOccupied':
                        require(objects[obj]['kind']=='pool','occupancy requires pool'); require(metrics[check['metric']]==sum(x is not None for x in s['slots']),'occupancy check failed')
                    elif kind=='metricEqualsNonNull':
                        require('values' in s,'nonnull check requires values'); require(metrics[check['metric']]==sum(x is not None for x in s['values']),'nonnull check failed')
                    elif kind=='constantValues':
                        baseline=first['objects'][obj]; require({k:v for k,v in s.items() if k not in ('status','window')}=={k:v for k,v in baseline.items() if k not in ('status','window')},'constant values changed')
                    elif kind=='nondecreasingMetric' and previous:
                        oldmetric=next(m['value'] for m in previous['metrics'] if m['id']==check['metric']); number(oldmetric); number(metrics[check['metric']]); require(metrics[check['metric']]>=oldmetric,'metric decreased')
                    elif kind=='sum':
                        values=[snap['objects'][x]['value'] for x in check['sources']]; target=snap['objects'][check['target']]['value']
                        if target is not None and all(x is not None for x in values):
                            for x in values+[target]: number(x)
                            require(math.isclose(sum(values),target,rel_tol=1e-10,abs_tol=1e-10),'sum check failed')
                previous=snap
            except (ValueError,KeyError,TypeError) as exc:
                raise ValueError(f"{lesson['key']}/{case.get('id','base')}/{step.get('id',index)}: {exc}") from exc
    if 'details' in lesson:
        d=lesson['details']; fields(d,'summary paragraphs'); string(d['summary'],True); array(d['paragraphs'])
        for p in d['paragraphs']: string(p,True)
    if 'prediction' in lesson:
        p=lesson['prediction']; fields(p,'before question answers correct'); string(p['question'],True); array(p['answers'],2,4); answer_ids=ids(p['answers'])
        for a in p['answers']: fields(a,'id label feedback'); string(a['label'],True); string(a['feedback'],True)
        require(p['correct'] in answer_ids,'prediction answer missing'); require(all(p['before'] in seq for seq in case_ids),'prediction step absent in case')
    return lesson


def strict_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,f'duplicate JSON field {key}'); result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'nonfinite JSON {x}')))


def load_chunks(directory, run_calculations=True):
    lessons={}
    directory=Path(directory).resolve()
    require(directory.is_dir(),f'mechanism directory does not exist: {directory}')
    chunks=sorted(directory.glob('vol*-*.json'))
    require(chunks,f'no mechanism chunks found in {directory}')
    for path in chunks:
        rows=strict_json(path); array(rows)
        for row in rows:
            try:
                validate_lesson(row)
                require(row['key'] not in lessons,'duplicate lesson key')
                lessons[row['key']]=row
            except (ValueError,KeyError,TypeError) as error:
                key=row.get('key','unknown') if isinstance(row,dict) else 'unknown'
                raise ValueError(f'{path.name}/{key}: {error}') from error
        calc=path.with_suffix('.calc.py'); require(calc.is_file(),f'{path.name}: missing calculation companion')
        if run_calculations:
            result=subprocess.run([sys.executable,str(calc)],cwd=path.parent,capture_output=True,text=True,timeout=60)
            require(result.returncode==0,f'{calc.name}: calculation failed\n{result.stdout}\n{result.stderr}')
    require(lessons,'mechanism chunks contain no lessons')
    return lessons


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path,help='Existing directory containing vol*-*.json chunks and calculation companions.')
    args=parser.parse_args()
    try:
        lessons=load_chunks(args.directory); print(json.dumps({'validated':len(lessons),'keys':sorted(lessons)},indent=2))
    except (ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as error:
        sys.exit(str(error))
