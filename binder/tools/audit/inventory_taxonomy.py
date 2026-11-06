"""Inventory review coverage and structural/lexical diagnostics; never infer truth.

Usage: python3 binder/tools/audit/inventory_taxonomy.py OUTPUT.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[3]


def inventory(root):
    path = root / 'taxonomy/taxonomy.yml'
    raw = path.read_bytes()
    doc = yaml.safe_load(raw)
    provenance = yaml.safe_load((root / 'taxonomy/provenance.yml').read_text())
    decisions = {}
    for entry in provenance['applied_campaigns']:
        if entry['mode'] == 'editorial-review':
            for change in entry['replayed_patch']['changes']:
                decisions.setdefault(change['id'], []).append(entry['campaign_id'])
    neighbors = {c['id']: set() for c in doc['concepts']}
    for edge in doc['relationships']:
        neighbors[edge['source']].add(edge['target'])
        neighbors[edge['target']].add(edge['source'])
    terms = {}
    concepts = []
    for concept in doc['concepts']:
        for term in [concept['label'], *concept.get('aliases', [])]:
            norm = re.sub(r'[^a-z0-9]+', ' ', term.lower()).strip()
            terms.setdefault(norm, set()).add(concept['id'])
        concepts.append(dict(id=concept['id'], home=concept['home'],
            incident_count=len(neighbors[concept['id']]),
            compound_label_candidate=bool(re.search(r'\band\b|/', concept['label'])),
            definition_present=bool(concept['definition'].strip()),
            entry_terms=[dict(term=t, decision=provenance.get('concepts', {}).get(concept['id'], {}).get('entry_term_decisions', {}).get(t), equivalence='source-backed' if provenance.get('concepts', {}).get(concept['id'], {}).get('entry_term_decisions', {}).get(t, {}).get('kind') in {'synonym', 'acronym'} else 'not-asserted') for t in concept.get('aliases', [])],
            editorial_changes=decisions.get(concept['id'], []),
            semantic_review='targeted-change-only' if concept['id'] in decisions else 'not-attested'))
    ids = {'mlsys.memory-coalescing', 'mlsys.tracing-jit', 'mlsys.tolerable-risk'}
    decisions_path = root / 'taxonomy/reviews/boundary-resolution/assignment-decisions.json'
    assignment_decisions = {r['path']: r for r in json.loads(decisions_path.read_text())['records']} if decisions_path.exists() else {}
    assignments = []
    for path in sorted((root / 'staffml/vault/questions').rglob('*.yaml')):
        text = path.read_text()
        affected = sorted(cid for cid in ids if cid in text)
        relative = str(path.relative_to(root))
        if affected or relative in assignment_decisions:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            decision = assignment_decisions.get(relative, {})
            disposition = decision.get('decision') if decision.get('after_sha256') == digest else 'requires-scope-review'
            assignments.append(dict(path=relative, sha256=digest, concepts=affected, disposition=disposition))
    return dict(status='mechanical-inventory-not-semantic-certification', source_sha256=hashlib.sha256(raw).hexdigest(),
        counts=dict(concepts=len(concepts), relationships=len(doc['relationships']), entry_terms=sum(len(c.get('aliases', [])) for c in doc['concepts']),
                    sparse_concepts=sum(len(n)<2 for n in neighbors.values()), affected_assignments=len(assignments)),
        lexical_collisions=[dict(normalized=k, concepts=sorted(v)) for k,v in sorted(terms.items()) if len(v)>1],
        concepts=concepts, relationships=[dict(id=e['id'], editorial_changes=decisions.get(e['id'], []), semantic_review='targeted-change-only' if e['id'] in decisions else 'not-attested') for e in doc['relationships']],
        assignment_review=assignments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = inventory(ROOT)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['counts'], indent=2))
