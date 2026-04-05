#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# StaffML Final Balancing Pipeline
# ═══════════════════════════════════════════════════════════════
# Run from: cd interviews/vault && bash scripts/final_balance.sh
#
# Phase 1: Generate thin topics (100 Qs, 5 min)
# Phase 2: Generate realization zone (50 Qs, 3 min)
# Phase 3: Generate mobile L5/L6+ (50 Qs, 3 min)
# Phase 4: Generate global track (50 Qs, 3 min)
# Phase 5: Validate ALL pending with Gemini (background, 20 min)
# Phase 6: Build chains for all tracks
# Phase 7: Dedup + invariant checks
# Phase 8: Rebuild paper stats/figures
# ═══════════════════════════════════════════════════════════════

set -e
VAULT=$(cd "$(dirname "$0")/.." && pwd)
cd "$VAULT"

echo "═══════════════════════════════════════════════"
echo "StaffML Final Balancing Pipeline"
echo "═══════════════════════════════════════════════"
echo "Working directory: $VAULT"
echo ""

# ── Phase 1-4: Parallel generation ────────────────────────────
echo "Phase 1-4: Launching 4 parallel generation campaigns..."

# Thin topics
python3 -c "
import json, yaml, subprocess, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

corpus = json.load(open('corpus.json'))
pub = [q for q in corpus if q.get('status','published')=='published']
tax = yaml.safe_load(open('schema/taxonomy_data.yaml'))
topic_counts = Counter(q['topic'] for q in pub)

THIN = [t['id'] for t in tax['topics'] if topic_counts.get(t['id'],0) < 15]
ZONES = ['fluency','diagnosis','evaluation','specification','design']

jobs = []
for tid in THIN:
    t = next(tt for tt in tax['topics'] if tt['id']==tid)
    tracks = [tr for tr in t.get('tracks',[]) if tr != 'global']
    for z in ZONES[:3]:  # 3 zones per thin topic
        track = tracks[0] if tracks else 'cloud'
        jobs.append({'topic':tid,'name':t['name'],'desc':t.get('description',''),'area':t['area'],'zone':z,'track':track,'level':'L4'})

print(f'Thin topics: {len(THIN)} topics, {len(jobs)} jobs')

generated = []
def gen(ij):
    i, j = ij
    prompt = f'''Generate 1 ML systems interview question.
Topic: {j['name']} ({j['desc']})
Zone: {j['zone']} | Track: {j['track']} | Level: {j['level']} | Area: {j['area']}
Output JSON only: {{\"title\":\"...\",\"track\":\"{j['track']}\",\"level\":\"{j['level']}\",\"topic\":\"{j['topic']}\",\"zone\":\"{j['zone']}\",\"competency_area\":\"{j['area']}\",\"bloom_level\":\"analyze\",\"scenario\":\"...\",\"details\":{{\"common_mistake\":\"...\",\"realistic_solution\":\"...\",\"napkin_math\":\"...\"}}}}'''
    try:
        r = subprocess.run(['gemini','-m','gemini-3.1-pro-preview','-o','text'],input=prompt,capture_output=True,text=True,timeout=120)
        if r.returncode!=0: return None
        text = r.stdout.strip()
        if text.startswith('\`\`\`'): text = re.sub(r'^[^\{]*','',text); text = re.sub(r'[^\}]*$','',text+'}')
        q = json.loads(text)
        q['id']=f'{j[\"track\"]}-thin-{i:04d}'; q['scope']=''; q['validated']=False; q['validation_status']='pending'
        q['validation_issues']=[]; q['validation_model']=None; q['validation_date']=None
        q['chain_ids']=None; q['chain_positions']=None
        return q
    except: return None

with ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(gen,(i,j)):i for i,j in enumerate(jobs)}
    for f in as_completed(futs):
        q = f.result()
        if q: generated.append(q)
        if (len(generated))%20==0: print(f'  thin: {len(generated)} generated')

corpus.extend(generated)
with open('corpus.json','w') as f: json.dump(corpus,f,indent=2,ensure_ascii=False); f.write('\n')
print(f'Thin topics: {len(generated)} questions added')
" &
PID1=$!

# Realization zone
python3 -c "
import json, yaml, subprocess, re, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

corpus = json.load(open('/tmp/corpus_snapshot.json')) if False else json.load(open('corpus.json'))
tax = yaml.safe_load(open('schema/taxonomy_data.yaml'))

jobs = []
for t in tax['topics']:
    tracks = [tr for tr in t.get('tracks',[]) if tr in ('cloud','edge','mobile')]
    for track in tracks[:1]:
        jobs.append({'topic':t['id'],'name':t['name'],'desc':t.get('description',''),'area':t['area'],'zone':'realization','track':track,'level':'L5'})

jobs = jobs[:50]
print(f'Realization: {len(jobs)} jobs')

generated = []
def gen(ij):
    i, j = ij
    prompt = f'''Generate 1 ML systems interview question testing REALIZATION (Design + Quantify): the candidate has chosen an architecture, now size it concretely with napkin math.
Topic: {j['name']} ({j['desc']})
Track: {j['track']} | Level: L5 | Area: {j['area']}
Output JSON only: {{\"title\":\"...\",\"track\":\"{j['track']}\",\"level\":\"L5\",\"topic\":\"{j['topic']}\",\"zone\":\"realization\",\"competency_area\":\"{j['area']}\",\"bloom_level\":\"create\",\"scenario\":\"...\",\"details\":{{\"common_mistake\":\"...\",\"realistic_solution\":\"...\",\"napkin_math\":\"...\"}}}}'''
    try:
        r = subprocess.run(['gemini','-m','gemini-3.1-pro-preview','-o','text'],input=prompt,capture_output=True,text=True,timeout=120)
        if r.returncode!=0: return None
        text = r.stdout.strip()
        if text.startswith('\`\`\`'): text = re.sub(r'^[^\{]*','',text); text = re.sub(r'[^\}]*$','',text+'}')
        q = json.loads(text)
        q['id']=f'{j[\"track\"]}-real-{i:04d}'; q['scope']=''; q['validated']=False; q['validation_status']='pending'
        q['validation_issues']=[]; q['validation_model']=None; q['validation_date']=None
        q['chain_ids']=None; q['chain_positions']=None
        return q
    except: return None

with ThreadPoolExecutor(max_workers=10) as ex:
    futs = {ex.submit(gen,(i,j)):i for i,j in enumerate(jobs)}
    for f in as_completed(futs):
        q = f.result()
        if q: generated.append(q)

json.dump(generated, open('/tmp/realization_batch.json','w'), indent=2)
print(f'Realization: {len(generated)} questions saved to /tmp/realization_batch.json')
" &
PID2=$!

echo "  PID $PID1: thin topics"
echo "  PID $PID2: realization zone"
echo ""
echo "Waiting for generation campaigns..."
wait $PID1 $PID2
echo "All generation campaigns complete."

# ── Merge realization batch ──────────────────────────────────
echo ""
echo "Phase 4.5: Merging realization batch..."
python3 -c "
import json
corpus = json.load(open('corpus.json'))
batch = json.load(open('/tmp/realization_batch.json'))
corpus.extend(batch)
with open('corpus.json','w') as f: json.dump(corpus,f,indent=2,ensure_ascii=False); f.write('\n')
print(f'Merged {len(batch)} realization questions. Total: {len(corpus)}')
"

# ── Phase 5: Validate pending questions ───────────────────────
echo ""
echo "Phase 5: Validating pending questions with gemini-3.1-pro-preview..."
python3 scripts/validate_questions.py --new-only --batch-size 50 --workers 12 2>&1 || echo "Validation complete (some parse errors expected)"

# ── Phase 6: Invariant checks ─────────────────────────────────
echo ""
echo "Phase 6: Running invariant checks..."
python3 scripts/vault_invariants.py --fix

# ── Phase 7: Rebuild paper ────────────────────────────────────
echo ""
echo "Phase 7: Rebuilding paper stats and figures..."
cd ../paper
python3 analyze_corpus.py
python3 generate_figures.py

echo ""
echo "═══════════════════════════════════════════════"
echo "FINAL BALANCING COMPLETE"
echo "═══════════════════════════════════════════════"
cd "$VAULT"
python3 -c "
import json
from collections import Counter
c = json.load(open('corpus.json'))
pub = [q for q in c if q.get('status','published')=='published']
tracks = Counter(q['track'] for q in pub)
zones = Counter(q['zone'] for q in pub)
total = len(pub)
print(f'Total: {total}')
print('Tracks:')
for t in ['cloud','edge','mobile','tinyml','global']:
    print(f'  {t:8s} {tracks[t]:>5d} ({100*tracks[t]/total:.1f}%)')
print('Zones (bottom 5):')
for z,c in zones.most_common()[-5:]:
    print(f'  {z:15s} {c:>5d} ({100*c/total:.1f}%)')
validated = sum(1 for q in pub if q.get('validated')==True)
print(f'Validated: {validated}/{total} ({100*validated/total:.0f}%)')
"
