#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Gap Fill: Generate questions for 37 underfilled cells + 86 missing napkin_math
# Uses gemini-3.1-pro-preview (or fallback to flash)
# Run when Pro quota resets
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
INTERVIEWS_DIR="$(cd "$(dirname "$0")" && pwd)"
VAULT_DIR="${INTERVIEWS_DIR}/_vault"
MAX_PARALLEL="${MAX_PARALLEL:-6}"

mkdir -p "$VAULT_DIR"

echo "═══ Gap Fill Script ═══"
echo "Model: ${MODEL}"
echo ""

# Step 1: Identify gaps
echo "── Step 1: Finding gaps ──"
python3 -c "
import json, sys
sys.path.insert(0, '${INTERVIEWS_DIR}/engine')
from taxonomy import normalize_tag, get_area_for_tag

c = json.load(open('${INTERVIEWS_DIR}/corpus.json'))

# Find underfilled cells
from collections import defaultdict
cube = defaultdict(int)
for q in c:
    if q['track'] in ['cloud','edge','mobile','tinyml'] and q.get('competency_area'):
        cube[(q['track'], q['level'], q['competency_area'])] += 1

gaps = [(k, v) for k, v in cube.items() if v < 3]
print(f'Underfilled cells: {len(gaps)}')
for (t, l, a), cnt in sorted(gaps):
    need = 3 - cnt
    print(f'  {t}/{l}/{a}: has {cnt}, need {need} more')

# Find missing napkin_math
missing_nm = [q for q in c if not q.get('details',{}).get('napkin_math','').strip()]
print(f'')
print(f'Missing napkin_math: {len(missing_nm)}')
for q in missing_nm[:10]:
    print(f'  {q[\"id\"][:60]} ({q[\"track\"]}/{q[\"level\"]})')
if len(missing_nm) > 10:
    print(f'  ... and {len(missing_nm)-10} more')

# Write gap list for the generator
import json as j
gap_list = []
for (t, l, a), cnt in sorted(gaps):
    gap_list.append({'track': t, 'level': l, 'area': a, 'have': cnt, 'need': 3-cnt})
j.dump(gap_list, open('/tmp/staffml-gaps.json','w'), indent=2)
print(f'')
print(f'Gap list written to /tmp/staffml-gaps.json')
"

echo ""
echo "── Step 2: Generate for gaps ──"
echo "(Run with: GEMINI_MODEL=gemini-3.1-pro-preview ./fill_gaps.sh)"
echo ""
echo "── Step 3: Backfill napkin_math ──"
echo "(Uses Gemini to add napkin math to questions that are missing it)"
echo ""
echo "This script is ready to run when Pro quota resets (~8h)."
echo "To run with Flash now: GEMINI_MODEL=gemini-2.5-flash ./fill_gaps.sh"
