# StaffML Taxonomy & Classification System

How the pieces fit together, from taxonomy to question generation to validation.

## Architecture

```
                    ┌─────────────────────────────┐
                    │  staffml_taxonomy.yaml       │  ← LinkML schema (defines the rules)
                    │  (LinkML)                    │
                    └──────────┬──────────────────┘
                               │ generates
               ┌───────────────┼───────────────────┐
               ▼               ▼                   ▼
        gen-pydantic     gen-json-schema     gen-typescript
        (Python)         (CI validation)     (StaffML app)

                    ┌─────────────────────────────┐
                    │  taxonomy_data.yaml          │  ← The data (79 topics + edges)
                    │  (the curated taxonomy)      │
                    └──────────┬──────────────────┘
                               │ used by
               ┌───────────────┼───────────────────┐
               ▼               ▼                   ▼
          resolve.py      graph.py           generate.py
          (map corpus     (explore &         (generate new
          to topics)      visualize)         questions)
               │                                   │
               ▼                                   ▼
        ┌─────────────┐                    ┌──────────────┐
        │ corpus.json  │◄───────────────── │ new questions │
        │ (5786 Qs)    │                   │ (topic+zone)  │
        └──────┬──────┘                    └──────────────┘
               │ validated by
               ▼
        ┌─────────────┐     ┌────────────────────┐
        │ schema.py    │     │ vault_invariants.py │
        │ (per-Q)      │     │ (cross-file)        │
        └─────────────┘     └────────────────────┘
```

## The Four Classification Axes

Every question is classified on four axes:

| Axis | Field | Values | What It Answers |
|------|-------|--------|-----------------|
| **Topic** | `topic` | 79 curated IDs | WHAT concept is tested? |
| **Zone** | `zone` | 11 ikigai zones | HOW is the concept tested? |
| **Track** | `track` | cloud, edge, mobile, tinyml, global | WHERE is it deployed? |
| **Level** | `level` | L1–L6+ | HOW HARD is it? |

### Topics (79)
Defined in `schema/taxonomy_data.yaml`. Human-curated, schema-enforced.
Each topic belongs to one of 13 competency areas and has typed edges
(prerequisite, broader, narrower, related) to other topics.

### Zones (11 — the Ikigai Model)
Defined in `schema/zones.py`. Based on four fundamental skills:

```
          Recall          ← facts & definitions
         /      \
   Diagnosis  Specification
       |    \  / |
    Analyze  \/  Design    ← reasoning / architecture
       |    /\   |
  Optimization  Realization
         \      /
         Implement        ← napkin math & building

          Mastery         ← all four skills
```

| Zone | Skills | Typical Levels |
|------|--------|----------------|
| recall | recall | L1-L2 |
| implement | implement | L2-L3 |
| fluency | recall + implement | L2-L3 |
| analyze | analyze | L3-L4 |
| diagnosis | recall + analyze | L3-L4 |
| design | design | L4-L5 |
| specification | recall + design | L4-L5 |
| optimization | analyze + implement | L4-L5 |
| evaluation | analyze + design | L5-L6+ |
| realization | design + implement | L5-L6+ |
| mastery | all four | L6+ |

---

## Workflows

### 1. Adding a New Topic

Rare — should happen maybe once a quarter.

```bash
# 1. Edit the taxonomy data
vim schema/taxonomy_data.yaml
# Add a new topic with edges to existing topics

# 2. Validate with LinkML
linkml-validate -s schema/staffml_taxonomy.yaml schema/taxonomy_data.yaml

# 3. Sync topics.json (backward compat)
cd schema && python3 -c "
import yaml, json
with open('taxonomy_data.yaml') as f:
    data = yaml.safe_load(f)
topics = [{'id': t['id'], 'name': t['name'], 'area': t['area'],
           'prerequisites': [e['target'] for e in t.get('edges',[])
                             if e['edge_type']=='prerequisite'],
           'description': t['description'].strip()}
          for t in data['topics']]
json.dump({'version': data['version'], 'description': data['description'],
           'last_updated': data['last_updated'],
           'areas': sorted({t['area'] for t in data['topics']}),
           'topics': topics}, open('../topics.json','w'), indent=2)
"

# 4. Validate topics.json
python3 ../topic_schema.py --stats

# 5. Visualize to check edges make sense
python3 graph.py --topic your-new-topic --output /tmp/check.dot
```

### 2. Generating New Questions

```bash
# 1. Identify coverage gaps
python3 schema/resolve.py    # Shows topic × zone distribution

# 2. Generate with explicit topic + zone targeting
python3 scripts/generate.py \
  --topic kv-cache-management \
  --zone diagnosis \
  --track cloud \
  --level L4 \
  --budget 5

# The generation prompt should include:
#   - Topic description from taxonomy_data.yaml
#   - Zone description from zones.py
#   - Track constraints
#   - Level expectations

# 3. Validate generated questions
python3 scripts/vault_invariants.py

# 4. Run scorecard
python3 scripts/scorecard.py
```

### 3. Exploring the Taxonomy

```bash
# What must you know before learning 3D parallelism?
python3 schema/graph.py --query "what leads to 3d-parallelism"

# What depends on knowing the roofline model?
python3 schema/graph.py --query "what needs roofline-analysis"

# Show the prerequisite path between two topics
python3 schema/graph.py --path roofline-analysis flash-attention

# Visualize one topic's neighborhood
python3 schema/graph.py --topic kv-cache-management --output kv.dot

# Visualize an entire competency area
python3 schema/graph.py --area parallelism --output para.dot

# Show only topics relevant to TinyML
python3 schema/graph.py --track tinyml --output tiny.dot

# Full graph statistics
python3 schema/graph.py --stats
```

### 4. Migrating Old Questions to New System

One-time operation to add `topic` and `zone` fields to existing corpus:

```bash
# Dry run — see mapping stats
python3 schema/resolve.py

# Check a specific question
python3 schema/resolve.py --question cloud-0042

# Apply migration
python3 schema/resolve.py --apply

# Verify
python3 scripts/vault_invariants.py
```

### 5. The QA Loop (Updated)

The existing QA loop from WORKFLOW.md, now with taxonomy integration:

```
MEASURE → DIAGNOSE → FIX → VALIDATE → INVARIANTS → MEASURE
    ↑                                                │
    └────────────────────────────────────────────────┘

Exit when: OK% > 95%, 0 FAIL invariants, all topics have 20+ Qs
```

```bash
# 1. Measure
python3 scripts/scorecard.py

# 2. Check topic coverage gaps
python3 schema/resolve.py

# 3. Generate to fill gaps (targeted by topic + zone + track)
python3 scripts/generate.py --topic [gap-topic] --zone [gap-zone]

# 4. Validate new questions
python3 scripts/validate_questions.py --new-only

# 5. Check invariants (MUST pass before commit)
python3 scripts/vault_invariants.py

# 6. Commit
git add corpus.json chains.json taxonomy.json
git commit -m "staffml: fill [topic] gaps, N new questions"
```

---

## File Reference

### Schema layer (source of truth)
| File | Purpose |
|------|---------|
| `schema/staffml_taxonomy.yaml` | LinkML schema — defines Topic, Question, Zone, Skill, Level, Edge types |
| `schema/taxonomy_data.yaml` | The 79 topics + 123 typed edges (the curated knowledge graph) |
| `schema/zones.py` | Ikigai zone model — zone ↔ skill mappings, migration from old fields |
| `schema/graph.py` | Graph explorer — queries, paths, visualization (DOT/SVG) |
| `schema/resolve.py` | Maps old corpus fields (primary_concept, reasoning_mode) → new (topic, zone) |

### Validation layer (enforcement)
| File | Purpose |
|------|---------|
| `schema.py` | Pydantic per-question validation (field types, enums, lengths) |
| `topic_schema.py` | Validates topics.json — DAG checks, ID format, duplicate detection |
| `scripts/vault_invariants.py` | 14 cross-file structural checks (taxonomy ↔ corpus consistency) |
| `scripts/gate.py` | Pipeline gate — wrap any script to block on regressions |

### Data layer
| File | Purpose |
|------|---------|
| `corpus.json` | The 5786 questions (each now has `topic` + `zone` fields) |
| `topics.json` | Simplified JSON view of topics (auto-generated from taxonomy_data.yaml) |
| `taxonomy.json` | OLD: 839 LLM-extracted concepts (kept for backward compat, not maintained) |
| `chains.json` | 1011 question chains (L3→L4→L5 progressions) |

### Pipeline layer
| File | Purpose |
|------|---------|
| `scripts/generate.py` | Question generation engine (should target topic + zone) |
| `scripts/validate_questions.py` | Gemini-powered math/content review |
| `scripts/scorecard.py` | Corpus health dashboard |
| `scripts/WORKFLOW.md` | The iterative QA loop |

---

## What Replaces What

| Old Field | Old Values | New Field | New Values |
|-----------|-----------|-----------|-----------|
| `primary_concept` | 839 uncurated IDs | `topic` | 79 curated IDs |
| `taxonomy_concept` | same as above | `topic` | same |
| `reasoning_mode` | 7 string modes | `zone` | 11 ikigai zones |
| `reasoning_competency` | RC-1 to RC-13 | `zone` | 11 ikigai zones |
| `knowledge_area` | A1-F1 (35 codes) | `topic` + `area` | topic inherits area from taxonomy |
| `competency_area` | 72 variants | `area` (via topic) | 13 canonical (from topic's area) |

The old fields remain in corpus.json for backward compatibility but are not maintained.
The new fields (`topic`, `zone`) are the canonical classification.
