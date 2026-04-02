# StaffML Web App Migration Plan

## Context

The StaffML vault (corpus, taxonomy, schema) has been completely redesigned:

**Old system** (what the app currently uses):
- `competency_area`: 70+ non-canonical values ("Compilation and Toolchains", "Cloud ML Architecture", etc.)
- `primary_concept` / `taxonomy_concept`: 839 LLM-extracted concept IDs
- `reasoning_mode`: 7 string modes
- `reasoning_competency`: RC-1 through RC-13
- `knowledge_area`: A1 through F1 (35 codes)
- `concept_tags`: multi-label free-form tags
- `canonical_topic`: inconsistent

**New system** (what the vault now uses):
- `topic`: 79 curated topic IDs (e.g., "roofline-analysis", "kv-cache-management")
- `zone`: 11 ikigai zones (recall, analyze, design, quantify, diagnosis, specification, fluency, evaluation, realization, optimization, mastery)
- `competency_area`: 13 canonical values (compute, memory, latency, precision, power, architecture, optimization, parallelism, networking, deployment, reliability, data, cross-cutting)
- `track`: 5 tracks (cloud, edge, mobile, tinyml, global)
- `level`: 6 levels (L1-L6+)

**Corpus stats**: 6,349 published questions, 83% validated, 79 topics, 11 zones

## Files to Update

### 1. Data Files (copy from vault)

| Source (vault) | Destination (staffml) | Action |
|---------------|----------------------|--------|
| `vault/corpus.json` | `staffml/src/data/corpus.json` | Copy (the vault version has topic+zone fields, no legacy fields) |
| `vault/schema/taxonomy_data.yaml` | `staffml/src/data/taxonomy.json` | Convert YAML→JSON, replace old 825-concept taxonomy |
| `vault/topics.json` | `staffml/src/data/topics.json` | NEW file — 79 curated topics with prereq edges |
| `vault/schema/zones.py` | `staffml/src/data/zones.json` | NEW file — convert zone definitions to JSON |
| — | `staffml/src/data/concept_tags_vocabulary.json` | DELETE — replaced by topics |
| — | `staffml/src/data/corpus-index.json` | REBUILD from new corpus |

### 2. TypeScript Types (`src/lib/corpus.ts`)

Replace the Question interface. The old interface has:
```typescript
// OLD
topic: string;           // was taxonomy_concept
competency_area: string; // 70+ values
reasoning_mode?: string;
reasoning_competency?: string;
knowledge_area?: string;
concept_tags?: string[];
canonical_topic?: string;
primary_concept?: string;
```

Replace with:
```typescript
// NEW
topic: string;            // one of 79 curated topic IDs
zone: string;             // one of 11 ikigai zones
competency_area: string;  // one of 13 canonical areas
track: string;            // cloud | edge | mobile | tinyml | global
level: string;            // L1 | L2 | L3 | L4 | L5 | L6+
bloom_level: string;      // remember | understand | apply | analyze | evaluate | create
```

Generate the valid values as TypeScript literals from the LinkML schema:
```bash
cd vault && python3 topic_schema.py --literal  # generates TopicID type
```

### 3. UI Components to Update

#### `src/app/practice/page.tsx` (~7 references)
- Replace `competency_area` filter with `topic` filter (79 topics grouped by 13 areas)
- Add `zone` filter (11 zones)
- The area filter sidebar should show 13 areas, each expandable to show its topics
- Add zone filter as a second dimension

#### `src/app/gauntlet/page.tsx` (~7 references)
- Replace `competency_area` with `topic` throughout
- Update the gauntlet scoring to use zones instead of reasoning_mode
- The gauntlet results should show zone-level performance profile

#### `src/app/plans/page.tsx` (~2 references)
- Replace `competency_area` with `topic`
- Study plans should use the prerequisite graph from topics.json to suggest learning order

### 4. Library Functions (`src/lib/`)

#### `src/lib/corpus.ts` (~21 references)
- `getQuestionsByTopic()`: currently filters by `taxonomy_concept` → change to `topic`
- Add `getQuestionsByZone(zone: string)` function
- Add `getTopicsByArea(area: string)` function
- Update all filter/search functions to use new fields

#### `src/lib/taxonomy.ts` (~2 references)
- Replace with new taxonomy based on `topics.json`
- The taxonomy should expose: topics, areas, prerequisite edges, zone definitions

#### `src/lib/plans.ts` (~2 references)
- Update study plan generation to use prerequisite graph
- Plans should follow the topic dependency chain

### 5. Filter/Search UI

The current AREA filter (shown in the screenshot) shows 70+ fragmented categories. Replace with a hierarchical filter:

```
Level 1: 13 Competency Areas (compute, memory, latency, ...)
  Level 2: Topics within each area (roofline-analysis, gpu-compute-architecture, ...)

Separate filter: 11 Zones (recall, diagnosis, fluency, ...)
Separate filter: 5 Tracks (cloud, edge, mobile, tinyml, global)
Separate filter: 6 Levels (L1-L6+)
```

### 6. Data Export Script

Create a script that converts the vault corpus to the app format:

```bash
# vault/scripts/export_to_staffml.py
# 1. Load corpus.json from vault
# 2. Filter to published questions only
# 3. Strip internal fields (validation_model, validation_date, etc.)
# 4. Keep only: id, title, track, level, topic, zone, competency_area,
#    bloom_level, scenario, details, scope, chain_ids
# 5. Write to staffml/src/data/corpus.json
# 6. Convert taxonomy_data.yaml → staffml/src/data/taxonomy.json
# 7. Generate corpus-index.json (topic counts, zone counts, etc.)
```

This script should be run whenever the vault corpus is updated.

## Migration Steps (for the implementing agent)

```
Step 1: Run export script to copy updated data files
Step 2: Update TypeScript types in src/lib/corpus.ts
Step 3: Update src/lib/taxonomy.ts with new topic/zone model
Step 4: Update practice page filters (area → topic hierarchy + zone)
Step 5: Update gauntlet page scoring (competency_area → zone profile)
Step 6: Update plans page (use prerequisite graph)
Step 7: Delete concept_tags_vocabulary.json
Step 8: Build and test: npm run build
Step 9: Verify: all filters work, questions display correctly, no 70+ area tags
```

## Key Principles

1. **topic** replaces: taxonomy_concept, primary_concept, canonical_topic, concept_tags
2. **zone** replaces: reasoning_mode, reasoning_competency
3. **competency_area** stays but is now ALWAYS one of 13 canonical values (no more "Cloud ML Architecture" etc.)
4. **knowledge_area** (A1-F1) is REMOVED — no replacement needed
5. The prerequisite graph in topics.json enables study path generation
6. The zone model enables interview loop design (phone screen → system design → debugging → architecture)

## Validation After Migration

- [ ] Area filter shows exactly 13 areas (not 70+)
- [ ] Each area expands to show its topics
- [ ] Zone filter shows 11 zones
- [ ] All 6,349 published questions display correctly
- [ ] Question cards show topic and zone (not old fields)
- [ ] Gauntlet scoring uses zones
- [ ] Study plans use prerequisite graph
- [ ] `npm run build` succeeds with no TypeScript errors
- [ ] No references to old fields remain in codebase
