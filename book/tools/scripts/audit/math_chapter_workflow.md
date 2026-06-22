# MLSysBook Math Accuracy Audit Workflow

This workflow builds PDF/TeX packets for Volume I and Volume II, then creates
small-batch prompts for extra-high-reasoning agents to audit calculations.

Run everything from the task worktree:

```bash
cd /Users/VJ/GitHub/MLSysBook-math-audit-workflow
```

## Smoke Test One Chapter

Build only Volume I introduction and package its PDF, TeX, source QMD, manifest,
math locator, and agent prompt together:

```bash
python3 book/tools/scripts/audit/math_chapter_workflow.py build \
  --only vol1:introduction \
  --run-id round-01-introduction \
  --force
```

The packet lands under:

```text
review/math-audit/runs/round-01-introduction/packets/vol1/
```

## Build Audit Packets

Default selection includes content chapters and appendices for both volumes. It
excludes frontmatter, part divider pages, and glossaries unless explicitly
requested.

```bash
python3 book/tools/scripts/audit/math_chapter_workflow.py build \
  --volumes vol1,vol2 \
  --run-id round-01 \
  --batch-size 3 \
  --continue-on-error \
  --force
```

To also build the full Volume I and Volume II PDFs before chapter packets:

```bash
python3 book/tools/scripts/audit/math_chapter_workflow.py build \
  --mode both \
  --volumes vol1,vol2 \
  --run-id round-01 \
  --batch-size 3 \
  --continue-on-error \
  --force
```

Useful listing command:

```bash
python3 book/tools/scripts/audit/math_chapter_workflow.py list --volumes vol1,vol2
```

## Agent Batches

Each run writes prompts under:

```text
review/math-audit/runs/<run-id>/agent_batches/
```

Spin up agents in small batches with extra-high reasoning. Give each agent one
batch prompt. The prompt tells the agent to read the packet TeX, copied source
QMD, rendered PDF, and math locator, then write one YAML audit file per packet
under:

```text
review/math-audit/runs/<run-id>/audits/
```

Agents should use `status: clear` with `issues: []` when a packet has no math
issues. If math is incorrect, the YAML should include the exact expression,
observed result, recomputed expected result, derivation, and proposed correction.

## Summarize A Round

After all agent YAML files are present:

```bash
python3 book/tools/scripts/audit/math_chapter_workflow.py summarize \
  review/math-audit/runs/round-01
```

This writes:

```text
review/math-audit/runs/round-01/summary.yml
```

Review `summary.yml`, apply accepted source fixes, then start `round-02` with the
same build command. Repeat until the summary has zero accepted math issues.
