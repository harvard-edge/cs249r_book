"""``vault generate`` — LLM-assisted question generation (ARCHITECTURE.md §12).

Draws exemplars ONLY from ``vault/exemplars/`` (never from the general corpus).
Outputs to ``vault/drafts/`` with ``status: draft``, ``provenance: llm-draft``.
Use ``vault promote`` to move drafts into the published corpus after human review.

Cost controls: hard cap of 25 per invocation, daily spend ledger, secrets from
~/.config/vault/secrets.toml (mode 0600 enforced).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from vault_cli.exit_codes import ExitCode
from vault_cli.yaml_io import dump_str, load_file

console = Console()

# ── Constants ────────────────────────────────────────────────────────────────

HARD_CAP = 25
DAILY_BUDGET_DEFAULT = 50.0  # USD
SECRETS_PATH = Path.home() / ".config" / "vault" / "secrets.toml"
MODEL_DEFAULT = "claude-sonnet-4-6"

# Approximate token costs (USD per 1M tokens) — conservative estimates.
_MODEL_COSTS: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_secrets() -> dict[str, str]:
    """Load API keys from ~/.config/vault/secrets.toml."""
    if not SECRETS_PATH.exists():
        return {}
    # Check permissions — refuse if world-readable.
    mode = SECRETS_PATH.stat().st_mode
    if mode & stat.S_IROTH or mode & stat.S_IWOTH:
        console.print(
            f"[red]error[/red]: {SECRETS_PATH} is world-readable (mode {oct(mode)}). "
            "Fix with: chmod 600 ~/.config/vault/secrets.toml"
        )
        raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)
    secrets: dict[str, str] = {}
    for line in SECRETS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            secrets[key.strip()] = val.strip().strip('"').strip("'")
    return secrets


def _load_spend_ledger(vault_dir: Path) -> dict[str, float]:
    """Load daily spend ledger from vault/.llm-spend.json."""
    ledger_path = vault_dir / ".llm-spend.json"
    if ledger_path.exists():
        return json.loads(ledger_path.read_text())
    return {}


def _save_spend_ledger(vault_dir: Path, ledger: dict[str, float]) -> None:
    path = vault_dir / ".llm-spend.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def _check_budget(vault_dir: Path, estimated_cost: float) -> None:
    """Fail if today's spend + estimated cost exceeds the daily budget."""
    budget_path = vault_dir / "llm-budget.yaml"
    ceiling = DAILY_BUDGET_DEFAULT
    if budget_path.exists():
        data = load_file(budget_path)
        ceiling = float(data.get("daily_ceiling_usd", DAILY_BUDGET_DEFAULT))

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    ledger = _load_spend_ledger(vault_dir)
    spent = ledger.get(today, 0.0)

    if spent + estimated_cost > ceiling:
        console.print(
            f"[red]error[/red]: daily budget exceeded. "
            f"Spent today: ${spent:.2f}, estimated: ${estimated_cost:.2f}, "
            f"ceiling: ${ceiling:.2f}/day. See vault/.llm-spend.json"
        )
        raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)


def _load_exemplars(
    vault_dir: Path,
    topic: str,
    track: str,
    level: str,
    zone: str,
) -> list[dict[str, Any]]:
    """Load exemplars from vault/exemplars/ for the given cell."""
    exemplars_root = vault_dir / "exemplars"
    if not exemplars_root.exists():
        return []

    results = []
    for path in exemplars_root.rglob("*.yaml"):
        data = load_file(path)
        # Match the cell loosely — topic is required, others optional
        if data.get("topic") != topic:
            continue
        results.append(data)

    return results


def _build_prompt(
    topic: str,
    track: str,
    level: str,
    zone: str,
    count: int,
    exemplars: list[dict[str, Any]],
    taxonomy_context: str,
) -> str:
    """Build a generation prompt using structural metadata only (§12.3)."""
    # Extract structural metadata from exemplars — NEVER include free text.
    exemplar_summaries = []
    for i, ex in enumerate(exemplars[:10], 1):
        scenario_words = len((ex.get("scenario") or "").split())
        solution_words = len((ex.get("details", {}).get("realistic_solution") or "").split())
        has_napkin = bool(ex.get("details", {}).get("napkin_math"))
        has_chain = bool(ex.get("chain"))
        exemplar_summaries.append(
            f"  Exemplar {i}: topic={ex.get('topic')}, level={ex.get('level', level)}, "
            f"zone={ex.get('zone', zone)}, scenario_words={scenario_words}, "
            f"solution_words={solution_words}, has_napkin_math={has_napkin}, "
            f"chained={has_chain}"
        )

    exemplar_block = "\n".join(exemplar_summaries) if exemplar_summaries else "  (no exemplars available — use structural guidelines only)"

    # Load generation guidelines if available
    guidelines = ""
    guidelines_dir = Path("interviews/vault/generation-guidelines")
    if guidelines_dir.exists():
        for md in sorted(guidelines_dir.glob("*.md")):
            guidelines += f"\n--- {md.name} ---\n{md.read_text()}\n"

    prompt = f"""You are generating ML systems interview questions for the StaffML corpus.

## Target Cell
- Topic: {topic}
- Track: {track}
- Level: {level}
- Zone: {zone}
- Count: {count} questions

## Taxonomy Context
{taxonomy_context}

## Exemplar Structural Patterns
These are structural summaries of existing high-quality questions in this topic.
Match their style (scenario length, solution depth, napkin math presence):
{exemplar_block}

## Output Format
Generate exactly {count} questions. Each question must be a valid YAML document
separated by `---`. Each document must have these fields:

```yaml
title: "Concise question title"
topic: "{topic}"
scenario: |
  A realistic interview scenario (2-4 paragraphs) that presents a concrete
  systems problem. Include specific numbers (memory sizes, throughput targets,
  model sizes) that a candidate would use in their analysis.
details:
  realistic_solution: |
    A thorough solution (3-6 paragraphs) explaining the reasoning, the key
    tradeoffs, and the quantitative analysis.
  common_mistake: |
    A common misconception or error that candidates make on this topic.
  napkin_math: |
    Step-by-step back-of-envelope calculation with concrete numbers.
    End with: => <final_answer>
```

## Quality Requirements
- Every scenario must include concrete numbers (memory, bandwidth, latency, etc.)
- Solutions must show quantitative reasoning, not just conceptual answers
- Napkin math should be step-by-step with a clear final answer
- Match the difficulty level: {level} (L1=recall, L2=understand, L3=apply, L4=analyze, L5=evaluate, L6+=create)
- Match the zone: {zone} (the cognitive skill being tested)
- Questions must be grounded in real hardware specs and deployment constraints
{guidelines}
"""
    return prompt


def _estimate_tokens(prompt: str, count: int) -> tuple[int, int]:
    """Rough token estimate: ~4 chars per token."""
    input_tokens = len(prompt) // 4
    output_tokens = count * 800  # ~800 tokens per question
    return input_tokens, output_tokens


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    costs = _MODEL_COSTS.get(model, _MODEL_COSTS[MODEL_DEFAULT])
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000


def _call_llm(
    prompt: str, model: str, api_key: str, max_tokens: int,
) -> tuple[str, float]:
    """Call the Anthropic API and return (response_text, cost_usd)."""
    try:
        import anthropic
    except ImportError:
        console.print(
            "[red]error[/red]: anthropic package not installed. "
            "Run: pip install anthropic"
        )
        raise typer.Exit(code=ExitCode.VALIDATION_FAILURE) from None

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text if response.content else ""
    usage = response.usage
    costs = _MODEL_COSTS.get(model, _MODEL_COSTS[MODEL_DEFAULT])
    cost = (usage.input_tokens * costs["input"] + usage.output_tokens * costs["output"]) / 1_000_000
    return text, cost


def _parse_questions(
    raw: str, topic: str, track: str, level: str, zone: str,
    model: str, prompt_hash: str, cost: float,
) -> list[dict[str, Any]]:
    """Parse LLM output into question YAML dicts."""
    import yaml

    # Split on YAML document separators
    docs = raw.split("---")
    questions: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")

    for doc in docs:
        doc = doc.strip()
        if not doc or len(doc) < 50:
            continue
        # Strip markdown code fences
        if doc.startswith("```"):
            doc = "\n".join(doc.split("\n")[1:])
        if doc.endswith("```"):
            doc = "\n".join(doc.split("\n")[:-1])
        doc = doc.strip()
        if not doc:
            continue

        try:
            data = yaml.safe_load(doc)
        except yaml.YAMLError:
            continue

        if not isinstance(data, dict):
            continue
        if "title" not in data or "scenario" not in data:
            continue

        # Generate content-addressed ID
        title_hash = hashlib.sha256(data["title"].encode()).hexdigest()[:6]
        qid = f"{topic}-{title_hash}-0001"

        question: dict[str, Any] = {
            "id": qid,
            "title": data["title"],
            "topic": topic,
            "status": "draft",
            "provenance": "llm-draft",
            "scenario": data.get("scenario", ""),
            "details": {
                "realistic_solution": data.get("details", {}).get("realistic_solution", ""),
                "common_mistake": data.get("details", {}).get("common_mistake", ""),
            },
            "created_at": now,
            "last_modified": now,
            "generation_meta": {
                "model": model,
                "prompt_hash": prompt_hash,
                "prompt_cost_usd": round(cost / max(len(questions) + 1, 1), 4),
                "generated_at": now,
            },
        }

        if data.get("details", {}).get("napkin_math"):
            question["details"]["napkin_math"] = data["details"]["napkin_math"]

        questions.append(question)

    return questions


# ── Command ──────────────────────────────────────────────────────────────────

def register(app: typer.Typer) -> None:
    @app.command("generate")
    def generate_cmd(
        topic: str = typer.Option(..., "--topic", help="Topic slug (e.g., kv-cache-management)."),
        zone: str = typer.Option(..., "--zone", help="Ikigai zone (e.g., specification, diagnosis)."),
        track: str = typer.Option("cloud", "--track", help="Deployment track."),
        level: str = typer.Option("L4", "--level", help="Difficulty level (L1-L6+)."),
        count: int = typer.Option(3, "--count", "-n", help="Number of questions to generate."),
        model: str = typer.Option(MODEL_DEFAULT, "--model", "-m", help="LLM model to use."),
        vault_dir: Path = typer.Option(Path("interviews/vault"), "--vault-dir"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print prompt and cost estimate; no API call."),
        yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmation."),
        no_context: bool = typer.Option(False, "--no-context", help="Skip exemplar loading."),
    ) -> None:
        """Generate ML systems interview questions using an LLM.

        Questions are written to vault/drafts/ as YAML with status=draft.
        Use ``vault promote`` to move them to the published corpus after review.
        """
        # ── Validate count ──
        if count > HARD_CAP:
            console.print(
                f"[red]error[/red]: --count {count} exceeds hard cap of {HARD_CAP}. "
                "Use smaller batches for quality."
            )
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # ── Load exemplars ──
        exemplars: list[dict[str, Any]] = []
        if not no_context:
            exemplars = _load_exemplars(vault_dir, topic, track, level, zone)
            if len(exemplars) < 3:
                console.print(
                    f"[yellow]warning[/yellow]: only {len(exemplars)} exemplars found "
                    f"for topic={topic}. Minimum recommended is 3. "
                    "Use --no-context to skip exemplar requirement, or add exemplars "
                    "with `vault mark-exemplar`."
                )
                if not exemplars and not no_context:
                    console.print(
                        "[dim]Tip: vault/exemplars/ directory doesn't exist or has no "
                        f"questions for topic={topic}. Proceeding with guidelines only.[/dim]"
                    )

        # ── Load taxonomy context ──
        taxonomy_context = ""
        topics_path = (vault_dir / ".." / "staffml" / "src" / "data" / "topics.json").resolve()
        if topics_path.exists():
            topics_data = json.loads(topics_path.read_text())
            topic_list = topics_data.get("topics", [])
            match = next((t for t in topic_list if t["id"] == topic), None)
            if match:
                taxonomy_context = (
                    f"Topic: {match['name']}\n"
                    f"Area: {match['area']}\n"
                    f"Description: {match.get('description', 'N/A')}"
                )

        # ── Build prompt ──
        prompt = _build_prompt(topic, track, level, zone, count, exemplars, taxonomy_context)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]

        # ── Estimate cost ──
        input_tokens, output_tokens = _estimate_tokens(prompt, count)
        estimated_cost = _estimate_cost(model, input_tokens, output_tokens)

        # ── Summary table ──
        table = Table(title="vault generate — plan")
        table.add_column("field", style="cyan")
        table.add_column("value")
        table.add_row("topic", topic)
        table.add_row("track / level / zone", f"{track} / {level} / {zone}")
        table.add_row("count", str(count))
        table.add_row("model", model)
        table.add_row("exemplars", str(len(exemplars)))
        table.add_row("prompt tokens (est)", f"~{input_tokens:,}")
        table.add_row("output tokens (est)", f"~{output_tokens:,}")
        table.add_row("estimated cost", f"${estimated_cost:.3f}")
        table.add_row("prompt hash", prompt_hash)
        console.print(table)

        if dry_run:
            console.print("\n[dim]── Full prompt ──[/dim]")
            console.print(prompt)
            console.print("\n[yellow]--dry-run[/yellow]: no API call made.")
            return

        # ── Budget check ──
        _check_budget(vault_dir, estimated_cost)

        # ── Confirmation ──
        if not yes:
            confirm = typer.prompt(
                f"Type the topic name to confirm generation ({topic})"
            )
            if confirm.strip() != topic:
                console.print("[red]aborted[/red]: confirmation mismatch.")
                raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # ── Load API key ──
        secrets = _load_secrets()
        api_key = secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            console.print(
                "[red]error[/red]: ANTHROPIC_API_KEY not found. "
                "Set it in ~/.config/vault/secrets.toml or ANTHROPIC_API_KEY env var."
            )
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # ── Log prompt ──
        log_dir = vault_dir / "generation-log" / datetime.now(UTC).strftime("%Y-%m-%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{prompt_hash}.txt"
        log_path.write_text(prompt, encoding="utf-8")
        console.print(f"[dim]prompt logged → {log_path}[/dim]")

        # ── Call LLM ──
        console.print(f"[bold]Generating {count} questions with {model}...[/bold]")
        max_output_tokens = count * 1200
        raw_response, actual_cost = _call_llm(prompt, model, api_key, max_output_tokens)

        # ── Parse response ──
        questions = _parse_questions(
            raw_response, topic, track, level, zone,
            model, prompt_hash, actual_cost,
        )

        if not questions:
            console.print("[red]error[/red]: no valid questions parsed from LLM response.")
            console.print("[dim]Raw response saved for debugging.[/dim]")
            (log_dir / f"{prompt_hash}-response.txt").write_text(raw_response)
            raise typer.Exit(code=ExitCode.VALIDATION_FAILURE)

        # ── Write drafts ──
        drafts_dir = vault_dir / "drafts" / track / level.lower() / zone
        drafts_dir.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        for q in questions:
            filename = f"{q['id']}.yaml"
            path = drafts_dir / filename
            path.write_text(dump_str(q), encoding="utf-8")
            written.append(path)

        # ── Update spend ledger ──
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        ledger = _load_spend_ledger(vault_dir)
        ledger[today] = ledger.get(today, 0.0) + actual_cost
        _save_spend_ledger(vault_dir, ledger)

        # ── Summary ──
        console.print(f"\n[green]Generated {len(written)} drafts[/green] (${actual_cost:.4f}):")
        for p in written:
            console.print(f"  {p}")
        console.print(
            "\nNext: review each draft, then run "
            "[bold]vault promote <id> --reviewed-by <email>[/bold]"
        )
