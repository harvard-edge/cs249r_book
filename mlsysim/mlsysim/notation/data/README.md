# The notation registry

Every mathematical symbol in the four volumes of *Machine Learning Systems*
is defined here, once. The book never types a symbol directly: it writes a
macro (`$\LinkBandwidth$`), and the macro's LaTeX comes from this registry.
Change an entry here and every page that uses it changes with it.

A check enforces this. `./binder/binder check notation --scope strict` reads
every math span in the book, the same way Pandoc does, and reports any symbol
that is not a registered macro.

## Where symbols live

```
data/
  README.md            this guide
  exempt.yaml          the few things math may contain that are not symbols
  core.yaml            symbols used by two or more volumes
  vol1.yaml … vol4.yaml  symbols used throughout one volume
  local/
    vol3/16_rlvr.yaml  symbols used inside one chapter only
```

Pick the file by **where the symbol is used**, not by topic:

| The symbol is used in | It goes in |
|---|---|
| two or more volumes | `core.yaml` |
| several chapters of one volume, or that volume's notation chapter | `volN.yaml` |
| one chapter only (a derivation's working variables) | `local/volN/<chapter>.yaml` |

A volume may use its own symbols and the core ones. A chapter may also use its
local ones. Using another volume's symbol is an error: if the meaning really is
the same, the symbol belongs in `core.yaml`.

Symbols in `core.yaml` and `volN.yaml` are **book-wide**: each one has a row in
that volume's notation chapter (`books/volN/frontmatter/notation.qmd`, written
by hand). Local symbols are not listed there; the chapter defines each one in
prose where it first appears ("where $\KsStatistic$ is the largest gap between
the two cumulative distributions").

## What an entry looks like

```yaml
- macro: \LinkBandwidth
  latex: \beta
  definition: bandwidth of one network link, in the alpha-beta communication model
  kind: scalar
  unit: bytes/s
  notes: beta is local to the alpha-beta model; elsewhere bandwidth is \Bandwidth (BW).
  aliases: []
```

| Field | Required | What it holds |
|---|---|---|
| `macro` | yes | The name the book writes. See *Naming*. |
| `latex` | yes | Exactly what readers see. Bold, calligraphic, hats and `\text{}` subscripts belong here, never in the chapter. |
| `definition` | yes | What the symbol **means**, as a plain noun phrase a reader would recognize. No LaTeX, no `$`. Never "(none)". |
| `kind` | yes | One of `scalar`, `count`, `vector`, `matrix`, `set`, `function`, `distribution`, `operator`. The check uses it: a `vector` or `matrix` must render bold, a `set` calligraphic or blackboard. |
| `unit` | yes for `scalar` | SI or book units (`s`, `bytes/s`, `FLOP`). `dimensionless` when it has none. |
| `notes` | no | Why this glyph, what it must not be confused with. LaTeX allowed here. |
| `aliases` | no | Old macro names that still compile to this entry. The check flags their use so chapters move to the canonical name. |

The volume and scope come from the file the entry sits in, so they are never
written by hand and cannot disagree with it.

## Naming

A macro name says what the symbol **means**, in the words a reader would use.
Someone who has never seen the registry should be able to guess the meaning
from the name, and find the name from the meaning.

1. **PascalCase noun phrase, qualifier first, quantity last.** `PeakRate`,
   `LinkBandwidth`, `PrebrakeDelay`, `AttemptPassProbability`, `ReplicaCount`.
2. **Two to five words.** Long enough to be unambiguous across the whole book,
   because LaTeX macro names are global.
3. **Never the glyph.** Not `Alpha`, `BigN`, `BoldQ`. The glyph lives in `latex:`.
4. **Never a number or a joining word.** No digits, no `Two`/`Three`/`Four`,
   no `And`/`Or`. Two quantities get two entries.
5. **No placeholders.** No `Var`, `Symbol`, `Value`, `Quantity`, `Term` on their own.
6. **Acronyms are words.** Write `KvCache`, `Tco`, `Mtbf`, `Gpu`, and only
   for acronyms the book itself defines; otherwise spell it out.
7. **Same glyph, different meaning, different name.** `p` is
   `AttemptPassProbability` in one place and `ValidCandidateRate` in another.
   Two entries may share `latex:`; they never share a name or a meaning.
8. **Kind stays out of the name** unless the meaning needs it
   (`JointTorque` is a vector; `BarrierConstraintMatrix` says matrix because a
   vector of the same name exists).

## Adding a symbol

1. **Search by meaning first.** `./binder/binder notation find "link bandwidth"`
   lists entries whose name, definition or notes match. Reuse an entry whose
   meaning is the same; never reuse one whose letter merely looks right.
2. **Choose the file** from the table in *Where symbols live*.
3. **Write the entry** with every required field, name it by the rules above,
   and keep the file's existing grouping (entries sit under `# --- group ---`
   comment headings).
4. **Compile and check.** `python3 tools/build_notation.py`, then
   `./binder/binder check notation --scope registry`. The registry check rejects
   a missing field, a name that breaks the rules, a duplicate meaning, and a
   `kind` that does not match the LaTeX.
5. **Book-wide symbols** also get a row in that volume's notation chapter.

## Renaming or retiring a symbol

Move the old name into the new entry's `aliases:`. Both names keep compiling,
the check reports every use of the old one, and once the count reaches zero the
alias is deleted. Never change an entry's `latex:` to fit a new use: that
silently changes every page that already uses it. Make a new entry instead.

## What math may contain besides macros

`exempt.yaml` lists it: index letters `i`, `j`, `k`; `d` in differentials;
Euler's `e`; `\pi`; the transpose `^T`; numerals; operators and standard
functions (`\sum`, `\log`, `\max`, `\Pr`, `\nabla`, ...); the number sets and
expectation (`\mathbb{R}`, `\mathbb{E}`); big-O `\mathcal{O}`; and anything
inside `\text{}`, `\mathrm{}` or `\operatorname{}`. Nothing else.

A line that genuinely must show raw LaTeX (the notation chapter explaining what
not to write, for instance) can carry `<!-- notation-ok: reason -->`. The check
counts these and prints them on every run.
