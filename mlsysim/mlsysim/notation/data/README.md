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
  README.md          this guide
  exempt.yaml        what math may contain that is not a symbol
  vocabulary.yaml    the units and quantity words entries are checked against
  performance.yaml   iron law, roofline, memory, serving, energy
  learning.yaml      data, models, training, loss, degradation
  distributed.yaml   collectives, communication models, reliability, scheduling
  agentic.yaml       trajectories, task contracts, tools, budgets, evaluation
  physical.yaml      plant state, kinematics, contact, control, perception
```

An entry goes in the file for **what it is**, never for where the book uses
it: the registry says what a symbol means; the book decides where to use it.

**How the book declares what it uses.** A chapter may write a macro when one of
these lists it:

1. **its volume's notation chapter** (`books/volN/frontmatter/notation.qmd`,
   written by hand): every symbol used across several chapters of that volume
   has a row there; or
2. **its own front matter**, for symbols only that chapter uses:

   ```yaml
   notation-local: [\ReplicaCount, \ComponentAvailability]
   ```

   The chapter defines each local symbol in prose where it first appears
   ("where $\ReplicaCount$ is the number of replicas the router probes").

Writing any other registry macro is an error (`out-of-scope`): borrowing a
symbol because its letter looks right is how meanings drift. If a second volume
needs the same meaning, add the row to that volume's notation chapter too.

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
| `kind` | yes | One of `scalar`, `count`, `index`, `vector`, `matrix`, `set`, `function`, `distribution`, `operator`, `record`. A `record` is a structured or textual object (a state, an action, a trajectory, a goal, a key) and has no unit. The check uses the kind: a `vector` or `matrix` must render bold, a `set` calligraphic or blackboard. |
| `unit` | yes for `scalar` | Built from the units in `vocabulary.yaml` (`s`, `bytes/s`, `USD/token`, `kg*m^2`); `dimensionless` when it has none; `varies` when it depends on what the symbol is applied to, with notes saying how. Counts, indices and records have none. See *Units*. |
| `notes` | no | Why this glyph, what it must not be confused with. LaTeX allowed here. |
| `render_exception` | no | Only when the LaTeX keeps a field convention the kind rule would reject: `\theta` as a non-bold parameter vector, upright `SE(3)` as a set. One sentence saying which convention. It excuses the kind-vs-LaTeX rule and nothing else. |
| `aliases` | no | Old macro names that still compile to this entry. The check flags their use so chapters move to the canonical name. |

The domain comes from the file the entry sits in. Nothing in an entry says which
volume uses it: the notation chapters and `notation-local` lists say that.

## Naming

A macro name says what the symbol **means**, in the words a reader would use.
Someone who has never seen the registry should be able to guess the meaning
from the name, and find the name from the meaning.

1. **PascalCase noun phrase, qualifier first, quantity last.** `PeakRate`,
   `LinkBandwidth`, `PrebrakeDelay`, `AttemptPassProbability`, `ReplicaCount`.
2. **Two to five words.** Long enough to be unambiguous across the whole book,
   because LaTeX macro names are global. One-word names are reserved for the
   series core (`Latency`, `Bandwidth`, `Energy`, `Policy` and a few more,
   listed in `types.py`).
3. **Never the glyph.** Not `Alpha`, `BigN`, `BoldQ`. The glyph lives in `latex:`.
4. **Never a number or a joining word.** No digits, no `Two` to `Ten`, no
   `Other`, no `And`/`Or`/`Of`/`The`, and no leading article (`AWorkerCount`).
   Two quantities get two entries.
5. **No placeholders or glyph words.** No `Var`, `Symbol`, `Value`, `Quantity`,
   `Term`, `Macro`; no Greek letter names (`Alpha` to `Omega`, so Cohen's kappa
   is `InterAnnotatorAgreement`); no decoration names (`Bold`, `Hat`, `Bar`,
   `Tilde`, `Dot`, `Cal`); no `Chapter` or `Appendix`. The check
   (`convention_problems` in `types.py`) holds the exact list.
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
2. **Choose the file** by what the symbol is (see *Where symbols live*).
3. **Write the entry** with every required field, name it by the rules above,
   and keep the file's existing grouping (entries sit under `# --- group ---`
   comment headings).
4. **Compile and check.** `python3 tools/build_notation.py`, then
   `./binder/binder check notation --scope registry`. The registry check rejects
   a missing field, a name that breaks the rules, a duplicate meaning, and a
   `kind` that does not match the LaTeX.
5. **Declare it where it is used**: a row in the volume's notation chapter, or
   the chapter's `notation-local` list if no other chapter uses it.

## Renaming or retiring a symbol

Move the old name into the new entry's `aliases:`. Both names keep compiling,
the check reports every use of the old one, and once the count reaches zero the
alias is deleted. Never change an entry's `latex:` to fit a new use: that
silently changes every page that already uses it. Make a new entry instead.

## What math may contain besides macros

`exempt.yaml` lists it, and nothing else is allowed:

- **Index letters by role.** `i`, `j`, `k`, `n`, `t` inside a subscript only
  (`x_i`, `\sum_{i=1}^{\NumberOfDevices}`, `\MechanicalPlantState_{t}`,
  `\ActionChunk_{k+1}`). The same letter outside a subscript is a symbol.
- **`d`** immediately before a variable in a differential (`dx`, `d\Time`),
  **`e`** as a base (`e^{x}`), **`T`** as a transpose superscript (`A^T`).
- **Constants and operators**: `\pi`, `\infty`, numerals, and the LaTeX
  built-ins that are not symbols (`\sum`, `\frac`, `\log`, `\max`, `\Pr`,
  `\nabla`, spacing).
- **Number sets and friends**: `\mathbb{R}`, `\mathbb{N}`, `\mathbb{Z}`,
  `\mathbb{E}`, `\mathbb{P}`, `\mathbb{1}`, big-O `\mathcal{O}`.
- **Upright text**: anything inside `\text{}`, `\mathrm{}` or
  `\operatorname{}`. Upright single letters are units (`5\,\text{s}`) or
  labels (group `\text{A}`); a variable is always italic, so writing one
  upright is a typesetting error the reader would see.

A macro may carry, without becoming a new symbol:

- an **index** subscript: index letters, digits, or index and count macros
  (`\Time_{t}`, `\LayerWidth_0`, `\GemmLeftOperandElement_{ik}`,
  `\WeightMatrix_{\LayerIndex}`);
- a **power**, including a power by another macro (`\Time^2`,
  `\DatasetSize^{-\DataScalingExponent}`), an **inverse** (`^{-1}`) or a
  **transpose** (`^\top`, `^T`);
- a **parenthesized index** naming a layer, block or iteration
  (`\WeightMatrix^{(\LayerIndex)}`, `\SoftmaxRowMaximum^{(1)}`);
- **`*`** for the optimal value (`\RetrainingInterval^*`);
- a **prime** on a function, meaning its derivative (`\ActivationFunction'`).

When the macro's LaTeX already carries a subscript (`p_{\text{gate}}`), group it
before adding an index: `{\RouterGateProbability}_{ij}`. Written bare,
`\RouterGateProbability_{ij}` is a double subscript, which LaTeX refuses; the
check reports it. The same holds for a power on a macro that already carries a
superscript.
A macro used as a subscript or superscript is always braced: `2^{\AttributionFeatureCount}`,
`\sup_{\MonitoredFeatureLevel}`. Unbraced, LaTeX takes only the first token of
its LaTeX, so a label lands on the base or the build stops.

Any other decoration, above all a text label (`\Bandwidth_{\text{bisect}}`,
`\SoftmaxRowMaximum^{\text{new}}`), or a prime on a quantity, makes a new
symbol, which needs its own entry.

**Mentioning a letter.** Sometimes prose talks about a letter rather than a
quantity: "the book writes $B$ for batch size, not bandwidth". Wrap it:
`$\Glyph{B}$`. It renders exactly like `B`, the check ignores its argument,
and anyone reading the source sees that the letter is mentioned, not used.

## Units

`vocabulary.yaml` lists every unit an entry may use, each with its base
dimensions. Units combine with `*`, `/`, `^n` and parentheses. Data, compute,
money and the countable things ML systems move (tokens, requests, samples) are
dimensions of their own, so `bytes/s` is never confused with `Hz`, nor
`USD/token` with a probability. Write the plural for an amount (`tokens`) and
the singular after a slash (`USD/token`). A unit not built from the list fails
the check; add it to the list only when no existing unit fits.

## Quantity words

Some words promise what kind of quantity a name is. When a name ends in one of
them, the entry must agree: a `...Time`, `...Latency` or `...Interval` is in
seconds, a `...Count` has kind `count`, a `...Probability` or `...Fraction` is
dimensionless, a `...Bandwidth` is data per second, a `...Cost` is in USD, a
`...Matrix` has kind `matrix`, and so on. A `...Rate` is per unit of time, or a
dimensionless share per trial as statistics uses the word (error rate). The
full list, with the kinds and dimensions each word allows, is in
`vocabulary.yaml`. Names that end in none of these words are free; this rule
only stops the common words from meaning two things. A few field terms that end
in such a word but mean something else (`LearningRate`) are listed there as
idioms, each with its reason.
