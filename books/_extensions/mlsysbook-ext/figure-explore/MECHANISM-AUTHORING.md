# Mechanism scenes

A mechanism scene adds inspectable state changes to an existing numbered figure.
The chapter opens with its complete Diagram. Explore performs one action at a
time; a reader can pause, inspect its result, and choose the next action.

Shared controls and rendering belong to `mechanism-scene.js` and
`mechanism-scene.css`. Authored JSON in `mechanisms/` supplies the mechanism,
spatial surfaces, complete snapshots, and verified numerical endpoints.
The runtime does not infer an algorithm from explanatory prose.

## Authoring contract

The following versioned contract is shared by all lessons. Unknown fields and
unsupported primitives require a deliberate contract change, validator change,
and renderer review. They cannot silently become narrated cards.

```text
FORMAT
Each disjoint author file is a UTF-8 JSON array of lessons. No comments, trailing
commas, functions, HTML, Markdown, LaTeX, or executable strings. Plain Unicode
text is supported (×, →, subscripts). Do not use serialized SVG or raw CSS.
Every lesson must have a unique key matching its enrolled volume/figure.
Every author chunk must have a companion executable Python calculation file
with the same basename plus .calc.py (example: vol2-a.json + vol2-a.calc.py).
See REPRODUCIBLE NUMERICAL AUTHORING below.
No additional fields beyond this contract. Optional means it may be omitted;
all other fields are required. Null values are allowed only where stated.

SCHEMA NOTATION
string = ordinary string; id = [A-Za-z][A-Za-z0-9_-]*; number = finite JSON
number; integer = finite integer. enum(a,b) means exactly one named string.
A|B means one of those JSON shapes, not an actual pipe character in JSON.
An array written [T] means zero or more T unless a minimum is stated.
TextValue = string | number | null. Null means not yet produced/unoccupied,
NEVER numeric zero. IDs are scoped to one lesson except its globally unique key.

LESSON (EXACT TOP LEVEL)
{
 "schema":1,
 "key":"vol2:fig-example",
 "title":"Short instructional title",
 "question":"What changing mechanism becomes visible?",
 "scope":"Explicitly identify any toy case and boundaries.",
 "family":"cache",
 "narrowColumns":1,
 "objects":[Object],
 "connections":[Connection],
 "steps":[Step],
 "invariants":["Specific technical invariant, not generic reassurance."],
 "checks":[Check],
 "variants":[Variant],
 "details":{"summary":"Optional model details","paragraphs":["Plain text."]},
 "prediction":{"before":"step-id","question":"Question?", "answers":[
   {"id":"a","label":"Answer","feedback":"Why."}],"correct":"a"}
}
Optional top-level fields: narrowColumns, connections, checks, variants, details, prediction.
narrowColumns is integer 1 or 2, default 1. Use 2 only for deliberately authored
compact geometry whose cell labels remain legible at actual chapter width.
family enum: sequence, cache, spatial, reduction, schedule, ownership, curve,
control, communication, representation. Family is metadata, not a generic
rendering template. Geometry comes from actual Object kinds below.
Limits: 2–12 objects (normally 3–6), 4–9 snapshots (3–8 actions), up to 48
resident tokens, 6 metrics, 8 connections. More requires maintainer review, not a
silent schema extension. Aim for 4–8 snapshots. One action may act on several
items in parallel if it is one causal operation. Never hide sequential writes,
reads, computations or release behind a single step with timed sub-actions.

OBJECT (FIXED EXPLANATORY SURFACE)
{
 "id":"cache", "kind":"pool", "label":"Request KV cache",
 "role":"memory", "config":PoolConfig,
 "layout":{
   "wide":{"row":1,"column":0,"columnSpan":2},
   "narrow":{"row":2,"column":0,"columnSpan":1}
 }
}
role enum: input, output, compute, memory, communication, control, reference.
kind enum: sequence, matrix, pool, lanes, plot, numberLine, register, chamber.
Layouts: wide is a 2-column grid, narrow uses lesson.narrowColumns (default 1). Rows are zero-based
integers 0–9, column is 0 or 1 in wide and less than narrowColumns in narrow; columnSpan defaults to
1 if omitted. No row spans. Every object must have both layouts. Within a
layout, occupied grid positions cannot overlap. DOM reading order follows
narrow row order. Labels must remain meaningful without color.
Tokens and changing state are NOT described in Object; they live in snapshots.

OBJECT CONFIGS (DISCRIMINATED BY kind; EXACT FIELDS)
sequence: {"slots":6,"slotLabels":["0","1","2","3","4","5"]}
matrix: {"rows":3,"columns":3,"rowLabels":["0","1","2"],
         "columnLabels":["0","1","2"]}
pool: {"slots":8,"slotLabels":["0","1","2","3","4","5","6","7"],
       "columns":4}
lanes: {"lanes":["Compute","Transfer"],"start":0,"end":8,
        "axisLabel":"Schematic progress","ticks":[0,2,4,6,8]}
plot: {"x":{"min":0,"max":8,"label":"Input","ticks":[0,2,4,6,8]},
       "y":{"min":0,"max":1,"label":"Response","ticks":[0,0.5,1]},
       "equalAxes":false}
numberLine: {"min":-1,"max":1,"ticks":[-1,-0.5,0,0.5,1],
             "axisLabel":"Represented real value"}
register: {"type":"INT32","unit":"integer units"}
chamber: {"slots":4}
Optional config fields: sequence.slotLabels; matrix.rowLabels,columnLabels;
pool.slotLabels,columns (default min(slots,4)); lanes.ticks; plot axis ticks;
numberLine.ticks; register.unit; plot.equalAxes (boolean, default false). All other config fields required.
Use equalAxes:true only for commensurate geometric axes such as two position
coordinates. It preserves world-unit lengths through responsive letterboxing;
never use it for unrelated dimensions such as time and velocity.
Limits: matrix up to 8×8, pool/sequence slots 1–24, chamber slots 1–12,
lanes 1–6, lane/plot/numberLine ranges strictly increasing. Label arrays must
match declared counts. Axis ticks must lie within declared ranges.
A matrix is not a card filled with a string rendering of an array: use its
cells. A plot is not a paragraph describing a curve: provide actual points.

ANCHOR (POSITION FOR TOKENS OR CONNECTION ENDPOINTS)
{"object":"input","index":4}
index is zero-based: matrix row-major, pool/sequence/chamber slot, lanes lane.
For plot and numberLine ONLY, use {"object":"line","x":0.31,"y":0}
(y is required for plot, omitted for numberLine). All coordinates must be
inside axes. For whole object center use {"object":"compute"}.
Index and x/y are mutually exclusive. No page pixels or CSS coordinates.

CONNECTION (FIXED RELATION; OPTIONAL TOP-LEVEL ARRAY)
{"id":"read-cache","from":{"object":"cache"},
 "to":{"object":"compute"},"kind":"read","label":"Read retained K/V"}
kind enum: read, write, copy, move, dependency, feedback, association.
Connections indicate relationships; they do not themselves count as changing
mechanism. A connection becomes active through snapshot.activeConnections.
Never claim latency or bandwidth through decorative distance or line speed.

STEP
{
 "id":"append", "title":"Append the current key and value",
 "explanation":"The request retains the current token state for later queries.",
 "snapshot":Snapshot,
 "transition":Transition
}
First step transition MUST be null. Every other step transition is required
and non-null. IDs unique. Title should describe one action, normally ≤75
characters. Explanation ≤280 characters, preferably one or two sentences.
Snapshot.result is a separate consequence, not a duplicate explanation.
P reverses ONE authored action (previous snapshot), never a whole phase.

SNAPSHOT (COMPLETE, AUTHORITATIVE, NO PATCHES)
{
 "objects":{
   "input":ObjectState,
   "cache":ObjectState
 },
 "tokens":[Token],
 "metrics":[{"id":"occupied","label":"Retained positions","value":3,"unit":"positions"}],
 "activeConnections":["read-cache"],
 "result":"Three positions remain cached; the next query can reuse them."
}
Every snapshot.objects map MUST contain exactly all lesson object IDs. Every
snapshot includes tokens, metrics, activeConnections, result, even if arrays
are empty. Every snapshot is independently renderable after a menu jump.
Metric value is number|string (not null); unit is optional plain string. Use
identical metric IDs/units throughout the lesson. No magic computed expressions
inside strings. Author numerical endpoints from declared inputs and verified
arithmetic; real specifications still come from the book's MLSysIM/LEGO source.

OBJECT STATES (DISCRIMINATED BY CORRESPONDING Object.kind)
All states require a short "status" string (may be empty). No additional fields.

SequenceState:
{"values":["t1","t2",null,null],"status":"Prompt available"}
values length MUST equal config.slots. Values are TextValue.

MatrixState:
{"values":[1,2,3,4,5,6,7,8,9],
 "window":{"row":0,"column":0,"rows":2,"columns":2},
 "status":"First receptive field"}
values length rows*columns, row-major. window optional or null; its complete
rectangle must fit the matrix. Source values never move when window moves.

PoolState:
{"slots":[
 {"value":"K1/V1","owner":"Request A","status":"retained"},
 null,null,null],"status":"One cached position"}
slots length config.slots. Each non-null slot is exactly {value:TextValue,
owner:string,status:enum(allocated,retained,ready,inflight,shared)}.
A null slot means free. A resident slot MUST have nonempty owner and non-null
value. Release changes slot to null, not merely a lighter fill color.

LanesState:
{"segments":[{"id":"task-a","lane":0,"start":0,"end":2,
 "label":"Layer 1","state":"done","role":"compute"}],
 "cursor":2,"status":"The transfer can begin"}
segments are full task records. state enum: waiting, running, done, blocked.
role uses Object.role enum. Cursor optional|null. Segment times inside range,
start<end, lane zero-based valid. A segment's geometry must remain stable
across snapshots unless the lesson explicitly changes a schedule assumption.
No requirement that schematic lane units represent measured seconds.

PlotState:
{"series":[{"id":"response","label":"Response","role":"output",
 "points":[[0,0],[1,0.3],[2,0.6]]}],
 "markers":[{"id":"current","x":2,"y":0.6,"label":"Current state"}],
 "status":"The response rises"}
series/markers may be empty. Points are finite [x,y] pairs in range, ≥1 point
per series; segment interpolation is schematic unless explicitly derived.
role uses Object.role enum. Do not animate observational plots into causal
experiments. Preserve the original evidence or mark the lesson blocked.

NumberLineState:
{"levels":[-1,-0.5,0,0.5,1],
 "markers":[{"id":"original","value":0.31,"label":"Original 0.31","role":"input"},
 {"id":"rounded","value":0.5,"label":"Rounded 0.50","role":"output"}],
 "status":"Rounded to the nearest represented level"}
levels may be empty, sorted ascending within range, no duplicates. Markers
have stable IDs, values inside range and role from Object.role enum. Original
markers remain visible when comparison requires them. Moving an existing
marker must not silently relabel an original value as a transformed value.

RegisterState:
{"value":24,"previous":0,"expression":"0 + 24","status":"One product accumulated"}
value is TextValue, previous optional TextValue, expression optional string.
Register.value stays at the old retained result during travel and commits only
at the step's end. INT32 is a type label, not a fake storage-speed claim.

ChamberState:
{"values":["q4","K4/V4",null,null],"status":"Current-token state computed"}
values length config.slots. Use tokens for visible travel between surfaces;
chamber values expose stable endpoint contents. Neither is generic prose.

TOKEN (VISIBLE MOVING DATA WITH IDENTITY)
{"id":"k1-read","label":"K1","value":"K1","role":"input",
 "owner":"Request A","at":{"object":"compute","index":0},
 "state":"copy","sourceId":"k1"}
Required fields: id,label,value,role,owner,at,state. sourceId optional and MUST
be present when state=copy. state enum: resident, copy, inflight, consumed.
value TextValue; role from Object.role enum. Owner may be empty only for a
nonowned illustrative input; memory-related tokens must name ownership.
sourceId points to a token present in the SAME snapshot. Copy IDs differ from
source IDs. For a source represented by a pool cell rather than token, use
optional sourceAt:Anchor instead of sourceId (exactly one for copies).
Tokens may overlay underlying cell values while moving. They are NOT counted
as extra physical allocations unless that allocation is explicitly modeled.
Do not duplicate unchanged labels unnecessarily: endpoint cells remain the
legible baseline; motion shows a read/copy/value transformation.

TRANSITION (ONE CAUSAL ACTION)
{
 "kind":"copy", "subjects":["k1"],
 "from":{"object":"cache","index":0},
 "to":{"object":"compute","index":0},
 "duration":900,
 "meaning":"Read K1 without evicting it."
}
Required: kind,subjects,meaning. Optional from,to,duration. duration defaults
900ms, allowed 300–1600ms. No delays, callbacks, chained animations or queues.
kind enum:
- move: subjects retain ID; location/owner changes from prior to next state.
- copy: prior source remains; a destination copy appears (token sourceId or
  sourceAt identifies provenance); original residency remains unchanged.
- write: destination cell/register/pool changes; old content persists until
  final commit. subjects may be [] for scalar writes with from/to anchors.
- transform: original and resulting representations visibly connected; subject
  token value or number-line marker/register changes. Original reference may
  remain as a separately identified object.
- release: resident token/slot disappears and ownership becomes free; no
  disappearance before last consumer. subjects may name tokens or object IDs.
- window: a matrix window's row/column changes; source values stay unchanged.
- execute: one bounded logical computation or task completes, causing actual
  numeric/segment/state change. Not a catchall for timed multi-action tours.
- reveal: an actual new output/input/state becomes available; cannot merely
  reveal an existing label, explanatory card, or prose paragraph.
subjects IDs may name token, object, segment, series, marker, or connection;
all must resolve in prior or next snapshot. Copy subject IDs normally name
source tokens and destination may be identified by new token.sourceId.
A group of parallel items can move together. Multiple sequential operations
require separate steps. In-place state changes are valid without token travel.
All semantic changes required to understand the mechanism must be in snapshots,
not encoded only in transition.meaning or step.explanation.

CHECK (OPTIONAL MACHINE-CHECKABLE INVARIANTS)
Supported exact forms:
{"kind":"metricEqualsOccupied","metric":"occupied","object":"cache"}
{"kind":"metricEqualsNonNull","metric":"written","object":"output"}
{"kind":"constantValues","object":"input"}
{"kind":"nondecreasingMetric","metric":"produced"}
{"kind":"sum","sources":["partialA","partialB"],"target":"total"}
For sum, sources/target name register objects, null values skip that snapshot.
Numeric values must be actual JSON numbers. Checks run on every snapshot.
Other invariants remain plain prose and receive independent technical review;
do not invent unsupported checker kinds.

VARIANT (OPTIONAL DECLARED CASES; NO UNMODELED SLIDERS)
{
 "id":"fine-grid", "label":"Finer activation spacing",
 "scope":"Same inputs; activation spacing changes from 0.10 to 0.05.",
 "steps":[Step],
 "invariants":["Additional case-specific invariant."]
}
Required: id,label,scope,steps. Optional: invariants. No other variant fields.
Top-level variants is an array of 1–4 ALTERNATIVE cases. Base lesson.steps and
lesson.scope remain the default case, labeled Original case by the renderer.
Alternative variant IDs must not be "base", and must be unique. All variants
reuse the SAME lesson objects/configs/connections. They have complete replacement
steps, NEVER patches. All snapshots must contain identical metric IDs/units to
the base case. Base checks run unchanged on ALL variants. Variants do not have
a checks field. Case-specific invariants supplement base invariants; they do not
waive them. Base invariants must therefore be valid across all declared cases.
Keep step IDs/count identical when corresponding actions exist. Switching case
preserves the current step only when the ENTIRE step-ID sequence matches the
base sequence; otherwise it resets to step 0 with explicit visible notice.
Always pause when changing case, settle/cancel old interpolation, and visibly
display the new scope. No hidden workload/precision/model change. Selectors offer
declared cases only; arbitrary sliders require a separate verified causal model
and maintainer review. Examples: cache enabled vs recompute, coarse/fine quantization,
monolithic/chunked schedules. Changing plot/numberLine axis ranges or matrix/pool
sizes is not supported by variants: choose compatible fixed configs or separate
lessons. Do not squeeze differing system models into misleading shared geometry.
Prediction.before must resolve in every variant if prediction is present. Only
use a shared prediction when its correct answer remains true for all cases.

EXPANDED WORKSPACE (RUNTIME BEHAVIOR, NOT LESSON DATA)
Every lesson may offer an Expand control through a native dialog workspace.
Move the SAME interactive component DOM into the workspace; do not clone controls
or duplicate IDs/state. Opening pauses and settles any in-flight action to its valid endpoint before
relocation, preventing stale motion geometry. The dialog exposes a clear Close control
and native Escape support, retains the current case/step, contains keyboard focus,
and restores focus to Expand on close. Closing pauses and restores the same DOM
to its original position. Chapter/static figure ownership remains unchanged.
Do not use a new app shell, iframe, separate page, or reset the user's sequence.
N/P/K/R remain scoped to the component inside the dialog. Native dialog inertness
must not prevent the original focus target from being restored after close.

REPRODUCIBLE NUMERICAL AUTHORING (REQUIRED)
Every author chunk has one companion .calc.py file; e.g. mechanisms-vol2-a.json
and mechanisms-vol2-a.calc.py. Run it with python3; no network or extra package
installation. It prints the derived endpoints with lesson/case/step IDs and
asserts that JSON snapshot outputs/metrics equal recomputed values. It must exit
nonzero on disagreement. It must NOT silently overwrite prose or authored JSON.
All derived toy values (sums, products, rounding, offsets, counts, bytes, window
outputs, occupancy, interval unions, comparisons, transformed coordinates, etc.)
are computed from explicit primitive inputs in this rerunnable code. Do not hand
copy arithmetic into a function that merely asserts the same hard-coded answer.
Declared inputs may be literal; derived outputs may not be invented literals in
the calculation. Python recomputes them and checks every occurrence in base and
variant JSON. A conceptual state/lifecycle example still computes displayed
counts from its actual objects/slots/task intervals rather than listing a known
answer. If there are no derived numbers, the script explicitly verifies that
fact and states the purely qualitative checks it performs.
Browser snapshot numbers remain plain JSON for deterministic rendering; the
companion file proves their derivation and catches drift. Declarative checks are
additional runtime/schema invariants, not a replacement for these calculations.
Real hardware/model specs and physical constants must come through the book's
MLSysIM/LEGO source, never a parallel author-created registry. Include precise
scope/units; illustrative motion durations are UI defaults, not claimed latency.

PREDICTION / DETAILS
Prediction is optional; it blocks neither stepping nor playback. Exactly one
correct answer ID; 2–4 answers. Feedback is text, no external service or stored
student data. details paragraphs explain optional arithmetic/model assumptions,
not essential interpretation hidden from the animation.

CONTROL CONTRACT (CORRECTION TO EARLIER PROPOSAL)
N=Next/finish current action; P=previous ONE action; K=explicit Play/Pause;
R=Restart. Letters work while a native BUTTON inside the focused component
has focus (clicking Next leaves button focus). Ignore keydown only for typing
inputs, select, textarea, contenteditable, modifier combinations, repeats and
defaultPrevented events. Do not intercept Space/Enter native button behavior.
No global letter-key listener. Real semantic buttons with visible help.
Next always pauses at the new snapshot. N during travel finishes that SAME
action, pauses, and does not queue another. P/menu/R paint an exact valid state
and pause. No autoplay on page load, opening Explore, visibility, theme change.
K explicitly permits continuous play with ≥3s stable hold per result. Reduced
motion retains states and stepping. View/visibility/offscreen pause stops both
WAAPI interpolation and advancement. Height-only resize does not advance.

VALIDATOR REQUIREMENTS (BEFORE ENROLLMENT)
1. Reject malformed JSON, unknown fields, duplicate keys/IDs, unsupported kinds,
   nonfinite/out-of-range values, layout overlap, mismatched cell lengths,
   missing state objects and unresolved anchors/subjects/connections.
2. Enforce step0 transition=null and every later transition non-null; stable
   metric IDs/units; same token ID cannot change owner silently under copy.
3. Require actual state change per action: token appearance/removal/value or
   position, matrix values/window, pool occupancy/owner/value, lane task state,
   plot data/markers, number-line markers, register value. Pure status text,
   narration, activeConnections or token color changes DO NOT qualify alone.
4. Run declared checks on all snapshots. Do not judge numerical correctness
   merely because JSON parses. Technical reviewer checks equations/dependencies.
5. Require scope, result and invariant text; exact figure key/source/art binding
   comes from enrollment manifest/compiler, not a guessed source in author data.
6. Validate every variant with the same structural and shared invariant checks.
   Run the companion .calc.py; reject inconsistent derived outputs or missing
   numerical derivation. Unknown/missing lesson/bundle/freshness fails closed to
   complete Diagram.
7. Original static technical meaning, units, identities, ownership/dependencies
   and evidence bind. NEW geometry/reflow/focused scenes ARE ALLOWED; no need
   to copy a print drawing's geometry. Mark toy scope explicitly; never silently
   replace a model, precision or workload during a tab switch.
8. Every distinct layout/state is rendered and PNG-inspected inside the actual
   themed chapter. Validation and screenshot generation alone are not visual
   approval. Simulated learner review is not actual student testing.

DISTINCT MECHANISM EXAMPLES (AUTHOR USING EXACT SCHEMA)
KV cache: use pool slots for persistent K/V positions, sequence for prompt and
output, chamber for current-token computation. Prefill computes/stores prompt
K/V and yields first output token. Decode consumes that emitted token, computes
its current query/K/V, reads retained plus current K/V, appends current-token
state, then emits next token. Do not cache an output token before it is consumed.
Separate read/copy from append. Cache occupancy grows; reads preserve contents.
Optional release is separate and only after request completion/last consumer.

Convolution: use actual 3×3 matrix [1..9], actual 2×2 kernel [1,0;0,1], register,
and 2×2 output. State0 blank output; move footprint to (0,0); compute 6; write
output[0]; move footprint to (0,1); compute 8; write output[1]. Lower row stays
null and scope says first row shown. Source cells and kernel stay unchanged.
Call operation cross-correlation (normal CNN convention), not a flipped kernel.

Checkpoint staging: lanes for execution/copy, pool for immutable staged state,
register for committed manifest generation. A copy completion does not imply
publication. Retain prior restore point until manifest commit; release staging
only after publication/consumer obligations. This is ownership and dependency,
not moving boxes through architecture labels.

Control overshoot: plot reference trajectory and sampled safe points, move a
continuous-state marker between samples to reveal bound violation. Show sampled
safety and intersample safety separately; preserve original assumption/model.
Never infer an actual physical safety guarantee from illustrative interpolation.

```

## Review and enrollment

The compiler packages each reviewed lesson as a canonical JSON string and its
SHA-1 digest. The registry binds that digest to the figure. Lua verifies the
volume bundle before injection; the asynchronous factory verifies the exact
lesson bytes before parsing. Missing or stale data leaves the static diagram.
These checks detect drift; they are not a security sandbox.

Inspect every distinct layout and state inside the actual themed chapter at
desktop and phone widths, in light and dark themes. Verify source artwork,
values, ownership, dependencies, controls, menu jumps, motion cancellation,
reduced motion, and static output. Simulated learner review must be described
as such; actual learning gains require actual student observations.

## Readable numerical displays

Authoritative values remain unchanged in snapshot data and `data-value`. The
visible display uses up to six significant digits and scientific notation for
extreme magnitudes. A displayed `≈` marks a rounded value; its title gives the
exact stored value. Nonzero values retain their sign and never become a displayed
zero through formatting. Symbolic identifiers and authored strings are unchanged.

Matrix rows retain their dimensions. Sequence and chamber surfaces may wrap
indexed slots into additional rows to keep words and values legible. Dense pools with short cell identities retain the declared physical geometry
and use a visible ownership key when per-cell owner text cannot fit. Other pools
wrap slots in index order so every resident and free slot stays visible. Completed copies integrate
with existing resident values rather than covering their metadata. Narrow task
intervals retain their duration and use linked task identifiers with a full key.

During a transfer, committed values stay visible until the action finishes.
Registers reserve real receiving cells labeled Incoming; these cells are separate
from the committed value and its units. Multiplication expressions remain at the
formation site while short result values travel. If a readable value chip cannot
pass between labels, a small carrier follows a visible track while the value
remains readable at its source and receiving cell. These routes illustrate data
transfer; their distance and duration do not represent physical latency.
Chambers show cell labels only when `slotLabels` are authored; their internal
indices remain available to transitions and inspection.
