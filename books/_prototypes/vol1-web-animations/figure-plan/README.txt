Figures and selective mechanisms across Volumes I–IV

The local chapter review now describes delivered figures and implemented
mechanisms. It covers all 71 numbered chapters across the four volumes. The
complete catalog contains 737 named body figures: 98 selected mechanisms and
639 retained static figures. Selection follows a specific changing-state
teaching question; there is no animation quota.

The static delivery manifest records 26 new or adapted figures with source,
PNG, actual themed HTML, and full-volume PDF review. Those reviews approve the
static delivery only. The 98 mechanism lessons are implemented and pass the
recorded source/schema, invariant, and companion calculation checks. All 98
now have bounded implementation acceptance against the exact final four-volume
HTML run 20261018-171632-19165. Every declared case endpoint was inspected in
desktop/narrow light/dark profiles, through fresh inspection or exact-pixel
prior-inspection proof; changed endpoints received fresh inspection. Fresh
representative in-flight motion samples were inspected in all four profiles.
This is finite implementation review, not every transition at every instant.
The separate focused runtime regression report records 35 passing tests.

Inputs and current views
- vol1.json through vol4.json preserve the original independent chapter source
  audits and their historical candidates. They do not control enrollment.
- ../animation-catalog/catalog.json is compiler-owned current figure metadata.
- The figure-explore mechanisms/selection.json and mechanisms/vol*-*.json files
  define the current selection, questions, scopes, steps, and authored cases.
- static-delivery.json records the reviewed static figures and build evidence.
- ../animation-catalog/evidence/acceptance-final.json binds the accepted scope,
  final build, registry, and per-lesson evidence. mechanism-technical-review.json
  records independent technical review; ../animation-catalog/evidence/ contains
  the final author, pilot-motion, mobile-control, and runtime-regression reports.
- review.json and index.html render the current chapter view. The complete
  inventory is linked at ../animation-catalog/index.html.

Lesson behavior
Lessons start paused. Next or N performs one atomic action and then pauses;
Back or P returns to the previous state. K toggles Play/Pause, and Restart or R
returns to the initial state. Letter shortcuts operate while the lesson has
focus and do not intercept typing in form fields. Authored cases are offered
only where the lesson defines them. Motion illustrates the declared mechanism;
its playback speed does not represent measured machine latency.

Expand moves the same lesson DOM into a larger workspace, preserving the
lesson rather than launching a separate simulator. Closing returns that same
component to its place in the chapter. The original diagram and caption remain
complete static counterparts, including the PDF fallback. The catalog's text
sequence preview is an authored description, not the actual chapter runtime.

Build the local review
  python3 books/_prototypes/vol1-web-animations/figure-plan/build-review.py

The checked-in pages link relative chapter routes in the final run
books/_build/parallel/20261018-171632-19165. All four HTML jobs passed.
No chapter build is selected automatically. With no explicit configuration,
chapter links remain disabled. To inspect a current parallel HTML build, use:
  python3 books/_prototypes/vol1-web-animations/figure-plan/build-review.py --build-root books/_build/parallel/RUN

For different runs per volume, pass --build-map PATH.json. The JSON maps volume
numbers to HTML output roots, for example:
  {"1": "books/_build/parallel/RUN/vol1-html/output"}
A value may instead be an object containing root and an optional link_base URL.
Chapter routes require the figure to exist; mechanism routes also require the
current mechanism-scene marker. Route presence does not establish approval.

Serve the books directory locally and open:
  /_prototypes/vol1-web-animations/figure-plan/index.html

Review boundary and learner feedback
The chapter page shows 714 chapter figures. The complete catalog also includes
23 figures in frontmatter and other supplementary sources whose visual needs
were outside the original numbered-chapter audit. Original source review is
not an exhaustive sentence-level audit or approval of unchanged artwork.

The acceptance is limited to the cases, profiles, samples, and controls recorded
in each report. Shared actual control sweeps cover two Volume I pilots. Two
genuine hidden-document headless attempts were inconclusive; no synthetic
replacement establishes hidden-tab pause behavior. Declared pulse/fade,
endpoint-commit, and unrouted incoming-identity fallbacks remain within the
reported scope. Some tall scenes require scrolling, and some motion captures
include sticky controls, partial upper context, or the host Back to top button.
These cosmetic and fallback limits are preserved in the acceptance evidence.

Simulated learner-perspective feedback is editorial feedback from reviewers
adopting a learner role. It is not feedback from recruited students. No actual
students participated in the recorded review, and no learning gains were
measured. Source, technical, runtime, visual, and student evidence remain
separate records.

Proposed actual-student protocol, not yet conducted
Recruit readers with the owning volume's prerequisites and record their prior
familiarity. Begin with the static figure and a short prediction question; then
ask each reader to use Next, Back, an authored case when present, and Expand to
explain one causal change. Include keyboard use and a phone where available.
Record incorrect predictions, explanations, navigation failures, and points
requiring help, with participant consent. Use a comparable transfer question
afterward and retain both successful and unsuccessful responses. Treat this as
formative usability and comprehension evidence. A learning-gain claim would
require a separately designed comparison study and analysis.

Author review of the new static figures

Open generated-figures.html for the 26 reviewed PNGs and their actual chapter
lead-ins, captions, and following explanations. Decisions and notes remain in
the reviewer's browser; Export my review downloads them without sending them.
Author sign-off is separate from the recorded implementation reviews.

Rebuild this focused gallery against an explicitly selected HTML run:
  python3 books/_prototypes/vol1-web-animations/figure-plan/build-generated-figures.py --build-root books/_build/parallel/RUN

Full-section integration review

section-integration-review.json records the complete containing-section reading
for the 26 new figures, including the neighboring section transitions. The
review produced targeted prose corrections in six chapter files; the record
preserves the exact replacements and pre/post source hashes. Drawings,
captions, equations, and executable cells remain unchanged.

The static provenance checker reverses only those exact approved prose edits
before comparing with the historical figure-review baseline. Its mutation
tests reject unlisted prose or changes to protected figure/math/code regions.
New HTML and PDF evidence is recorded separately from historical visual
reviews; inherited animation acceptance is explicitly distinguished from fresh
prose-context and one-action browser checks.
