// Physical AI Kit — Arduino partner brief (sendable PDF)
// Rebuild: typst compile brief.typ Physical-AI-Kit-Arduino-Brief.pdf

#set document(
  title: "Physical AI Kit — Partner Brief for Arduino",
  author: "Prof. Vijay Janapa Reddi",
)
#set page(
  paper: "us-letter",
  margin: (x: 0.6in, y: 0.5in),
)
#set text(font: "New Computer Modern", size: 9.2pt)
#set par(justify: true, leading: 0.62em)
#show heading.where(level: 1): it => {
  set text(size: 11.5pt, weight: "bold", fill: rgb("#1a365d"))
  block(above: 0.7em, below: 0.3em, it.body)
}

#let accent = rgb("#1a365d")
#let muted = rgb("#4a5568")

#grid(
  columns: (1fr, 1.25in),
  column-gutter: 0.3in,
  [
    #text(size: 17pt, weight: "bold", fill: accent)[Physical AI Kit]
    #v(-0.3em)
    #text(size: 10pt, fill: muted)[Partner brief for Arduino · successor to the TinyML Kit]

    #v(0.25em)
    #text(size: 8.2pt)[
      *From* Prof. Vijay Janapa Reddi · Harvard / ETH Zurich \
      *Contact* #link("mailto:vj\@eecs.harvard.edu")[vj\@eecs.harvard.edu] \
      *Curriculum* #link("https://physical.mlsysbook.ai")[physical.mlsysbook.ai] ·
      #link("https://github.com/harvard-edge/physical-ai")[github.com/harvard-edge/physical-ai]
    ]
  ],
  align(right, image("cover-brief.png", width: 1.25in)),
)

#line(length: 100%, stroke: 0.75pt + accent)
#v(0.15em)

= Where this sits for you

#grid(
  columns: (1fr, 1fr),
  column-gutter: 0.35in,
  [*TinyML Kit* — deploy a model on a small board.],
  [*Physical AI Kit* — stand behind a machine whose learned proposals may *act*, only when independently *permitted*.],
)

UNO Q’s dual-brain (Linux MPU + real-time MCU) is the teaching instrument: *MPU proposes · MCU permits*. That matches the silicon—not a bolted-on metaphor.

= The ask

Co-create a *Physical AI Kit* on *Arduino UNO Q*—backed by a university project course and an open book—so the TinyML education line has a clear next chapter. We are *not* asking Arduino to invent pedagogy. We want a *kit that embodies pedagogy that already exists*.

= Why now

AI is leaving the screen. Models command motors and contactors. In software, a bad output is a retry; in physics it is momentum—no `ctrl+z`. Embodied demos are everywhere; the missing piece is systems discipline: measure the full path, keep running when intelligence fails, and separate *proposal* from *permission* before kinetic energy hits the world.

= What learners practice

#table(
  columns: (0.95in, 1fr),
  inset: (x: 5pt, y: 3.2pt),
  stroke: 0.35pt + rgb("#cbd5e0"),
  fill: (_, y) => if y == 0 { rgb("#edf2f7") } else { white },
  [*Primitive*], [*Question*],
  [*Loop*], [Does action change the next observation?],
  [*Time*], [How old may belief be when the move happens?],
  [*Budget*], [What is shared among sensing, inference, link, actuation?],
  [*Permission*], [What may refuse a capable proposal?],
  [*Evidence*], [What measurement supports ship / condition / refuse?],
)

#v(0.1em)
#align(center)[
  #image("fig_mcu_sbc_boundary.svg", width: 82%)
  #text(size: 7.5pt, fill: muted)[Host proposals vs trusted physical permission]
]

= Pedagogy already built

#table(
  columns: (0.85in, 1fr),
  inset: (x: 5pt, y: 3.2pt),
  stroke: 0.35pt + rgb("#cbd5e0"),
  fill: (_, y) => if calc.odd(y) { rgb("#f7fafc") } else { white },
  [*Book*], [_Physical AI: Machine Learning Systems That Sense and Act_ — 11 chapters + capstone],
  [*Course*], [Project seminar / studio · dossier + midterm + defense · no classical written exam],
  [*Labs*], [Kit contracts: bring-up → measure → *MCU enforcer* → ship gate],
  [*Baseline*], [Quantize / serve → #link("https://mlsysbook.ai")[mlsysbook.ai] (not re-taught here)],
)

*Signature lab:* hang the MPU; the MCU still holds. Capstone ends in *deploy / condition / refuse*.

= Kit sketch

#table(
  columns: (1.75in, 1fr),
  inset: (x: 5pt, y: 3pt),
  stroke: 0.35pt + rgb("#cbd5e0"),
  [*Include*], [*Why*],
  [Arduino UNO Q (dual-brain)], [MPU proposes · MCU permits],
  [Camera + motion / actuation], [Real causal loop on the desk],
  [Sensors + safe-idle defaults], [Freshness, faults, studio safety],
  [Starter firmware + course link], [Failed week ≠ stranded · box → physical.mlsysbook.ai],
)

#block(
  width: 100%,
  inset: 7pt,
  fill: rgb("#edf2f7"),
  radius: 3pt,
  text(size: 8.5pt)[
    *Box line:* `PHYSICAL AI KIT` · Proposal ≠ permission · Dual-brain UNO Q · physical.mlsysbook.ai
  ],
)

= Partnership & next step

#grid(
  columns: (1fr, 1fr),
  column-gutter: 0.3in,
  [*Arduino* — kit SKU, store, bring-up quality, co-marketing TinyML → Physical AI.],
  [*Academic* — book + labs + syllabus, proposal–permission method ownership.],
)

1. Kit contents (studio MVP vs retail) · 2. Naming · 3. Pilot → open → store · 4. Narrative: *TinyML deployed the model; Physical AI stands behind the machine.*

#v(0.15em)
#block(
  width: 100%,
  inset: 8pt,
  stroke: 0.65pt + accent,
  radius: 3pt,
  [
    #set text(size: 8.4pt)
    *Forwardable.* Harvard/ETH Physical AI Systems is building the educational successor to TinyML: an open book and project course where students engineer dual-brain agents on Arduino UNO Q—Linux intelligence proposes; the real-time MCU independently permits or refuses action. We want to co-design a *Physical AI Kit* so the same pedagogy can ship worldwide the way the TinyML Kit did—bring-up, measurement, signature enforcer lab, and an evidence-backed deploy / condition / refuse decision.

    #v(0.2em)
    *Next:* 30-minute session on kit BOM + out-of-box lab path. · *Email:* #link("mailto:vj\@eecs.harvard.edu")[vj\@eecs.harvard.edu]
  ],
)
