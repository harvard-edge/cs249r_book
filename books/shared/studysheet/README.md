# Chapter study sheets

One double-sided sheet per chapter: the front is what to learn, the back is practice, with the
answers printed upside down at the foot of the back.

## Where things live

| What | Where |
|---|---|
| Shared template (layout, cards, tiles, practice boxes, drawing styles) | `books/shared/studysheet/studysheet.cls` |
| Build and layout check | `books/shared/studysheet/build.py` |
| A chapter's sheet (plain LaTeX) | `books/volN/NN_chapter/<stem>_studysheet.tex` |
| Its numbers, computed from mlsysim | `books/volN/NN_chapter/<stem>_studysheet_numbers.py` |
| Generated numbers (not committed) | `<stem>_studysheet_numbers.tex` |
| PDFs | `books/_build/studysheets/volN/` |

There is one template for all four volumes. A sheet names its volume with `\sheetvolume{N}`. The
accent color is the volume's `accentcolor` from `books/volN/tex/theme-colors.tex`, read at build
time, so a sheet always matches its book. Each sheet's header carries a QR code and a link to its chapter online
(`build.py` passes the chapter URL, read from the book's own HTML config, as `\SheetURL`); the
footer points to the free book and the GitHub repository. Every printed or forwarded copy leads
back to the book. Math uses the book's notation macros
(`books/shared/_includes/_macros.tex`).

## Numbers

Every number on a sheet comes from `\val{key}`. The sheet's `_numbers.py` computes each value from
mlsysim, as the chapter's computation cells do, and writes `\setnum{key}{value}` lines. An unset key
stops the build. Inputs a sheet chooses for its own questions are marked assumed on the sheet and
`ASSUMED` in the numbers file.

## Build

```bash
python3 books/shared/studysheet/build.py            # every sheet
python3 books/shared/studysheet/build.py vol1       # one volume
python3 books/shared/studysheet/build.py books/vol1/01_introduction/introduction_studysheet.tex
```

The build regenerates the numbers, runs lualatex, and fails a sheet that is not exactly two pages,
leaves less than 24 pt free at the foot of any column, runs a key idea past three lines, or has an
overfull box.
`binder/tests/test_studysheet.py` checks every sheet without LaTeX.

## Publishing

Study sheets publish like the lecture slides: as PDFs on GitHub releases, linked from the book and
the slides site. Three workflows in `.github/workflows/`:

| Workflow | When | What |
|---|---|---|
| `studysheets-validate-dev.yml` | push or PR touching the sheets, their numbers, or mlsysim | builds every sheet in the book's TeX container, runs the tests, stages the release files |
| `studysheets-preview-dev.yml` | push to `dev` | validate, then all four volumes to the `studysheets-dev` prerelease |
| `studysheets-publish-live.yml` | manual (type `PUBLISH`), or `publish-all-live.yml` | live volumes to the `studysheets-latest` release |

A release holds each chapter's sheet as `volN_<stem>_studysheet.pdf` (releases are flat) and each
volume's sheets combined in book order as `MLSysBook-StudySheets-VolN.pdf`.

**Live volumes** are `LIVE_VOLUMES` in `build.py`, kept equal to the slides' list (a test checks).
A released volume's links point at `studysheets-latest`; an unreleased volume's point at
`studysheets-dev`, which only the dev preview shows. To release a volume, add it there and in
`slides-publish-live.yml`, then publish the sheets before the book.

**Links.** Under each chapter title the online book shows "For this chapter: Lecture slides ·
Study sheet (PDF)". `build.py --manifest` writes the chapter-to-URL map at the book's HTML
pre-render (`books/config/_quarto-html-volN.yml`), and `books/shared/filters/inject_slides.lua`
adds the line. The slides site's chapter rows link the same URL through `slides/tools/lectures.py`.

```bash
python3 books/shared/studysheet/build.py --stage DIR vol1 vol2   # flat + combined PDFs
python3 books/shared/studysheet/build.py --manifest PATH         # chapter -> URL map, no LaTeX
python3 books/shared/studysheet/build.py --live-volumes          # "vol1 vol2"
```
