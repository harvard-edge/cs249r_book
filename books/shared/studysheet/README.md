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
time, so a sheet always matches its book. Math uses the book's notation macros
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
leaves less than 24 pt free at the foot of any column, or has an overfull box.
`binder/tests/test_studysheet.py` checks every sheet without LaTeX.
