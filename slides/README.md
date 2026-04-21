<!-- DEV-BANNER-START -->
<div align="center">
<table border="0" cellspacing="0" cellpadding="0" width="92%">
<tr><td align="center" bgcolor="#f6f8fa">

<table border="0" cellspacing="0" cellpadding="20" width="100%">
<tr><td align="center">

<h3>📌 Early release (2026)</h3>
<p align="center">These Beamer decks ship with the <b>2026</b> MLSysBook refresh. Slides, figures, and speaker notes are <b>actively iterated</b> as the textbook evolves.</p>
<p align="center"><b>Feedback</b> — <a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub issues</a> (teaching notes, errata, and requests).</p>
<p align="center">
<a href="https://github.com/harvard-edge/cs249r_book/tree/dev"><img src="https://img.shields.io/badge/branch-dev-orange?logo=git&logoColor=white" alt="dev branch"></a>
&nbsp;
<a href="https://mlsysbook.ai"><img src="https://img.shields.io/badge/live_site-mlsysbook.ai-blue?logo=safari&logoColor=white" alt="live site"></a>
</p>

</td></tr>
</table>

</td></tr>
</table>
</div>
<!-- DEV-BANNER-END -->

# ML Systems Lecture Slides

Beamer slide decks for the ML Systems textbook. One deck per chapter, ready to drop into your course.

## Quick Start

```bash
cd slides/vol1/01_introduction
xelatex 01_introduction.tex   # Compile (run twice for navigation)
```

## Coverage

<table>
  <thead>
    <tr>
      <th width="25%">Volume</th>
      <th width="15%">Chapters</th>
      <th width="60%">Topics</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Volume I</b></td>
      <td align="center">17 decks</td>
      <td>Course overview, Introduction through Responsible Engineering, Conclusion</td>
    </tr>
    <tr>
      <td><b>Volume II</b></td>
      <td align="center">9 decks</td>
      <td>Course overview, Compute Infrastructure through Inference</td>
    </tr>
  </tbody>
</table>

## Structure

Each chapter is **self-contained**. All images live in `<chapter>/images/`. The only shared assets are the Beamer theme and logos.

```
slides/
├── assets/
│   ├── beamerthememlsys.sty    # Beamer theme (Crimson)
│   └── img/                    # Shared logos
├── vol1/                       # Volume I decks (17 chapters)
│   ├── 01_introduction/
│   │   ├── 01_introduction.tex
│   │   └── images/
│   └── ...
└── vol2/                       # Volume II decks (9 chapters)
    ├── 01_introduction/
    └── ...
```

## Theme

The default theme is **Crimson**: Harvard crimson accents, white background, branded footer. All decks share a common API (`\mlsystitle`, `\mlsysfocus`, `\mlsyscard`, etc.).

Adopting instructors who want a different look can create their own variant by copying `beamerthememlsys.sty` and adjusting the color definitions. The API stays the same, so no `.tex` files need to change.

## Contributing

Agent instructions and production guidelines live in `.claude/docs/slides/`.

## Contributors

Thanks to these wonderful people who have helped build the slide decks!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```
@all-contributors please add @username for code, design, doc in slides
```

## License

These slides accompany the ML Systems textbook and follow the same license terms.
