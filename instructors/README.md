<div align="center">
  <h1>🗺️ The Blueprint — Instructor Site</h1>
  <!-- EARLY-RELEASE-CALLOUT:START -->
  <table border="0" cellspacing="0" cellpadding="0" width="92%">
  <tr><td align="center" bgcolor="#f6f8fa">
  <table border="0" cellspacing="0" cellpadding="20" width="100%">
  <tr><td align="center">
  <h3>📌 Early release (2026)</h3>
  <p align="center">This instructor site shipped with the <b>2026</b> MLSysBook refresh and is <b>actively iterated</b>: syllabi, rubrics, layouts, and diagrams (including SVGs) will keep improving as we teach from it.</p>
  <p align="center"><b>Feedback</b> — <a href="https://github.com/harvard-edge/cs249r_book/issues">GitHub issues</a> or pull requests.</p>
  </td></tr>
  </table>
  </td></tr>
  </table>
  <!-- EARLY-RELEASE-CALLOUT:END -->
</div>

---

## 📖 What This Is

"The Blueprint" is the instructor-facing companion site for the [ML Systems textbook](https://mlsysbook.ai). It provides everything needed to teach AI Engineering as a university course.

<table>
  <tbody>
    <tr>
      <td width="15%" align="center"><b>📅 Syllabi</b></td>
      <td><b>Week-by-week syllabi</b> for two semesters (Foundations + Scale).</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>⚖️ Assessment</b></td>
      <td><b>Assessment rubrics</b> with sample student work.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>🧠 Pedagogy</b></td>
      <td><b>Pedagogy guide</b> with learning science and facilitation strategies.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>👨‍🏫 TA Guide</b></td>
      <td><b>TA training guide</b> with grading workflows and common student struggles.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>⚙️ Config</b></td>
      <td><b>Customization guide</b> for quarters, seminars, and different emphases.</td>
    </tr>
    <tr>
      <td width="15%" align="center"><b>❓ FAQ</b></td>
      <td><b>FAQ</b> for adopting instructors.</td>
    </tr>
  </tbody>
</table>

---

## 🛠️ Building Locally

<kbd>cd instructors/</kbd>
<kbd>quarto preview</kbd>

Requires [Quarto](https://quarto.org/) 1.4+.

---

## 📂 File Structure

```text
instructors/
├── _quarto.yml              # Site configuration
├── index.qmd                # Landing page (hero + overview)
├── getting-started.qmd      # 8-step adoption checklist
├── course-map.qmd           # Integration matrix + SVG diagrams
├── foundations-syllabus.qmd # Semester 1: 16-week schedule
├── scale-syllabus.qmd       # Semester 2: 16-week schedule
├── pedagogy.qmd             # Learning science + facilitation
├── assessment.qmd           # Rubrics + sample work
├── ta-guide.qmd             # TA training + grading workflows
├── customization.qmd        # Quarter/seminar/emphasis variants
├── faq.qmd                  # Common instructor questions
└── assets/                  # Styles and SVGs
```

---

## 🎨 Style Conventions

This site follows the MLSysBook ecosystem design system:

*   **Accent color**: Indigo (`#6366f1` light, `#818cf8` dark)
*   **Callout geometry**: 5px left border, 0.5rem radius, 8% opacity backgrounds
*   **Typography**: Inter (body), Outfit (headings), JetBrains Mono (code)
*   **Navbar**: Dark background (`#0a0a0f`), matching book sites

See `assets/styles/style.scss` for full details.

---

## Contributors

Thanks to these wonderful people who have helped build the Instructor Site!

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
@all-contributors please add @username for code, design, doc in instructors
```
