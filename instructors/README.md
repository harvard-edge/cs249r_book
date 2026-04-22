<div align="center">
  <h1>🗺️ The Blueprint — Instructor Site</h1>
</div>

<!-- EARLY-RELEASE-CALLOUT:START -->
> [!NOTE]
> **📌 Early release (2026)**
>
> This instructor site shipped with the **2026** MLSysBook refresh and is **actively iterated**: syllabi, rubrics, layouts, and diagrams (including SVGs) will keep improving as we teach from it.
>
> **Feedback** — [GitHub issues](https://github.com/harvard-edge/cs249r_book/issues) or pull requests.
<!-- EARLY-RELEASE-CALLOUT:END -->

---

## 📖 What This Is

"The Blueprint" is the instructor-facing companion site for the [ML Systems textbook](https://mlsysbook.ai). It provides everything needed to teach AI Engineering as a university course.

<table width="100%">
  <thead>
    <tr>
      <th align="left" width="20%">Area</th>
      <th align="left" width="80%">What you get</th>
    </tr>
  </thead>
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
<table width="100%">
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
