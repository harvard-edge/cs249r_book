<div align="center">
  <h1>🗺️ MLSysBook Instructor Hub</h1>
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

The Instructor Hub is the instructor-facing companion site for the [Machine Learning Systems textbook](https://mlsysbook.ai). It holds the syllabi, shared materials, and teaching guidance for every course built on the book. Course descriptions and community offerings live in the separate [course catalog](https://mlsysbook.ai/courses/) (`../courses/`).

<table width="100%" style="width:100%">
  <thead>
    <tr>
      <th align="left" width="20%">Area</th>
      <th align="left" width="80%">What you get</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td width="15%" align="center"><b>📅 Syllabi</b></td>
      <td><b>One-page syllabi</b> for every course: introductory, scaling, TinyML, Edge AI, Physical AI, and Agentic AI.</td>
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
├── index.qmd                # Hub page: one kit per course
├── foundations-syllabus.qmd # Introduction course syllabus
├── scale-syllabus.qmd       # Scaling course syllabus
├── tinyml-syllabus.qmd      # TinyML course syllabus
├── edge-ai-syllabus.qmd     # Edge AI reading list
├── agentic-syllabus.qmd     # Agentic AI reading list
├── vol4/                    # Physical AI syllabus (index.qmd) and forwarding stub
└── assets/                  # Styles and images
```

---

## 🎨 Style Conventions

This site uses the shared ecosystem theme (`shared/styles/style-site.scss` and
`shared/styles/dark-mode-site.scss`), the same as the main site and the course
pages. Do not add site-specific styles; change the shared theme instead so every
page keeps one layout.

---

## Contributors

Thanks to these wonderful people who have helped build the Instructor Site!

**Legend:** 🪲 Bug Hunter · ⚡ Code Warrior · 📚 Documentation Hero · 🎨 Design Artist · 🧠 Idea Generator · 🔎 Code Reviewer · 🧪 Test Engineer · 🛠️ Tool Builder

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table width="100%" style="width:100%">
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/farhan523"><img src="https://avatars.githubusercontent.com/u/62025759?v=4?v=4?s=80" width="80px;" alt="Farhan Asghar"/><br /><sub><b>Farhan Asghar</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🧪</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Shashank-Tripathi-07"><img src="https://avatars.githubusercontent.com/u/178375647?v=4?v=4?s=80" width="80px;" alt="Rocky"/><br /><sub><b>Rocky</b></sub></a><br />🪲 🧑‍💻 🎨 ✍️ 🔎</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/profvjreddi"><img src="https://avatars.githubusercontent.com/profvjreddi?v=4?s=80" width="80px;" alt="Vijay Janapa Reddi"/><br /><sub><b>Vijay Janapa Reddi</b></sub></a><br />🧑‍💻 🎨 ✍️</td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/coyaSONG"><img src="https://avatars.githubusercontent.com/u/66289470?v=4?v=4?s=80" width="80px;" alt="coyaSONG"/><br /><sub><b>coyaSONG</b></sub></a><br />🪲 🧑‍💻 ✍️</td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

**Recognize a contributor:** Comment on any issue or PR:
```text
@all-contributors please add @username for code, design, doc in instructors
```
