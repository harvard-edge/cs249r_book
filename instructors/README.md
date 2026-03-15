# The Blueprint — Instructor Site

> **Status: Under Active Development**
>
> This site is being actively built and refined. Content, structure, and styling may change frequently. SVG diagrams are placeholder quality and will be professionally refined before launch.

## What This Is

"The Blueprint" is the instructor-facing companion site for the [ML Systems textbook](https://mlsysbook.ai). It provides everything needed to teach AI Engineering as a university course:

- **Week-by-week syllabi** for two semesters (Foundations + Scale)
- **Assessment rubrics** with sample student work
- **Pedagogy guide** with learning science and facilitation strategies
- **TA training guide** with grading workflows and common student struggles
- **Customization guide** for quarters, seminars, and different emphases
- **FAQ** for adopting instructors

## Building Locally

```bash
cd instructors/
quarto preview
```

Requires [Quarto](https://quarto.org/) 1.4+.

## File Structure

```
instructors/
├── _quarto.yml              # Site configuration
├── index.qmd                # Landing page (hero + overview)
├── getting-started.qmd      # 8-step adoption checklist
├── course-map.qmd           # Integration matrix + SVG diagrams
├── foundations-syllabus.qmd  # Semester 1: 16-week schedule
├── scale-syllabus.qmd       # Semester 2: 16-week schedule
├── pedagogy.qmd             # Learning science + facilitation
├── assessment.qmd           # Rubrics + sample work
├── ta-guide.qmd             # TA training + grading workflows
├── customization.qmd        # Quarter/seminar/emphasis variants
├── faq.qmd                  # Common instructor questions
└── assets/
    ├── images/
    │   ├── logo.png
    │   └── svg/              # Diagrams (under development)
    │       ├── four-pillar-loop.svg
    │       ├── semester-timeline.svg
    │       ├── lab-abc-flow.svg
    │       └── assessment-tiers.svg
    └── styles/
        ├── style.scss        # Light mode (ecosystem-aligned)
        └── dark-mode.scss    # Dark mode overrides
```

## SVG Diagrams

The SVG diagrams in `assets/images/svg/` are functional but need visual polish. They follow the textbook's SVG conventions (see `.claude/rules/svg-style.md`) adapted for the Blueprint's indigo accent color.

## Style Conventions

This site follows the MLSysBook ecosystem design system:

- **Accent color**: Indigo (`#6366f1` light, `#818cf8` dark)
- **Callout geometry**: 5px left border, 0.5rem radius, 8% opacity backgrounds
- **Typography**: Inter (body), Outfit (headings), JetBrains Mono (code)
- **Navbar**: Dark background (`#0a0a0f`), matching book sites
- **Callout colors**: Shared semantic palette (info/success/caution/secondary)

See `assets/styles/style.scss` for full details.
