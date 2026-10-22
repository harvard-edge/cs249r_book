# MLSysBook Course Websites

This directory holds student-facing course websites built around the MLSysBook
curriculum. Course names identify the subject; volume numbers identify the
supporting textbooks.

## Course paths

| Directory | Course title | Supporting textbook |
|:---|:---|:---|
| `intro/` | Introduction to Machine Learning Systems | Volume I |
| `scaling/` | Scaling Machine Learning Systems | Volume II |
| `agentic/` | Agentic Machine Learning Systems | Volume III |
| `physical-ai/` | Physical AI | Volume IV |
| `tinyml/` | Tiny Machine Learning | Historical Harvard CS249r offerings and resources |
| `edge-ai/` | Edge AI learning path | Volume I and author-maintained hardware materials |

The planned public URL pattern is `https://mlsysbook.ai/courses/<course>/`.
A standalone Quarto project renders the course pages locally. The unified site
build stages these pages under `/courses/` for the existing preview and production
workflows.

## Course and offering structure

A course's `index.qmd` provides a brief introduction and links to current
and past offerings. Dated offerings use `<institution>-<term>-<year>` directories:

```text
courses/
├── README.md
├── intro/
├── scaling/
├── agentic/
├── tinyml/
├── edge-ai/
└── physical-ai/
    └── eth-spring-2027/
```

An offering's `index.qmd` is the student homepage. It links to its syllabus,
schedule, staff information, deadlines, and course policies. Use Quarto source
files to generate HTML. Keep dated offering URLs stable when a new term begins.

## Content ownership

- The [course directory](index.qmd) is the single discovery page for hosted
  offerings, archives, and courses at other institutions. The former
  `community/courses.html` address forwards here, preserving filters and anchors.
  Adopter data still comes from `site/community/adopters.yml` and its refresh cache.
- The [Instructor Hub](../instructors/index.qmd)
  holds reusable syllabi, pedagogy, assessment guidance, and adoption resources.
- The [labs directory](../labs/README.md) holds reusable experiments. Course
  offerings link to those instructions and specify their local deadlines.
- This directory holds the student-facing details of the offerings maintained
  in this repository. Staff preparation notes stay with their existing materials.

The first planned offering is [Physical AI at ETH in spring 2027](physical-ai/eth-spring-2027/index.qmd).
Its [syllabus](physical-ai/eth-spring-2027/syllabus.qmd) is the canonical source for
the schedule and policies. The former syllabus location links to this page;
reusable lab instructions and staff qualification notes remain under `labs/vol4/`.

The [TinyML archive](tinyml/index.qmd) links historical course websites and verified
public resources in their original locations. Course assignments, student projects,
and related HarvardX materials are identified separately. It does not mirror the
external repositories or replace the original semester websites.

## Local build and preview

From this directory, run `quarto render` to build `_build/`, or `quarto preview`
to inspect the course website. The explicit render list publishes only student
pages; this README and future staff notes are excluded. Cross-project book and
lab links use their public URLs. Those destinations require their own deployments.

The unified site post-render hook (`site/scripts/build_courses.py`) builds this
project and copies its output into `site/_build/courses/`. The site preview and
production workflows deploy that artifact alongside the community pages.

`courses/scripts/build_directory.py` reuses the existing adopter payload and map
builder, stages the canonical community stylesheet in ignored `_generated/`, and
embeds a usable directory at build time. The browser then refreshes from the
site-root `adopters.json`, preserving the existing scheduled-update workflow.

The ETH offering uses the official ETH Blue (`#215CAF`) for course-content
accents. Its styles are scoped to the offering; all course pages import the same
shared navbar configuration as the volumes.

## Shared data refresh

The scheduled `site-refresh-stats.yml` workflow runs one entry point,
`site/scripts/refresh_site_data.py`, every six hours. It delegates collection to
the existing metrics and adopter collectors, then exports `stats.json` and
`adopters.json` at the site root. Course/institution data is assembled by
`site/scripts/build_adopters.py` from curated `site/community/adopters.yml` and
the generated `site/config/adopters-cache.json`; do not maintain another list
or enter totals in page copy.

The landing ticker, community page, and course directory all fetch the same
site-root `adopters.json`. The Schools view derives its institution and country
counts from that payload after deduplication, so its totals and rows change
together on page load. Search filters do not change the overall totals. If the
refresh cannot be fetched, the directory uses its embedded build-time payload.

School icons are cached in `site/config/adopters-icons.json` by the same refresh
entry point. Curated logos take precedence; other schools use their site icons.
Icons are embedded in the shared adopter payload so newly discovered schools
can display them without deploying static assets. Collection uses indexed
favicons, then the school homepage; unsuccessful requests retain the initials
fallback and are retried on the next refresh.

The BUPT seal was extracted from its [official international admissions
brochure](https://sem.bupt.edu.cn/__local/D/D5/17/3E86F4C3F4F53F21A638CC46336_8A912C6B_A8DA4B.pdf).
Manually resolved icons use Tongji's blue footer logo, Ain Shams' current
`asu.edu.eg` domain, and institutional icons from Helwan, HUST's SoICT, and
North South's admissions site. Existing cached icons are retained on refresh.

Draft reading links are resolved at render time by
`shared/scripts/resolve-draft-links.py`, called by both the course and instructor
projects: Volumes III and IV and the Volume IV labs
currently use the working development site because their production routes are
not yet published. Course navigation stays relative to this Quarto project.
Remove the draft URL overrides when those production routes become available.

## Course families and navigation

The directory groups hosted material by course family. Each card links the
family overview and its offerings or resources. All course pages share one
sidebar (`courses` in `_quarto.yml`) that lists every course, so a reader can move
between them; the page's table of contents covers its own sections. Offering pages
can hide the sidebar when they provide their own local navigation.

Every course overview uses the same sections, in this order, and omits any that
do not apply: Overview (`#overview`), Syllabus and schedule (`#syllabus`),
Materials (`#materials`), For instructors (`#for-instructors`), and Offerings
(`#offerings`). The subtitle states the course's status and volume, matching the
labels in the directory and the Instructor Hub. A new course family receives a
card, a sidebar entry, and an overview page in this form.

The Edge AI page is a learning path through existing textbook and hardware
materials, not a dated course offering. Keep its external materials attributed
to their maintainers. Add an institutional offering only when its syllabus and
student resources are available.

The Physical AI family links to `instructors/vol4/index.qmd` for reusable teaching
guidance. Course rendering resolves this link relative to the combined site.
Build and stage the Instructor Hub alongside `/courses/` for a complete local
preview, and deploy the updated instructor artifact with the course artifact.

## Instructor entry point

The main site keeps Instructor Hub at `/instructors/`. Its landing page starts
with a short getting-started note, then one kit per course linking its page here,
its syllabus or guide, slides, and hands-on materials, then the shared guides. The
introductory and scaling courses can run as a two-semester sequence; the hub's
course map covers that case. `/courses/intro/` and `/courses/scaling/` provide student-facing
overviews of the existing curriculum; the instructor syllabi remain canonical.
Course packages can combine reusable materials with dated institutional examples.
