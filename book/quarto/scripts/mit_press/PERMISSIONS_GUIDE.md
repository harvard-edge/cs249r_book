# Permissions Tracking — MIT Press Submission

**Status:** Schema as of 2026-05-20. Author-facing guide for what to track and how.

MIT Press requires print + electronic + world-language permissions for any
third-party material in the book. Volume editors are responsible for ensuring
the compliance of contributing authors. Three separate logs cover the three
categories of third-party material.

## When permission is required

Per MIT Press author guidelines, permission must be obtained for:

| Category | Threshold | Log |
|---|---|---|
| **Illustrations / figures** | Any third-party figure | `PERMISSIONS_FIGURES_*.csv` |
| **Tables** | Any third-party table | `PERMISSIONS_TABLES_*.csv` |
| **Prose excerpts** | More than 300 words | `PERMISSIONS_TEXT_*.csv` |
| **Poetry / song lyrics** | Any length — even one line | `PERMISSIONS_TEXT_*.csv` |

Permission must cover **all three** scopes: print, electronic (HTML/EPUB/PDF),
and world-language (translation rights).

## When permission is NOT required

- **Original work** by the author or coauthors (most TikZ figures, original
  schematic diagrams) — log as `source_type: original`, blank permission fields.
- **Public-domain material** (US government works, pre-1928 works, explicitly
  CC0-released material) — log as `source_type: public_domain`, blank
  permission fields.
- **Fair use** — borderline; MIT Press will advise on a case-by-case basis.
  When in doubt, treat as requiring permission.
- **Brief, properly attributed quotations** under 300 words in scholarly
  context — typically falls under fair use; log as `source_type: fair_use`
  with the citation.

## Source-type categories (canonical values)

Use these exact strings in the `source_type` column for downstream filtering:

| Value | Meaning | Permission needed? |
|---|---|---|
| `original` | Created by the author or named coauthor / contributor | No |
| `adapted` | Author's redrawing of a third-party figure; usually requires permission unless substantially transformed | Usually yes |
| `used_with_permission` | Third-party material reproduced with explicit grant | Yes (logged) |
| `public_domain` | Pre-1928 US, US government work, or explicitly CC0 | No |
| `cc_licensed` | Creative Commons license (record specific license in `license` column) | Depends on license terms |
| `fair_use` | Brief quotation under fair use; cite source, do not need permission | No (but cite) |

## License column (when `source_type` is `cc_licensed`)

| Value | Notes |
|---|---|
| `CC0` | Public domain dedication — no attribution required, no permission needed |
| `CC-BY` | Attribution required; permission generally not needed for non-commercial reproduction |
| `CC-BY-SA` | Attribution + share-alike; MIT Press will need to confirm compatibility |
| `CC-BY-NC` | Non-commercial — **incompatible with commercial publication; do not use** |
| `CC-BY-ND` | No derivatives — usable only if no adaptation; flag for review |
| `all_rights_reserved` | Third-party copyright held; permission required |

## Permission scope columns

Three columns (`permission_print`, `permission_electronic`, `permission_world_lang`)
take one of:

- `yes` — permission obtained for this scope
- `no` — explicitly denied (figure must be removed or substituted)
- `pending` — request sent, awaiting response
- `na` — not applicable (e.g., source is original work or public domain)

## Workflow

1. **Initial pass** — for each figure/table/excerpt, the author identifies the
   source type. If `original` or `public_domain`, the row is complete with no
   permission work needed.
2. **Permission requests** — for `used_with_permission` and `adapted` entries,
   draft and send permission requests using MIT Press's standard letter. The
   acquisitions assistant (currently Susan Hartman's team) provides the
   template.
3. **Tracking** — record outcomes in the log. Pending requests must be
   resolved before final manuscript submission.
4. **Final review** — before submission, run the audit script (TBD) to flag
   any row where the source requires permission but `permission_print`,
   `permission_electronic`, or `permission_world_lang` is still `pending` or
   blank.

## Templates

- `PERMISSIONS_FIGURES_TEMPLATE.csv` — figure log schema with example rows
- `PERMISSIONS_TABLES_TEMPLATE.csv` — table log (same shape, different label
  column)
- `PERMISSIONS_TEXT_TEMPLATE.csv` — prose / lyrics / poetry log

Pre-populate from existing data:

```bash
# After regenerating the figure list:
python3 scripts/mit_press/generate_figure_list.py
# Then merge labels into permissions log (script TBD; manual for now)
```
