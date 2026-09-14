# MIT Press Assets

Scripts and generated files for MIT Press submission.

## Figure List Generator

Extracts figure metadata from QMD source files and combines it with the
figure-number and page manifest produced by the PDF build:

```bash
# From the quarto/ directory, before rendering:
python3 scripts/mit_press/generate_figure_list.py --clear

# Post-render runs automatically for the production PDF. To rerun it:
python3 scripts/mit_press/generate_figure_list.py
```

The generator writes `FIGURE_LIST.txt` into the active PDF output directory.
Run it only after a fresh render so the LaTeX manifest, figure numbers, and
page numbers correspond to the current manuscript.

### Output Fields

<table width="100%">
  <thead>
<tr>
<th width="25%"><b>Field</b></th>
<th width="75%">Description</th>
</tr>
</thead>
<tbody>
<tr><td><b>Figure Number</b></td><td>Chapter.Figure format (e.g., 1.1, 2.3)</td></tr>
<tr><td><b>Label</b></td><td>Source reference (e.g., fig-ai-timeline)</td></tr>
<tr><td><b>Caption</b></td><td>Full caption text</td></tr>
<tr><td><b>Alt-Text</b></td><td>Accessibility description</td></tr>
</tbody>
</table>

## Tracked submission logs

- `PERMISSIONS_FIGURES_VOL1.csv` records source and permission status for
  every Volume I figure. It is source-based and can be maintained without a
  render.
- The generated `FIGURE_LIST.txt` is a build artifact and is intentionally
  not tracked in this directory.

Validate that the permissions log covers the current QMD figure inventory:

```bash
python3 scripts/mit_press/check_permissions_log.py
```

Before final submission, require every permission scope to be resolved:

```bash
python3 scripts/mit_press/check_permissions_log.py --require-resolved
```
