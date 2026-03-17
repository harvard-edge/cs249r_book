# MIT Press Assets

Scripts and generated files for MIT Press submission.

## Figure List Generator

Extracts figure metadata from QMD source files:

```bash
# From the quarto/ directory:
python3 scripts/mit_press/figure_list_for_press.py --vol 1 -o scripts/mit_press/FIGURE_LIST_VOL1.txt
python3 scripts/mit_press/figure_list_for_press.py --vol 1 --format csv -o scripts/mit_press/FIGURE_LIST_VOL1.csv
```

### Output Formats

- **Text** (default): Human-readable, organized by chapter
- **CSV**: Spreadsheet format with columns: Figure Number, Label, Caption, Alt-Text
- **Markdown**: Formatted for documentation

### Output Fields

<table>
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

## Generated Files

- `FIGURE_LIST_VOL1.txt` - Text format, 152 figures
- `FIGURE_LIST_VOL1.csv` - CSV format for spreadsheets
