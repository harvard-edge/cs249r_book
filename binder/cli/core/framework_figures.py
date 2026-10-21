"""Deterministic, neutral print views of the authored candidate framework.

Geometry is equal sized: concept counts never encode importance. These views
show editorial collections and relevance facets, not asserted SKOS broader edges.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import textwrap

_FONT = 'Helvetica Neue, Helvetica, Arial, sans-serif'
_INK = '#333333'
_MUTED = '#666666'
_LINE = '#bbbbbb'


def _label(item: dict) -> str:
    return item.get('display_label', item['label'])


def _start(width: int, height: int, title: str, desc: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="{_FONT}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(desc)}</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]


def _text(x: int | float, y: int | float, value: str, size: int = 18,
          weight: int = 400, anchor: str = 'start', fill: str = _INK) -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}">{escape(value)}</text>')


def _rect(x: int, y: int, width: int, height: int, fill: str = '#ffffff',
          stroke: str = _LINE, radius: int = 0) -> str:
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1" rx="{radius}"/>')


def _numbers(framework: dict) -> dict[str, str]:
    return {row['id']: f'{i:02d}' for i, row in enumerate(
        (row for area in framework['areas'] for row in area['rows']), 1)}


def _end(parts: list[str]) -> bytes:
    return ('\n'.join([*parts, '</svg>', ''])).encode('utf-8')


def _atlas(framework: dict) -> bytes:
    areas = framework['areas']
    columns, width, gutter, margin = 3, 1200, 40, 20
    card_width = (width - 2 * margin - (columns - 1) * gutter) // columns
    max_rows = max((len(a['rows']) for a in areas), default=0)
    card_height = 100 + max_rows * 40
    bands = (len(areas) + columns - 1) // columns
    height = margin * 2 + bands * card_height + (bands - 1) * gutter
    numbers = _numbers(framework)
    parts = _start(width, height, 'ML systems candidate framework: field atlas',
                   'Equal sized editorial area cards organize candidate knowledge rows. '
                   'The geometry does not encode importance or book coverage. Row numbers '
                   'link the three print views. An open square marks a row whose concept '
                   'inventory is not yet populated; it does not indicate missing book coverage. '
                   'All placements are an editorial draft, not a validated field taxonomy.')
    for i, area in enumerate(areas):
        x = margin + (i % columns) * (card_width + gutter)
        y = margin + (i // columns) * (card_height + gutter)
        parts += [f'<g><title>{escape(area["label"])}</title>',
                  _rect(x, y, card_width, card_height, '#ffffff', radius=3),
                  _rect(x, y, card_width, 80, '#f7f7f7', radius=3)]
        label_lines = textwrap.wrap(_label(area), width=24)
        for line_i, line in enumerate(label_lines):
            parts.append(_text(x + 20, y + 32 + line_i * 26, line, 22, 700))
        for j, row in enumerate(area['rows']):
            baseline = y + 110 + j * 40
            parts.append(f'<g><title>{escape(row["label"])}</title>')
            parts.append(_text(x + 20, baseline, numbers[row['id']], fill=_MUTED))
            lines = textwrap.wrap(_label(row), width=30)
            # Curated labels stay single line; unknown future labels wrap safely.
            for line_i, line in enumerate(lines):
                parts.append(_text(x + 60, baseline + line_i * 18, line))
            if not row.get('concepts'):
                parts.append(_rect(x + card_width - 18, baseline - 11, 8, 8, stroke=_MUTED))
            parts.append('</g>')
        parts.append('</g>')
    return _end(parts)


def _matrix(framework: dict) -> bytes:
    areas = framework['areas']
    activities = framework['dimensions']['activities']
    numbers = _numbers(framework)
    label_width, col_width, margin, row_height, area_height, top = 540, 90, 20, 34, 40, 100
    width = margin * 2 + label_width + len(activities) * col_width
    count = sum(len(a['rows']) for a in areas)
    height = top + len(areas) * area_height + count * row_height + margin
    parts = _start(width, height, 'ML systems candidate framework: activity matrix',
                   'Rows are candidate bodies of engineering knowledge; columns are activity '
                   'facets. A filled square is an authored relevance assertion. Blank cells '
                   'mean no relevance assertion has been authored, not that a relationship '
                   'is absent. These marks do not establish book coverage or prerequisite '
                   'order. An open square beside a row identifies an unpopulated concept inventory.')
    headings = {'define': ('Design',), 'data': ('Prepare',),
                'learn': ('Learn',), 'evaluate': ('Evaluate',),
                'deploy': ('Deploy',), 'operate': ('Operate',),
                'retire': ('Retire',)}
    parts.append(_text(margin, 50, 'Knowledge area / subject', 20, 700))
    for i, activity in enumerate(activities):
        x = margin + label_width + i * col_width + col_width / 2
        parts.append(f'<g><title>{escape(activity["label"])}</title>')
        for j, line in enumerate(headings.get(activity['id'], (activity['label'],))):
            parts.append(_text(x, 40 + j * 24, line, 18, 700, 'middle'))
        parts.append('</g>')
    y = top
    for area in areas:
        parts += [_rect(margin, y, width - margin * 2, area_height, '#e8e8e8', '#e8e8e8'),
                  _text(margin + 12, y + 27, area['label'], 20, 700)]
        y += area_height
        for row in area['rows']:
            parts.append(f'<g><title>{escape(row["label"])}</title>')
            parts.append(f'<path d="M{margin},{y + row_height} H{width - margin}" '
                         'fill="none" stroke="#dddddd" stroke-width="1"/>')
            parts.append(_text(margin + 12, y + 23, numbers[row['id']], fill=_MUTED))
            parts.append(_text(margin + 55, y + 23, _label(row)))
            if not row.get('concepts'):
                parts.append(_rect(margin + label_width - 28, y + 12, 9, 9, stroke=_MUTED))
            for i, activity in enumerate(activities):
                x = margin + label_width + i * col_width
                if activity['id'] in row.get('activities', []):
                    parts.append(_rect(x + 38, y + 11, 14, 14, _INK, _INK))
            parts.append('</g>')
            y += row_height
    return _end(parts)


def _layers(framework: dict) -> bytes:
    areas = framework['areas']
    layers = framework['dimensions']['layers']
    numbers = _numbers(framework)
    width, margin, gutter, columns, tile_width = 1200, 20, 40, 3, 360
    rows = (len(areas) + columns - 1) // columns
    band_height = 70 + rows * 64
    height = margin * 2 + len(layers) * band_height + (len(layers) - 1) * 30
    parts = _start(width, height, 'ML systems candidate framework: systems layer lens',
                   'Five equal bands show authored relevance to systems layers. Each area '
                   'appears in every layer touched by one of its subject rows. References '
                   'within each tile name the relevant numbered subject rows from the atlas '
                   'and matrix. Repeated placements express overlap, not exclusive ownership. '
                   'Layer order is an interpretive systems lens, not causality, data flow, '
                   'prerequisite order, or a hierarchy of the knowledge areas.')
    for li, layer in enumerate(layers):
        y = margin + li * (band_height + 30)
        parts += [_rect(margin, y, width - 2 * margin, band_height, '#f7f7f7', '#cccccc', 3),
                  _text(margin + 20, y + 36, layer['label'], 24, 700)]
        ai = 0
        for area in areas:
            relevant = [row for row in area['rows'] if layer['id'] in row.get('layers', [])]
            if not relevant:
                continue
            x = margin + 20 + (ai % columns) * (tile_width + 20)
            yy = y + 60 + (ai // columns) * 64
            parts += [f'<g><title>{escape(area["label"])}</title>',
                      _rect(x, yy, tile_width, 56, '#ffffff', '#cccccc', 2),
                      _text(x + 12, yy + 23, _label(area), 18, 700),
                      _text(x + 12, yy + 45, ' · '.join(numbers[row['id']] for row in relevant), 18, fill=_MUTED),
                      '</g>']
            ai += 1
    return _end(parts)


def framework_figures(root: Path, framework: dict) -> dict[Path, bytes]:
    """Return three reproducible SVG exports without modifying the filesystem."""
    output = root / 'site' / 'explore'
    return {output / 'framework-atlas.svg': _atlas(framework),
            output / 'framework-matrix.svg': _matrix(framework),
            output / 'framework-layers.svg': _layers(framework)}
