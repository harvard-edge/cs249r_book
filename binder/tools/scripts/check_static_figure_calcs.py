#!/usr/bin/env python3
"""Recompute the declared toy quantities in the static figure additions.

Hardware/model specifications remain owned by MLSysIM. These checks cover only
illustrative allocation/address examples, schematic cache-position counts,
and a chapter's existing LEGO inputs.
"""
import argparse
import ast
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
STATIC_REVIEW_REVISION = '99cc75bfed2c434e73b30be25523ce85d4c9d2dd'
STATIC_DELIVERY = ROOT / 'books/_prototypes/vol1-web-animations/figure-plan/static-delivery.json'


def source(path):
    return (ROOT / path).read_text()


def require(text, value):
    assert value in text, f'Computed label missing from source: {value!r}'


def static_source_provenance(refresh=False):
    """Preserve historical raw hashes; permit only two HTML enrollment attributes."""
    manifest = json.loads(STATIC_DELIVERY.read_text())
    digest = lambda data: hashlib.sha256(data).hexdigest()
    normalize = lambda data: re.sub(rb' data-explore(?:-key)?="[^"]*"', b'', data)
    assert manifest['count'] == len(manifest['figures']) == 26
    proofs = {}
    for entry in manifest['figures']:
        reviewed = subprocess.check_output(
            ['git', 'show', STATIC_REVIEW_REVISION + ':' + entry['source']], cwd=ROOT)
        current = (ROOT / entry['source']).read_bytes()
        assert digest(reviewed) == entry['source_sha256'], (entry['key'], 'historical raw source drift')
        assert normalize(current) == normalize(reviewed), (entry['key'], 'static source content changed')
        png = (STATIC_DELIVERY.parent / entry['png']).read_bytes()
        assert digest(png) == entry['png_sha256'], (entry['key'], 'reviewed PNG changed')
        proof = dict(current_raw_source_sha256=digest(current),
                     reviewed_normalized_source_sha256=digest(normalize(reviewed)),
                     current_normalized_source_sha256=digest(normalize(current)),
                     normalized_static_source_equal=True, reviewed_png_hash_matches=True)
        if refresh:
            entry['current_static_source_proof'] = proof
        else:
            assert entry['current_static_source_proof'] == proof, (entry['key'], 'stale provenance proof')
        proofs[entry['key']] = proof
    note = ('source_sha256 preserves the raw reviewed QMD at reviewed_source_revision. '
            'Current static equivalence removes only space-prefixed data-explore and '
            'data-explore-key double-quoted HTML enrollment attributes, then compares '
            'all remaining bytes. PNG hashes remain the historical reviewed artifacts. '
            'This provenance check does not claim a new PDF build or visual review.')
    if refresh:
        manifest['reviewed_source_revision'] = STATIC_REVIEW_REVISION
        manifest['source_hash_scope'] = note
        STATIC_DELIVERY.write_text(json.dumps(manifest, indent=2) + '\n')
    else:
        assert manifest['reviewed_source_revision'] == STATIC_REVIEW_REVISION
        assert manifest['source_hash_scope'] == note
    return dict(reviewed_source_revision=STATIC_REVIEW_REVISION,
                checked_figures=len(proofs), historical_raw_hashes_match=True,
                normalized_static_sources_equal=True, reviewed_png_hashes_match=True,
                normalization='Remove only data-explore and data-explore-key HTML enrollment attributes.',
                figures=proofs)


def figure_tikz(chapter, figure_id):
    """Inspect only the bound figure, rather than matching another chapter label."""
    start = chapter.index('::: {#' + figure_id + ' ')
    match = re.search(r'^```\{\.tikz\}\n(.*?)^```', chapter[start:], re.M | re.S)
    assert match, f'No TikZ body for {figure_id}'
    return match.group(1)


def foreach_rows(tikz, variables):
    signature = '/'.join('\\' + name for name in variables)
    matches = re.findall(r'\\foreach\s+' + re.escape(signature) + r'\s+in\s+\{([^}]*)\}', tikz)
    assert len(matches) == 1, (variables, len(matches))
    return [tuple(part.strip() for part in row.split('/')) for row in matches[0].split(',')]


def request_local_kv(svg, chapter):
    """Bind schematic counts to actual snapshot cells and their panel labels."""
    xml = ET.fromstring(svg)
    assert not any('transform' in node.attrib for node in xml.iter()), 'Normalize transformed SVG coordinates first.'
    texts = [node for node in xml.iter() if node.tag.endswith('text')]
    rects = [node for node in xml.iter() if node.tag.endswith('rect')]
    def number(node, key):
        return Decimal(node.get(key, '0'))
    def text(label):
        matches = [node for node in texts if ''.join(node.itertext()) == label]
        assert len(matches) == 1, label
        return matches[0]
    # Three prompt positions are a declared schematic input, not model dimensions.
    prompt_positions, decode_passes = 3, 2
    expected = [prompt_positions + step for step in range(decode_passes + 1)]
    counts, offsets = [], []
    for step, count in enumerate(expected):
        label = text(f'KV snapshot {step}')
        cx, cy = number(label, 'x'), number(label, 'y')
        containers = [r for r in rects if 'stroke' in r.attrib
                      and number(r, 'x') < cx < number(r, 'x') + number(r, 'width')
                      and number(r, 'y') < cy < number(r, 'y') + number(r, 'height')]
        assert len(containers) == 1, (step, 'ambiguous snapshot panel')
        panel = containers[0]
        x, y, w, h = [number(panel, k) for k in ['x', 'y', 'width', 'height']]
        cells = [r for r in rects if r is not panel and number(r, 'width') == number(r, 'height')
                 and x < number(r, 'x') and number(r, 'x') + number(r, 'width') < x + w
                 and y < number(r, 'y') and number(r, 'y') + number(r, 'height') < y + h]
        cells.sort(key=lambda r: number(r, 'x'))
        assert len(cells) == count, (step, len(cells), count)
        assert len({number(r, 'x') for r in cells}) == count, 'Overlapping cache slots'
        assert len({number(r, 'y') for r in cells}) == 1, 'Cache slots must share a row'
        local = [(number(r, 'x') - x, number(r, 'y') - y, number(r, 'width')) for r in cells]
        if offsets:
            assert local[:-1] == offsets[-1], 'Prior-position slots changed during append'
        offsets.append(local)
        counts.append(len(cells))
        stage = text('Prefill' if step == 0 else f'Decode pass {step}')
        assert number(stage, 'x') == cx and number(stage, 'y') < y, 'Snapshot bound to wrong pass'
        if step:
            sampled = text(f'Sampled token {step}')
            assert number(sampled, 'x') == cx and number(sampled, 'y') < number(stage, 'y')
    require(chapter, 'Three cache snapshots grow from three to five cells.')
    for label in ['Next independent intent evaluation', 'New observation', 'New prefill', 'New KV state',
                  'Previous KV does not carry across control evaluations.']:
        text(label)
    return dict(prompt_positions=prompt_positions, decode_passes=decode_passes,
                snapshot_positions=counts, appended_positions=counts[-1] - counts[0],
                reset='New independent context; the SVG does not draw or quantify reset slots.',
                guard='Schematic positions in successive snapshots of one request, not bytes or model dimensions.')


def calculate():
    result = {}
    # Declared toy pool: three allocations, then free B, split it for D, free A.
    allocation = {'A': 4, 'B': 4, 'C': 4}
    reserved = sum(allocation.values())
    live_after_free = allocation['A'] + allocation['C']
    d_size = 2
    free_regions = [allocation['A'], allocation['B'] - d_size]
    live_final = d_size + allocation['C']
    request = 6
    allocator = source('books/vol1/07_frameworks/07_frameworks.qmd')
    require(allocator, f'Reserved: {reserved} units\\qquad Live: {reserved}\\qquad Reusable: {reserved - reserved}')
    require(allocator, f'Reserved: {reserved} units\\qquad Live: {live_after_free}\\qquad Reusable: {reserved - live_after_free}')
    require(allocator, f'Reserved: {reserved} units\\qquad Live: {live_final}\\qquad Reusable: {sum(free_regions)}\\qquad Largest free block: {max(free_regions)}')
    require(allocator, f'Request: {request} contiguous units')
    assert sum(free_regions) >= request > max(free_regions)
    result['allocator'] = dict(reserved=reserved, live_after_release=live_after_free,
        final_live=live_final, reusable=sum(free_regions), largest_free=max(free_regions),
        requested=request, request_fits=any(size >= request for size in free_regions), unit='illustrative pool units')

    # A declared interface with four-byte words and four-word aligned sectors.
    word_bytes, words_per_sector, request_count, sector_count = 4, 4, 4, 4
    sector_bytes = word_bytes * words_per_sector
    adjacent = [i * word_bytes for i in range(request_count)]
    scattered = [i * sector_bytes for i in range(request_count)]
    useful = request_count * word_bytes
    fetched = lambda addresses: len({a // sector_bytes for a in addresses}) * sector_bytes
    irregular = source('books/vol1/11_hw_acceleration/11_hw_acceleration.qmd')
    irregular = figure_tikz(irregular, 'fig-irregular-address-transactions')
    for addresses in [adjacent, scattered]:
        require(irregular, 'Requested byte addresses: ' + ', '.join(map(str, addresses)))
    # Check the displayed fetched-sector claims, not only the output evidence.
    claims = re.findall(r'Useful payload: (\d+) B\\qquad Fetched: (\w+) (\d+) B sectors?', irregular)
    words = {'one': 1, 'two': 2, 'three': 3, 'four': 4}
    assert len(claims) == 2, claims
    for (payload, count_word, displayed_sector_bytes), addresses in zip(claims, [adjacent, scattered]):
        assert count_word in words, f'Unsupported sector-count spelling: {count_word}'
        count = len({a // sector_bytes for a in addresses})
        assert (int(payload), words[count_word], int(displayed_sector_bytes)) == (useful, count, sector_bytes)
        assert words[count_word] * int(displayed_sector_bytes) == fetched(addresses)
    require(irregular, f'Useful payload: {useful} B')
    for i in range(sector_count):
        require(irregular, f'{i * sector_bytes}--{(i + 1) * sector_bytes - 1}')
    result['cold_sector_example'] = dict(word_bytes=word_bytes, sector_bytes=sector_bytes,
        adjacent_addresses=adjacent, scattered_addresses=scattered, useful_bytes=useful,
        adjacent_fetched_bytes=fetched(adjacent), scattered_fetched_bytes=fetched(scattered),
        guard='Cold cache, no reuse, merged requests at the declared illustrative sector interface.')

    # Same eight-block source example: E can use scattered free blocks.
    owners = ['E', 'E', 'C', 'C', 'E', 'E', 'D', 'D']
    pages_e = [i for i, owner in enumerate(owners) if owner == 'E']
    released = [None if owner == 'E' else owner for owner in owners]
    serving = source('books/vol1/13_model_serving/13_model_serving.qmd')
    paged = figure_tikz(serving, 'fig-serving-paged-kv-mapping')
    drawn_blocks = foreach_rows(paged, ['x', 'b', 'owner'])
    assert [int(block) for _, block, _ in drawn_blocks] == list(range(len(owners)))
    assert [owner for _, _, owner in drawn_blocks] == owners
    centers = {int(block): Decimal(x) for x, block, _ in drawn_blocks}
    drawn_mapping = foreach_rows(paged, ['x', 'p', 'b'])
    assert [int(page) for _, page, _ in drawn_mapping] == list(range(len(pages_e)))
    assert [int(block) for _, _, block in drawn_mapping] == pages_e
    # Independently verify the arrow destinations hit the mapped physical blocks.
    targets = foreach_rows(paged, ['p', 'x'])
    assert {int(page): Decimal(x) for page, x in targets} == {
        page: centers[block] for page, block in enumerate(pages_e)}
    drawn_release = foreach_rows(paged, ['x', 'owner'])
    assert [(Decimal(x), owner) for x, owner in drawn_release] == [
        (centers[block], 'Free' if owner is None else owner)
        for block, owner in enumerate(released)]
    # White repainting must free exactly E's blocks, not another owner's block.
    repaint_lists = re.findall(r'\\foreach\s+\\x\s+in\s+\{([^}]*)\}', paged)
    assert len(repaint_lists) == 2
    assert [Decimal(x) for x in repaint_lists[-1].split(',')] == [centers[b] for b in pages_e]
    # The first repaint shows full E blocks; the final E block has a partial tail.
    assert [Decimal(x) for x in repaint_lists[0].split(',')] == [centers[b] for b in pages_e[:-1]]
    result['paged_example'] = dict(blocks=len(owners), e_mapping=pages_e,
        owned_counts=dict(Counter(owners)), released_blocks=sum(v is None for v in released),
        retained_owners=[v for v in released if v is not None])

    # Reuse exact input tuples from the existing LEGO cell, not a second registry.
    chapter = source('books/vol3/14_agent_operations/14_agent_operations.qmd')
    blocks = re.findall(r'^```\{python\}[^\n]*\n(.*?)^```', chapter, re.M | re.S)
    tree = ast.parse(next(b for b in blocks if 'class ObservedCriticalPath:' in b))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ObservedCriticalPath')
    names = {'retrieval', 'planning', 'tests', 'lint', 'synthesis', 'segment'}
    intervals = {}
    for n in cls.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names:
            intervals[n.targets[0].id] = tuple(Decimal(str(v)) for v in ast.literal_eval(n.value))
    assert set(intervals) == names
    durations = {name: b - a for name, (a, b) in intervals.items()}
    gap = intervals['synthesis'][0] - max(intervals['tests'][1], intervals['lint'][1])
    svg = source('books/vol3/14_agent_operations/images/svg/overlapping_trace_intervals.svg')
    for value in [durations['tests'], durations['lint']]:
        require(svg, f'{value:.1f} s')
    require(svg, f'{gap:.1f} s gap')
    accounted = durations['retrieval'] + durations['planning'] + max(durations['tests'], durations['lint']) + gap + durations['synthesis']
    assert accounted == durations['segment']
    result['hypothetical_trace'] = dict(durations_s={k: str(v) for k, v in durations.items()},
        unattributed_gap_s=str(gap), accounted_wall_s=str(accounted),
        source='ObservedCriticalPath LEGO inputs in Volume III Chapter 14; hypothetical, not measured.')

    # Ownership transpose preserves all token/head pairs on every surface.
    ranks = range(4)
    views = [[f'R{r}' for r in ranks for _ in ranks],
             [f'R{h}' for _ in ranks for h in ranks],
             [f'R{r}' for r in ranks for _ in ranks]]
    xml = ET.fromstring(source('books/vol2/06_distributed_training/images/svg/ulysses-ownership-transpose.svg'))
    assert not any('transform' in node.attrib for node in xml.iter()), 'Transformed SVG needs explicit coordinate normalization.'
    labels = [n for n in xml.iter() if n.tag.endswith('text') and re.fullmatch(r'R\d+', n.text or '')]
    # Recover actual cells and assign each visible rank label by its coordinates.
    # Aggregate label counts alone cannot detect swapped ownership.
    cells = []
    for rect in xml.iter():
        if not rect.tag.endswith('rect') or 'x' not in rect.attrib:
            continue
        x, y, w, h = (Decimal(rect.get(k, '0')) for k in ['x', 'y', 'width', 'height'])
        inside = [n for n in labels if x < Decimal(n.get('x')) < x + w
                  and y < Decimal(n.get('y')) < y + h]
        if inside:
            assert len(inside) == 1 and w == h, 'Each token/head cell must contain exactly one rank label.'
            cells.append((x, y, inside[0].text))
    rows = sorted({y for _, y, _ in cells})
    columns = sorted({x for x, _, _ in cells})
    assert len(rows) == len(ranks) and len(columns) == len(views) * len(ranks)
    assert len(cells) == len(labels) == sum(map(len, views))
    actual = {(x, y): owner for x, y, owner in cells}
    assert len(actual) == len(cells), 'Overlapping token/head cells.'
    for panel, view in enumerate(views):
        for row, y in enumerate(rows):
            for column in ranks:
                x = columns[panel * len(ranks) + column]
                assert actual[(x, y)] == view[row * len(ranks) + column], (panel, row, column, actual[(x, y)])
    result['ulysses_toy'] = dict(ranks=len(ranks), tokens=len(ranks), heads=len(ranks),
        token_head_pairs=len(views[0]), per_rank_pairs=dict(Counter(views[0])),
        verified_surfaces=['sequence ownership', 'head ownership', 'restored sequence ownership'])
    result['request_local_kv'] = request_local_kv(
        source('books/vol4/03_brain/images/svg/request_local_prefill_decode.svg'),
        source('books/vol4/03_brain/03_brain.qmd'))
    result['static_source_provenance'] = static_source_provenance()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--refresh-static-provenance', action='store_true',
                        help='Refresh manifest proofs only after all historical/equivalence/PNG guards pass.')
    args = parser.parse_args()
    if args.refresh_static_provenance:
        static_source_provenance(refresh=True)
    data = calculate()
    rendered = json.dumps(data, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
