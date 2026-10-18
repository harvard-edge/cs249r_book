#!/usr/bin/env python3
"""Recompute the declared toy quantities in the static figure additions.

Hardware/model specifications remain owned by MLSysIM. These checks cover only
illustrative allocation/address examples and a chapter's existing LEGO inputs.
"""
import argparse
import ast
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]


def source(path):
    return (ROOT / path).read_text()


def require(text, value):
    assert value in text, f'Computed label missing from source: {value!r}'


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
        requested=request, request_fits=False, unit='illustrative pool units')

    # A declared interface with four-byte words and four-word aligned sectors.
    word_bytes, words_per_sector, request_count, sector_count = 4, 4, 4, 4
    sector_bytes = word_bytes * words_per_sector
    adjacent = [i * word_bytes for i in range(request_count)]
    scattered = [i * sector_bytes for i in range(request_count)]
    useful = request_count * word_bytes
    fetched = lambda addresses: len({a // sector_bytes for a in addresses}) * sector_bytes
    irregular = source('books/vol1/11_hw_acceleration/11_hw_acceleration.qmd')
    for addresses in [adjacent, scattered]:
        require(irregular, 'Requested byte addresses: ' + ', '.join(map(str, addresses)))
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
    expected = Counter(v for view in views for v in view)
    xml = ET.fromstring(source('books/vol2/06_distributed_training/images/svg/ulysses-ownership-transpose.svg'))
    actual = Counter(n.text for n in xml.iter() if n.tag.endswith('text') and n.text in expected)
    assert actual == expected, (actual, expected)
    result['ulysses_toy'] = dict(ranks=len(ranks), tokens=len(ranks), heads=len(ranks),
        token_head_pairs=len(views[0]), per_rank_pairs=dict(Counter(views[0])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = calculate()
    rendered = json.dumps(data, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')
