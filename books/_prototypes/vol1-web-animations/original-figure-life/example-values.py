#!/usr/bin/env python3
"""Recompute the chapter's small vertical-line example for the optional inset."""
import json
from pathlib import Path

# The nearby equivariance worked example specifies this input and filter.
inputs = [[int(column == 2) for column in range(7)] for _ in range(7)]
weights = [[-1, 0, 1] for _ in range(3)]
positions = []
for row in range(len(inputs) - len(weights) + 1):
    for column in range(len(inputs[0]) - len(weights[0]) + 1):
        patch = [line[column:column + len(weights[0])]
                 for line in inputs[row:row + len(weights)]]
        products = [[x * w for x, w in zip(a, b)] for a, b in zip(patch, weights)]
        row_sums = [sum(line) for line in products]
        positions.append(dict(row=row, column=column, patch=patch,
                              products=products, row_sums=row_sums, output=sum(row_sums)))
data = dict(
    label="Values from the vertical-line example below",
    source_anchor="exmp-nn-architectures-equivariance-feature-detection",
    scope="Illustrative inputs added to this drawing. One channel, stride 1, no padding or bias; the filter is not flipped.",
    inputs=inputs, weights=weights, positions=positions,
)
Path(__file__).with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps([entry['output'] for entry in positions]))
