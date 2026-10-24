#!/usr/bin/env python3
"""Run Fairlearn's MetricFrame on the chapter's credit service (Lecture 15).

The labels and decisions are the constructed confusion matrices of the
chapter's credit service (Responsible Engineering, Sec. 15.3.2.1):
Group A TP 4,500, FN 500, FP 1,000, TN 4,000; Group B TP 600, FN 400, FP 200,
TN 800. The library is real; the data are the chapter's constructed example.

The slide quotes the printed output exactly. Fairlearn is not in the book's
venv; run with any Python that has it installed, for example
    python3 -m venv flvenv && flvenv/bin/pip install fairlearn
    flvenv/bin/python fairlearn_gate.py
Output recorded on 2026-10-19 with fairlearn 0.15.0, scikit-learn 1.9.1,
pandas 3.0.6 in fairlearn_gate_output.txt.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, false_positive_rate,
                               selection_rate, true_positive_rate)

CELLS = {"A": (4500, 500, 1000, 4000), "B": (600, 400, 200, 800)}


def rows(tp, fn, fp, tn):
    y = [1] * (tp + fn) + [0] * (fp + tn)
    yhat = [1] * tp + [0] * fn + [1] * fp + [0] * tn
    return y, yhat


y, y_hat, group = [], [], []
for g, cells in CELLS.items():
    a, b = rows(*cells)
    y += a
    y_hat += b
    group += [g] * len(a)
y, y_hat, group = np.array(y), np.array(y_hat), pd.Series(group, name="group")

# --- the lines shown on the slide ---
mf = MetricFrame(
    metrics={"approve": selection_rate,
             "tpr": true_positive_rate,
             "fpr": false_positive_rate},
    y_true=y, y_pred=y_hat,
    sensitive_features=group)
print(mf.by_group)
print(mf.difference())
