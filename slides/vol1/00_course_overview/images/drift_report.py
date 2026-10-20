#!/usr/bin/env python3
"""Real drift report for Lecture 0 (The Stack): evidently-drift-report.png.

Data: UCI Bike Sharing Dataset, hour.csv (Fanaee-T and Gama 2013, CC BY 4.0),
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Reference: January 2011. Current: June 2011. A model trained on winter
inputs is served in summer; Evidently tests each input column for drift.

Run (Evidently 0.7.23 in its own venv; not part of the slides venv):
    python -m venv evenv && evenv/bin/pip install evidently==0.7.23
    evenv/bin/python drift_report.py hour.csv drift_report.html
Screenshot: headless Chrome, --window-size=1000,1250
--force-device-scale-factor=2, cropped to the per-column table
(Column, Type, Reference and Current Distribution).
"""
import sys

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv(sys.argv[1], parse_dates=["dteday"])
cols = ["temp", "atemp", "hum", "windspeed", "hr", "weathersit"]
ref = df[(df.dteday >= "2011-01-01") & (df.dteday < "2011-02-01")][cols]
cur = df[(df.dteday >= "2011-06-01") & (df.dteday < "2011-07-01")][cols]
print(f"reference rows {len(ref)}, current rows {len(cur)}")
snap = Report([DataDriftPreset()]).run(cur, ref)
snap.save_html(sys.argv[2])
for m in snap.dict()["metrics"]:
    print(m.get("metric_name"), m.get("value"))
