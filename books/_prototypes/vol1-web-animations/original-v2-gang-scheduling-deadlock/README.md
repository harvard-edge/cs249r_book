# Why must a rigid job receive its whole gang?

Source-art prototype for `vol2:fig-gang-scheduling-deadlock`. Static source and caption stay unchanged.

Original pool: GPU 0–7. Both J1 and J2 require all eight GPUs. The two scheduling policies are alternative runs.

Invariant: No claimed wall time, utilization gain, fairness, or general deadlock freedom.

Run `python3 books/_prototypes/vol1-web-animations/original-v2-gang-scheduling-deadlock/check.py` for actual-chapter fixtures and browser checks. Review screenshots are written only under `/tmp`.

## Frozen review assets

`original-v2-gang-scheduling-deadlock.js`: `bd3b34cc8ef1afa2bbb6a0c45962a172859a52f868ceb8fc4e0eb7ce481135c4`
`original-v2-gang-scheduling-deadlock.css`: `4790cba1c551a7065f57dc89c292d522e357e0bcf49f5e955172131a0271a191`
`entry.json`: `f4c4ff85411b627eb4da54e76b2da397a99fbc486b9edf7f039a7642fc413e9d`
