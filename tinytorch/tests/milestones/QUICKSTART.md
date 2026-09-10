# Milestones Quick Start

## Fast Check

```bash
tito dev export --all
pytest tests/milestones/test_milestones_smoke.py -v
```

Expected: milestone scripts import and model classes construct without running
long training jobs.

## Release Check

```bash
tito dev export --all
pytest tests/milestones/test_milestones_run.py -v
```

Expected: milestone scripts run through the `tito milestone run` command path.
This is slower and is intended for release validation.

## Run One Milestone Test

```bash
pytest tests/milestones/test_milestones_smoke.py -v -k milestone_03
pytest tests/milestones/test_milestones_run.py -v -k milestone_03
```

## Run Student Commands Directly

```bash
tito milestone list
tito milestone info 03
tito milestone run 03
tito milestone run 03 --part 1
tito milestone run 03 --part 2
```

Use `--skip-checks` only in release tests or controlled maintainer runs:

```bash
tito milestone run 03 --skip-checks
```

## Files

```text
tests/milestones/
├── milestone_tracker.py        # Compatibility tracker for older hooks
├── test_milestones_smoke.py    # Fast import/model construction tests
├── test_milestones_run.py      # Full milestone execution tests
├── README.md
├── API.md
├── PROGRESSION.md
└── QUICKSTART.md
```

The removed learning-verification test file is no longer part of this release
surface. Do not add new docs or commands that reference removed milestone tests.
