# Optional Forge integration

Compare MLSysBook experiments in Forge while keeping the normal local and browser
lessons usable without an account. Python 3.10 or newer is required.

```sh
python -m pip install -e integrations/forge
# Only when publishing online:
python -m pip install -e 'integrations/forge[tracking]'
```

## Capture a result

```python
import subprocess
from mlsysbook_forge import ForgeSession, ResultKind, RunContext

revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
context = RunContext(
    lab_id="my-lab", source_commit=revision,
    result_kind=ResultKind.ANALYTICAL,
    config={"batch_size": 8}, seed=42, units={"latency": "ms"},
)
with ForgeSession(context, mode="offline", session_id="comparison-1") as session:
    session.capture({"baseline": 12, "alternative": 9}, action_id="saved-comparison")
```

`disabled` is the default and creates no files or cloud connections. `offline`
writes sanitized evidence and receipts under `.forge/runs`. `online` also needs
explicit `entity` and `project` arguments and an authenticated W&B SDK. Create a
private project in Forge first; this integration does not change project visibility.
SDK credentials stay in the environment or the SDK credential store.

Use `analytical_simulation` for simulator outputs, `supplied_case` for provided
case studies, `functional_smoke` for install/fixture checks, `measured_experiment`
for actual local measurements, and `hosted_model_execution` for live model calls.
These labels travel with the run and every evidence artifact. Simulations do not
produce model-call traces.

## Explicit actions and recovery

Keep a session ID stable across notebook reevaluation. Use a new action ID for
each intentional experiment; reuse it for rerenders of the same saved evidence.
Reusing an action ID with changed evidence is an error. Changing experiment
configuration requires a new session ID. The SQLite journal claims each action
before contacting the service, preventing concurrent duplicate delivery.

A remote timeout cannot prove whether the service accepted an event. Such an
event remains `unconfirmed` (or `sending` after process interruption) and is never
automatically resent. Inspect the remote run and local receipt before intentionally
publishing again with a new action ID. Offline sessions are local records, not an
automatic upload queue. SDK failures preserve local evidence and emit a warning.

## Existing MLSys·im pipelines

Register `PipelineCallback(session, experiment_id="baseline")` using the existing
`Pipeline.register_callback` API. The adapter observes public stage results and
preserves their units. It does not change equations or solver return values.

## Evidence boundaries

Identity, credentials, and student interpretation fields are omitted recursively.
Avoid putting personal information in experimental settings or arbitrary strings;
field filtering cannot recognize every kind of sensitive text. `log_artifact`
accepts structured JSON only and applies the same filtering. Original course
reports and student writing remain under the course's existing submission workflow.
Source-code capture and console upload are disabled in the W&B backend.

## Tests

```sh
python -m pytest integrations/forge/tests -q
```

The package uses the repository's code license (Apache-2.0); teaching content
retains its original attribution and content license.
