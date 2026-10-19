"""A repair trajectory as a Temporal workflow, for Lecture 11 (Durable Execution).

The workflow is the harness: it asks a model for the next step (an activity
that calls a local model through Ollama), then dispatches the integration
test of candidate r18 (an activity whose side effect is a job launched on a
"remote executor", here a ledger file shared by every worker).

Two test activities differ only in how they reach the executor:
  run_tests_naive     launches a job on every attempt;
  run_tests_keyed     launches with operation_id=test_r18, and the executor
                      returns the existing job if that id was already launched.

Run by crash_run.py, which kills the worker in the middle of the test.
"""

import json
import os
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

LEDGER = Path(os.environ.get("L11_LEDGER", "executor_ledger.jsonl"))
MARK = Path(os.environ.get("L11_MARK", "test_started.flag"))
MODEL = os.environ.get("L11_MODEL", "qwen2.5:7b")

PROMPT = (
    "You are a repair agent. Candidate patch r18 fixes a race in a connection "
    "pool. The workspace snapshot is ws_r18. Reply with exactly one line: the "
    "next tool call, as tool_name(arguments). Tools: run_tests(snapshot), "
    "read_file(path), edit_file(path), finish()."
)


def _ledger_append(entry):
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _ledger_entries():
    if not LEDGER.exists():
        return []
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]


def _launches():
    return sum(1 for e in _ledger_entries() if e.get("event") == "job_launched")


@activity.defn
async def call_model(prompt: str) -> dict:
    """One sampled model call (temperature 0.8), recorded by the engine."""
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.8, "num_predict": 40}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    _ledger_append({"t": time.time(), "event": "model_call",
                    "worker_pid": os.getpid(),
                    "attempt": activity.info().attempt,
                    "reply": out["response"].strip().splitlines()[0][:120]})
    return {"reply": out["response"].strip(), "eval_count": out.get("eval_count")}


def _run_job(op_id, keyed):
    info = activity.info()
    if keyed:
        prior = [e for e in _ledger_entries()
                 if e.get("event") == "job_launched" and e.get("operation_id") == op_id]
        if prior:
            _ledger_append({"t": time.time(), "event": "job_reused",
                            "operation_id": op_id, "job": prior[0]["job"],
                            "worker_pid": os.getpid(), "attempt": info.attempt})
            job = prior[0]["job"]
        else:
            job = f"job_tests_{_launches() + 1}"
            _ledger_append({"t": time.time(), "event": "job_launched",
                            "operation_id": op_id, "job": job,
                            "worker_pid": os.getpid(), "attempt": info.attempt})
    else:
        job = f"job_tests_{_launches() + 1}"
        _ledger_append({"t": time.time(), "event": "job_launched",
                        "operation_id": None, "job": job,
                        "worker_pid": os.getpid(), "attempt": info.attempt})
    MARK.write_text(str(os.getpid()))
    # The test job runs for about 6 s; heartbeats tell the server the worker lives.
    for _ in range(12):
        time.sleep(0.5)
        activity.heartbeat()
    return {"job": job, "result": "passed", "tests": 214}


@activity.defn
def run_tests_naive(snapshot: str) -> dict:
    return _run_job(None, keyed=False)


@activity.defn
def run_tests_keyed(snapshot: str) -> dict:
    return _run_job("test_r18", keyed=True)


OPTS = dict(start_to_close_timeout=timedelta(seconds=60),
            heartbeat_timeout=timedelta(seconds=2),
            retry_policy=RetryPolicy(initial_interval=timedelta(seconds=1),
                                     maximum_attempts=5))


@workflow.defn
class RepairNaive:
    @workflow.run
    async def run(self, snapshot: str) -> dict:
        proposal = await workflow.execute_activity(call_model, PROMPT, **OPTS)
        test = await workflow.execute_activity(run_tests_naive, snapshot, **OPTS)
        return {"proposal": proposal["reply"], "test": test}


@workflow.defn
class RepairKeyed:
    @workflow.run
    async def run(self, snapshot: str) -> dict:
        proposal = await workflow.execute_activity(call_model, PROMPT, **OPTS)
        test = await workflow.execute_activity(run_tests_keyed, snapshot, **OPTS)
        return {"proposal": proposal["reply"], "test": test}


@workflow.defn(name="RepairKeyed")
class RepairKeyedReordered:
    """A revised harness that runs the test before asking the model."""
    @workflow.run
    async def run(self, snapshot: str) -> dict:
        test = await workflow.execute_activity(run_tests_keyed, snapshot, **OPTS)
        proposal = await workflow.execute_activity(call_model, PROMPT, **OPTS)
        return {"proposal": proposal["reply"], "test": test}
