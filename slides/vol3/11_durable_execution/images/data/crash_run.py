"""Lecture 11 crash run: kill a Temporal worker in the middle of the r18 test.

For each workflow (RepairNaive, RepairKeyed):
  1. start a worker process and the workflow;
  2. wait until the test activity has launched its job on the executor;
  3. SIGKILL the worker (no cleanup, no goodbye: a host crash);
  4. start a replacement worker process;
  5. wait for the workflow to finish and save its event history and the
     executor ledger.
Then replay the keyed history against a reordered workflow (a revised
harness) and record the error the replayer raises.

Usage: python crash_run.py OUTDIR
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from google.protobuf.json_format import MessageToDict
from temporalio.client import Client, WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repair_wf as wf  # noqa: E402


def spawn(target, env):
    return subprocess.Popen([sys.executable, str(HERE / "worker.py"), target],
                            cwd=HERE, env=env)


async def one_run(client, target, name, out):
    ledger = out / f"ledger_{name}.jsonl"
    mark = out / f"mark_{name}.flag"
    for p in (ledger, mark):
        if p.exists():
            p.unlink()
    env = dict(os.environ, L11_LEDGER=str(ledger), L11_MARK=str(mark))
    w1 = spawn(target, env)
    t0 = time.time()
    handle = await client.start_workflow(name, "ws_r18", id=f"repair-r18-{name}",
                                         task_queue="repair")
    while not mark.exists():
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.0)            # the job is running remotely
    t_kill = time.time()
    os.kill(w1.pid, signal.SIGKILL)
    w1.wait()
    await asyncio.sleep(0.5)
    w2 = spawn(target, env)
    t_restart = time.time()
    result = await handle.result()
    t_done = time.time()
    hist = await handle.fetch_history()
    w2.terminate()
    w2.wait()
    events = []
    for e in hist.events:
        d = MessageToDict(e)
        events.append({"id": e.event_id, "type": d.get("eventType"),
                       "time": d.get("eventTime"),
                       "attrs": {k: v for k, v in d.items()
                                 if k.endswith("Attributes")}})
    (out / f"history_{name}.json").write_text(hist.to_json())
    summary = {"workflow": name, "worker1_pid": w1.pid, "worker2_pid": w2.pid,
               "kill_after_s": round(t_kill - t0, 3),
               "restart_after_s": round(t_restart - t0, 3),
               "done_after_s": round(t_done - t0, 3), "result": result,
               "events": [{"id": x["id"], "type": x["type"]} for x in events],
               "ledger": [json.loads(x) for x in ledger.read_text().splitlines()]}
    (out / f"summary_{name}.json").write_text(json.dumps(summary, indent=1))
    return summary


async def main(out):
    out.mkdir(parents=True, exist_ok=True)
    env = await WorkflowEnvironment.start_local(port=7299,
                                                download_dest_dir=str(out / "devserver"))
    target = "127.0.0.1:7299"
    client = await Client.connect(target)
    try:
        for name in ("RepairNaive", "RepairKeyed"):
            s = await one_run(client, target, name, out)
            print(name, "events:", len(s["events"]), "ledger:",
                  [e["event"] for e in s["ledger"]])
        # Harness regression replay: the recorded history, a revised harness.
        hist = WorkflowHistory.from_json("repair-r18-RepairKeyed",
                                         (out / "history_RepairKeyed.json").read_text())
        rep = {}
        for label, cls in (("original", wf.RepairKeyed),
                           ("reordered", wf.RepairKeyedReordered)):
            r = await Replayer(workflows=[cls]).replay_workflow(hist, raise_on_replay_failure=False)
            rep[label] = {"ok": r.replay_failure is None,
                          "error": None if r.replay_failure is None else
                          f"{type(r.replay_failure).__name__}: {r.replay_failure}"}
        (out / "replay.json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
    finally:
        await env.shutdown()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
