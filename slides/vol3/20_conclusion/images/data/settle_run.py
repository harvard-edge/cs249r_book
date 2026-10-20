"""Run incident_42's lost-reply dilemma for real, on a localhost HTTP endpoint.

Conclusion §20.2 asks what a runtime does when the configuration endpoint
drops the connection before acknowledging a change. This script runs four
runtimes against a real HTTP server (Python http.server) that injects the two
faults of the lost-reply estimate (§20.6): the request is lost before the endpoint applies
it (probability 0.001), or the endpoint applies it and the reply is lost
(probability 0.004). Each runtime handles 25,000 incidents, one mutating
change each, the daily volume of that estimate.

  retry   : on a lost reply, send the change again (a fresh request; the
            endpoint keeps no operation identity)
  assume  : on a lost reply, record the change as applied and close
  halt    : on a lost reply, stop and leave the outcome unrecorded
  settle  : the endpoint deduplicates by operation identity and offers a
            read-only status lookup; on a lost reply, ask for the status of
            the same operation id, and resend the same id only if the
            endpoint has no record of it

The endpoint's own log is the ground truth: how many times each incident's
change was applied. Outcomes per runtime:
  duplicates      : incidents whose change was applied twice or more
  false closures  : incidents recorded as applied whose change never applied
  unowned unknowns: incidents left with no recorded outcome

Run: python3 settle_run.py   (writes settle_run.json beside this file)
"""

import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "mlsysim"))
from mlsysim import Scenarios  # noqa: E402

# The fault rates and daily volume are MLSysIM's (Scenarios.BlindRetryAgainstStatusLookup).
LOST = Scenarios.BlindRetryAgainstStatusLookup
P_REQ = LOST.request_loss   # request lost before the endpoint applies it
P_REP = LOST.reply_loss     # endpoint applies, then the reply is lost
N = LOST.calls              # incidents per runtime: one day's mutating calls
SEED = 20261019

STATE = {"applied": {}, "ops": {}}
LOCK = threading.Lock()
RNG = random.Random(SEED)


class Endpoint(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def drop(self):
        # Close the socket without any reply: the client sees a dropped
        # connection, the "connection dropped before acknowledging" of §20.2.
        self.close_connection = True
        try:
            self.connection.shutdown(2)
        except OSError:
            pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query)
        inc = q["incident"][0]
        op = q.get("op", [None])[0]
        with LOCK:
            r = RNG.random()
        if r < P_REQ:
            return self.drop()              # lost before it is applied
        with LOCK:
            if op is not None and op in STATE["ops"]:
                pass                        # a repeat of a known operation
            else:
                STATE["applied"][inc] = STATE["applied"].get(inc, 0) + 1
                if op is not None:
                    STATE["ops"][op] = "applied"
            r2 = RNG.random()
        if r2 < P_REP:
            return self.drop()              # applied, reply lost
        self.reply(200, {"status": "applied"})

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        op = q["op"][0]
        with LOCK:
            known = STATE["ops"].get(op)
        self.reply(200, {"status": known or "no record"})


def call(url, method="POST"):
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.load(r)
    except (urllib.error.URLError, ConnectionError, OSError):
        return None                         # the runtime sees only a lost reply


def run(policy, base):
    recorded = {}                           # what the runtime believes
    lost = 0
    for i in range(N):
        inc = f"{policy}-{i}"
        if policy == "settle":
            op = f"op-{inc}"
            res = call(f"{base}/apply?incident={inc}&op={op}")
            while res is None:
                lost += 1
                st = call(f"{base}/status?op={op}", method="GET")
                if st and st["status"] == "applied":
                    res = st
                elif st and st["status"] == "no record":
                    res = call(f"{base}/apply?incident={inc}&op={op}")
            recorded[inc] = "applied"
            continue
        res = call(f"{base}/apply?incident={inc}")
        if res is not None:
            recorded[inc] = "applied"
            continue
        lost += 1
        if policy == "retry":
            while res is None:
                res = call(f"{base}/apply?incident={inc}")
            recorded[inc] = "applied"
        elif policy == "assume":
            recorded[inc] = "applied"
        elif policy == "halt":
            recorded[inc] = "unknown"
    applied = STATE["applied"]
    keys = [f"{policy}-{i}" for i in range(N)]
    return {
        "incidents": N,
        "lost_replies_seen": lost,
        "duplicates": sum(1 for k in keys if applied.get(k, 0) >= 2),
        "false_closures": sum(1 for k in keys if recorded[k] == "applied" and applied.get(k, 0) == 0),
        "unowned_unknowns": sum(1 for k in keys if recorded[k] == "unknown"),
        "applied_but_unknown": sum(1 for k in keys if recorded[k] == "unknown" and applied.get(k, 0) > 0),
    }


def main():
    srv = HTTPServer(("127.0.0.1", 0), Endpoint)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    out = {"p_request_lost": P_REQ, "p_reply_lost": P_REP, "seed": SEED,
           "date": time.strftime("%Y-%m-%d"), "runtimes": {}}
    for policy in ("retry", "assume", "halt", "settle"):
        t0 = time.time()
        out["runtimes"][policy] = run(policy, base)
        out["runtimes"][policy]["seconds"] = round(time.time() - t0, 1)
        print(policy, out["runtimes"][policy], flush=True)
    srv.shutdown()
    Path(__file__).with_name("settle_run.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
