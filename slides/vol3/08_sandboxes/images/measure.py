#!/usr/bin/env python3
"""Real runs for Lecture 8 (Agent Sandboxes): every transcript and timing
the deck shows as "our run" comes from this script, not from typing.

Writes into data/:
  escape.txt        the in-process escape of Agent Sandboxes listing 1, run
                    in a real CPython interpreter
  seatbelt.txt      a dummy credentials file read and an outbound connection,
                    first unconfined, then under a kernel-enforced macOS
                    sandbox profile (sandbox-exec)
  docker_limits.txt one Linux container with no network, a read-only test
                    mount, a 64-process cap, and a 256 MiB memory ceiling:
                    what the code inside sees when it hits each limit
  startup.csv       wall-clock start-to-exit time of an empty command under
                    four envelopes, N runs each
  clone.csv         full copy against a copy-on-write clone of a 2.4 GB base
                    file on APFS, and the cost of a 1-byte change to a clone

The credentials file is a dummy written to a temporary directory; no real
secret is read. Needs: macOS with sandbox-exec, Docker with the alpine image.
Run from this folder: python3 measure.py
"""

import csv
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
N = 30          # runs per envelope for the startup timing
N_DOCKER = 15   # docker runs are slower; fewer samples

# Agent Sandboxes, Napkin Math 8.2: the pinned base the clone run copies
BASE_GB = 2.4


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def header():
    return (f"# host: {platform.platform()} ({platform.machine()}), "
            f"python {platform.python_version()}, "
            f"run {time.strftime('%Y-%m-%d')}\n")


# 1. The in-process escape ---------------------------------------------------
ESCAPE = r'''
API_KEY = "sk-demo-not-a-real-key"     # the supervisor's own state

def total(values):                     # the one helper we hand over
    return sum(values)

restricted = {"__builtins__": {}, "total": total}

print(eval("total([1, 2, 3])", restricted))
g = eval("total.__globals__", restricted)
print(type(g).__name__, "with", len(g), "names; API_KEY =", g["API_KEY"])
subs = eval("().__class__.__base__.__subclasses__()", restricted)
print(len(subs), "classes reachable from an empty tuple")
'''


def escape():
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "supervisor.py"
        src.write_text(ESCAPE)
        r = run([sys.executable, "-I", str(src)])
    (DATA / "escape.txt").write_text(header() + "$ python3 supervisor.py\n" + r.stdout + r.stderr)


# 2. Kernel-enforced profile on macOS ----------------------------------------
def seatbelt():
    out = [header()]
    with tempfile.TemporaryDirectory() as d:
        home = Path(d) / "home"
        (home / ".aws").mkdir(parents=True)
        cred = home / ".aws" / "credentials"
        cred.write_text("[default]\naws_secret_access_key = DUMMY-NOT-A-SECRET\n")
        aws = (home / ".aws").resolve()
        profile = ("(version 1)(allow default)"
                   f'(deny file-read* (subpath "{aws}"))'
                   "(deny network-outbound (remote ip))")
        cases = [
            ("cat ~/.aws/credentials", ["cat", str(cred)]),
            ("curl -sS -m 3 -o /dev/null -w '%{http_code}' https://example.com",
             ["curl", "-sS", "-m", "3", "-o", "/dev/null", "-w", "%{http_code}",
              "https://example.com"]),
        ]
        for label, cmd in cases:
            for mode, full in (("unconfined", cmd),
                               ("sandboxed", ["sandbox-exec", "-p", profile] + cmd)):
                r = run(full)
                text = (r.stdout + r.stderr).strip().replace(str(home), "~")
                out.append(f"[{mode}] $ {label}\n{text}\n(exit {r.returncode})\n")
        out.append("profile: (allow default) (deny file-read* (subpath \"~/.aws\")) "
                   "(deny network-outbound (remote ip))\n")
    (DATA / "seatbelt.txt").write_text("".join(out))


# 3. Linux limits in one container -------------------------------------------
INSIDE = r'''
exec 2>&1
echo '$ cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/pids.max'
cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/pids.max
echo '$ wget -q -T 3 -O /dev/null http://example.com'
wget -q -T 3 -O /dev/null http://example.com; echo "(exit $?)"
echo "$ echo 'assert True' > /tests/test_pool.py"
echo 'assert True' > /tests/test_pool.py; echo "(exit $?)"
echo '$ for i in $(seq 100); do sleep 30 & done'
( for i in $(seq 100); do sleep 30 & echo $i > /tmp/n; done )
echo "sleeps started before the cap: $(cat /tmp/n)"
echo '$ unshare -U true'
unshare -U true; echo "(exit $?)"
'''


def docker_limits():
    with tempfile.TemporaryDirectory() as d:
        tests = Path(d) / "tests"
        tests.mkdir()
        (tests / "test_pool.py").write_text("def test_pool():\n    assert True\n")
        cmd = ["docker", "run", "--rm", "--network", "none", "--pids-limit", "64",
               "--memory", "256m", "-v", f"{tests}:/tests:ro", "alpine", "sh", "-c", INSIDE]
        r = run(cmd)
    flags = "docker run --rm --network none --pids-limit 64 --memory 256m -v ./tests:/tests:ro alpine"
    ver = run(["docker", "version", "--format", "{{.Server.Version}}"]).stdout.strip()
    (DATA / "docker_limits.txt").write_text(
        header() + f"# docker engine {ver}, default seccomp profile\n# {flags}\n"
        + r.stdout + r.stderr)


# 4. Start-to-exit time of an empty command ----------------------------------
def startup():
    envs = [
        ("process", ["/usr/bin/true"], N),
        ("macOS sandbox profile", ["sandbox-exec", "-p", "(version 1)(allow default)",
                                   "/usr/bin/true"], N),
        ("CPython interpreter", [sys.executable, "-I", "-c", "pass"], N),
        ("Linux container (Docker)", ["docker", "run", "--rm", "--network", "none",
                                      "alpine", "true"], N_DOCKER),
    ]
    rows = []
    for name, cmd, n in envs:
        run(cmd)  # warm caches once
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            subprocess.run(cmd, capture_output=True)
            ts.append((time.perf_counter() - t0) * 1000)
        rows.append((name, n, statistics.median(ts), min(ts), max(ts)))
    with open(DATA / "startup.csv", "w", newline="") as fh:
        fh.write(header())
        w = csv.writer(fh)
        w.writerow(["envelope", "runs", "median_ms", "min_ms", "max_ms"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}"])


# 5. Copy-on-write clone against a full copy (APFS) --------------------------
def clone():
    size = int(BASE_GB * 1e9)
    with tempfile.TemporaryDirectory(dir=HERE) as d:
        base = Path(d) / "base.bin"
        with open(base, "wb") as fh:  # incompressible bytes, written once
            chunk = os.urandom(64 * 2**20)
            left = size
            while left > 0:
                fh.write(chunk[:min(left, len(chunk))])
                left -= len(chunk)
        os.sync()
        rows = []
        t0 = time.perf_counter()
        shutil.copyfile(base, Path(d) / "copy.bin")  # full copy of every byte
        os.sync()
        rows.append(("full copy", (time.perf_counter() - t0) * 1000))
        t0 = time.perf_counter()
        subprocess.run(["cp", "-c", str(base), str(Path(d) / "clone.bin")], check=True)
        os.sync()
        rows.append(("copy-on-write clone (cp -c)", (time.perf_counter() - t0) * 1000))
        t0 = time.perf_counter()
        with open(Path(d) / "clone.bin", "r+b") as fh:  # 1-byte change mid-file
            fh.seek(size // 2)
            fh.write(b"\x00")
            fh.flush()
            os.fsync(fh.fileno())
        rows.append(("1-byte change to the clone", (time.perf_counter() - t0) * 1000))
    with open(DATA / "clone.csv", "w", newline="") as fh:
        fh.write(header() + f"# base file {BASE_GB} GB, APFS, internal SSD\n")
        w = csv.writer(fh)
        w.writerow(["operation", "ms"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.1f}"])


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    steps = [globals()[a] for a in sys.argv[1:]] or [escape, seatbelt, docker_limits, startup, clone]
    for step in steps:
        print("running", step.__name__)
        step()
    print("done; see data/")
