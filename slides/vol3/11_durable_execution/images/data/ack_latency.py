"""Lecture 11: how long does each acknowledgment take on one laptop?

Appends a 512-byte ToolIntent-sized record to a log file and times three
acknowledgments, each repeated N times:
  write        os.write returns (bytes in the kernel page cache)
  fsync        os.fsync returns (on macOS: data sent to the drive, which may
               still hold it in its volatile cache; see man fsync)
  full_fsync   fcntl(F_FULLFSYNC) returns (macOS: the drive is asked to flush
               its cache to stable media)
Then group commit: k records written, one F_FULLFSYNC for the batch, for
k in 1, 2, 4, 8, 16, 32, 64; reports the flush time per batch and per record.

Usage: python ack_latency.py OUT.json [DIR]   (DIR holds the scratch log file)
"""

import fcntl
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time

N = 300
REC = (b'{"type":"ToolIntent","operation_id":"test_r18","attempt_id":"attempt1",'
       b'"snapshot":"ws_r18","owner_epoch":7}').ljust(511, b" ") + b"\n"


def timed(fn):
    t = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t) / 1e6   # ms


def main(out, d):
    path = os.path.join(d, "trajectory.log")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o644)
    # warm the file and the drive
    for _ in range(50):
        os.write(fd, REC)
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    res = {"write": [], "fsync": [], "full_fsync": []}
    for _ in range(N):
        res["write"].append(timed(lambda: os.write(fd, REC)))
        os.write(fd, REC)
        res["fsync"].append(timed(lambda: os.fsync(fd)))
        os.write(fd, REC)
        res["full_fsync"].append(timed(lambda: fcntl.fcntl(fd, fcntl.F_FULLFSYNC)))
    group = {}
    for k in (1, 2, 4, 8, 16, 32, 64):
        flush = []
        for _ in range(60):
            for _ in range(k):
                os.write(fd, REC)
            flush.append(timed(lambda: fcntl.fcntl(fd, fcntl.F_FULLFSYNC)))
        group[k] = {"flush_ms_median": statistics.median(flush),
                    "per_record_ms": statistics.median(flush) / k}
    os.close(fd)
    os.unlink(path)

    def summ(v):
        v = sorted(v)
        return {"median_ms": statistics.median(v), "p90_ms": v[int(0.9 * len(v))],
                "n": len(v)}

    hw = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True, text=True).stdout.strip()
    data = {"machine": f"{hw}, macOS {platform.mac_ver()[0]}, internal SSD, APFS",
            "record_bytes": len(REC), "measured": time.strftime("%Y-%m-%d"),
            **{k: summ(v) for k, v in res.items()}, "group_commit": group}
    with open(out, "w") as f:
        json.dump(data, f, indent=1)
    print(json.dumps(data, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else tempfile.gettempdir())
