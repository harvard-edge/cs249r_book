#!/usr/bin/env python3
"""A targeted reproducer for the repair agent's connection-pool stall.

Written for Lecture 3 (Test-Time Compute) and run for real; its printed
trace is shown on the slides (pool_trace.txt). It forces the interleaving
the chapter describes (Test-Time Compute, section 3.1): one worker waits for
a connection while another returns it.

  python3 pool_reproducer.py > pool_trace.txt

The pool is a minimal stand-in, not the book's code. The return path of a
bounded pool must wake one waiter, as CPython's queue.Queue.put does with
self.not_empty.notify() (Lib/queue.py). The buggy pool omits that call.
"""

import threading
import time


class Pool:
    def __init__(self, size, notify_on_return):
        self.free = [f"conn{i}" for i in range(size)]
        self.cv = threading.Condition()
        self.notify_on_return = notify_on_return

    def acquire(self, timeout):
        with self.cv:
            end = time.monotonic() + timeout
            while not self.free:
                left = end - time.monotonic()
                if left <= 0:
                    return None  # timed out
                self.cv.wait(left)
            return self.free.pop()

    def release(self, conn):
        with self.cv:
            self.free.append(conn)
            if self.notify_on_return:
                self.cv.notify()


def probe(notify_on_return, hold_ms=5, timeout_ms=30):
    """Worker A holds the only connection; B waits; A returns it."""
    pool = Pool(1, notify_on_return)
    t0 = time.monotonic()
    ms = lambda: (time.monotonic() - t0) * 1000.0  # noqa: E731
    log = []
    conn = pool.acquire(timeout=1.0)
    log.append(f"{ms():5.1f} ms  A acquires {conn}")
    result = {}

    def waiter():
        log.append(f"{ms():5.1f} ms  B waits")
        result["conn"] = pool.acquire(timeout=timeout_ms / 1000.0)
        result["at"] = ms()

    b = threading.Thread(target=waiter)
    b.start()
    time.sleep(hold_ms / 1000.0)
    pool.release(conn)
    returned_at = ms()
    log.append(f"{returned_at:5.1f} ms  A returns {conn}; pool holds {len(pool.free)} idle")
    b.join()
    if result["conn"] is None:
        log.append(f"{result['at']:5.1f} ms  B times out; idle connections: {len(pool.free)}")
    else:
        log.append(f"{result['at']:5.1f} ms  B acquires {result['conn']}")
    blocked = result["at"] - returned_at
    log.append(f"B blocked {blocked:.1f} ms after the return")
    # The probe's question: does a waiter wake when a connection returns?
    ok = result["conn"] is not None and blocked < 5.0
    return log, ms(), ok


if __name__ == "__main__":
    cases = (("pool without notify on return, 30 ms timeout", False, 30),
             ("patch 1: timeout raised to 60 ms", False, 60),
             ("patch 2: notify on return", True, 30))
    for label, flag, timeout_ms in cases:
        log, total, ok = probe(flag, timeout_ms=timeout_ms)
        print(f"# {label}")
        for line in log:
            print(line)
        print(f"probe {'PASS' if ok else 'FAIL'}, {total:.1f} ms wall clock")
        print()
