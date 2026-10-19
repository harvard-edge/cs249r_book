import threading


class Pool:
    def __init__(self, size):
        self._idle = [object() for _ in range(size)]
        self._cond = threading.Condition()

    def acquire(self, timeout=None):
        with self._cond:
            while not self._idle:
                if not self._cond.wait(timeout):
                    raise TimeoutError("no connection")
            return self._idle.pop()

    def release(self, conn):
        with self._cond:
            self._idle.append(conn)
            return None
