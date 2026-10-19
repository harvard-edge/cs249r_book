import threading
import time

from src.pool import Pool


def test_waiter_notified_on_return():
    pool = Pool(size=1)
    held = pool.acquire()
    resumed = threading.Event()

    def waiter():
        pool.acquire(timeout=0.5)
        resumed.set()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)                 # waiter registered before return
    pool.release(held)               # connection available after return
    assert resumed.wait(0.2), "waiter_resumed_at_notification_check"
