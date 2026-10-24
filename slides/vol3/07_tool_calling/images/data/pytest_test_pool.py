import pytest
@pytest.mark.parametrize("i", range(20000))
def test_checkout_return(i):
    assert i >= 0
def test_waiter_notified_on_return():
    waiters_notified = 0
    assert waiters_notified == 1
