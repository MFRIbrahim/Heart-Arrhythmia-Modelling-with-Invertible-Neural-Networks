from .solver import get_signals_from_intervals
from .solver import get_intervals_from_signals


def test_should_compute_signals_from_intervals():
    intervals = [100, 100, 100]
    signals = [0, 100, 200, 300]

    assert (get_signals_from_intervals(intervals) == signals).all()


def test_should_compute_intervals_from_signals():
    signals = [0, 100, 200, 300]
    intervals = [100, 100, 100]

    assert (get_intervals_from_signals(signals) == intervals).all()
