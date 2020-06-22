import numpy as np

from .forward import compute_heat_error


measured_intervals = [
    586, 552, 556, 900, 580, 566, 938, 620, 904, 606, 928, 600,
    952, 610, 910, 592, 568, 896, 590, 572, 892, 582, 586, 880
]


simulated_intervals = [
    562, 562, 562, 884, 586, 586, 884, 658, 884, 658, 884, 658,
    884, 658, 884, 586, 586, 884, 586, 586, 884, 586, 586, 884
]


def test_should_compute_the_correct_heat_error():
    error = compute_heat_error(measured_intervals, simulated_intervals)

    assert np.round(error, decimals=3) == 20.673
