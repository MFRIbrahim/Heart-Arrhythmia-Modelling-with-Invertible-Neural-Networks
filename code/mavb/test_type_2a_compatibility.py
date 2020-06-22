from mavb.ratios import *
from mavb.forward import simulate_type_2a

offset = 0
atrial_cycle_length = 257
conduction_constant = 370

expected_intervals = [
    562, 562, 562, 884, 586, 586, 884, 658, 884, 658, 884, 658,
    884, 658, 884, 586, 586, 884, 586, 586, 884, 586, 586, 884
]

block_pattern = [
    RATIO_5_TO_4, RATIO_4_TO_3, RATIO_3_TO_2, RATIO_3_TO_2, RATIO_3_TO_2, RATIO_3_TO_2,
    RATIO_4_TO_3, RATIO_4_TO_3, RATIO_4_TO_3, RATIO_3_TO_2,
]


def test_should_return_the_correct_intervals():
    intervals = simulate_type_2a(
        block_pattern, atrial_cycle_length, conduction_constant)

    intervals = intervals[offset:]
    intervals = intervals[:len(expected_intervals)]

    distance = expected_intervals - intervals

    assert max(distance) <= 1
