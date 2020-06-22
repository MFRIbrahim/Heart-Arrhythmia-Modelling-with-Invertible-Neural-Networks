from mavb.ratios import *
from mavb.forward import simulate_type_1

offset = 0
atrial_cycle_length = 279
conduction_constant = 250

expected_intervals = [
    529, 294, 294, 557, 529, 294, 294, 557, 558, 529, 308, 558,
    529, 308, 558, 529, 294, 294, 557, 558, 529, 294, 294, 528
]

block_pattern = [
    RATIO_2_TO_1, RATIO_4_TO_3, RATIO_2_TO_1, RATIO_4_TO_3, RATIO_2_TO_1, RATIO_2_TO_1,
    RATIO_3_TO_2, RATIO_2_TO_1, RATIO_3_TO_2, RATIO_2_TO_1, RATIO_4_TO_3, RATIO_2_TO_1,
    RATIO_2_TO_1, RATIO_4_TO_3, RATIO_3_TO_2,
]


def test_should_return_the_correct_intervals():
    intervals = simulate_type_1(
        block_pattern, atrial_cycle_length, conduction_constant)

    intervals = intervals[offset:]
    intervals = intervals[:len(expected_intervals)]

    distance = expected_intervals - intervals

    assert max(distance) <= 1
