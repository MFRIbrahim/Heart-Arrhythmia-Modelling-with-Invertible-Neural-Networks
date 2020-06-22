from mavb.ratios import *
from mavb.forward import simulate_type_2c

offset = 0
atrial_cycle_length = 216
conduction_constant = 10

expected_intervals = [
    432, 432, 432, 432, 432, 433, 863, 432, 432, 432, 432, 432,
    432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 432, 433
]

l1_block_pattern = [
    RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1,
    RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1,
    RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1,
    RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1, RATIO_2_TO_1,
]

l2_block_pattern = [
    RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_2_TO_1,
    RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1,
    RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1,
    RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1, RATIO_2_TO_1,
]


def test_should_return_the_correct_intervals():
    intervals = simulate_type_2c(
        l1_block_pattern, l2_block_pattern, atrial_cycle_length, conduction_constant)

    intervals = intervals[offset:]
    intervals = intervals[:len(expected_intervals)]

    distance = expected_intervals - intervals

    assert max(distance) <= 1
