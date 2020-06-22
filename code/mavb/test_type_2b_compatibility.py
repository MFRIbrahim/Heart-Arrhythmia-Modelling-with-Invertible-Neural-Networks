from mavb.ratios import *
from mavb.forward import simulate_type_2b

offset = 0
atrial_cycle_length = 251
conduction_constant = 210

expected_intervals = [
    753, 753, 753, 753, 753, 753, 530, 725, 753, 544, 962, 544,
    732, 732, 530, 725, 753, 530, 725, 753, 530, 725, 753, 544
]

block_pattern = [
    RATIO_3_TO_2, RATIO_3_TO_2, RATIO_3_TO_2, RATIO_3_TO_2, RATIO_3_TO_2, RATIO_3_TO_2,
    RATIO_5_TO_4, RATIO_3_TO_2, RATIO_4_TO_3, RATIO_2_TO_1, RATIO_4_TO_3, RATIO_4_TO_3,
    RATIO_5_TO_4, RATIO_3_TO_2, RATIO_5_TO_4, RATIO_3_TO_2, RATIO_5_TO_4, RATIO_3_TO_2,
    RATIO_4_TO_3,
]


def test_should_return_the_correct_intervals():
    intervals = simulate_type_2b(
        block_pattern, atrial_cycle_length, conduction_constant)

    intervals = intervals[offset:]
    intervals = intervals[:len(expected_intervals)]

    distance = expected_intervals - intervals

    assert max(distance) <= 1
