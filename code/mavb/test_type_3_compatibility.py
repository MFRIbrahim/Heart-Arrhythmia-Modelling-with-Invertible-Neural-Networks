from mavb.ratios import *
from mavb.forward import simulate_type_3

offset = 0
atrial_cycle_length = 279
conduction_constant = 10

expected_intervals = [
    559, 838, 1393, 559, 837, 1116, 1116, 1116, 1117, 1673, 1116, 1117,
    1393, 559, 838, 1673, 1117, 1673, 1116, 1116, 837, 1116, 1116, 1116
]

l1_block_pattern = [
    1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 1, 1, 1, 1, 1, 1
]

l2_block_pattern = [
    0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1,
    1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1
]

l3_block_pattern = [
    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 0
]


def test_should_return_the_correct_intervals():
    intervals = simulate_type_3(
        l1_block_pattern, l2_block_pattern, l3_block_pattern, atrial_cycle_length, conduction_constant)

    intervals = intervals[offset:]
    intervals = intervals[:len(expected_intervals)]

    distance = expected_intervals - intervals

    assert max(distance) <= 1
