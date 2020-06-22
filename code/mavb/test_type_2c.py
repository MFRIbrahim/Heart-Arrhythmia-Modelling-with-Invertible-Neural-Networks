import numpy as np

from .ratios import *
from .type_2c import create_propagation_matrix
from .type_2c import extend_block_pattern


def test_constraints():
    """

    a1    a2    a3    a4    a5
    |     |     |     |     |
    |     |     |     |     |
    |----3:2----|     |-2:1-|
    |     |           |
    |     |           |
    b1    b2          b3
    |     |           |
    |     |           |
    |-2:1-|           |-1:1-|
    |                 |
    |                 |
    v1                v2


    v1 = d(2_TO_1, level=2) + b1
       = d(2_TO_1, level=2) + d(3_TO_2, level=1, conducted=0) + a1
       = 2*AA               + cc                              + ac
       = 2*AA + ac + cc

    v2 = d(1_TO_1, level=2) + b2
       = d(1_TO_1, level=2) + d(2_TO_1, level=1) + a4
       = 2*AA               + AA                 + ac + 3*AA
       = 6*AA + ac
    """
    l1_block_pattern = [
        RATIO_3_TO_2,
        RATIO_2_TO_1,
    ]

    l2_block_pattern = [
        RATIO_2_TO_1,
        RATIO_1_TO_1,
    ]

    constraints = create_propagation_matrix(
        l1_block_pattern, l2_block_pattern, 2)

    expected = np.array([
        [2, 1, 1],
        [6, 1, 0],
    ])

    assert (expected == constraints).all()


def test_should_extend_the_block_pattern():
    l1_pattern = [RATIO_3_TO_2]
    l2_pattern = [RATIO_1_TO_1]

    patterns = extend_block_pattern(l1_pattern, l2_pattern)
    patterns = list(patterns)

    assert (
        [RATIO_3_TO_2],
        [RATIO_1_TO_1, RATIO_1_TO_1]
    ) in patterns
    assert (
        [RATIO_3_TO_2, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns
    assert (
        [RATIO_3_TO_2, RATIO_3_TO_2],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns
    assert (
        [RATIO_3_TO_2, RATIO_4_TO_3],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns
    assert (
        [RATIO_3_TO_2, RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert len(patterns) == 5
