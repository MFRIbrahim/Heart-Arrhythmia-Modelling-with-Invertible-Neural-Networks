import numpy as np

from .ratios import *
from .type_3 import create_propagation_matrix
from .type_3 import extend_block_pattern


def test_constraints():
    """

    a1    a2    a3    a4    a5    a6    a7
    |     |     |     |     |     |     |
    |     |     |     |     |     |     |
    |----3:2----|     |-2:1-|     |-2:1-|
    |     |           |           |     
    |     |           |           |
    b1    b2          b3          b4
    |     |           |           |
    |     |           |           |
    |-1:1-|----2:1----|           |-1:1-|
    |     |                       |
    |     |                       |
    c1    c2                      c3
    |     |                       |
    |     |                       |
    |-2:1-|                       |-1:1-|
    |                             |
    |                             |
    v1                            v2


    v1 = c1                                                        + d(2_TO_1, level=3)
       = b1                                   + d(1_TO_1, level=2) + d(2_TO_1, level=3)
       = a1 + d(3_TO_2, level=1, conducted=0) + d(1_TO_1, level=2) + d(2_TO_1, level=3)
       = ac + cc                              + 2*AA               + 4*AA
       = ac + cc + 6*AA

    v2 = c3                                                  + d(1_TO_1, level=3)
       = b4                             + d(2_TO_2, level=2) + d(1_TO_1, level=3)
       = a6        + d(2_TO_1, level=1) + d(2_TO_2, level=2) + d(1_TO_1, level=3)
       = ac + 5*AA + 1*AA +             + 2*AA               + 4*AA
       = ac + 12*AA
    """
    l1_block_pattern = [
        RATIO_3_TO_2,
        RATIO_2_TO_1,
        RATIO_2_TO_1,
    ]

    l2_block_pattern = [
        RATIO_1_TO_1,
        RATIO_2_TO_1,
        RATIO_1_TO_1,
    ]

    l3_block_pattern = [
        RATIO_2_TO_1,
        RATIO_1_TO_1
    ]

    constraints = create_propagation_matrix(
        l1_block_pattern,
        l2_block_pattern,
        l3_block_pattern,
        total_signals=2
    )

    expected = np.array([
        [6, 1, 1],
        [12, 1, 0],
    ])

    assert (expected == constraints).all()


def test_should_extend_the_block_pattern():
    l1_pattern = [RATIO_5_TO_4]
    l2_pattern = [RATIO_1_TO_1]
    l3_pattern = [RATIO_1_TO_1]

    patterns = extend_block_pattern(l1_pattern, l2_pattern, l3_pattern)
    patterns = list(patterns)

    assert (
        [RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_1_TO_1],
        [RATIO_1_TO_1, RATIO_1_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_1_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_1_TO_1, RATIO_1_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_1_TO_1, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_2_TO_1, RATIO_1_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4, RATIO_3_TO_2],
        [RATIO_1_TO_1, RATIO_2_TO_1, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4, RATIO_4_TO_3],
        [RATIO_1_TO_1, RATIO_2_TO_1, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4, RATIO_5_TO_4],
        [RATIO_1_TO_1, RATIO_2_TO_1, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert (
        [RATIO_5_TO_4, RATIO_6_TO_5],
        [RATIO_1_TO_1, RATIO_2_TO_1, RATIO_2_TO_1],
        [RATIO_1_TO_1, RATIO_2_TO_1]
    ) in patterns

    assert len(patterns) == 9
