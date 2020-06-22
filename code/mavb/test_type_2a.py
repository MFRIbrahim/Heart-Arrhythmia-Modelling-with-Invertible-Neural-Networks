import numpy as np

from .ratios import *
from .type_2a import create_propagation_matrix


def test_constraints():
    """

    a1    a2    a3    a4    a5    a6    a7    a8    a9    a10
    |     |     |     |     |     |     |     |     |     |
    |     |     |     |     |     |     |     |     |     |
    |-2:1-|     |-2:1-|     |-2:1-|     |-2:1-|     |-2:1-|
    |           |           |           |           |
    |           |           |           |           |
    b1          b2          b3          b4          b5
    |           |           |           |           |
    |           |           |           |           |
    |----2:1----|           |----------3:2----------|
    |                       |           |
    |                       |           |
    v1                      v2          v3


    v1 = d(2_TO_1, level=2) + b1
       = d(2_TO_1, level=2) + d(2_TO_1, level=1) + a1
       = 2*AA               + AA                 + ac
       = 3*AA + ac

    v2 = d(3_TO_2, level=2, conducted=0) + b3
       = d(3_TO_2, level=2, conducted=0) + d(2_TO_1, level=1) + a5
       = cc                              + AA                 + ac + 4*AA
       = 5*AA + ac + cc

    v3 = d(3_TO_2, level=2, conducted=1) + b4
       = d(3_TO_2, level=2, conducted=1) + d(2_TO_1, level=1) + a7
       = cc + 2*AA - cc                  + AA                 + ac + 6*AA
       = 9*AA + ac
    """
    block_pattern = [
        RATIO_2_TO_1,
        RATIO_3_TO_2
    ]

    constraints = create_propagation_matrix(block_pattern, 3)

    expected = np.array([
        [3, 1, 0],
        [5, 1, 1],
        [9, 1, 0]
    ])

    assert (expected == constraints).all()


def test_should_use_the_correct_offset():
    block_pattern = [
        RATIO_3_TO_2
    ]

    constraints = create_propagation_matrix(block_pattern, 1, offset=1)

    expected = np.array([
        [5, 1, 0]
    ])

    assert (expected == constraints).all()
