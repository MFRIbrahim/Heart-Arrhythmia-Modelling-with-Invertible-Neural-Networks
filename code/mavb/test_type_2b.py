import numpy as np

from .ratios import *
from .type_2b import create_propagation_matrix


def test_constraints():
    """

    a1    a2    a3    a4    a5    a6
    |     |     |     |     |     |
    |     |     |     |     |     |
    |----3:2----|     |----3:2----|
    |     |           |     |
    |     |           |     |
    b1    b2          b3    b4
    |     |           |     |
    |     |           |     |
    |-2:1-|           |-2:1-|
    |                 |
    |                 |
    v1                v2


    v1 = d(2_TO_1, level=2) + b1
       = d(2_TO_1, level=2) + d(3_TO_2, level=1, conducted=0) + a1
       = 2*AA               + cc                              + ac
       = 2*AA + ac + cc

    v2 = d(1_TO_1, level=2) + b3
       = d(1_TO_1, level=2) + d(3_TO_2, level=1, conducted=0) + a4
       = 2*AA               + cc                              + ac + 3*AA
       = 5*AA + ac + cc
    """
    block_pattern = [
        RATIO_3_TO_2,
        RATIO_3_TO_2,
    ]

    constraints = create_propagation_matrix(block_pattern, 2)

    expected = np.array([
        [2, 1, 1],
        [5, 1, 1],
    ])

    assert (expected == constraints).all()
