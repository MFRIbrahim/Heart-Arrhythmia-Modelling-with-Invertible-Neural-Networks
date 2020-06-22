import numpy as np

from .ratios import *
from .levels import create_atrial_level
from .levels import create_block_constraints


def test_2_to_1_constraints():
    atrium = create_atrial_level(20)
    constraints, input_index = create_block_constraints(
        RATIO_2_TO_1, atrium, input_index=0)

    expected = np.array([
        [1, 1, 0]
    ])

    assert input_index == 2
    assert (expected == constraints).all()


def test_3_to_2_constraints():
    atrium = create_atrial_level(20)
    constraints, input_index = create_block_constraints(
        RATIO_3_TO_2, atrium, input_index=0)

    expected = np.array([
        [0, 1, 1],
        [2, 1, 0]
    ])

    assert input_index == 3
    assert (expected == constraints).all()


def test_4_to_3_constraints():
    atrium = create_atrial_level(20)
    constraints, input_index = create_block_constraints(
        RATIO_4_TO_3, atrium, input_index=0)

    expected = np.array([
        [0, 1, 1],
        [1.5, 1, 0.5],
        [3, 1, 0]
    ])

    assert input_index == 4
    assert (expected == constraints).all()
