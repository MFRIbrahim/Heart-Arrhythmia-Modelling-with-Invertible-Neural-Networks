import numpy as np

from .ratios import *


def create_pattern_level(input_constraints, block_pattern, total_signals, level=1):
    assert total_signals > 0

    constraints = []
    input_index = 0
    for ratio in block_pattern:
        remaining_signals = total_signals - len(constraints)
        assert remaining_signals > 0, "Too few measurements to reach the last block"

        new_constraints, input_index = create_block_constraints(
            ratio, input_constraints, input_index, level=level)
        constraints += new_constraints

    return np.array(constraints)[:total_signals]


def create_block_constraints(ratio, input_constraints, input_index, level=1):
    assert ratio >= RATIO_2_TO_1, "Ratio must be at least 2:1"
    assert ratio <= RATIO_8_TO_7, "Ratio must be at most 8:7"

    remaining_input_signals = len(input_constraints) - input_index
    assert remaining_input_signals > 0, "Not enough input signals to each the last block"

    if ratio == RATIO_2_TO_1:
        input_constraint = input_constraints[input_index]
        constraints = [conduct_single(input_constraint, level)]
        input_index += 2
    else:
        total_signals = min(ratio, remaining_input_signals)

        constraints = []
        for i in range(total_signals):
            input_constraint = input_constraints[input_index]

            constraint = conduct_multiple(
                input_constraint, ratio, conducted=i, level=level)
            constraints.append(constraint)

            input_index += 1
        input_index += 1

    return constraints, input_index


def create_regular_2_to_1_level(input_constraints, total_signals, level=1):
    assert total_signals > 0

    constraints = []
    for i in range(0, len(input_constraints), 2):
        input_constraint = input_constraints[i]
        constraint = conduct_single(input_constraint, level)
        constraints.append(constraint)

    return np.array(constraints)[:total_signals]


def create_changing_1_to_1_level(input_constraints, block_pattern, total_signals, level=1):
    assert total_signals > 0

    input_index = 0
    constraints = []
    for ratio in block_pattern:
        assert ratio in [RATIO_1_TO_1, RATIO_2_TO_1]

        assert input_index < len(input_constraints)
        input_constraint = input_constraints[input_index]

        constraint = conduct_single(input_constraint, level)
        constraints.append(constraint)

        input_index += 1 + ratio

    return np.array(constraints)[:total_signals]


def create_atrial_level(total_signals):
    assert total_signals > 0

    constraints = [constraint(i, 1, 0) for i in range(total_signals)]
    return np.array(constraints)


def conduct_single(input_constraint, level=1):
    """
    Create the output constraint for either 2:1 or 1:1 conduction.
    """
    AA, ac, cc = input_constraint

    AA += get_longest_input_difference(level)

    return constraint(AA, ac, cc)


def conduct_multiple(input_constraint, ratio, conducted, level=1):
    """
    Create the output constraint for n+1:n with n > 1.
    """
    assert ratio > 1
    assert conducted >= 0
    assert conducted < ratio

    conduction_increment = 1 / (ratio - 1)

    AA, ac, cc = input_constraint

    factor = conducted * conduction_increment
    AA += get_longest_input_difference(level) * factor
    cc += 1 - factor

    return constraint(AA, ac, cc)


def get_longest_input_difference(level):
    """
    TODO: Needs proof!

    Return the longest possible interval between two incoming signals
    relative to the AA.

    max_intervall = k_l * AA

    Returns the factor k_l.
    """
    assert level > 0
    return 2 ** (level - 1)


def constraint(AA, ac, cc):
    return [AA, ac, cc]


def extend_variable_pattern(pattern):
    if pattern:
        last_ratio = pattern[-1]
        min_ratio = max(RATIO_2_TO_1, last_ratio - 2)
        max_ratio = min(RATIO_6_TO_5, last_ratio + 2)
    else:
        min_ratio = RATIO_2_TO_1
        max_ratio = RATIO_6_TO_5

    ratios = range(min_ratio, max_ratio + 1)
    return map(lambda r: [*pattern, r], ratios)
