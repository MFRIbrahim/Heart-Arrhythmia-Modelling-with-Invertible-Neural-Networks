from .levels import create_atrial_level
from .levels import create_regular_2_to_1_level
from .levels import create_pattern_level
from .levels import extend_variable_pattern
from .solver import forward_simulate_solution
from .solver import get_signals_from_intervals
from .solver import fit_parameters


def simulate_signals(solution, block_pattern):
    A = create_propagation_matrix(block_pattern, sum(block_pattern))
    return forward_simulate_solution(solution, A)


def solve(real_intervals, block_pattern):
    real_signals = get_signals_from_intervals(real_intervals)
    A = create_propagation_matrix(block_pattern, len(real_signals))

    return fit_parameters(real_signals, A, levels=2)


def create_propagation_matrix(block_pattern, total_signals, offset=0):
    assert block_pattern
    assert offset < block_pattern[0]

    atrium_length = 2 * (sum(block_pattern) + len(block_pattern))
    atrial_constraints = create_atrial_level(atrium_length)

    av_length = atrium_length // 2
    av_constraints = create_regular_2_to_1_level(
        atrial_constraints, av_length, level=1)

    constraints = create_pattern_level(
        av_constraints, block_pattern, total_signals, level=2)

    return constraints[offset:]


def extend_block_pattern(pattern):
    return extend_variable_pattern(pattern)
