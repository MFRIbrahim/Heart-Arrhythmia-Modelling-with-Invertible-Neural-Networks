from cvxopt import matrix
import numpy as np

from . import type_1
from . import type_2a
from . import type_2b
from . import type_2c
from . import type_3
from .solver import get_signals_from_intervals
from .solver import get_intervals_from_signals


def simulate_type_1(block, atrial_cycle_length, conduction_constant):
    solution = __create_solution(atrial_cycle_length, conduction_constant)
    signals = type_1.simulate_signals(solution, block)
    return get_intervals_from_signals(signals)


def simulate_type_2a(block, atrial_cycle_length, conduction_constant):
    solution = __create_solution(atrial_cycle_length, conduction_constant)
    signals = type_2a.simulate_signals(solution, block)
    return get_intervals_from_signals(signals)


def simulate_type_2b(block, atrial_cycle_length, conduction_constant):
    solution = __create_solution(atrial_cycle_length, conduction_constant)
    signals = type_2b.simulate_signals(solution, block)
    return get_intervals_from_signals(signals)


def simulate_type_2c(l1_block, l2_block, atrial_cycle_length, conduction_constant):
    solution = __create_solution(atrial_cycle_length, conduction_constant)
    signals = type_2c.simulate_signals(solution, l1_block, l2_block)
    return get_intervals_from_signals(signals)


def simulate_type_3(l1_block, l2_block, l3_block, atrial_cycle_length, conduction_constant):
    solution = __create_solution(atrial_cycle_length, conduction_constant)
    signals = type_3.simulate_signals(solution, l1_block, l2_block, l3_block)
    return get_intervals_from_signals(signals)


def __create_solution(atrial_cycle_length, conduction_constant):
    x = matrix([atrial_cycle_length, 0, conduction_constant], tc="d")
    return {"x": x}


def compute_heat_error(measured_intervals, simulated_intervals):
    measured_signals = get_signals_from_intervals(measured_intervals)
    simulated_signals = get_signals_from_intervals(simulated_intervals)

    num_measured_signals = len(measured_signals)
    num_simulated_signals = len(simulated_signals)

    max_offset = num_simulated_signals - num_measured_signals
    assert max_offset >= 0

    def compute_error(offset):
        start = offset
        end = offset + num_measured_signals
        shifted_simulated_signals = simulated_signals[start:end]

        return compute_root_mean_squared_error(measured_signals, shifted_simulated_signals)

    errors = [compute_error(o) for o in range(max_offset + 1)]
    return min(errors)


def compute_root_mean_squared_error(measured_signals, simulated_signals):
    distance = measured_signals - simulated_signals
    error = np.linalg.norm(distance) ** 2
    error = error / len(measured_signals)
    error = np.sqrt(error)

    return error
