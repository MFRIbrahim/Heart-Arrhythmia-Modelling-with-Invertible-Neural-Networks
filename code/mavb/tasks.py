from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List
import itertools
import math

from . import type_1
from . import type_2a
from . import type_2b
from . import type_2c
from . import type_3


@dataclass
class MAVBTask(ABC):
    error: float

    @abstractmethod
    def print(self):
        pass

    @abstractmethod
    def generate_extensions(self, error):
        pass

    @abstractmethod
    def simulation_length(self):
        pass

    @abstractmethod
    def _invoke_solver(self, problem_intervals):
        pass

    def solve(self, measured_intervals):
        simulation_length = self.simulation_length()
        length = min(simulation_length, len(measured_intervals) + 1)

        problem_intervals = measured_intervals[:(length - 1)]
        is_subproblem = len(problem_intervals) < len(measured_intervals)

        solution = self._invoke_solver(problem_intervals)

        return is_subproblem, solution


@dataclass
class MAVB1Task(MAVBTask):
    block_pattern: List[int]

    def __lt__(self, other):
        return self.error < other.error

    def print(self):
        print("MAVB 1")
        print(self.block_pattern)

    def simulation_length(self):
        return sum(self.block_pattern)

    def _invoke_solver(self, problem_intervals):
        return type_1.solve(problem_intervals, self.block_pattern)

    def generate_extensions(self, error):
        patterns = type_1.extend_block_pattern(self.block_pattern)
        return [MAVB1Task(error, bp) for bp in patterns]


@dataclass
class MAVB2aTask(MAVBTask):
    block_pattern: List[int]

    def __lt__(self, other):
        return self.error < other.error

    def print(self):
        print("MAVB 2a")
        print(self.block_pattern)

    def simulation_length(self):
        return sum(self.block_pattern)

    def _invoke_solver(self, problem_intervals):
        return type_2a.solve(problem_intervals, self.block_pattern)

    def generate_extensions(self, error):
        patterns = type_2a.extend_block_pattern(self.block_pattern)
        return [MAVB2aTask(error, bp) for bp in patterns]


@dataclass
class MAVB2bTask(MAVBTask):
    block_pattern: List[int]

    def __lt__(self, other):
        return self.error < other.error

    def print(self):
        print("MAVB 2b")
        print(self.block_pattern)

    def simulation_length(self):
        return math.ceil(sum(self.block_pattern) / 2)

    def _invoke_solver(self, intervals):
        return type_2b.solve(intervals, self.block_pattern)

    def generate_extensions(self, error):
        patterns = type_2b.extend_block_pattern(self.block_pattern)
        return [MAVB2bTask(error, bp) for bp in patterns]


@dataclass
class MAVB2cTask(MAVBTask):
    type_1_pattern: List[int]
    two_to_one_pattern: List[int]

    def __lt__(self, other):
        return self.error < other.error

    def print(self):
        print("MAVB 2c")
        print(self.type_1_pattern)
        print(self.two_to_one_pattern)

    def simulation_length(self):
        return len(self.two_to_one_pattern)

    def _invoke_solver(self, intervals):
        return type_2c.solve(intervals, self.type_1_pattern, self.two_to_one_pattern)

    def generate_extensions(self, error):
        patterns = type_2c.extend_block_pattern(
            self.type_1_pattern, self.two_to_one_pattern)
        return [MAVB2cTask(error, l1_bp, l2_bp) for l1_bp, l2_bp in patterns]


@dataclass
class MAVB3Task(MAVBTask):
    type_1_pattern: List[int]
    l2_pattern: List[int]
    l3_pattern: List[int]

    def __lt__(self, other):
        return self.error < other.error

    def print(self):
        print("MAVB 3")
        print(self.type_1_pattern)
        print(self.l2_pattern)
        print(self.l3_pattern)

    def simulation_length(self):
        return len(self.l3_pattern)

    def _invoke_solver(self, intervals):
        return type_3.solve(intervals, self.type_1_pattern, self.l2_pattern, self.l3_pattern)

    def generate_extensions(self, error):
        patterns = type_3.extend_block_pattern(
            self.type_1_pattern, self.l2_pattern, self.l3_pattern)
        return [MAVB3Task(error, l1, l2, l3) for l1, l2, l3 in patterns]
