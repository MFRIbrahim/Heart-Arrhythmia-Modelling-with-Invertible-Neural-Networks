import numpy as np
from cvxopt import matrix
from cvxopt import solvers


def forward_simulate_solution(solution, A):
    x = solution["x"]
    return [np.dot(c, x)[0] for c in A]


def fit_parameters(real_signals, A, levels):
    """
    We want to solve the Problem:

    minimize    ||Ax - b||
    subject to  G*x <= h

    where Gx <= h describes the physiological parameter limits.
    However, the solver cvxopt.solvers.qp can only solve problems of the form:

    minimize    (1/2)*x'Px + q'x
    subject to  Gx <= h
                Ax = b

    We will therefore translate the problem into a suitable form for cvxopt.

      ||Ax - b||
    = x'A'Ax - 2b'Ax + b'b
    = x'Px + 2q'x + b'b
    = 2*( (1/2)*x'Px + q'x) + b'b

    with P = A'A and q = -A'b

    Since b'b is constant it can be removed when minimizing the expression.

    """
    assert len(real_signals) == len(A), "Pattern has the wrong size"

    P = A.transpose() @ A
    P = matrix(P, tc="d")

    q = - A.transpose() @ np.array(real_signals).transpose()
    q = matrix(q, tc="d")

    tight_cc_bound = get_tight_cc_bound(levels)
    G = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 0, 1],
        [0, 0, -1],
        [-tight_cc_bound, 0, 1],
    ])
    G = matrix(G, tc="d")

    h = np.array([400, -188, 400, -10, 0])
    h = matrix(h, tc="d")

    solution = solvers.qp(P, q, G, h)

    return extend_solution(solution, real_signals)


def get_signals_from_intervals(intervals):
    return np.array([sum(intervals[:i]) for i in range(len(intervals) + 1)])


def get_intervals_from_signals(signals):
    return np.array([signals[i] - signals[i-1] for i in range(1, len(signals))])


def get_tight_cc_bound(levels):
    """
    See Diss. Kehrle, Eq.4.10:

    cc <= k_l * AA, with k_l = 2^(l-1)

    Returns the factor k_l.
    """
    return 2 ** (levels - 1)


def extend_solution(solution, real_signals):
    real_signals = np.array(real_signals)

    dot = np.dot(real_signals, real_signals)
    objective = solution['primal objective']
    adjusted_objective = max(0, 2 * objective + dot)
    solution["adjusted objective"] = adjusted_objective

    rmse = np.sqrt(adjusted_objective / len(real_signals))
    solution["rmse"] = rmse

    return solution
