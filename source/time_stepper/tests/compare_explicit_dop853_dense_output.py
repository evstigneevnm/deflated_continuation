#!/usr/bin/env python3
"""Compare fixed-step samples against SciPy's native DOP853 interpolant."""

import argparse

import numpy as np
from scipy.integrate import DOP853


def exponential_rhs(time, state):
    return state


def forced_rhs(time, state):
    return state + time


def quadratic_rhs(time, state):
    return state * state


def compare(path):
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape != (126, 5) or not np.isfinite(rows).all():
        raise ValueError("Expected 126 finite dense samples with five columns")
    maximum_error = 0.0
    for group in rows.reshape(-1, 7, 5):
        problem, start, dt = group[0, :3]
        if not (group[:, :3] == group[0, :3]).all():
            raise ValueError("Inconsistent dense sample group")
        if problem == 0:
            rhs, initial = exponential_rhs, np.exp(start)
        elif problem == 1:
            rhs, initial = forced_rhs, 2 * np.exp(start) - start - 1
        elif problem == 2:
            rhs, initial = quadratic_rhs, 1 / (1 - start)
        else:
            raise ValueError("Unknown analytical problem")
        end = start + dt
        magnitude = abs(end - start)
        solver = DOP853(rhs, start, [initial], end, first_step=magnitude,
                        max_step=magnitude, rtol=0.5, atol=0.5)
        solver.step()
        if solver.status != "finished":
            raise AssertionError("SciPy must take the same single fixed step")
        values = solver.dense_output()(start + dt * group[:, 3])[0]
        error = np.max(np.abs(values - group[:, 4]))
        maximum_error = max(maximum_error, float(error))
        np.testing.assert_allclose(group[:, 4], values, rtol=2e-12, atol=2e-13)
    print(f"Native SciPy DOP853 dense comparison: PASS, max difference {maximum_error:.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", help="C++ DOP853 dense-output sample file")
    compare(parser.parse_args().samples)
