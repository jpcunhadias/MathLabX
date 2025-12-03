import numpy as np
from sympy import Interval, sympify

from labs.lab5_precalculus import (
    build_transformed_expression,
    compose_functions,
    invert_function,
    piecewise_values,
    analyze_rational,
    sign_samples,
    solve_scalar_inequality,
)


def test_build_transformed_expression():
    expr = build_transformed_expression("x**2", a=2, b=3, c=1, d=-4)
    assert expr.subs({"x": 2}) == 14


def test_compose_and_inverse():
    f_of_g, g_of_f = compose_functions("x**2 + 1", "2*x")
    assert sympify("4*x**2 + 1") == f_of_g
    assert sympify("2*x**2 + 2") == g_of_f
    inv_f = invert_function("2*x + 3")
    assert sympify("(x - 3)/2") == inv_f


def test_piecewise_values():
    x_vals = np.array([-1.0, 1.0])
    y_vals = piecewise_values("x", "-x", 0, x_vals)
    np.testing.assert_allclose(y_vals, [-1.0, -1.0])


def test_solve_scalar_inequality():
    solution = solve_scalar_inequality("x**2 - 4 <= 0")
    assert solution == Interval(-2, 2)


def test_analyze_rational():
    analysis = analyze_rational("x**2 - 1", "x - 1")
    assert sympify("1") in analysis["holes"]
    assert analysis["vertical_asymptotes"] == []
    assert analysis["horizontal_asymptote"] is None
    assert analysis["oblique_asymptote"] is None


def test_sign_samples():
    expr = sympify("1/(x - 1)")
    samples = sign_samples(expr, [1], x_min=-3, x_max=3)
    labels = [s["sign"] for s in samples]
    assert labels[0] < 0
    assert labels[1] > 0


def test_oblique_detection():
    analysis = analyze_rational("x**2 + 1", "x")
    assert analysis["horizontal_asymptote"] is None
    assert sympify("x") == analysis["oblique_asymptote"]
