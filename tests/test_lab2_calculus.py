import numpy as np
from sympy import sympify
from labs.lab2_calculus import calculate_derivative_data


def test_calculate_derivative_data():
    x0 = 1
    data = calculate_derivative_data("x**2", -5, 5, x0)
    assert data["f_prime"] == sympify("2*x")
    assert data["f_prime_x0"] == sympify("2")
    assert data["f_x0"] == sympify("1")
    assert len(data["x_vals"]) == 1000
    assert len(data["y_vals"]) == 1000
    assert len(data["tangent_line"]) == 1000

    # Find the index of the x_vals array that is closest to x0
    x0_index = np.argmin(np.abs(data["x_vals"] - x0))
    # Check that the value of the tangent line at x0 is equal to f(x0)
    assert abs(data["tangent_line"][x0_index] - float(data["f_x0"])) < 0.01
