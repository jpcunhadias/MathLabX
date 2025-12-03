from sympy import sympify, symbols
from labs.lab4_multivariable import calculate_multivariable_data

def test_calculate_multivariable_data():
    data = calculate_multivariable_data("x**2 + y**2", -3, 3, -3, 3, 1, 1)
    x, y = symbols("x y")
    assert data["grad_f"][0] == sympify("2*x")
    assert data["grad_f"][1] == sympify("2*y")
    assert data["grad_f_x0_y0"][0] == sympify("2")
    assert data["grad_f_x0_y0"][1] == sympify("2")
    assert data["X"].shape == (100, 100)
    assert data["Y"].shape == (100, 100)
    assert data["Z"].shape == (100, 100)
