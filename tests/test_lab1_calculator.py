from sympy import sympify

from labs.lab1_calculator import (
    calculate_derivative,
    calculate_integral,
    calculate_limit,
)


def test_calculate_derivative():
    assert calculate_derivative("x**2") == sympify("2*x")
    assert calculate_derivative("sin(x)") == sympify("cos(x)")
    assert calculate_derivative("cos(x)") == sympify("-sin(x)")


def test_calculate_integral():
    assert calculate_integral("x**2") == sympify("x**3/3")
    assert calculate_integral("cos(x)") == sympify("sin(x)")


def test_calculate_limit():
    assert calculate_limit("1/x", "oo") == sympify("0")
    assert calculate_limit("sin(x)/x", "0") == sympify("1")
