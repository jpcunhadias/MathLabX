import unittest
import sys
from unittest.mock import MagicMock
from sympy import sympify
from labs.lab1_calculator import (
    calculate_derivative,
    calculate_integral,
    calculate_limit,
)

# Mock the streamlit module
sys.modules["streamlit"] = MagicMock()


class TestLab1Calculator(unittest.TestCase):

    def test_calculate_derivative(self):
        self.assertEqual(calculate_derivative("x**2"), sympify("2*x"))
        self.assertEqual(calculate_derivative("sin(x)"), sympify("cos(x)"))
        self.assertEqual(calculate_derivative("cos(x)"), sympify("-sin(x)"))

    def test_calculate_integral(self):
        self.assertEqual(calculate_integral("x**2"), sympify("x**3/3"))
        self.assertEqual(calculate_integral("cos(x)"), sympify("sin(x)"))

    def test_calculate_limit(self):
        self.assertEqual(calculate_limit("1/x", "oo"), sympify("0"))
        self.assertEqual(calculate_limit("sin(x)/x", "0"), sympify("1"))


if __name__ == "__main__":
    unittest.main()
