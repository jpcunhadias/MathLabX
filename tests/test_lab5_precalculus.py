import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from sympy import Interval, sympify

# Mock streamlit to import the lab module safely
sys.modules["streamlit"] = MagicMock()

from labs.lab5_precalculus import (
    build_transformed_expression,
    compose_functions,
    invert_function,
    piecewise_values,
    solve_scalar_inequality,
)


class TestLab5Precalculus(unittest.TestCase):
    def test_build_transformed_expression(self):
        expr = build_transformed_expression("x**2", a=2, b=3, c=1, d=-4)
        self.assertEqual(expr.subs({"x": 2}), 14)

    def test_compose_and_inverse(self):
        f_of_g, g_of_f = compose_functions("x**2 + 1", "2*x")
        self.assertEqual(sympify("4*x**2 + 1"), f_of_g)
        self.assertEqual(sympify("2*x**2 + 2"), g_of_f)
        inv_f = invert_function("2*x + 3")
        self.assertEqual(sympify("(x - 3)/2"), inv_f)

    def test_piecewise_values(self):
        x_vals = np.array([-1.0, 1.0])
        y_vals = piecewise_values("x", "-x", 0, x_vals)
        np.testing.assert_allclose(y_vals, [-1.0, -1.0])

    def test_solve_scalar_inequality(self):
        solution = solve_scalar_inequality("x**2 - 4 <= 0")
        self.assertEqual(solution, Interval(-2, 2))


if __name__ == "__main__":
    unittest.main()
