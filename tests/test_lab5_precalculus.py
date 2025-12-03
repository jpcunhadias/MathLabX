import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from sympy import Interval, sympify

# Mock heavy UI dependencies to import the lab module safely
sys.modules["streamlit"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()

from labs.lab5_precalculus import (
    build_transformed_expression,
    compose_functions,
    invert_function,
    piecewise_values,
    analyze_rational,
    sign_samples,
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

    def test_analyze_rational(self):
        analysis = analyze_rational("x**2 - 1", "x - 1")
        self.assertIn(sympify("1"), analysis["holes"])
        self.assertEqual(analysis["vertical_asymptotes"], [])
        self.assertIsNone(analysis["horizontal_asymptote"])
        self.assertIsNone(analysis["oblique_asymptote"])

    def test_sign_samples(self):
        expr = sympify("1/(x - 1)")
        samples = sign_samples(expr, [1], x_min=-3, x_max=3)
        labels = [s["sign"] for s in samples]
        self.assertLess(labels[0], 0)
        self.assertGreater(labels[1], 0)

    def test_oblique_detection(self):
        analysis = analyze_rational("x**2 + 1", "x")
        self.assertIsNone(analysis["horizontal_asymptote"])
        self.assertEqual(sympify("x"), analysis["oblique_asymptote"])


if __name__ == "__main__":
    unittest.main()
