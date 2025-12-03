import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
from sympy import sympify
from labs.lab2_calculus import calculate_derivative_data

# Mock the streamlit and matplotlib modules
sys.modules["streamlit"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()


class TestLab2Calculus(unittest.TestCase):
    def test_calculate_derivative_data(self):
        x0 = 1
        data = calculate_derivative_data("x**2", -5, 5, x0)
        self.assertEqual(data["f_prime"], sympify("2*x"))
        self.assertEqual(data["f_prime_x0"], sympify("2"))
        self.assertEqual(data["f_x0"], sympify("1"))
        self.assertEqual(len(data["x_vals"]), 1000)
        self.assertEqual(len(data["y_vals"]), 1000)
        self.assertEqual(len(data["tangent_line"]), 1000)

        # Find the index of the x_vals array that is closest to x0
        x0_index = np.argmin(np.abs(data["x_vals"] - x0))
        # Check that the value of the tangent line at x0 is equal to f(x0)
        self.assertAlmostEqual(
            data["tangent_line"][x0_index], float(data["f_x0"]), delta=0.01
        )


if __name__ == "__main__":
    unittest.main()
