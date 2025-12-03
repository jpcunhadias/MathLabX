import unittest
import sys
from unittest.mock import MagicMock
from sympy import sympify, symbols
from labs.lab4_multivariable import calculate_multivariable_data

# Mock the streamlit and matplotlib modules
sys.modules["streamlit"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["mpl_toolkits.mplot3d"] = MagicMock()


class TestLab4Multivariable(unittest.TestCase):

    def test_calculate_multivariable_data(self):
        data = calculate_multivariable_data("x**2 + y**2", -3, 3, -3, 3, 1, 1)
        x, y = symbols("x y")
        self.assertEqual(data["grad_f"][0], sympify("2*x"))
        self.assertEqual(data["grad_f"][1], sympify("2*y"))
        self.assertEqual(data["grad_f_x0_y0"][0], sympify("2"))
        self.assertEqual(data["grad_f_x0_y0"][1], sympify("2"))
        self.assertEqual(data["X"].shape, (100, 100))
        self.assertEqual(data["Y"].shape, (100, 100))
        self.assertEqual(data["Z"].shape, (100, 100))


if __name__ == "__main__":
    unittest.main()
