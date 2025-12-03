import unittest
import sys
from unittest.mock import MagicMock
import numpy as np
from labs.lab3_linear_algebra import (
    calculate_2d_vectors_data,
    calculate_matrix_transformation_data,
)

# Mock the streamlit and matplotlib modules
sys.modules["streamlit"] = MagicMock()
sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.patches"] = MagicMock()


class TestLab3LinearAlgebra(unittest.TestCase):

    def test_calculate_2d_vectors_data(self):
        data = calculate_2d_vectors_data(1, 2, 3, 1)
        self.assertTrue(np.array_equal(data["v"], np.array([1, 2])))
        self.assertTrue(np.array_equal(data["w"], np.array([3, 1])))
        self.assertTrue(np.array_equal(data["v_plus_w"], np.array([4, 3])))

    def test_calculate_matrix_transformation_data(self):
        A = np.array([[1, 1], [0, 1]])
        data = calculate_matrix_transformation_data(A)
        expected_transformed_square = np.array(
            [[0, 0], [1, 0], [2, 1], [1, 1]]
        )
        self.assertTrue(
            np.array_equal(
                data["transformed_square"], expected_transformed_square
            )
        )
        self.assertTrue(
            np.allclose(data["eigenvalues"], np.array([1.0, 1.0]))
        )


if __name__ == "__main__":
    unittest.main()
