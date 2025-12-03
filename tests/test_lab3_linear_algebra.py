import numpy as np
from labs.lab3_linear_algebra import (
    calculate_2d_vectors_data,
    calculate_matrix_transformation_data,
)

def test_calculate_2d_vectors_data():
    data = calculate_2d_vectors_data(1, 2, 3, 1)
    assert np.array_equal(data["v"], np.array([1, 2]))
    assert np.array_equal(data["w"], np.array([3, 1]))
    assert np.array_equal(data["v_plus_w"], np.array([4, 3]))


def test_calculate_matrix_transformation_data():
    A = np.array([[1, 1], [0, 1]])
    data = calculate_matrix_transformation_data(A)
    expected_transformed_square = np.array([[0, 0], [1, 0], [2, 1], [1, 1]])
    assert np.array_equal(data["transformed_square"], expected_transformed_square)
    assert np.allclose(data["eigenvalues"], np.array([1.0, 1.0]))
