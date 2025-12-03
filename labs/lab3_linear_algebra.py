import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def calculate_2d_vectors_data(v_x, v_y, w_x, w_y):
    v = np.array([v_x, v_y])
    w = np.array([w_x, w_y])
    v_plus_w = v + w
    return {"v": v, "w": w, "v_plus_w": v_plus_w}

def calculate_matrix_transformation_data(A):
    # Unit square vertices
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # Transformed square
    transformed_square = unit_square @ A.T

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = None, None

    return {
        "unit_square": unit_square,
        "transformed_square": transformed_square,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }

def run_lab3():
    st.title("Lab 3: Linear Algebra: Vectors & Transformations")

    sub_lab = st.radio("Select a page:", ("2D Vectors", "Matrix as Transformation"))

    if sub_lab == "2D Vectors":
        st.subheader("2D Vectors")

        v_x_str, v_y_str = st.columns(2)
        v_x = float(v_x_str.text_input("Vector v: x", "1"))
        v_y = float(v_y_str.text_input("Vector v: y", "2"))
        
        w_x_str, w_y_str = st.columns(2)
        w_x = float(w_x_str.text_input("Vector w: x", "3"))
        w_y = float(w_y_str.text_input("Vector w: y", "1"))

        show_basis = st.checkbox("Show basis vectors (e1, e2)")

        try:
            vector_data = calculate_2d_vectors_data(v_x, v_y, w_x, w_y)
            v = vector_data["v"]
            w = vector_data["w"]
            v_plus_w = vector_data["v_plus_w"]

            fig, ax = plt.subplots()

            # Plot vectors
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='v')
            ax.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='b', label='w')
            ax.quiver(0, 0, v_plus_w[0], v_plus_w[1], angles='xy', scale_units='xy', scale=1, color='g', label='v+w')

            if show_basis:
                ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='gray', label='e1', linestyle='--')
                ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='gray', label='e2', linestyle='--')

            ax.set_xlim([-max(abs(v_x), abs(w_x), abs(v_plus_w[0]))-1, max(abs(v_x), abs(w_x), abs(v_plus_w[0]))+1])
            ax.set_ylim([-max(abs(v_y), abs(w_y), abs(v_plus_w[1]))-1, max(abs(v_y), abs(w_y), abs(v_plus_w[1]))+1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvline(0, color='grey', lw=0.5)
            ax.grid(True)
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

    elif sub_lab == "Matrix as Transformation":
        st.subheader("Matrix as Transformation")

        st.write("Enter the 2x2 matrix A:")
        a11, a12 = st.columns(2)
        a11 = float(a11.text_input("a11", "1"))
        a12 = float(a12.text_input("a12", "1"))
        a21, a22 = st.columns(2)
        a21 = float(a21.text_input("a21", "0"))
        a22 = float(a22.text_input("a22", "1"))

        A = np.array([[a11, a12], [a21, a22]])

        try:
            data = calculate_matrix_transformation_data(A)
            unit_square = data["unit_square"]
            transformed_square = data["transformed_square"]

            fig, ax = plt.subplots()

            # Plot original and transformed squares
            ax.add_patch(Polygon(unit_square, closed=True, fill=True, color='lightblue', label='Unit Square'))
            ax.add_patch(Polygon(transformed_square, closed=True, fill=True, color='lightcoral', alpha=0.7, label='Transformed'))

            # Plot basis vectors
            ax.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='gray', linestyle='--')
            ax.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='gray', linestyle='--')
            
            # Plot transformed basis vectors
            ax.quiver(0, 0, A[0,0], A[1,0], angles='xy', scale_units='xy', scale=1, color='r', label='T(e1)')
            ax.quiver(0, 0, A[0,1], A[1,1], angles='xy', scale_units='xy', scale=1, color='b', label='T(e2)')

            all_points = np.vstack([unit_square, transformed_square])
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)

            ax.set_xlim([min(x_min, -1) - 1, max(x_max, 1) + 1])
            ax.set_ylim([min(y_min, -1) - 1, max(y_max, 1) + 1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axhline(0, color='grey', lw=0.5)
            ax.axvline(0, color='grey', lw=0.5)
            ax.grid(True)
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            st.pyplot(fig)
            
            if data["eigenvalues"] is not None:
                st.write("Eigenvalues:", data["eigenvalues"])
                st.write("Eigenvectors:")
                st.write(data["eigenvectors"])
            else:
                st.write("Matrix has no real eigenvalues/eigenvectors.")

        except Exception as e:
            st.error(f"Error: {e}")
