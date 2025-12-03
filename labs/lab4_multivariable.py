import streamlit as st
from sympy import sympify, diff, latex, symbols
import numpy as np
import matplotlib.pyplot as plt


def calculate_multivariable_data(
    f_str, x_min, x_max, y_min, y_max, x0, y0
):
    x, y = symbols("x y")
    f = sympify(f_str)

    grad_f = [diff(f, x), diff(f, y)]
    grad_f_x0_y0 = [g.subs([(x, x0), (y, y0)]) for g in grad_f]

    x_vals = np.linspace(x_min, x_max, 100)
    y_vals = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    Z = np.array(
        [[f.subs([(x, xv), (y, yv)]) for xv in x_vals] for yv in y_vals],
        dtype=float,
    )

    return {
        "grad_f": grad_f,
        "grad_f_x0_y0": grad_f_x0_y0,
        "X": X,
        "Y": Y,
        "Z": Z,
    }


def run_lab4():
    st.title("Lab 4: Multivariable Calculus")

    f_str = st.text_input("Enter a function f(x, y)", "sin(x**2 + y**2)")

    x_range_col, y_range_col = st.columns(2)
    x_min = float(x_range_col.text_input("x min", "-3"))
    x_max = float(x_range_col.text_input("x max", "3"))
    y_min = float(y_range_col.text_input("y min", "-3"))
    y_max = float(y_range_col.text_input("y max", "3"))

    x0_col, y0_col = st.columns(2)
    x0 = float(x0_col.text_input("x0", "1"))
    y0 = float(y0_col.text_input("y0", "1"))

    try:
        data = calculate_multivariable_data(
            f_str, x_min, x_max, y_min, y_max, x0, y0
        )

        st.latex(f"\\nabla f(x, y) = {latex(data['grad_f'])}")
        st.write(f"Gradient at ({x0}, {y0}): `{data['grad_f_x0_y0']}`")

        # 3D Surface Plot
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        ax_3d.plot_surface(
            data["X"], data["Y"], data["Z"], cmap="viridis"
        )
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("f(x, y)")
        st.pyplot(fig_3d)

        # Contour Plot
        fig_contour, ax_contour = plt.subplots()
        cp = ax_contour.contour(
            data["X"], data["Y"], data["Z"], 20, cmap="viridis"
        )
        plt.colorbar(cp)
        ax_contour.set_xlabel("x")
        ax_contour.set_ylabel("y")

        # Plot gradient
        ax_contour.quiver(
            x0,
            y0,
            data["grad_f_x0_y0"][0],
            data["grad_f_x0_y0"][1],
            color="r",
            scale=10,
        )
        st.pyplot(fig_contour)

    except Exception as e:
        st.error(f"Error: {e}")
