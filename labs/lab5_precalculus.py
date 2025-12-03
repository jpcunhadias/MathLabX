import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sympy import (
    Abs,
    Eq,
    Symbol,
    lambdify,
    solve,
    solve_univariate_inequality,
    sympify,
)

import config

BASE_FUNCTIONS = {
    "x^2": "x**2",
    "|x|": "Abs(x)",
    "sqrt(x)": "sqrt(x)",
    "1/x": "1/x",
    "sin(x)": "sin(x)",
    "cos(x)": "cos(x)",
}


def build_transformed_expression(base_expr_str, a, b, c, d):
    """Return transformed expression a * f(b * (x - c)) + d."""
    x = Symbol("x")
    base_expr = sympify(base_expr_str)
    return a * base_expr.subs(x, b * (x - c)) + d


def compose_functions(f_str, g_str):
    x = Symbol("x")
    f_expr = sympify(f_str)
    g_expr = sympify(g_str)
    return f_expr.subs(x, g_expr), g_expr.subs(x, f_expr)


def invert_function(f_str):
    x = Symbol("x")
    y = Symbol("y")
    f_expr = sympify(f_str)
    solutions = solve(Eq(f_expr, y), x)
    return solutions[0].subs(y, x) if solutions else None


def solve_scalar_inequality(inequality_str):
    x = Symbol("x")
    for op in ("<=", ">=", "<", ">"):
        if op in inequality_str:
            left, right = inequality_str.split(op)
            left_expr = sympify(left)
            right_expr = sympify(right)
            if op == "<=":
                relation = left_expr <= right_expr
            elif op == ">=":
                relation = left_expr >= right_expr
            elif op == "<":
                relation = left_expr < right_expr
            else:
                relation = left_expr > right_expr
            return solve_univariate_inequality(relation, x, relational=False)
    raise ValueError("Inequality must contain one of: <=, >=, <, >")


def piecewise_values(expr_left, expr_right, breakpoint, x_vals):
    x = Symbol("x")
    left_fn = lambdify(x, sympify(expr_left), "numpy")
    right_fn = lambdify(x, sympify(expr_right), "numpy")
    y_vals = np.where(x_vals < breakpoint, left_fn(x_vals), right_fn(x_vals))
    return y_vals


def run_lab5():
    st.title("Precalculus")

    st.header("Function Transformations")
    base_choice = st.selectbox("Base function", list(BASE_FUNCTIONS.keys()))
    a = st.slider("Vertical scale (a)", -3.0, 3.0, 1.0, 0.1)
    b = st.slider("Horizontal scale (b)", 0.1, 3.0, 1.0, 0.1)
    c = st.slider("Horizontal shift (c)", -5.0, 5.0, 0.0, 0.5)
    d = st.slider("Vertical shift (d)", -5.0, 5.0, 0.0, 0.5)

    base_expr_str = BASE_FUNCTIONS[base_choice]
    transformed_expr = build_transformed_expression(base_expr_str, a, b, c, d)

    x_range = (-10, 10)
    if base_choice == "sqrt(x)":
        x_vals = np.linspace(0, 10, 400)
    elif base_choice == "1/x":
        x_vals = np.concatenate(
            [np.linspace(-10, -0.1, 200), np.linspace(0.1, 10, 200)]
        )
    else:
        x_vals = np.linspace(*x_range, 400)

    x = Symbol("x")
    base_fn = lambdify(x, sympify(base_expr_str), "numpy")
    transformed_fn = lambdify(x, transformed_expr, "numpy")

    fig, ax = plt.subplots()
    ax.plot(
        x_vals,
        base_fn(x_vals),
        label=f"f(x) = {base_choice}",
        linestyle="--",
        color="#999999",
    )
    ax.plot(
        x_vals,
        transformed_fn(x_vals),
        label=rf"g(x) = {a}·f({b}(x - {c})) + {d}",
        color=config.DEFAULT_PLOT_COLOR,
        linewidth=config.DEFAULT_LINE_WIDTH,
    )
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(c, color="#cccccc", linewidth=0.8, linestyle=":")
    ax.set_title("Transformations")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
    ax.legend()
    st.latex(f"g(x) = {transformed_expr}")
    st.pyplot(fig)

    st.header("Composition and Inverses")
    f_str = st.text_input("f(x) =", "x**2 + 1")
    g_str = st.text_input("g(x) =", "2*x - 3")
    try:
        f_of_g, g_of_f = compose_functions(f_str, g_str)
        st.latex(f"f(g(x)) = {f_of_g}")
        st.latex(f"g(f(x)) = {g_of_f}")

        inv_f = invert_function(f_str)
        if inv_f is not None:
            st.latex(f"f^{{-1}}(x) = {inv_f}")
        else:
            st.info("Could not find an inverse symbolically for f(x).")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.header("Piecewise, Absolute Value, and Inequalities")
    st.subheader("Piecewise example")
    expr_left = st.text_input("f1(x) for x < breakpoint", "-x - 1")
    expr_right = st.text_input("f2(x) for x ≥ breakpoint", "0.5*x + 2")
    breakpoint = st.slider("Breakpoint", -5.0, 5.0, 0.0, 0.5)
    x_vals_piecewise = np.linspace(-10, 10, 400)
    try:
        y_vals_piecewise = piecewise_values(
            expr_left, expr_right, breakpoint, x_vals_piecewise
        )
        fig_pw, ax_pw = plt.subplots()
        ax_pw.plot(
            x_vals_piecewise,
            y_vals_piecewise,
            color=config.DEFAULT_PLOT_COLOR,
            linewidth=config.DEFAULT_LINE_WIDTH,
        )
        ax_pw.axvline(breakpoint, color="red", linestyle="--", linewidth=1.0)
        ax_pw.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
        ax_pw.set_title("Piecewise Function")
        st.pyplot(fig_pw)
    except Exception as e:
        st.error(f"Error in piecewise setup: {e}")

    st.subheader("Absolute value with horizontal line")
    abs_a = st.slider("a (scale)", -3.0, 3.0, 1.0, 0.1, key="abs_a")
    abs_b = st.slider("b (shift)", -5.0, 5.0, 0.0, 0.5, key="abs_b")
    abs_c = st.slider("c (vertical shift)", -5.0, 5.0, 0.0, 0.5, key="abs_c")
    line_k = st.slider("y = k", -5.0, 5.0, 0.0, 0.5, key="line_k")
    x_vals_abs = np.linspace(-10, 10, 400)
    abs_expr = abs_a * Abs(Symbol("x") - abs_b) + abs_c
    abs_fn = lambdify(Symbol("x"), abs_expr, "numpy")
    fig_abs, ax_abs = plt.subplots()
    ax_abs.plot(x_vals_abs, abs_fn(x_vals_abs), color=config.DEFAULT_PLOT_COLOR)
    ax_abs.axhline(line_k, color="red", linestyle="--")
    ax_abs.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
    ax_abs.set_title(r"$a|x - b| + c$")
    st.latex(f"y = {abs_expr}")
    st.pyplot(fig_abs)

    st.subheader("Inequality solver (single variable)")
    inequality_str = st.text_input("Enter inequality (e.g., x**2 - 4 <= 0)", "x**2 - 4 <= 0")
    try:
        solution = solve_scalar_inequality(inequality_str)
        st.latex(f"Solution: {solution}")
    except Exception as e:
        st.error(f"Could not solve inequality: {e}")

    st.header("Polynomials")

    st.write(
        "Enter the coefficients of the polynomial, separated by commas. "
        "For example, for the polynomial x^2 - 2x + 1, enter '1, -2, 1'."
    )
    coefficients = st.text_input("Coefficients:", config.DEFAULT_POLY_COEFFS)

    try:
        # Convert the string of coefficients to a list of floats
        coeffs = [float(c.strip()) for c in coefficients.split(",")]

        # Create a numpy polynomial object
        p = np.poly1d(coeffs)

        # Create a symbolic expression for printing
        x_sym = Symbol("x")
        poly_expr = sum(c * x_sym**i for i, c in enumerate(reversed(coeffs)))

        st.latex(f"p(x) = {poly_expr}")

        # Plot the polynomial
        x = np.linspace(-10, 10, 400)
        y = p(x)

        fig, ax = plt.subplots()
        ax.plot(
            x,
            y,
            color=config.DEFAULT_PLOT_COLOR,
            linewidth=config.DEFAULT_LINE_WIDTH,
        )
        ax.set_title("Polynomial Plot")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
        st.pyplot(fig)

        # Find and display the roots
        roots = np.roots(coeffs)
        st.write("Roots:", roots)

    except Exception as e:
        st.error(f"An error occurred: {e}")

    st.header("Trigonometry")

    trig_function = st.selectbox(
        "Select a function:",
        ["sin", "cos", "tan"],
        index=["sin", "cos", "tan"].index(config.DEFAULT_TRIG_FUNCTION),
    )
    amplitude = st.slider("Amplitude", -5.0, 5.0, config.DEFAULT_AMPLITUDE)
    frequency = st.slider("Frequency", 0.1, 10.0, config.DEFAULT_FREQUENCY)
    phase_shift = st.slider("Phase Shift", -np.pi, np.pi, config.DEFAULT_PHASE_SHIFT)
    c = st.slider("y = c", -5.0, 5.0, 0.5)

    x = np.linspace(-2 * np.pi, 2 * np.pi, 400)

    if trig_function == "sin":
        y = amplitude * np.sin(frequency * (x + phase_shift))
    elif trig_function == "cos":
        y = amplitude * np.cos(frequency * (x + phase_shift))
    elif trig_function == "tan":
        y = amplitude * np.tan(frequency * (x + phase_shift))
        # For tan, we need to handle the vertical asymptotes
        y[np.abs(y) > 20] = np.nan

    fig, ax = plt.subplots()
    ax.plot(
        x,
        y,
        color=config.DEFAULT_PLOT_COLOR,
        linewidth=config.DEFAULT_LINE_WIDTH,
    )
    ax.axhline(y=c, color='r', linestyle='--')

    # Find intersection points
    intersections_x = []
    intersections_y = []
    for i in range(len(x) - 1):
        if (y[i] - c) * (y[i+1] - c) < 0:
            intersections_x.append((x[i] + x[i+1]) / 2)
            intersections_y.append(c)
    
    if intersections_x:
        ax.scatter(intersections_x, intersections_y, color='red', zorder=5)
        st.write("Intersection points (approximate):")
        for i, (ix, iy) in enumerate(zip(intersections_x, intersections_y)):
            st.write(f"  ({ix:.2f}, {iy:.2f})")


    ax.set_title(
        f"{amplitude} * {trig_function}({frequency} * (x + {phase_shift:.2f}))"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-5, 5)
    ax.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
    st.pyplot(fig)

    st.header("Exponentials and Logarithms")

    exp_log_function = st.selectbox(
        "Select a function:",
        ["Exponential", "Logarithmic"],
        index=["Exponential", "Logarithmic"].index(config.DEFAULT_EXP_LOG_FUNCTION),
    )
    use_e = st.checkbox("Use Euler's number (e) as base")

    if use_e:
        base = np.e
        st.slider("Base", 0.1, 10.0, float(base), disabled=True)
    else:
        base = st.slider("Base", 0.1, 10.0, config.DEFAULT_BASE)

    x = np.linspace(0.01, 10, 400)

    if exp_log_function == "Exponential":
        y = base**x
        title = f"${base:.2f}^x$"
        if use_e:
            title = "$e^x$"
    elif exp_log_function == "Logarithmic":
        y = np.log(x) / np.log(base)
        title = f"$\\log_{{{base:.2f}}}(x)$"
        if use_e:
            title = "$\\ln(x)$"

    fig, ax = plt.subplots()
    ax.plot(
        x,
        y,
        color=config.DEFAULT_PLOT_COLOR,
        linewidth=config.DEFAULT_LINE_WIDTH,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
    st.pyplot(fig)
