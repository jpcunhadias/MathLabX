import streamlit as st
from sympy import sympify, diff, latex, symbols, integrate, limit


def calculate_derivative(expr_str):
    x = symbols("x")
    expr = sympify(expr_str)
    return diff(expr, x)


def calculate_integral(expr_str):
    x = symbols("x")
    expr = sympify(expr_str)
    return integrate(expr, x)


def calculate_limit(expr_str, a_str):
    x = symbols("x")
    expr = sympify(expr_str)
    a = sympify(a_str)
    return limit(expr, x, a)


def run_lab1():
    st.title("Lab 1: Scientific + Symbolic Calculator")

    expr_str = st.text_input(
        "Enter a mathematical expression (e.g., sin(x) + x**2)",
        "sin(x) + x**2",
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Symbolic Operations")
        if st.button("Differentiate"):
            try:
                derivative = calculate_derivative(expr_str)
                st.latex(
                    f"\\frac{{d}}{{dx}}({latex(sympify(expr_str))}) = "
                    f"{latex(derivative)}"
                )
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Integrate"):
            try:
                integral = calculate_integral(expr_str)
                st.latex(
                                    f"\\int {latex(sympify(expr_str))}\\,dx = "
                                    f"{latex(integral)}"
                                )
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        a_str = st.text_input("Value of a for limit (e.g., 0, oo)", "0")
        if st.button("Calculate Limit"):
            try:
                lim = calculate_limit(expr_str, a_str)
                st.latex(
                    f"\\lim_{{x \\to {latex(sympify(a_str))}}}}} "
                    f"{latex(sympify(expr_str))} = {latex(lim)}"
                )
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Numeric Operations")
        x_val_str = st.text_input("Value of x for evaluation", "1.0")
        if st.button("Evaluate at x"):
            try:
                x = symbols("x")
                expr = sympify(expr_str)
                x_val = float(x_val_str)
                result = expr.subs(x, x_val)
                st.write(f"Result: `{result}`")
            except Exception as e:
                st.error(f"Error: {e}")
