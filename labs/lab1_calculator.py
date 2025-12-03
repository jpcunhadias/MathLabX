import streamlit as st
from sympy import (
    cos,
    diff,
    integrate,
    latex,
    limit,
    sin,
    symbols,
    sympify,
    tan,
)
from sympy.integrals.manualintegrate import integral_steps


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


def describe_derivative_rules(expr):
    rules = []
    if expr.is_Add:
        rules.append("Linearity: differentiate each term separately.")
    if expr.is_Mul:
        rules.append("Product rule across multiplied factors.")
    if expr.is_Pow:
        rules.append("Power rule on polynomial-like terms.")
    if expr.has(sin, cos, tan):
        rules.append("Standard trigonometric derivatives.")
    if not rules:
        rules.append("Symbolic differentiation with respect to x.")
    return rules


def describe_integral_rules(expr):
    notes = []
    if expr.is_Add:
        notes.append("Split the integral across summed terms (linearity).")
    if expr.is_Mul:
        notes.append("Look for constant factors to pull out or use substitution.")
    if expr.is_Pow:
        notes.append("Apply the power rule where applicable.")
    if expr.has(sin, cos, tan):
        notes.append("Use standard trigonometric antiderivatives.")
    if not notes:
        notes.append("Symbolic integration strategy chosen by SymPy.")
    return notes


def format_integral_steps(expr, variable):
    try:
        steps = integral_steps(expr, variable)
        return summarize_integral_step(steps)
    except Exception:
        return None


def summarize_integral_step(step, depth=0):
    indent = "  " * depth
    label = step.__class__.__name__
    lines = []
    var_latex = latex(getattr(step, "variable", symbols("x")))

    if label == "PartsRule":
        lines.append(
            f"{indent}- Integration by parts with $u = {latex(step.u)}$ and "
            f"$dv = {latex(step.dv)}$."
        )
        if step.v_step:
            lines.extend(summarize_integral_step(step.v_step, depth + 1))
        if step.second_step:
            lines.extend(summarize_integral_step(step.second_step, depth + 1))
    elif label == "ConstantTimesRule":
        lines.append(
            f"{indent}- Pull out constant {latex(step.constant)} and integrate "
            f"{latex(step.other)}."
        )
        if step.substep:
            lines.extend(summarize_integral_step(step.substep, depth + 1))
    elif label == "PowerRule":
        lines.append(
            f"{indent}- Power rule on $({latex(step.base)})^{latex(step.exp)}$."
        )
    elif label == "ExpRule":
        lines.append(
            f"{indent}- Exponential rule for $e^{{{latex(step.exp)}}}$."
        )
    elif label == "SinRule":
        lines.append(
            f"{indent}- Integrate sine: $\\int \\sin({var_latex})\\,d{var_latex} = "
            f"-\\cos({var_latex})$."
        )
    elif label == "CosRule":
        lines.append(
            f"{indent}- Integrate cosine: $\\int \\cos({var_latex})\\,d{var_latex} = "
            f"\\sin({var_latex})$."
        )
    elif label == "URule":
        lines.append(
            f"{indent}- Use substitution with $u = {latex(step.u_func)}$."
        )
        if step.substep:
            lines.extend(summarize_integral_step(step.substep, depth + 1))
    elif label == "AlternativeRule":
        lines.append(
            f"{indent}- Multiple strategies detected; using a workable alternative."
        )
        if step.alternatives:
            lines.extend(summarize_integral_step(step.alternatives[0], depth + 1))
    elif label == "CyclicPartsRule":
        lines.append(f"{indent}- Repeated integration by parts (cyclic).")
        for sub in getattr(step, "parts_rules", []):
            lines.extend(summarize_integral_step(sub, depth + 1))
    else:
        lines.append(f"{indent}- Strategy: {label}.")
    return lines


def run_lab1():
    st.title("Lab 1: Scientific + Symbolic Calculator")

    expr_str = st.text_input(
        "Enter a mathematical expression (e.g., sin(x) + x**2)",
        "sin(x) + x**2",
    )

    show_resolution = st.checkbox(
        "Show resolution steps",
        value=False,
        help="Display the steps SymPy uses to compute results.",
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
                if show_resolution:
                    st.markdown("**Resolution**")
                    for rule in describe_derivative_rules(sympify(expr_str)):
                        st.write(f"- {rule}")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Integrate"):
            try:
                integral = calculate_integral(expr_str)
                st.latex(f"\\int {latex(sympify(expr_str))}\\,dx = {latex(integral)}")
                if show_resolution:
                    st.markdown("**Resolution**")
                    expr = sympify(expr_str)
                    for note in describe_integral_rules(expr):
                        st.write(f"- {note}")
                    pretty_steps = format_integral_steps(expr, symbols("x"))
                    if pretty_steps:
                        st.markdown("\n".join(pretty_steps))
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("---")
        a_str = st.text_input("Value of a for limit (e.g., 0, oo)", "0")
        if st.button("Calculate Limit"):
            try:
                lim = calculate_limit(expr_str, a_str)
                st.latex(
                    f"\\lim_{{x \\to {latex(sympify(a_str))}}} "
                    f"{latex(sympify(expr_str))} = {latex(lim)}"
                )
                if show_resolution:
                    st.markdown("**Resolution**")
                    st.write(
                        "- SymPy evaluates the limit symbolically; if direct "
                        "substitution fails it applies algebraic rewrites and "
                        "L'Hôpital-style steps internally."
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
