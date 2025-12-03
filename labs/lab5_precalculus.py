import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sympy import (
    Abs,
    Eq,
    Poly,
    Symbol,
    cancel,
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


def analyze_rational(numerator_str, denominator_str):
    """Analyze rational function features: asymptotes, holes, and simplification."""
    x = Symbol("x")
    num = sympify(numerator_str)
    den = sympify(denominator_str)
    if den == 0:
        raise ValueError("Denominator cannot be zero.")

    num_poly = Poly(num, x)
    den_poly = Poly(den, x)
    common_poly = num_poly.gcd(den_poly)
    holes = solve(Eq(common_poly.as_expr(), 0), x) if common_poly.degree() >= 1 else []

    cancelled = cancel(num / den)
    num_simplified, den_simplified = cancelled.as_numer_denom()

    vertical_asymptotes = solve(Eq(den_simplified, 0), x)

    p_num = Poly(num_simplified, x)
    p_den = Poly(den_simplified, x)
    deg_num = p_num.degree()
    deg_den = p_den.degree()

    horizontal_asymptote = None
    oblique_asymptote = None
    if deg_num < deg_den:
        horizontal_asymptote = sympify(0)
    elif deg_num == deg_den:
        horizontal_asymptote = p_num.LC() / p_den.LC()
    elif deg_den > 0 and deg_num == deg_den + 1:
        quotient = p_num.quo(p_den)
        oblique_asymptote = quotient.as_expr()

    simplified_expr = num_simplified / den_simplified
    return {
        "holes": holes,
        "vertical_asymptotes": vertical_asymptotes,
        "horizontal_asymptote": horizontal_asymptote,
        "oblique_asymptote": oblique_asymptote,
        "simplified_expr": simplified_expr,
    }


def sign_samples(rational_expr, critical_points, x_min=-8.0, x_max=8.0):
    """Sample sign on intervals defined by critical points (asymptotes/holes)."""
    x = Symbol("x")
    fn = lambdify(x, rational_expr, "numpy")
    sorted_points = sorted(set(critical_points))
    intervals = []
    boundaries = [x_min] + sorted_points + [x_max]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        mid = (left + right) / 2
        try:
            val = fn(mid)
            sign = np.sign(val)
        except Exception:
            sign = np.nan
        intervals.append({"interval": (left, right), "sign": sign})
    return intervals


def run_lab5():
    st.title("Precalculus")

    # Shared plot styling
    def base_figure():
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#d1d5db"),
            yaxis=dict(showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#d1d5db"),
        )
        return fig

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

    fig = base_figure()
    fig.add_scatter(
        x=x_vals,
        y=base_fn(x_vals),
        name=f"f(x) = {base_choice}",
        line=dict(color="#9ca3af", dash="dash"),
    )
    fig.add_scatter(
        x=x_vals,
        y=transformed_fn(x_vals),
        name=rf"g(x) = {a}·f({b}(x - {c})) + {d}",
        line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
    )
    fig.add_shape(type="line", x0=c, x1=c, y0=min(transformed_fn(x_vals)), y1=max(transformed_fn(x_vals)), line=dict(color="#d1d5db", dash="dot"))
    fig.update_layout(title="Transformations", xaxis_title="x", yaxis_title="y")
    st.latex(f"g(x) = {transformed_expr}")
    st.plotly_chart(fig, width=900)

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
        fig_pw = base_figure()
        fig_pw.add_scatter(
            x=x_vals_piecewise,
            y=y_vals_piecewise,
            mode="lines",
            name="Piecewise",
            line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
        )
        fig_pw.add_shape(type="line", x0=breakpoint, x1=breakpoint, y0=min(y_vals_piecewise), y1=max(y_vals_piecewise), line=dict(color="red", dash="dash"))
        fig_pw.update_layout(title="Piecewise Function", xaxis_title="x", yaxis_title="y")
        st.plotly_chart(fig_pw, width=900)
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
    fig_abs = base_figure()
    fig_abs.add_scatter(
        x=x_vals_abs,
        y=abs_fn(x_vals_abs),
        mode="lines",
        name="Absolute",
        line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
    )
    fig_abs.add_shape(type="line", x0=x_vals_abs.min(), x1=x_vals_abs.max(), y0=line_k, y1=line_k, line=dict(color="red", dash="dash"))
    fig_abs.update_layout(title=r"$a|x - b| + c$", xaxis_title="x", yaxis_title="y")
    st.latex(f"y = {abs_expr}")
    st.plotly_chart(fig_abs, width=900)

    st.subheader("Inequality solver (single variable)")
    inequality_str = st.text_input("Enter inequality (e.g., x**2 - 4 <= 0)", "x**2 - 4 <= 0")
    try:
        solution = solve_scalar_inequality(inequality_str)
        st.latex(f"Solution: {solution}")
    except Exception as e:
        st.error(f"Could not solve inequality: {e}")

    st.header("Rational Functions")
    st.write("Analyze asymptotes, holes, and sign behavior of f(x) = p(x) / q(x).")
    numerator_str = st.text_input("Numerator p(x)", "x**2 - 1")
    denominator_str = st.text_input("Denominator q(x)", "x - 1")
    try:
        analysis = analyze_rational(numerator_str, denominator_str)
        st.latex(f"f(x) = {analysis['simplified_expr']}")
        if analysis["holes"]:
            st.write(f"Holes at x = {analysis['holes']}")
        if analysis["vertical_asymptotes"]:
            st.write(f"Vertical asymptotes at x = {analysis['vertical_asymptotes']}")
        if analysis["horizontal_asymptote"] is not None:
            st.write(f"Horizontal asymptote: y = {analysis['horizontal_asymptote']}")
        if analysis["oblique_asymptote"] is not None:
            st.write(f"Oblique asymptote: y = {analysis['oblique_asymptote']}")

        x_vals = np.linspace(-8, 8, 800)
        x = Symbol("x")
        fn = lambdify(x, analysis["simplified_expr"], "numpy")
        y_vals = fn(x_vals)

        # Mask near holes and vertical asymptotes for plot clarity
        for a in analysis["holes"] + analysis["vertical_asymptotes"]:
            y_vals[np.abs(x_vals - float(a)) < 0.05] = np.nan

        fig_r = base_figure()
        fig_r.add_scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name="f(x)",
            line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
        )
        for va in analysis["vertical_asymptotes"]:
            fig_r.add_shape(
                type="line",
                x0=float(va),
                x1=float(va),
                y0=-10,
                y1=10,
                line=dict(color="red", dash="dash"),
            )
        if analysis["horizontal_asymptote"] is not None:
            fig_r.add_shape(
                type="line",
                x0=x_vals.min(),
                x1=x_vals.max(),
                y0=float(analysis["horizontal_asymptote"]),
                y1=float(analysis["horizontal_asymptote"]),
                line=dict(color="#666666", dash="dot"),
            )
        if analysis["oblique_asymptote"] is not None:
            oa_fn = lambdify(x, analysis["oblique_asymptote"], "numpy")
            fig_r.add_scatter(
                x=x_vals,
                y=oa_fn(x_vals),
                mode="lines",
                name="Oblique asymptote",
                line=dict(color="#999999", dash="dot"),
            )
        for hole in analysis["holes"]:
            try:
                y_hole = fn(float(hole))
                fig_r.add_scatter(
                    x=[float(hole)],
                    y=[y_hole],
                    mode="markers",
                    marker=dict(color="red", size=9, symbol="circle-open"),
                    name="Hole",
                    showlegend=False,
                )
            except Exception:
                pass

        fig_r.update_layout(
            title="Rational Function",
            xaxis_title="x",
            yaxis_title="f(x)",
            yaxis=dict(range=[-10, 10]),
        )
        st.plotly_chart(fig_r, width=900)

        # Sign chart sampling
        crit_points = [float(p) for p in analysis["holes"] + analysis["vertical_asymptotes"]]
        sign_info = sign_samples(analysis["simplified_expr"], crit_points)
        st.subheader("Sign by interval")
        for entry in sign_info:
            left, right = entry["interval"]
            sign = entry["sign"]
            label = "positive" if sign > 0 else "negative" if sign < 0 else "undefined"
            st.write(f"{left:.1f} to {right:.1f}: {label}")

    except Exception as e:
        st.error(f"Error analyzing rational function: {e}")

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

        fig_poly = base_figure()
        fig_poly.add_scatter(
            x=x,
            y=y,
            mode="lines",
            name="p(x)",
            line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
        )
        fig_poly.update_layout(title="Polynomial Plot", xaxis_title="x", yaxis_title="y")
        st.plotly_chart(fig_poly, width=900)

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

    fig_trig = base_figure()
    fig_trig.add_scatter(
        x=x,
        y=y,
        mode="lines",
        name=f"{trig_function}(x)",
        line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
    )
    fig_trig.add_shape(
        type="line",
        x0=x.min(),
        x1=x.max(),
        y0=c,
        y1=c,
        line=dict(color="red", dash="dash"),
    )

    # Find intersection points
    intersections_x = []
    intersections_y = []
    for i in range(len(x) - 1):
        if (y[i] - c) * (y[i + 1] - c) < 0:
            intersections_x.append((x[i] + x[i + 1]) / 2)
            intersections_y.append(c)

    if intersections_x:
        fig_trig.add_scatter(
            x=intersections_x,
            y=intersections_y,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Intersections",
        )
        st.write("Intersection points (approximate):")
        for ix, iy in zip(intersections_x, intersections_y):
            st.write(f"  ({ix:.2f}, {iy:.2f})")

    fig_trig.update_layout(
        title=f"{amplitude} * {trig_function}({frequency} * (x + {phase_shift:.2f}))",
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(range=[-5, 5]),
    )
    st.plotly_chart(fig_trig, width=900)

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

    fig_exp = base_figure()
    fig_exp.add_scatter(
        x=x,
        y=y,
        mode="lines",
        name=exp_log_function,
        line=dict(color=config.DEFAULT_PLOT_COLOR, width=3),
    )
    fig_exp.update_layout(title=title, xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig_exp, width=900)
