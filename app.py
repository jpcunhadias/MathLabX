import streamlit as st

st.set_page_config(layout="wide")

st.sidebar.title("MathLabX")
lab = st.sidebar.radio("Select a Lab:", ("Scientific + Symbolic Calculator", "Single-variable Calculus Visualizer", "Linear Algebra: Vectors & Transformations", "Multivariable Calculus"))

if lab == "Scientific + Symbolic Calculator":
    st.title("Lab 1: Scientific + Symbolic Calculator")

    expr_str = st.text_input("Enter a mathematical expression (e.g., sin(x) + x**2)", "sin(x) + x**2")
    
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Symbolic Operations")
        if st.button("Differentiate"):
            try:
                from sympy import sympify, diff, latex, symbols
                x = symbols('x')
                expr = sympify(expr_str)
                derivative = diff(expr, x)
                st.latex(f"\\frac{{d}}{{dx}}({latex(expr)}) = {latex(derivative)}")
            except Exception as e:
                st.error(f"Error: {e}")

        if st.button("Integrate"):
            try:
                from sympy import sympify, integrate, latex, symbols
                x = symbols('x')
                expr = sympify(expr_str)
                integral = integrate(expr, x)
                st.latex(f"\\int {latex(expr)}\\,dx = {latex(integral)}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("---")
        a_str = st.text_input("Value of a for limit (e.g., 0, oo)", "0")
        if st.button("Calculate Limit"):
            try:
                from sympy import sympify, limit, latex, symbols, oo
                x = symbols('x')
                expr = sympify(expr_str)
                a = sympify(a_str)
                lim = limit(expr, x, a)
                st.latex(f"\\lim_{{x \\to {latex(a)}}}} {latex(expr)} = {latex(lim)}")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.subheader("Numeric Operations")
        x_val_str = st.text_input("Value of x for evaluation", "1.0")
        if st.button("Evaluate at x"):
            try:
                from sympy import sympify, symbols
                x = symbols('x')
                expr = sympify(expr_str)
                x_val = float(x_val_str)
                result = expr.subs(x, x_val)
                st.write(f"Result: `{result}`")
            except Exception as e:
                st.error(f"Error: {e}")
elif lab == "Single-variable Calculus Visualizer":
    st.title("Lab 2: Single-variable Calculus Visualizer")

    sub_lab = st.radio("Select a page:", ("Function & Derivative", "Limits", "Integrals & Area"))

    if sub_lab == "Function & Derivative":
        st.subheader("Function & Derivative")

        f_str = st.text_input("Enter a function f(x)", "x**2")
        x_min_str, x_max_str = st.columns(2)
        x_min = float(x_min_str.text_input("x min", "-5"))
        x_max = float(x_max_str.text_input("x max", "5"))
        x0 = st.slider("Select a point x0", min_value=x_min, max_value=x_max, value=(x_min + x_max) / 2)

        try:
            from sympy import sympify, diff, latex, symbols
            import numpy as np
            import matplotlib.pyplot as plt

            x = symbols('x')
            f = sympify(f_str)
            f_prime = diff(f, x)
            f_prime_x0 = f_prime.subs(x, x0)
            f_x0 = f.subs(x, x0)

            st.latex(f"f'(x) = {latex(f_prime)}")
            st.write(f"Slope at x0={x0}: `f'({x0}) = {f_prime_x0}`")

            x_vals = np.linspace(x_min, x_max, 400)
            y_vals = np.array([f.subs(x, val) for val in x_vals], dtype=float)

            tangent_line = f_prime_x0 * (x_vals - x0) + f_x0

            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label=f"f(x) = {f_str}")
            ax.plot(x_vals, tangent_line, label=f"Tangent at x0={x0}", linestyle='--')
            ax.scatter([x0], [f_x0], color='red')
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

    elif sub_lab == "Limits":
        st.subheader("Limits")

        f_str = st.text_input("Enter a function f(x)", "1/x")
        a_str = st.text_input("Point a", "0")

        h_left = st.slider("h_left (approaching 0 from the left)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        h_right = st.slider("h_right (approaching 0 from the right)", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

        try:
            from sympy import sympify, latex, symbols, limit
            import numpy as np
            import matplotlib.pyplot as plt

            x = symbols('x')
            f = sympify(f_str)
            a = float(a_str)

            f_a_left = f.subs(x, a - h_left)
            f_a_right = f.subs(x, a + h_right)

            st.write(f"f(a - h_left) = f({a - h_left:.2f}) = {f_a_left}")
            st.write(f"f(a + h_right) = f({a + h_right:.2f}) = {f_a_right}")

            left_limit = limit(f, x, a, dir='-')
            right_limit = limit(f, x, a, dir='+')

            st.latex(f"\\lim_{{x \\to {a}^-}} {latex(f)} = {latex(left_limit)}")
            st.latex(f"\\lim_{{x \\to {a}^+}} {latex(f)} = {latex(right_limit)}")

            x_vals = np.linspace(a - 2, a + 2, 400)
            y_vals = np.array([f.subs(x, val) for val in x_vals], dtype=float)
            
            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label=f"f(x) = {f_str}")
            ax.axvline(x=a, color='gray', linestyle='--')
            ax.scatter([a - h_left, a + h_right], [f_a_left, f_a_right], color='red')
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.legend()
            ax.grid(True)
            ax.set_ylim([-10, 10])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

    elif sub_lab == "Integrals & Area":
        st.subheader("Integrals & Area")

        f_str = st.text_input("Enter a function f(x)", "x**2")
        a_str, b_str = st.columns(2)
        a = float(a_str.text_input("a", "0"))
        b = float(b_str.text_input("b", "2"))
        n = st.slider("Number of rectangles (n)", min_value=1, max_value=100, value=10)
        method = st.selectbox("Riemann sum method", ("left", "midpoint", "right"))

        try:
            from sympy import sympify, integrate, latex, symbols
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            x = symbols('x')
            f = sympify(f_str)

            indefinite_integral = integrate(f, x)
            definite_integral = integrate(f, (x, a, b))

            st.latex(f"\\int {latex(f)}\\,dx = {latex(indefinite_integral)}")
            st.latex(f"\\int_{{{a}}}^{{{b}}} {latex(f)}\\,dx = {latex(definite_integral)}")

            x_vals = np.linspace(a, b, 400)
            y_vals = np.array([f.subs(x, val) for val in x_vals], dtype=float)

            fig, ax = plt.subplots()
            ax.plot(x_vals, y_vals, label=f"f(x) = {f_str}")

            dx = (b - a) / n
            riemann_sum = 0
            
            if method == "left":
                points = np.linspace(a, b - dx, n)
            elif method == "midpoint":
                points = np.linspace(a + dx / 2, b - dx / 2, n)
            else: # right
                points = np.linspace(a + dx, b, n)

            for point in points:
                height = f.subs(x, point)
                riemann_sum += height * dx
                ax.add_patch(Rectangle((point - (dx if method == 'right' else 0 if method == 'left' else dx/2)), 0, dx, height, edgecolor='black', facecolor='blue', alpha=0.3))

            st.write(f"Riemann Sum ({method}): `{riemann_sum}`")

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
elif lab == "Linear Algebra: Vectors & Transformations":
    st.title("Lab 3: Linear Algebra: Vectors & Transformations")
    st.write("Work in progress...")
elif lab == "Multivariable Calculus":
    st.title("Lab 4: Multivariable Calculus")
    st.write("Work in progress...")
