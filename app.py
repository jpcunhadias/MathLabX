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
            import numpy as np
            import matplotlib.pyplot as plt

            v = np.array([v_x, v_y])
            w = np.array([w_x, w_y])
            v_plus_w = v + w

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
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon

            # Unit square vertices
            unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            # Transformed square
            transformed_square = unit_square @ A.T

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
            
            try:
                eigenvalues, eigenvectors = np.linalg.eig(A)
                st.write("Eigenvalues:", eigenvalues)
                st.write("Eigenvectors:")
                st.write(eigenvectors)
            except np.linalg.LinAlgError:
                st.write("Matrix has no real eigenvalues/eigenvectors.")


        except Exception as e:
            st.error(f"Error: {e}")
elif lab == "Multivariable Calculus":
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
        from sympy import sympify, diff, latex, symbols
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        x, y = symbols('x y')
        f = sympify(f_str)
        
        grad_f = [diff(f, x), diff(f, y)]
        grad_f_x0_y0 = [g.subs([(x, x0), (y, y0)]) for g in grad_f]

        st.latex(f"\\nabla f(x, y) = {latex(grad_f)}")
        st.write(f"Gradient at ({x0}, {y0}): `{grad_f_x0_y0}`")

        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        Z = np.array([[f.subs([(x, xv), (y, yv)]) for xv in x_vals] for yv in y_vals], dtype=float)

        # 3D Surface Plot
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.plot_surface(X, Y, Z, cmap='viridis')
        ax_3d.set_xlabel('x')
        ax_3d.set_ylabel('y')
        ax_3d.set_zlabel('f(x, y)')
        st.pyplot(fig_3d)

        # Contour Plot
        fig_contour, ax_contour = plt.subplots()
        cp = ax_contour.contour(X, Y, Z, 20, cmap='viridis')
        plt.colorbar(cp)
        ax_contour.set_xlabel('x')
        ax_contour.set_ylabel('y')
        
        # Plot gradient
        ax_contour.quiver(x0, y0, grad_f_x0_y0[0], grad_f_x0_y0[1], color='r', scale=10)
        st.pyplot(fig_contour)


    except Exception as e:
        st.error(f"Error: {e}")
