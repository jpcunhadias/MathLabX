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
    st.write("Work in progress...")
elif lab == "Linear Algebra: Vectors & Transformations":
    st.title("Lab 3: Linear Algebra: Vectors & Transformations")
    st.write("Work in progress...")
elif lab == "Multivariable Calculus":
    st.title("Lab 4: Multivariable Calculus")
    st.write("Work in progress...")
