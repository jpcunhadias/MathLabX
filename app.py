import streamlit as st

st.set_page_config(layout="wide")

st.sidebar.title("MathLabX")
lab = st.sidebar.radio("Select a Lab:", ("Scientific + Symbolic Calculator", "Single-variable Calculus Visualizer", "Linear Algebra: Vectors & Transformations", "Multivariable Calculus"))

if lab == "Scientific + Symbolic Calculator":
    st.title("Lab 1: Scientific + Symbolic Calculator")
    st.write("Work in progress...")
elif lab == "Single-variable Calculus Visualizer":
    st.title("Lab 2: Single-variable Calculus Visualizer")
    st.write("Work in progress...")
elif lab == "Linear Algebra: Vectors & Transformations":
    st.title("Lab 3: Linear Algebra: Vectors & Transformations")
    st.write("Work in progress...")
elif lab == "Multivariable Calculus":
    st.title("Lab 4: Multivariable Calculus")
    st.write("Work in progress...")
