import streamlit as st

from labs.lab1_calculator import run_lab1
from labs.lab2_calculus import run_lab2
from labs.lab3_linear_algebra import run_lab3
from labs.lab4_multivariable import run_lab4
from labs.lab5_precalculus import run_lab5
from labs.lab6_sequences_and_series import run_lab6

st.set_page_config(layout="wide")

st.sidebar.title("MathLabX")
lab = st.sidebar.radio(
    "Select a Lab:",
    (
        "Scientific + Symbolic Calculator",
        "Single-variable Calculus Visualizer",
        "Linear Algebra: Vectors & Transformations",
        "Multivariable Calculus",
        "Precalculus",
        "Sequences and Series",
    ),
)

if lab == "Scientific + Symbolic Calculator":
    run_lab1()
elif lab == "Single-variable Calculus Visualizer":
    run_lab2()
elif lab == "Linear Algebra: Vectors & Transformations":
    run_lab3()
elif lab == "Multivariable Calculus":
    run_lab4()
elif lab == "Precalculus":
    run_lab5()
elif lab == "Sequences and Series":
    run_lab6()
