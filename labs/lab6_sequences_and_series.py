import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import sympify, Symbol

import config

def run_lab6():
    st.title("Sequences and Series")

    st.header("Sequences")
    
    n = Symbol('n')
    sequence_expr_str = st.text_input("Enter a sequence a(n):", "1/n")
    
    n_min = st.slider("n min", 1, 100, 1)
    n_max = st.slider("n max", 1, 100, 10)

    try:
        a_n = sympify(sequence_expr_str)
        
        n_vals = np.arange(n_min, n_max + 1)
        seq_vals = [a_n.subs(n, val) for val in n_vals]
        
        st.write("First few terms:")
        st.write([float(val) for val in seq_vals[:5]])
        
        fig, ax = plt.subplots()
        ax.plot(n_vals, seq_vals, 'o', color=config.DEFAULT_PLOT_COLOR)
        ax.set_title(f"Sequence: $a_n = {a_n}$")
        ax.set_xlabel("n")
        ax.set_ylabel("a_n")
        ax.grid(True, linestyle=config.DEFAULT_GRID_STYLE)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
