# MathLabX

A web-based application for mathematical exploration, built with Python and Streamlit.

## Features

MathLabX is organized into four "labs":

1.  **Scientific + Symbolic Calculator**: A versatile calculator for both numeric and symbolic computations, including differentiation, integration, and limits.
2.  **Single-variable Calculus Visualizer**: Tools for building intuition around calculus concepts, including:
    *   **Function & Derivative**: Visualize functions, their derivatives, and tangent lines.
    *   **Limits**: Explore the concept of limits by approaching a point from both sides.
    *   **Integrals & Area**: Visualize Riemann sums and the area under a curve.
3.  **Linear Algebra: Vectors & Transformations**: Visualize vectors and the effect of matrix transformations in 2D space.
4.  **Multivariable Calculus**: Explore functions of two variables with 3D surface plots and contour plots, including gradient vectors.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jpcunhadias/MathLabX.git
    cd MathLabX
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

The application will be accessible in your web browser at `http://localhost:8501`.
