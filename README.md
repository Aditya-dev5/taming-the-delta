# FEM Solver – Tent String Example

This repo contains the Python code (`fem_solver.py`) used in my blog post:  
**[Relax the Rules to Solve Hard Problems – A Hands-on Introduction to FEM]([https://medium.com/@adityajabade1/relax-the-rules-to-solve-hard-problems-080880538b00])**

The script demonstrates how the **Finite Element Method (FEM)** can be applied to solve a 1D string displacement problem with a point load (Dirac delta), showing the transition from physics → weak form → matrix system → numerical solution.

## Files
- `fem_solver.py` – Core Python script (NumPy + Matplotlib). Builds the stiffness matrix and load vector, solves the system `Ax = b`, and visualizes the FEM vs true solution for different mesh sizes.

## Usage
python fem_solver.py
