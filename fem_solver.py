# =============================================================================
# 1D Finite Element Method (FEM) Solver
# Solves the problem: -u''(x) = delta(x - 0.5)
# With boundary conditions: u(0) = 0, u(1) = 0
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

def u_true(x):
    """
    Calculates the exact analytical solution for the 1D point load problem.
    This function is vectorized using np.where for efficiency.
    """
    return np.where(x <= 0.5, 0.5 * x, 0.5 * (1 - x))

if __name__ == "__main__":
    
    # --- Parameters ---
    m = 21          # Number of internal mesh points (nodes)
    load_pos = 0.5  # Position of the point load

    # --- 1. Set up the mesh ---
    h = 1 / (m + 1)  # Calculate the size of each element
    # Create the node coordinates (m+2 total points, including x=0 and x=1)
    x_nodes = np.linspace(0, 1, m + 2)

    # --- 2. Assemble the Stiffness Matrix A ---
    # The matrix A represents the discretized -d^2/dx^2 operator.
    # Its entries are derived from the integrals of the basis function derivatives.
    diag = np.ones(m) * (2 / h)
    off_diag = np.ones(m - 1) * (-1 / h)
    A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    # --- 3. Assemble the Load Vector b ---
    # The vector b represents the effect of the point load on each node.
    # Its entries are calculated by evaluating each basis function at the load position.
    b = np.zeros(m)
    
    # Find the element that contains the point load.
    # 'k' will be the index of the node to the left of the load.
    k = int(np.floor(load_pos / h))
    
    # Get the coordinates of the two nodes that bracket the load
    x_k = x_nodes[k]
    x_k_plus_1 = x_nodes[k+1]
    
    # Calculate the linear interpolation weights to distribute the point load
    # to the two adjacent nodes. This is equivalent to phi_k(load_pos).
    weight_k_plus_1 = (load_pos - x_k) / h
    weight_k = (x_k_plus_1 - load_pos) / h
    
    # Add weights to the load vector b.
    if k > 0:
        b[k - 1] = weight_k
    if k < m:
        b[k] = weight_k_plus_1
        
    # --- 4. Solve the linear system Ax = b ---
    U = np.linalg.solve(A, b)
    
    # Create the full approximate solution vector by adding the zero boundary conditions
    u_approx = np.concatenate(([0], U, [0]))
    
    # --- 5. Plot the results ---
    # Create a fine mesh to plot the smooth, exact solution
    x_fine = np.linspace(0, 1, 1000)
    u_actual_fine = u_true(x_fine)

    plt.figure(figsize=(6, 4))
    plt.plot(x_fine, u_actual_fine, 'r-', label='Exact Solution')
    plt.plot(x_nodes, u_approx, 'bo', label=f'FEM Solution ({m} internal nodes)')
    plt.title('1D FEM Solution vs. Exact Solution')
    plt.xlabel('Position x')
    plt.ylabel('Displacement u(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.savefig(f'fem_solution_m{m}.png', dpi=300) # Optional: save the figure
    plt.show()