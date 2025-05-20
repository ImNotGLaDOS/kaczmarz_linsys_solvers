import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple

def kaczmarz_2d(A: np.ndarray, b: np.ndarray, x0: np.ndarray, max_iter: int = 100) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Implements the Kaczmarz algorithm for 2D systems and returns the solution path.
    Works with overdetermined systems (more equations than variables).
    
    Args:
        A: m x 2 matrix representing the system (m can be > 2)
        b: m x 1 vector representing the right-hand side
        x0: Initial guess
        max_iter: Maximum number of iterations
        
    Returns:
        Tuple of (final solution, list of points visited during iteration)
    """
    x = x0.copy()
    path = [x.copy()]
    
    for _ in range(max_iter):
        for i in range(len(b)):
            # Get the i-th row of A and corresponding element of b
            ai = A[i]
            bi = b[i]
            
            # Compute the projection
            alpha = (bi - np.dot(ai, x)) / np.dot(ai, ai)
            x = x + alpha * ai
            
            path.append(x.copy())
    
    return x, path

def visualize_kaczmarz(A: np.ndarray, b: np.ndarray, x0: np.ndarray, solution: np.ndarray, path: List[np.ndarray], title: str = "Kaczmarz Algorithm Visualization in 2D"):
    """
    Visualizes the Kaczmarz algorithm's convergence path in 2D.
    
    Args:
        A: m x 2 matrix representing the system
        b: m x 1 vector representing the right-hand side
        x0: Initial guess
        solution: True solution of the system (or least squares solution for overdetermined systems)
        path: List of points visited during iteration
        title: Title for the plot
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the lines representing the equations
    x_range = np.linspace(-5, 5, 100)
    
    for i in range(len(b)):
        if abs(A[i, 1]) > 1e-10:  # Avoid division by zero
            y = (b[i] - A[i, 0] * x_range) / A[i, 1]
            ax.plot(x_range, y, label=f'Equation {i+1}: {A[i,0]}x + {A[i,1]}y = {b[i]}', 
                   linewidth=2.5)  # Increased line width
    
    # Plot the solution path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'r-', alpha=0.5, label='Solution path', linewidth=2.0)  # Increased line width
    ax.scatter(path[:, 0], path[:, 1], c='r', s=30, alpha=0.5)  # Increased point size
    
    # Plot initial point and solution
    ax.scatter(x0[0], x0[1], c='g', s=100, label='Initial point')
    
    # Add a circle at each step to show the projection
    # for i in range(len(path)-1):
    #     circle = Circle((path[i][0], path[i][1]), 
    #                    np.linalg.norm(path[i+1] - path[i]),
    #                    fill=False, color='gray', alpha=0.3)
    #     ax.add_patch(circle)
    
    # Set up the plot
    ax.grid(True, linewidth=1)  # Made grid lines thinner
    ax.set_aspect('equal')
    ax.set_title(title, pad=20)  # Added padding to title
    
    
    # Set reasonable limits
    ax.set_xlim(min(-3, min(path[:, 0])-1), max(3, max(path[:, 0])+1))
    ax.set_ylim(min(-3, min(path[:, 1])-1), max(3, max(path[:, 1])+1))

    plt.savefig("kaczmarz_2d" + str(A[0][0]) + ".eps")
    plt.show()
    # Save the plot to a file in high resolution

def main():
    # Example 1: Overdetermined but solvable system
    # System of 4 equations that have a common solution
    A1 = np.array([[-1, 2],
                   [1, 8],
                   [2, 3],
                   [7, -1],])
    b1 = np.array([0, 0, 0, 0])
    x0 = np.array([-2, -2])
    
    # Solve the system using numpy's least squares for comparison
    true_solution = np.linalg.lstsq(A1, b1, rcond=None)[0]
    
    # Run Kaczmarz algorithm
    solution, path = kaczmarz_2d(A1, b1, x0, max_iter=20)
    
    # Visualize
    print("Example 1: Overdetermined but solvable system")
    print("System:")
    for i in range(len(b1)):
        print(f"{A1[i,0]}x + {A1[i,1]}y = {b1[i]}")
    print(f"\nLeast squares solution: {true_solution}")
    print(f"Kaczmarz solution: {solution}")
    print(f"Residual norm: {np.linalg.norm(A1 @ solution - b1):.6f}")
    visualize_kaczmarz(A1, b1, x0, true_solution, path, 
                       "Kaczmarz Algorithm - Overdetermined Solvable System")
    input("Press Enter to continue...")
    # Example 2: Overdetermined and unsolvable (inconsistent) system
    # System of 4 equations that are inconsistent (no common solution)
    A2 = np.array([[1, 2],
                   [10, 1],
                   [-1, 1.5]])
    b2 = np.array([-2.5, 30, 4])
    x0 = np.array([3.5, 3])
    
    # Solve the system using numpy's least squares for comparison
    true_solution2 = np.linalg.lstsq(A2, b2, rcond=None)[0]
    
    # Run Kaczmarz algorithm
    solution2, path2 = kaczmarz_2d(A2, b2, x0, max_iter=30)
    
    # Visualize
    print("\nExample 2: Overdetermined and unsolvable (inconsistent) system")
    print("System:")
    for i in range(len(b2)):
        print(f"{A2[i,0]}x + {A2[i,1]}y = {b2[i]}")
    print(f"\nLeast squares solution: {true_solution2}")
    print(f"Kaczmarz solution: {solution2}")
    print(f"Residual norm: {np.linalg.norm(A2 @ solution2 - b2):.6f}")
    visualize_kaczmarz(A2, b2, x0, true_solution2, path2,
                       "Kaczmarz Algorithm - Overdetermined Unsolvable System")

if __name__ == "__main__":
    main() 