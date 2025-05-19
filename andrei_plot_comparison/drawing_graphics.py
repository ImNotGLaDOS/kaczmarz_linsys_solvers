import os
import matplotlib.pyplot as plt
import numpy as np
from kaczmarz_realisation import (
    classical_kaczmarz,
    randomized_kaczmarz,
    maximal_residual_kaczmarz,
    preconditioned_kaczmarz,
    sorted_kaczmarz,
)

method_names = ["classical_kaczmarz", "randomized_kaczmarz", "maximal_residual_kaczmarz",
                "preconditioned_kaczmarz", "sorted_kaczmarz"]


def relative_solution_error(A, b, x_true, directory='', max_iter=1000):
    plt.figure(figsize=(12, 7))
    methods = {
        "classical_kaczmarz": (classical_kaczmarz, "blue"),
        "randomized_kaczmarz": (randomized_kaczmarz, "red"),
        "maximal_residual_kaczmarz": (maximal_residual_kaczmarz, "green"),
        "preconditioned_kaczmarz": (preconditioned_kaczmarz, "purple"),
        "sorted_kaczmarz": (sorted_kaczmarz, "black"),
    }
    for name in method_names:
        method, color = methods[name]
        _, _, errors = method(A, b, x_true, max_iter=max_iter)
        errors_normed = errors / np.linalg.norm(x_true)
        plt.semilogy(errors_normed, color=color, linewidth=2, label=name)

    plt.xlabel('Iteration number', fontsize=16)
    plt.ylabel('Relative error $\\frac{\\Vert x_k - x^* \\Vert}{\\Vert x^* \\Vert}$', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, which="both", ls="--")
    plt.title('Convergence to true solution', fontsize=16)
    if directory:
        plt.savefig(os.path.join(directory, "relative_solution_error.png"), dpi=300, bbox_inches="tight")
