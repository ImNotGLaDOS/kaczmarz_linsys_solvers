import numpy as np


def generate_random_system(m=100, n=100, seed=42):
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true
    return A, b, x_true


def generate_well_conditioned_system(m=100, n=100):
    A = np.random.randn(m, n)
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    s = np.linspace(1, 0.5, min(m,n))  # Спектр плавно убывает
    A = U @ np.diag(s) @ Vt
    x_true = np.random.randn(n)
    b = A @ x_true
    return A, b, x_true


def generate_vandermond_system(m=100, n=100):
    t = np.sort(np.random.uniform(0, 1, n))
    A = np.vander(t, N=n, increasing=True)[:m]
    x_true = np.random.randn(n)
    b = A @ x_true
    return A, b, x_true
