# kaczmarz_comparison.py

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm

# ------------------ Helper Functions ------------------
def generate_random_matrix(m, n, condition_number=1e2, sparse=False, toeplitz=False):
    if toeplitz:
        from scipy.linalg import toeplitz
        c = np.random.randn(m)
        r = np.random.randn(n)
        A = toeplitz(c, r)
    elif sparse:
        from scipy.sparse import rand as sprand
        A = sprand(m, n, density=0.05).toarray()
    else:
        U, _, Vt = np.linalg.svd(np.random.randn(m, n), full_matrices=False)
        s = np.logspace(0, -np.log10(condition_number), min(m, n))
        A = U @ np.diag(s) @ Vt
    x_true = np.random.randn(n)
    b = A @ x_true
    return A, b, x_true

# ------------------ Kaczmarz Variants ------------------
def kaczmarz(A, b, x0, max_iter=500):
    m, n = A.shape
    x = x0.copy()
    errors = []
    t_start = time.time()
    for i in range(max_iter):
        i_k = i % m
        a = A[i_k]
        x += (b[i_k] - np.dot(a, x)) / np.dot(a, a) * a
        errors.append(norm(A @ x - b))
    t_total = time.time() - t_start
    return x, errors, t_total

def randomized_kaczmarz(A, b, x0, max_iter=500):
    m, n = A.shape
    x = x0.copy()
    errors = []
    row_norms = np.sum(A**2, axis=1)
    probs = row_norms / np.sum(row_norms)
    t_start = time.time()
    for _ in range(max_iter):
        i = np.random.choice(m, p=probs)
        a = A[i]
        x += (b[i] - np.dot(a, x)) / np.dot(a, a) * a
        errors.append(norm(A @ x - b))
    t_total = time.time() - t_start
    return x, errors, t_total

def block_kaczmarz(A, b, x0, block_size=10, max_iter=500):
    m, n = A.shape
    x = x0.copy()
    errors = []
    t_start = time.time()
    for i in range(max_iter):
        idx = np.random.choice(m, size=block_size, replace=False)
        A_blk = A[idx]
        b_blk = b[idx]
        x += np.linalg.pinv(A_blk) @ (b_blk - A_blk @ x)
        errors.append(norm(A @ x - b))
    t_total = time.time() - t_start
    return x, errors, t_total

def preconditioned_kaczmarz(A, b, x0, precond, max_iter=500):
    A_pc = A @ precond
    x = x0.copy()
    errors = []
    t_start = time.time()
    for i in range(max_iter):
        i_k = i % A.shape[0]
        a = A_pc[i_k]
        x += (b[i_k] - np.dot(a, x)) / np.dot(a, a) * a
        errors.append(norm(A_pc @ x - b))
    t_total = time.time() - t_start
    return precond @ x, errors, t_total

# ------------------ Plotting ------------------
def plot_errors(errors_dict, title, log_time=False):
    plt.figure(figsize=(10, 6))
    for label, (errors, time_taken) in errors_dict.items():
        x = np.arange(len(errors))
        plt.plot(x, errors, label=f"{label} (t={time_taken:.2f}s)")
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------ Example Run ------------------
if __name__ == '__main__':
    A, b, x_true = generate_random_matrix(1000, 100)
    x0 = np.zeros(100)

    res_k = kaczmarz(A, b, x0)
    res_rk = randomized_kaczmarz(A, b, x0)
    res_bk = block_kaczmarz(A, b, x0, block_size=20)
    P = np.linalg.pinv(A)
    res_pk = preconditioned_kaczmarz(A, b, x0, precond=P)

    plot_errors({
        'Classic': (res_k[1], res_k[2]),
        'Randomized': (res_rk[1], res_rk[2]),
        'Block': (res_bk[1], res_bk[2]),
        'Preconditioned': (res_pk[1], res_pk[2]),
    }, title="Comparison of Kaczmarz Variants")
