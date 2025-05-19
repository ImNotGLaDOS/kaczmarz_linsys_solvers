import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm
from scipy.linalg import toeplitz, orth
from scipy.sparse import rand as sprand

def generate_random_matrix(
    m, n,
    condition_number=1e2,
    sparse=False,
    toeplit=False,
    orthogonal=False,
    important_rows_ratio=0,
    important_rows_scale=1.0
):

    if orthogonal:
        if m < n:
            raise ValueError("Для ортогональной матрицы m >= n (чтобы столбцы были ортонормированы).")
        A = np.random.randn(m, n)
        Q, _ = np.linalg.qr(A) 
        A = Q[:, :n] 
    elif toeplit:
        c = np.random.randn(m)
        r = np.random.randn(n)
        A = toeplitz(c, r)
    elif sparse:
        A = sprand(m, n, density=0.3).toarray()
    else:
        U, _, Vt = np.linalg.svd(np.random.randn(m, n), full_matrices=False)
        s = np.logspace(0, -np.log10(condition_number), min(m, n))
        A = U @ np.diag(s) @ Vt
    if important_rows_ratio > 0:
        num_important = int(m * important_rows_ratio)
        important_indices = np.random.choice(m, size=num_important, replace=False)
        A[important_indices] *= important_rows_scale
    x_true = np.random.randn(n)
    b = A @ x_true

    return A, b, x_true

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
        if (time.time() - t_start > 0.25):
            t_total = time.time() - t_start
            #return x, errors, t_total
            
    t_total = time.time() - t_start
    return x, errors, t_total

def randomized_kaczmarz(A, b, x0, max_iter=500):
    m, n = A.shape
    x = x0.copy()
    errors = []
    row_norms = np.linalg.norm(A, axis=1)
    probs = row_norms / np.sum(row_norms)
    t_start = time.time()
    for _ in range(max_iter):
        i = np.random.choice(m, p=probs)
        a = A[i]
        x += (b[i] - np.dot(a, x)) / np.dot(a, a) * a
        errors.append(norm(A @ x - b))
        if (time.time() - t_start > 0.25):
            t_total = time.time() - t_start
            #return x, errors, t_total
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
        if (time.time() - t_start > 0.25):
            t_total = time.time() - t_start
            #return x, errors, t_total
    t_total = time.time() - t_start
    return x, errors, t_total

def preconditioned_kaczmarz(A, b, x0, precond, max_iter=500):
    t_start = time.time()
    print(time.time() - t_start)
    precondx = np.linalg.pinv(precond)
    A_pc = A @ precondx
    x = np.zeros(A_pc.shape[1])
    errors = []
    for i in range(max_iter):
        i_k = i % A.shape[0]
        a = A_pc[i_k]
        x += (b[i_k] - np.dot(a, x)) / np.dot(a, a) * a
        errors.append(norm(A_pc @ x - b))
        if (time.time() - t_start > 0.25):
            t_total = time.time() - t_start
            #return x, errors, t_total
    t_total = time.time() - t_start
    return precondx @ x, errors, t_total

def plot_errors(errors_dict, title, log_time=False):
    plt.figure(figsize=(10, 6))
    for label, (errors, time_taken) in errors_dict.items():
        x = np.arange(len(errors))
        linestyle = '--' if "Precond" in label else '-'
        plt.plot(x, errors, label=f"{label} (t={time_taken:.2f}s)", linestyle=linestyle)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(str(title)+".eps", dpi=150)
    plt.show()

def run_scenario(name, m=300, n=100, **kwargs):
    print(f"\n=== {name} ===")
    A, b, x_true = generate_random_matrix(m, n, **kwargs)
    x0 = np.zeros(A.shape[1])
    mx_iter = 1000
    results = {
        'Classic': kaczmarz(A, b, x0, max_iter=mx_iter),
        'Randomized': randomized_kaczmarz(A, b, x0, max_iter=mx_iter),
        'Block': block_kaczmarz(A, b, x0, block_size=5, max_iter=mx_iter),
        'Precond': preconditioned_kaczmarz(A, b, x0, A, max_iter=mx_iter),
    }
    plot_errors({k: (v[1], v[2]) for k, v in results.items()}, title=name)

if __name__ == '__main__':
    # 5. Sparse matrices
    run_scenario("Sparse Matrix", sparse=True)

    # 6. Ill-conditioned matrix
    run_scenario("High Condition Number", condition_number=1e8)

    # 7. Well-conditioned matrix
    run_scenario("Low Condition Number", condition_number=2)

    # 8. Overdetermined system
    run_scenario("Overdetermined Matrix", m=1200, n=100)

    # 9. Toeplitz matrix
    run_scenario("Toeplitz Matrix", sparse=False, toeplit=True)

    #10. Orthogonal matrix
    run_scenario("Orthogonal Matrix", orthogonal=True)

