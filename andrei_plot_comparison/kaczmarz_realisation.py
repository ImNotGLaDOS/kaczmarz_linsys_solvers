import numpy as np
from sklearn.random_projection import GaussianRandomProjection


def classical_kaczmarz(A, b, x_true, x0=None, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    residuals = []
    errors = []
    for it in range(max_iter):
        i = it % m
        a_i = A[i]
        x += (b[i] - np.dot(a_i, x)) / np.dot(a_i, a_i) * a_i
        residual = np.linalg.norm(A @ x - b)
        residuals.append(residual)
        errors.append(np.linalg.norm(x - x_true))
        if residual < tol:
            break
    return x, np.array(residuals), np.array(errors)


def randomized_kaczmarz(A, b, x_true, x0=None, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    row_norms = np.linalg.norm(A, axis=1)**2
    probabilities = row_norms / np.sum(row_norms)
    residuals = []
    errors = []
    for _ in range(max_iter):
        i = np.random.choice(m, p=probabilities)
        a_i = A[i]
        x += (b[i] - np.dot(a_i, x)) / np.dot(a_i, a_i) * a_i
        residual = np.linalg.norm(A @ x - b)
        residuals.append(residual)
        errors.append(np.linalg.norm(x - x_true))
        if residual < tol:
            break
    return x, np.array(residuals), np.array(errors)


def maximal_residual_kaczmarz(A, b, x_true, x0=None, max_iter=1000, tol=1e-6):
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    residuals = []
    errors = []
    for _ in range(max_iter):
        temp_residuals = b - A @ x
        i = np.argmax(np.abs(temp_residuals))
        a_i = A[i]
        x += (temp_residuals[i] / np.dot(a_i, a_i)) * a_i
        residual = np.linalg.norm(A @ x - b)
        residuals.append(residual)
        errors.append(np.linalg.norm(x - x_true))
        if np.linalg.norm(residuals) < tol:
            break
    return x, np.array(residuals), np.array(errors)


def preconditioned_kaczmarz(A, b, x_true, sketch_dim=50, max_iter=1000, tol=1e-6):
    m, n = A.shape
    transformer = GaussianRandomProjection(n_components=min(sketch_dim, n))
    S = transformer.fit_transform(A)  # S has shape (m, sketch_dim)
    SA = S.T @ A  # shape: (sketch_dim, n)
    Sb = S.T @ b  # shape: (sketch_dim,)
    return randomized_kaczmarz(SA, Sb, x_true, max_iter=max_iter, tol=tol)

