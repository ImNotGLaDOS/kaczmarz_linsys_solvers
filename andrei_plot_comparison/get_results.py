import os
import numpy as np
from datetime import datetime
from generation_system import (
    generate_random_system,
    generate_well_conditioned_system,
    generate_vandermond_system,
)
from drawing_graphics import relative_solution_error


def get_experiment_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_system(directory, A, b, x_true):
    np.savez(os.path.join(directory, "system.npz"), A=A, b=b, x_true=x_true)


def get_system(generation_type, n, m):
    generation = [
        generate_random_system,
        generate_well_conditioned_system,
        generate_vandermond_system
    ]
    return generation[generation_type](m, n)


def save_results(generation_type, n, m, max_iter):
    A, b, x_true = get_system(generation_type, n, m)
    experiment_id = get_experiment_id()
    ensure_dir_exists(f'graphics/{experiment_id}')

    save_system(f'graphics/{experiment_id}', A, b, x_true)
    relative_solution_error(A, b, x_true, directory=f'graphics/{experiment_id}', max_iter=max_iter)
