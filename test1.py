import KaczmarzSolvers.py as kaczmarz
import numpy as np
import pytest

def test1():
  A = np.matrix(
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  )
  b = np.array([1, 2, 3])
  assert np.isclose(kaczmarz.Kaczmarz(A, b), np.array([-1/3, 2/3, 0]))

if __name__ == "__main__":
  test1()