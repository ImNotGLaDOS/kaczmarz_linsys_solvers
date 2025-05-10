import numpy as np

def Kaczmarz(A: np.matrix, b: np.ndarray, iterations: int = 0, relaxation: np.float32 = np.float32(1)) -> np.ndarray:
  """Classical Kaczmarz algorithm [ADDITIONAL INFO]"""
  x = np.zeros(b.shape)
  def iterate(ix: int):
    ix %= A.shape[1]
    row = A[:ix]
    projection = b[ix] - row @ x
    projection /= np.linalg.norm(projection)
    projection *= relaxation
    x += projection

  if iterations == 0:
    ix = 0
    while np.isclose(A @ x, b):
      iterate(ix)
      ix += 1
  else:
    for ix in range(iterations):
      iterate(ix)
  return x