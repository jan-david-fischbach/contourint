from contourint import hankel_matrix, vandermonde_matrix, locate_poles
from contourint.contour import Ellipse
import jax.numpy as np
import chex

def test_hankel():
  c = np.arange(3)
  r = np.arange(3)
  expected = np.array([
    [0, 1, 2],
    [1, 2, 1],
    [2, 1, 2],
  ])
  assert np.array_equal(hankel_matrix(c, r), expected)

def test_vandermonde():
  evs = np.arange(3)
  expected = np.array([
    [1, 1, 1],
    [0, 1, 2],
    [0, 1, 4],
  ])
  assert np.array_equal(vandermonde_matrix(evs), expected)

def test_poles_residues():
  def f_mero(z: chex.ArrayDevice):
      """example of a meromorphic function with a single pole"""
      return np.sum(example_residues.reshape(1, -1)/(z.reshape(-1, 1)-example_poles.reshape(1, -1)), axis=1)

  contour = Ellipse(center = 1+1j, radius = 1+2j, num_points = 128)

  example_poles = np.array([1+1j, 1+1.1j, 1.1+1j])
  example_residues = np.array([1, 1e-3, 1e-4])
  poles, residues = locate_poles(contour, f_mero)
  assert np.allclose(poles, example_poles)
  assert np.allclose(residues, example_residues)

if __name__ == "__main__":
  test_hankel()
  test_vandermonde()
  test_poles_residues()
