from contourint import hankel_matrix, vandermonde_matrix, locate_poles
from contourint.contour import Ellipse, Stack
import jax.numpy as np
import math
import jax.config
jax.config.update("jax_enable_x64", True)
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
      """example of a meromorphic function"""
      return np.sum(example_residues.reshape(1, -1)/(z.reshape(-1, 1)-example_poles.reshape(1, -1)), axis=1)

  contour = Ellipse(center = 1+1j, radius = 1+2j, num_points = 128)

  example_poles = np.array([1+1j, 1+1.1j, 1.1+1j])
  example_residues = np.array([1, 1e-3, 1e-4])
  poles, residues = locate_poles(contour, f_mero)
  print(f"{poles=}")
  print(f"{residues=}")
  assert np.allclose(poles, example_poles)
  assert np.allclose(residues, example_residues)

def test_numerics():
  example_poles = np.array([1+1j, 1+1.1j, 1.1+1j])
  example_residues = np.array([1, 1e-3, 1e-4])
  essential_scaling = 10
  def f_mero(z: chex.ArrayDevice):
    """example of a meromorphic function with a single pole"""
    f_val = np.exp(1/(z*essential_scaling)) + np.sum(example_residues.reshape(1, -1)/(z.reshape(-1, 1)-example_poles.reshape(1, -1)), axis=1)
    if not np.isfinite(f_val).all():
      raise RuntimeError("evaluating f failed")
    return f_val

  contour = Stack(contours=[
    Ellipse(center = 0, radius = 23+23j, num_points = 512),
    Ellipse(center = 0, radius = 0.03-0.03j, num_points = 512)
  ])

  poles, residues = locate_poles(contour, f_mero, M=10, regularize=False)
  print(np.abs(poles-example_poles))

  poles, residues = locate_poles(contour, f_mero, M=10, regularize=True)
  print(np.abs(poles-example_poles))
  assert np.allclose(poles, example_poles)
  assert np.allclose(residues, example_residues)
  #[2.13939921e-05 7.63910591e-03 9.41356433e-02]

def test_numerics_2():
  example_poles = np.array([1+1j, 1+1.1j, 1.1+1j])
  example_residues = np.array([1, 1e-3, 1e-4])

  essential_scaling = 10
  def f_mero(z: chex.ArrayDevice):
    """example of a meromorphic function with a single pole"""
    f_val = np.exp(1/((z)*10)) #+ np.sum(example_residues.reshape(1, -1)/(z.reshape(-1, 1)-example_poles.reshape(1, -1)), axis=1)
    return f_val

  contour = Ellipse(center = 0.1, radius = 10-10j, num_points = 2**6)
  r = contour.radius
  scale = 1/np.sqrt(r.real)/np.sqrt(np.abs(r.imag))
  shift = -contour.center
  reg_contour = (contour+shift)*scale

  def integrand(k):
    def _inner(omega):
      val = omega**k * f_mero(omega)
      #print(val)
      return val
    return _inner

  for k in range(5):
    f = integrand(k)
    def reg_f(omega):
      return f((omega/scale)-shift)
    i = contour.integrate(f)
    i_reg = reg_contour.integrate(reg_f)
    i_reg = i_reg/scale

    ref = -2j*np.pi/math.factorial(k+1)/(essential_scaling**(k+1))
    print(f"i    :{np.abs(i-ref)}:{np.abs((i-ref)/ref)}")
    print(f"i_reg:{np.abs(i_reg-ref)}:{np.abs((i_reg-ref)/ref)}")
  assert False

if __name__ == "__main__":
  test_hankel()
  test_vandermonde()
  test_poles_residues()
