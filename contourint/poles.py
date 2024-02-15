from contourint import Contour
import jax.numpy as np
import scipy.linalg
import chex

def locate_poles(contour: Contour, f: callable, M: int=10, cutoff=1e-9):
  """finds the poles of f within the given contour
  largely based on https://arxiv.org/abs/2307.04654 ([1]), which is based on
  algorithm [Cp] from  https://doi.org/10.1137/130931035 ([2])
  Args:
      contour (Contour):
          the contour used for integration
      f (callable):
          the meromorphic function, of which the poles are to be found.
      M (int):
          maximum number of poles
  """

  s_k = hankel_entries(contour, f, M)
  H  = hankel_matrix(s_k[ :M  ], s_k[M-1:-1])
  H2 = hankel_matrix(s_k[1:M+1], s_k[M:])

  poles, R = scipy.linalg.eig(H2, H)

  V = vandermonde_matrix(poles)
  residues = np.diagonal(R.T@H@R) / np.sum(V.T*R.T, 1)**2

  selection = residues/np.max(residues) > cutoff

  poles = poles[selection]
  residues = residues[selection]

  sorting = np.argsort(-np.abs(residues))
  #TODO quadratic quantities, Real valued hankel matrices, derivatives

  return poles[sorting], residues[sorting]


def hankel_entries(contour: Contour, f: callable, M: int):
  """elements of the hankel matrices according to eq. (2) of [1]
  Args:
      contour (Contour):
          the contour used for integration
      f (callable):
          the meromorphic function, of which the poles are to be found.
      M (int):
          maximum number of poles
  """
  def integrand(k):
    def _inner(omega):
      return omega**k * f(omega)
    return _inner

  return 1/(2j * np.pi) * np.array([
    contour.integrate(integrand(k)) for k in range(2*M)
  ])

def hankel_matrix(c: chex.ArrayDevice, r: chex.ArrayDevice):
  """constructs the hankel matrix as detailed here:
  https://de.mathworks.com/help/matlab/ref/hankel.html

  Args:
      c (chex.ArrayDevice): the first column
      r (chex.ArrayDevice): the last row
  """
  assert len(c) == len(r)
  M = len(c)
  tot = np.concatenate([c, r[1:]])

  hankel = [tot[i:i+M] for i in range(M)]
  return np.array(hankel)


def vandermonde_matrix(w: chex.ArrayDevice):
  """constructs the vandermonde matrix according to eq. (3.7) of [2]

  Args:
      w (chex.ArrayDevice): the eigenvalues
  """
  M = len(w)
  return np.array([np.power(np.where(np.isinf(w), 0, w), k) for k in range(M)])
