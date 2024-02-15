import chex
import jax.config
jax.config.update("jax_enable_x64", True)
import jax.numpy as np

class Contour:
  points: chex.ArrayDevice

@chex.dataclass
class Ellipse(Contour):
  """Elliptical contour along which a quantity can be integrated
  Note that the phase of the clockwise and counterclockwise circular components
  is forced to zero, leading to the major and minor axis to be oriented along
  the imaginary/real axis or vice versa.
  ...

  Attributes
  ----------
  center : complex
      center of the ellipse
  radius : stcomplexr
      radius of the ellipse as a complex number. The imaginary part specifies
      the radius in the imaginary direction and the real part accordingly.
  num_points : int
      number of samplepoints on the contour (preferably an integer power of 2)
  """
  center: complex
  radius: complex
  num_points: int

  @property
  def A(self) -> float:
    """Coefficient of the counterclockwise (positive rotation) component"""
    return (self.radius.real + self.radius.imag)/2

  @property
  def B(self) -> float:
    """Coefficient of the clockwise (negative rotation) component"""
    return (self.radius.real - self.radius.imag)/2

  @property
  def alpha(self) -> chex.ArrayDevice:
    """Parametrization of the ellipse"""
    return  np.linspace(0, 2*np.pi, num=self.num_points, endpoint=False)

  @property
  def points(self) -> chex.ArrayDevice:
    """The complex valued sample points on the elliptical contour """
    points = (
      self.center +
      self.A*np.exp(1j*self.alpha) +
      self.B*np.exp(-1j*self.alpha)
    )
    return points

  @property
  def d_points(self) -> chex.ArrayDevice:
    """Derivative of the sampling positions wrt to alpha"""
    return 1j*(
      self.A*np.exp(1j*self.alpha) -  # Derivative of points wrt. alpha
      self.B*np.exp(-1j*self.alpha)
    )

  @property
  def weights(self) -> chex.ArrayDevice:
    """Integration weights for trapezoidal quadrature"""
    weights = self.d_points * 2*np.pi/self.num_points
    return weights

  def integrate(self, f: callable) -> complex:
    return np.sum(self.weights*f(self.points))
