from __future__ import annotations
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

  def integrate(self, f: callable, **kwargs) -> complex:
    return np.sum(self.weights*f(self.points))

  def __add__(self, shift: complex) -> Ellipse:
    center = self.center + shift
    return Ellipse(center=center, radius=self.radius, num_points=self.num_points)

  def __sub__(self, shift:complex) -> Ellipse:
    return self + -1*shift

  def __mul__(self, scale: float) -> Ellipse:
    return Ellipse(center=self.center, radius=self.radius*scale, num_points=self.num_points)

  def __div__(self, scale: float) -> Ellipse:
    return self * (1/scale)

@chex.dataclass
class Stack(Contour):
  """Stack of contours. Last one should be the outermost one"""
  contours: list[Contour]

  @property
  def radius(self) -> complex:
    radii = np.array([contour.radius for contour in self.contours])
    return np.mean(radii.real) + 1j*np.mean(np.abs(radii.imag)) # np.sqrt(np.mean(radii**2))

  @property
  def center(self) -> complex:
    return self.contours[-1].center

  @property
  def points(self) -> chex.ArrayDevice:
    points = np.concatenate([inner.points for inner in self.contours])
    return points

  def integrate(self, f: callable, regularize: bool=False) -> complex:
    inner_integrals = []
    for partial_contour in self.contours:
      #print(f"{partial_contour.radius=}")
      if regularize:
        r = partial_contour.radius
        scale = 1/np.sqrt(r.real)/np.sqrt(np.abs(r.imag))
        shift = -partial_contour.center
      else:
        scale = 1
        shift = 0

      def reg_f(omega):
        return f((omega/scale)-shift)

      reg_partial_contour = (partial_contour+shift)*scale

      inner_integrals.append(reg_partial_contour.integrate(reg_f)/scale - shift)
    #print(f"{inner_integrals=}")
    return np.sum(np.array(inner_integrals))

  def __add__(self, shift: complex) -> Stack:
    contours = [inner + shift for inner in self.contours]
    return Stack(contours = contours)

  def __sub__(self, shift:complex) -> Stack:
    return self + -1*shift

  def __mul__(self, scale: float) -> Stack:
    contours = [inner * scale for inner in self.contours]
    return Stack(contours = contours)

  def __div__(self, scale: float) -> Stack:
    return self * (1/scale)
