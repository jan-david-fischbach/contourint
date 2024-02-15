from contourint.contour import Ellipse
import jax.numpy as jnp
import matplotlib.pyplot as plt

def f_mero(z):
    """example of a meromorphic function with a single pole"""
    return 1/(z-(1+1j))

def test_ellipse_integration():
    elli = Ellipse(center = 1+1j, radius = 1+2j, num_points = 128)
    plt.scatter(elli.points.real, elli.points.imag, marker='X')
    plt.figure()
    plt.plot(elli.d_points.real)
    plt.plot(elli.d_points.imag)

    assert jnp.isclose(jnp.sum(elli.weights*f_mero(elli.points)), 2j*jnp.pi)
    assert jnp.isclose(elli.integrate(f_mero), 2j*jnp.pi)

if __name__ == "__main__":
    test_ellipse_integration()
