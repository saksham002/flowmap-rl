"""Flow-matching interpolation + one-step (or few-step) Euler sampling.

Convention: noise z ~ N(0, I) at t=0, data action a at t=1,
x_t = (1 - t) * z + t * a, target velocity v* = a - z.
Sampling integrates dx/dt = v(x, t) from t=0 (z) to t=1.
"""

from typing import Callable
import jax.numpy as jnp
from jaxtyping import Array


def interpolate(action: Array, z: Array, t: Array) -> tuple[Array, Array]:
    """Return (x_t, target_velocity) for flow-matching at time t."""
    x_t = (1.0 - t) * z + t * action
    v_target = action - z
    return x_t, v_target


def sample_action(velocity_fn: Callable[[Array, Array], Array], z: Array, steps: int = 1) -> Array:
    """Euler-integrate the flow from noise z to an action chunk.

    steps=1 gives the one-step sample  a_hat = z + v(z, t=0).
    velocity_fn(x, t) returns the velocity at state x and scalar time t.
    """
    dt = 1.0 / steps
    x = z
    for i in range(steps):
        t = jnp.asarray(i * dt, dtype = jnp.float32)
        x = x + velocity_fn(x, t) * dt
    return x
