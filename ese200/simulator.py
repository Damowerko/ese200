import numpy as np
from matplotlib.patches import Circle

from ese200.config import Config


class Simulator:
    def __init__(self) -> None:
        self.A, self.B = dynamics_ca_drag(Config.time_step, Config.drag_coefficient)
        self.rng = np.random.default_rng()

    def step(self, x, u):
        noise = self.rng.normal(
            scale=np.array([Config.noise_position] * 2 + [Config.noise_velocity] * 2)
            * Config.time_step,
            size=x.shape,
        )
        return x @ self.A.T + u @ self.B.T + noise


def dynamics_ca(dt: float):
    # Define dynamical model parameters
    # This is a 2D model with position and velocity
    # state x = [x, y, vx, vy]
    # input u = [ax, ay]
    dt = Config.time_step
    A = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Input is acceleration
    B = np.array(
        [
            [dt**2 / 2, 0.0],
            [0.0, dt**2 / 2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return A, B


def dynamics_ca_drag(dt: float, mu: float):
    A, B = dynamics_ca(dt)
    A[0, 2] -= mu * dt**2 / 2
    A[1, 3] -= mu * dt**2 / 2
    A[2, 2] -= mu * dt
    A[3, 3] -= mu * dt
    return A, B
