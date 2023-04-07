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


def plot_obstacles(ax):
    config = Config()
    ax.set_aspect("equal")
    for i in range(2):
        for j in range(2):
            ax.add_patch(
                Circle(
                    (
                        config.trajectory_radius * (2 * i),
                        config.trajectory_radius * (2 * j),
                    ),
                    config.trajectory_radius - config.trajectory_margin,
                    color="k",
                    fill=False,
                )
            )
