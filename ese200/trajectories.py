import cvxpy as cp
import numpy as np
import numpy.typing as npt

from ese200.config import Config


def generate():
    rng = np.random.default_rng()
    config = Config()
    A, B = get_dynamics(config.time_step)
    t = np.arange(0, config.duration, config.time_step)
    x = np.zeros((config.n_trajectories, len(t), 4))
    u = np.zeros((config.n_trajectories, len(t), 2))
    points = rng.uniform(0, config.width, (config.n_trajectories, config.n_points, 2))
    for i in range(config.n_trajectories):
        x[i], u[i] = optimize(A, B, t, points[i])
    np.save("data/states.npy", x)
    np.save("data/inputs.npy", u)
    np.save("data/points.npy", points)


def optimize(A, B, t, points):
    """
    Fit a polynomial trajectory to a set of points.
    Find a trajectory that goes through all the points.
    The trajectory should have minimum acceleration.
    """
    u = cp.Variable((len(t), 2))
    x = cp.Variable((len(t), 4))

    # initial state at points[0] and velocity is zero
    x_start = np.zeros(4)
    x_start[:2] = points[0]
    # final state at points[-1] and velocity is zero
    x_end = np.zeros(4)
    x_end[:2] = points[-1]

    # assume we pass through all points at equal time intervals
    point_idx = np.linspace(0, len(t) - 1, len(points)).astype(int)
    point_idx[0] = 0
    point_idx[-1] = len(t) - 1

    # objective is to minimize the sum of squared accelerations
    objective = cp.sum_squares(u)

    constraints = [
        # dynamics
        x[1:] == x[:-1] @ A.T + u[:-1] @ B.T,
        # pass through points
        x[point_idx, :2] == points,
        # initial and final state
        x[0] == x_start,
        x[-1] == x_end,
    ]

    # solve optimization problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    return x.value, u.value


def get_dynamics(dt: float):
    # Define dynamical model parameters
    # This is a 2D model with position and velocity
    # state x = [x, y, vx, vy]
    # input u = [ax, ay]
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
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return A, B


if __name__ == "__main__":
    generate()
