import numpy as np

from ese200.config import Config
from ese200.simulator import dynamics_ca


def optimize(A, B, t, points):
    """
    Fit a polynomial trajectory to a set of points.
    Find a trajectory that goes through all the points.
    The trajectory should have minimum acceleration.
    """
    import cvxpy as cp

    u = cp.Variable((len(t), 2))
    x = cp.Variable((len(t), 4))

    # assume we pass through all points at equal time intervals
    # we also assume that the first point is also the last point for loop closure
    point_idx = np.linspace(0, len(t) - 1, len(points) + 1).astype(int)[:-1]
    point_idx[0] = 0

    # objective is to minimize the sum of squared accelerations
    objective = cp.sum_squares(u)
    constraints = [
        # dynamics
        x[1:] == x[:-1] @ A.T + u[:-1] @ B.T,
        # pass through points
        x[point_idx, :2] == points,
        # final state should be equal to initial state
        x[-1] == x[0],
    ]

    # solve optimization problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    return x.value, u.value


def dynamics_ca_drag(dt: float, mu: float):
    A, B = dynamics_ca(dt)
    A[0, 2] -= mu * dt**2 / 2
    A[1, 3] -= mu * dt**2 / 2
    A[2, 2] -= mu * dt
    A[3, 3] -= mu * dt
    return A, B


def generate_expert_trajectories():
    rng = np.random.default_rng()
    config = Config()
    A, B = dynamics_ca_drag(Config.time_step, Config.drag_coefficient)
    t = np.arange(0, config.duration, config.time_step)
    x = np.zeros((config.n_trajectories, len(t), 4))
    u = np.zeros((config.n_trajectories, len(t), 2))

    # generate random trajectories
    points = (
        np.asarray(
            [
                [-1, 0],
                [-1, 2],
                [0, 3],
                [1, 2],
                [1, 1],
                [2, 1],
                [3, 0],
                [2, -1],
                [0, -1],
            ]
        )
        * config.trajectory_radius
    )
    points = points[None, ...] + rng.normal(
        scale=config.trajectory_noise,
        size=(config.n_trajectories,) + points.shape,
    )

    for i in range(config.n_trajectories):
        x[i], u[i] = optimize(A, B, t, points[i])
    noise = rng.normal(
        scale=np.array([config.noise_position] * 2 + [config.noise_velocity] * 2)
        * config.time_step,
        size=x.shape,
    )
    return x + noise, u, points


if __name__ == "__main__":
    x, u, points = generate_expert_trajectories()
    np.save("data/states.npy", x)
    np.save("data/inputs.npy", u)
