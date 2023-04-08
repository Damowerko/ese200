import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation as R

from ese2000_dynamical.config import Config
from ese2000_dynamical.simulator import dynamics_ca_drag


def optimize(A, B, t, points, noise_position=0.0, noise_velocity=0.0):
    """
    Find a trajectory that goes through loops through all the points.
    Assuming that all points are visited at equal time intervals.
    The trajectory should have minimum acceleration.
    """
    import cvxpy as cp

    u = cp.Variable((len(t), 2))
    x = cp.Variable((len(t), 4))

    # assume we pass through all points at equal time intervals
    # we also assume that the first point is also the last point for loop closure
    point_idx = np.linspace(0, len(t) - 1, len(points) + 1).astype(int)[:-1]
    point_idx[0] = 0

    noise = np.random.normal(
        scale=np.array([noise_position] * 2 + [noise_velocity] * 2)
        * np.diff(t)[:, None],
        size=(len(t) - 1, 4),
    )

    # objective is to minimize the sum of squared accelerations
    objective = cp.sum_squares(u)
    constraints = [
        # dynamics
        x[1:] == x[:-1] @ A.T + u[:-1] @ B.T + noise,
        # pass through points
        x[point_idx, :2] == points,
        # final state should be equal to initial state
        x[-1] == x[0],
    ]

    # solve optimization problem
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()
    return x.value, u.value


def generate_expert_trajectory(noise=True):
    rng = np.random.default_rng()
    A, B = dynamics_ca_drag(Config.time_step, Config.drag_coefficient)
    t = np.arange(0, Config.duration, Config.time_step)

    # generate random trajectories
    points = (
        np.asarray(
            [
                [-1, 0],
                [-1, 2],
                [0, 3],
                [1, 2.5],
                [1, 0],
                [2, 0],
                [2, 1],
                [3, 2],
                [3, 0],
                [2, -1],
                [0, -1],
            ]
        )
        * Config.trajectory_scale
    )
    if noise:
        points += rng.normal(
            scale=Config.trajectory_noise,
            size=points.shape,
        )
        x, u = optimize(A, B, t, points, Config.noise_position, Config.noise_velocity)
    else:
        x, u = optimize(A, B, t, points)
    return x, u


def generate_track_bounds():
    t = np.arange(0, Config.duration, Config.time_step)
    x, _ = generate_expert_trajectory(noise=False)

    spline = make_interp_spline(t, x[:, :2])
    dx = spline.derivative(1)(t)

    rot = R.from_euler("z", np.pi / 2).as_matrix()[:2, :2]
    right = dx @ rot
    right /= np.linalg.norm(right, axis=1)[:, None]

    # generate track bounds
    inside = x[:, :2] + right * Config.trajectory_width
    outside = x[:, :2] - right * Config.trajectory_width
    return inside, outside


def generate_expert_trajectories():
    x, u = zip(
        *[generate_expert_trajectory(noise=True) for _ in range(Config.n_trajectories)]
    )
    # stack x and u
    x = np.stack(x, axis=0)
    u = np.stack(u, axis=0)
    return x, u


if __name__ == "__main__":
    inside, outside = generate_track_bounds()
    np.savez("data/track.npz", inside=inside, outside=outside)

    x, u = generate_expert_trajectories()
    np.save("data/states.npy", x)
    np.save("data/inputs.npy", u)
