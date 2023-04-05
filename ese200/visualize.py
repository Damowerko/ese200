from matplotlib.patches import Circle

from ese200.config import Config


def plot_obstacles(ax):
    config = Config()
    ax.set_aspect("equal")
    ax.add_patch(
        Circle(
            (-config.trajectory_radius, 0),
            config.trajectory_radius - config.trajectory_margin,
            color="k",
            fill=False,
        )
    )
    ax.add_patch(
        Circle(
            (config.trajectory_radius, 0),
            config.trajectory_radius - config.trajectory_margin,
            color="k",
            fill=False,
        )
    )
