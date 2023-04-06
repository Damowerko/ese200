class Config:
    duration: float = 30.0
    time_step: float = 0.1
    drag_coefficient: float = 0.1
    # parameters used for trajectory generation
    n_trajectories = 10
    noise_position = 0.1
    noise_velocity = 0.1
    trajectory_radius = 5.0
    trajectory_margin = 1.5
    trajectory_safety = 0.1
    trajectory_noise = 0.4
