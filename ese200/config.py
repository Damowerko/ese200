class Config:
    duration: float = 30.0
    time_step: float = 0.1
    drag_coefficient: float = 0.1
    # parameters used for trajectory generation
    n_trajectories = 10
    noise_position = 0.1
    noise_velocity = 0.2
    trajectory_scale = 5.0
    trajectory_width = 1.0
    trajectory_noise = 0.3
