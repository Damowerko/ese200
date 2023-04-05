from dataclasses import dataclass


@dataclass
class Config:
    duration: float = 30.0
    time_step: float = 0.1
    width: float = 10.0
    n_trajectories = 10
    trajectory_radius = 5.0
    trajectory_margin = 1.5
    trajectory_safety = 0.1
    trajectory_noise = 0.4
    state_noise = 0.1
