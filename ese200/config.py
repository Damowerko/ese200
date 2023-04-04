from dataclasses import dataclass


@dataclass
class Config:
    duration: float = 30.0
    time_step: float = 0.1
    n_points: int = 5
    width: float = 10.0
    n_trajectories = 10
