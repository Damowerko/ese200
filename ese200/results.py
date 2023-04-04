import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from ese200.config import Config


def main():
    config = Config()
    t = np.arange(0, config.duration, config.time_step)
    x_truth = np.load("data/states.npy")
    u = np.load("data/inputs.npy")
    points = np.load("data/points.npy")
    A = np.load("data/A.npy")
    B = np.load("data/B.npy")

    x_estimate = np.zeros((config.n_trajectories, len(t), A.shape[0]))
    # integrate model forward using the starting point and the provided inputs
    x_estimate[:, 0, :2] = points[:, 0, :2]
    for i in range(len(t) - 1):
        x_estimate[:, i + 1] = x_estimate[:, i] @ A.T + u[:, i] @ B.T
    for idx in trange(config.n_trajectories):
        # plot true and estimated trajectories
        plt.figure()
        plt.plot(x_truth[idx, :, 0], x_truth[idx, :, 1], "-", label="true")
        plt.plot(x_estimate[idx, :, 0], x_estimate[idx, :, 1], "-", label="estimated")
        plt.plot(points[idx, :, 0], points[idx, :, 1], "o", label="points")
        plt.legend()
        plt.grid(True)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.savefig(f"figures/{idx}.png")
        plt.close()


if __name__ == "__main__":
    main()
