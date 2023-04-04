from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yaml


@dataclass
class Track:
    map: npt.NDArray[np.uint8]
    map_data: dict
    # x_m, y_m, w_tr_right_m, w_tr_left_m
    centerline: npt.NDArray[np.float64]
    # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
    raceline: npt.NDArray[np.float64]


def load_track(path: str) -> Track:
    map = plt.imread(f"{path}_map.png")
    with open(f"{path}_map.yaml") as f:
        map_data = yaml.load(f, Loader=yaml.CLoader)
    centerline = np.loadtxt(f"{path}_centerline.csv", delimiter=",")
    raceline = np.loadtxt(f"{path}_raceline.csv", delimiter=";")
    return Track(map, map_data, centerline, raceline)


def plot_track(track: Track):
    plt.figure(figsize=(10, 10))

    resolution = track.map_data["resolution"]
    plt.imshow(
        track.map,
        origin="upper",
        extent=[
            track.map_data["origin"][0],
            track.map_data["origin"][0] + track.map.shape[1] * resolution,
            track.map_data["origin"][1],
            track.map_data["origin"][1] + track.map.shape[0] * resolution,
        ],
        cmap="gray",
    )
    plt.plot(track.raceline[:, 1], track.raceline[:, 2], "r")
    plt.show()


if __name__ == "__main__":
    track = load_track()
    plot_track(track)
    plt.show()
