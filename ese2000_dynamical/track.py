from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class Track:
    def __init__(self, inside, outside) -> None:
        self.inside = inside
        self.outside = outside

    def plot(self, ax=None):
        ax = ax or plt.gca()
        ax.plot(self.inside[:, 0], self.inside[:, 1], color="k")
        ax.plot(self.outside[:, 0], self.outside[:, 1], color="k")


def load_track(bounds_file: Union[str, Path]) -> Track:
    data = np.load(bounds_file)
    return Track(data["inside"], data["outside"])
