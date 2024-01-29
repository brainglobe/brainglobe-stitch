from typing import Tuple

import dask.array as da
import numpy as np
import numpy.typing as npt


class Tile:
    def __init__(self, tile_name: str, tile_id: int):
        self.name: str = tile_name
        self.id: int = tile_id
        self.position: Tuple[int, int, int] = (0, 0, 0)
        self.stitched_position: Tuple[int, int, int] = (0, 0, 0)
        self.downsampled_factors: npt.ArrayLike = np.array([4, 4, 4])
        self.downsampled_data: da.Array | None = None
        self.data: da.Array | None = None
        self.channel_id: int = -1
        self.channel_name: str | None = None
        self.tile_id: int = -1
        self.illumination_id: int = -1
        self.angle: float = -1000.0

    def set_position(self, position: Tuple[int, int, int]):
        self.position = position

    @property
    def data(self):
        if self.data is None:
            raise ValueError("Tile data not set")

        return self.data

    @data.setter
    def data(self, data: da.Array):
        self.data = data

    @property
    def downsampled_data(self):
        if self.downsampled_data is None:
            raise ValueError("Tile downsampled data not set")

        return self.downsampled_data

    @downsampled_data.setter
    def downsampled_data(self, data: da.Array):
        self.downsampled_data = data
