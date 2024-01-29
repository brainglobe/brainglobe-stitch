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
        self.downsampled_data: da.Array | None = None
        self.downsampled_factors: npt.ArrayLike = np.array([4, 4, 4])
        self.data: da.Array | None = None
        self.channel_id: int | None = None
        self.channel_name: str | None = None
        self.tile_id: int | None = None
        self.illumination_id: int | None = None
        self.angle: float | None = None

    def set_position(self, position: Tuple[int, int, int]):
        self.position = position
