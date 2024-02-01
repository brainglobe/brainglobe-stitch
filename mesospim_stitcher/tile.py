from typing import Dict, List, Tuple

import dask.array as da
import numpy as np
import numpy.typing as npt


class Tile:
    def __init__(
        self,
        tile_name: str,
        tile_id: int,
        attributes: Dict[str, str | int | float],
    ):
        self.name: str = tile_name
        self.id: int = tile_id
        self.position: List[int] = [0, 0, 0]
        self.neighbours: List[int] = []
        self.stitched_position: List[int] = [0, 0, 0]
        self.data_pyramid: List[da.Array] = []
        self.resolution_pyramid: npt.NDArray = np.array([1, 1, 1])
        self.downsampled_factors: npt.ArrayLike = np.array([4, 4, 4])
        self._downsampled_data: da.Array | None = None
        self._data: da.Array | None = None
        self.channel_id: int = int(attributes["channel"])
        self.channel_name: str | None = None
        self.tile_id: int = int(attributes["tile"])
        self.illumination_id: int = int(attributes["illumination"])
        self.angle: float = float(attributes["angle"])

    def set_position(self, position: Tuple[int, int, int]):
        self.position = list(position)

    @property
    def data(self):
        if self._data is None:
            raise ValueError("Tile data not set")

        return self._data

    @data.setter
    def data(self, data: da.Array):
        self._data = data

    @property
    def downsampled_data(self):
        if self._downsampled_data is None:
            raise ValueError("Tile downsampled data not set")

        return self._downsampled_data

    @downsampled_data.setter
    def downsampled_data(self, downsampled_data: da.Array):
        self._downsampled_data = downsampled_data


class Overlap:
    def __init__(
        self,
        overlap_coordinates: npt.NDArray,
        overlap_size: npt.NDArray,
        tile_i: Tile,
        tile_j: Tile,
    ):
        self.coordinates: npt.NDArray = overlap_coordinates
        self.size: List[npt.NDArray] = [overlap_size]
        self.tiles: Tuple[Tile, Tile] = (tile_i, tile_j)
        self.local_coordinates: Tuple[npt.NDArray, npt.NDArray] = (
            np.zeros((self.coordinates.shape[0], 2)),
            np.zeros((self.coordinates.shape[0], 2)),
        )

        self.local_coords: List[Tuple[npt.NDArray, npt.NDArray]] = [
            (
                np.zeros(self.coordinates.shape),
                np.zeros(self.coordinates.shape),
            ),
        ]
        self.get_local_overlap_indices(self.tiles)

    def get_local_overlap_indices(self, tiles: Tuple[Tile, Tile]) -> None:
        tile_shape = self.tiles[0].data_pyramid[0].shape
        resolution_pyramid = tiles[0].resolution_pyramid

        for i in range(self.coordinates.shape[0]):
            if tiles[0].position[i] < tiles[1].position[i]:
                self.local_coords[0][0][i] = tile_shape[i] - self.size[0][i]
                self.local_coords[0][1][i] = 0
            else:
                self.local_coords[0][0][i] = 0
                self.local_coords[0][1][i] = tile_shape[i] - self.size[0][i]

        for j in range(1, len(resolution_pyramid)):
            self.local_coords.append(
                (
                    self.local_coords[0][0] // resolution_pyramid[j],
                    self.local_coords[0][1] // resolution_pyramid[j],
                )
            )
            self.size.append(self.size[0] // resolution_pyramid[j])
