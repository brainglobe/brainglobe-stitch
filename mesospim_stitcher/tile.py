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
        self.channel_id: int = int(attributes["channel"])
        self.channel_name: str | None = None
        self.tile_id: int = int(attributes["tile"])
        self.illumination_id: int = int(attributes["illumination"])
        self.angle: float = float(attributes["angle"])


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

    def extract_tile_overlaps(
        self, resolution_level: int
    ) -> Tuple[da.Array, da.Array]:
        scaled_coordinates = self.local_coords[resolution_level]
        scaled_size = self.size[resolution_level]

        i_overlap = self.tiles[0].data_pyramid[resolution_level][
            scaled_coordinates[0][0] : scaled_coordinates[0][0]
            + scaled_size[0],
            scaled_coordinates[0][1] : scaled_coordinates[0][1]
            + scaled_size[1],
            scaled_coordinates[0][2] : scaled_coordinates[0][2]
            + scaled_size[2],
        ]

        j_overlap = self.tiles[1].data_pyramid[resolution_level][
            scaled_coordinates[1][0] : scaled_coordinates[1][0]
            + scaled_size[0],
            scaled_coordinates[1][1] : scaled_coordinates[1][1]
            + scaled_size[1],
            scaled_coordinates[1][2] : scaled_coordinates[1][2]
            + scaled_size[2],
        ]

        return i_overlap, j_overlap

    def replace_overlap_data(self, resolution_level: int, new_data: da.Array):
        scaled_coordinates = self.local_coords[resolution_level]
        scaled_size = self.size[resolution_level]

        self.tiles[0].data_pyramid[resolution_level][
            scaled_coordinates[0][0] : scaled_coordinates[0][0]
            + scaled_size[0],
            scaled_coordinates[0][1] : scaled_coordinates[0][1]
            + scaled_size[1],
            scaled_coordinates[0][2] : scaled_coordinates[0][2]
            + scaled_size[2],
        ] = new_data

        self.tiles[1].data_pyramid[resolution_level][
            scaled_coordinates[1][0] : scaled_coordinates[1][0]
            + scaled_size[0],
            scaled_coordinates[1][1] : scaled_coordinates[1][1]
            + scaled_size[1],
            scaled_coordinates[1][2] : scaled_coordinates[1][2]
            + scaled_size[2],
        ] = new_data
