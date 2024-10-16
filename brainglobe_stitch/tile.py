from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import numpy.typing as npt


class Tile:
    """
    Tile class to store information about a single tile of an image.

    Attributes
    ----------
    name : str
        The name of the tile.
    id : int
        The id of the tile (unique for each tile).
    position : List[int]
        The position of the tile in the fused image. Initialized to 0, 0, 0.
    neighbours : List[int]
        The ids of the neighbouring tiles.
    data_pyramid : List[da.Array]
        The pyramid of data arrays for the tile. The full resolution data
        is at index 0.
    resolution_pyramid : npt.NDArray
        The downsample factors for each pyramid level.
    channel_id : int
        The id of the channel.
    channel_name : str
        The name of the channel.
    tile_id : int
        The id of the tile based on its position in the image.
        Two tiles with the same tile_id are in the same position in
        different channels.
    illumination_id : int
        The id of the illumination side.
    angle : float
        The angle of sample rotation.
    """

    def __init__(
        self,
        tile_name: str,
        tile_id: int,
        attributes: Dict[str, Union[str, int, float]],
    ):
        self.name: str = tile_name
        self.id: int = tile_id
        self.position: List[int] = [0, 0, 0]
        self.neighbours: List[int] = []
        self.data_pyramid: List[da.Array] = []
        self.resolution_pyramid: npt.NDArray = np.array([])
        self.channel_name: Optional[str] = None
        self.channel_id: int = int(attributes["channel"])
        self.tile_id: int = int(attributes["tile"])
        self.illumination_id: int = int(attributes["illumination"])
        self.angle: float = float(attributes["angle"])


class Overlap:
    """
    Overlap class to store information about the overlap between two tiles.

    Attributes
    ----------
    coordinates : npt.NDArray
        The coordinates for the start of the overlap in the fused image.
    size : List[npt.NDArray]
        The size of the overlap in the fused image for each resolution pyramid.
    tiles : Tuple[Tile, Tile]
        The overlapping tiles.
    local_coordinates : List[Tuple[npt.NDArray, npt.NDArray]]
        The coordinates for the start of the overlap in each tile's
        data pyramid.
    """

    def __init__(
        self,
        overlap_coordinates: npt.NDArray,
        overlap_size: npt.NDArray,
        tile_i: Tile,
        tile_j: Tile,
    ):
        """
        Initialize the overlap.

        Parameters
        ----------
        overlap_coordinates: npt.NDArray
            The starting coordinates for the overlap in the fused image.
        overlap_size: npt.NDArray
            The size of the overlap in the fused image [z,y,x].
        tile_i: Tile
            The first tile.
        tile_j: Tile
            The second tile.
        """
        self.coordinates: npt.NDArray = overlap_coordinates
        self.size: List[npt.NDArray] = [overlap_size]
        self.tiles: Tuple[Tile, Tile] = (tile_i, tile_j)
        self.local_coordinates: List[Tuple[npt.NDArray, npt.NDArray]] = [
            (
                np.zeros(self.coordinates.shape),
                np.zeros(self.coordinates.shape),
            ),
        ]
        self.get_local_overlap_coordinates()

    def get_local_overlap_coordinates(self) -> None:
        """
        Calculate the starting coordinates for the overlap in
        each tile's data pyramid.
        """
        tile_shape = self.tiles[0].data_pyramid[0].shape
        resolution_pyramid = self.tiles[0].resolution_pyramid

        for i in range(self.coordinates.shape[0]):
            if self.tiles[0].position[i] < self.tiles[1].position[i]:
                self.local_coordinates[0][0][i] = (
                    tile_shape[i] - self.size[0][i]
                )
                self.local_coordinates[0][1][i] = 0
            else:
                self.local_coordinates[0][0][i] = 0
                self.local_coordinates[0][1][i] = (
                    tile_shape[i] - self.size[0][i]
                )

        for j in range(1, len(resolution_pyramid)):
            self.local_coordinates.append(
                (
                    self.local_coordinates[0][0] // resolution_pyramid[j],
                    self.local_coordinates[0][1] // resolution_pyramid[j],
                )
            )
            self.size.append(self.size[0] // resolution_pyramid[j])

    def extract_tile_overlaps(
        self, resolution_level: int
    ) -> Tuple[da.Array, da.Array]:
        """
        Extract the overlap data from both tiles at a given resolution level.

        Parameters
        ----------
        resolution_level: int
            The resolution level to extract the overlap data from.

        Returns
        -------
        Tuple[da.Array, da.Array]
            The overlapping data for both tiles.
        """
        scaled_coordinates = self.local_coordinates[resolution_level]
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
        """
        Replace the overlapping data for both tiles at a given resolution
        level.

        Parameters
        ----------
        resolution_level: int
            The resolution level to replace the overlapping data at.
        new_data:
            The new data to replace the overlapping data with.
        """
        scaled_coordinates = self.local_coordinates[resolution_level]
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

    def linear_interpolation(
        self, resolution_level: int, tile_shape: Tuple[int, ...]
    ):
        # Extract overlapping data from both tiles
        i_overlap, j_overlap = self.extract_tile_overlaps(resolution_level)

        x_overlap_size = self.size[resolution_level][2]
        y_overlap_size = self.size[resolution_level][1]

        z_size, y_size, x_size = tile_shape
        tile_i, tile_j = self.tiles

        if x_overlap_size / x_size < 0.4 and y_overlap_size / y_size < 0.4:
            # Skip the small diagonal overlaps
            return
        elif x_overlap_size / x_size < 0.4:
            # The thin overlap is in the x dimension
            x_lin = np.linspace(1, 0, x_overlap_size)

            # 1 in the first column,
            # linearly decreasing to 0 in the last column
            yx_grid = np.tile(x_lin, (y_overlap_size, 1))

            # The decreasing image is the one for which the overlap
            # is at the end of the x dimension
            if tile_i.position[2] < tile_j.position[2]:
                decreasing_image = i_overlap
                increasing_image = j_overlap
            else:
                decreasing_image = j_overlap
                increasing_image = i_overlap
        else:
            # The thin overlap is in the y dimension
            y_lin = np.linspace(1, 0, y_overlap_size)

            # 1 in the first row,
            # linearly decreasing to 0 in the last row
            yx_grid = np.tile(y_lin, (x_overlap_size, 1)).T

            # The decreasing image is the one for which the overlap
            # is at the end of the y dimension
            if tile_i.position[1] < tile_j.position[1]:
                decreasing_image = i_overlap
                increasing_image = j_overlap
            else:
                decreasing_image = j_overlap
                increasing_image = i_overlap

        interp = (
            np.multiply(
                decreasing_image.compute(),
                yx_grid,
                dtype=np.float16,
            )
            + np.multiply(
                increasing_image.compute(),
                1 - yx_grid,
                dtype=np.float16,
            )
        ).astype(np.int16)

        # Replace the overlap data with the interpolated data
        # Allows future interpolation to be done on the
        # interpolated data
        self.replace_overlap_data(resolution_level, interp)
