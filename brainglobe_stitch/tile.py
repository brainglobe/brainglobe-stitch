from typing import Dict, List

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
        attributes: Dict[str, str],
    ):
        self.name: str = tile_name
        self.id: int = tile_id
        self.position: List[int] = [0, 0, 0]
        self.neighbours: List[int] = []
        self.data_pyramid: List[da.Array] = []
        self.resolution_pyramid: npt.NDArray = np.array([])
        self.channel_name: str = ""
        self.channel_id: int = int(attributes["channel"])
        self.tile_id: int = int(attributes["tile"])
        self.illumination_id: int = int(attributes["illumination"])
        self.angle: float = float(attributes["angle"])
