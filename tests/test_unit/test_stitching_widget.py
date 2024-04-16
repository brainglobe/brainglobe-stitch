import dask.array as da
import napari.layers
import numpy as np

from brainglobe_stitch.stitching_widget import (
    add_tiles_from_mosaic,
)


def test_add_tiles_from_mosaic():
    num_tiles = 4

    test_data = []
    for i in range(num_tiles):
        test_data.append((da.ones((10, 10, 10)), np.array([i, i, i])))

    tile_names = [f"s{i:02}" for i in range(num_tiles)]

    for data, tile in zip(
        test_data, add_tiles_from_mosaic(test_data, tile_names)
    ):
        assert isinstance(tile, napari.layers.Image)
        assert (tile.data == data[0]).all()
        assert (tile.translate == data[1]).all()
