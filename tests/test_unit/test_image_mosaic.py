import os

import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic

# The tiles lie in one z-plane and are arranged in a 2x2 grid with 2 channels.
# Each tile is 128x128x107 pixels (x, y, z).
# The tiles overlap by 10% in x and y (13 pixels).
# The tiles are arranged in the following pattern:
# channel 0   | channel 1
# 00 10          | 04 05
# 01 11           | 14 15
NUM_TILES = 8
NUM_RESOLUTIONS = 5
NUM_CHANNELS = 2
TILE_SIZE = (107, 128, 128)
# Expected tile config for the test data in test_data_bdv.h5
# The tile positions are in pixels in x, y, z order
EXPECTED_TILE_CONFIG = [
    "dim=3",
    "00;;(0,0,0)",
    "01;;(0,115,0)",
    "04;;(0,0,0)",
    "05;;(0,115,0)",
    "10;;(115,0,0)",
    "11;;(115,115,0)",
    "14;;(115,0,0)",
    "15;;(115,115,0)",
]
# Expected tile positions for the test data in test_data_bdv.h5
# The tile positions are in pixels in z, y, x order
EXPECTED_TILE_POSITIONS = [
    [0, 0, 0],
    [0, 115, 0],
    [0, 0, 0],
    [0, 115, 0],
    [0, 0, 115],
    [0, 115, 115],
    [0, 0, 115],
    [0, 115, 115],
]


@pytest.fixture(scope="module")
def image_mosaic(naive_bdv_directory):
    os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")
    image_mosaic = ImageMosaic(naive_bdv_directory)

    yield image_mosaic

    # Explicit call to clean up open h5 files
    image_mosaic.__del__()


def test_image_mosaic_init(image_mosaic, naive_bdv_directory):
    image_mosaic = image_mosaic
    assert image_mosaic.xml_path == naive_bdv_directory / "test_data_bdv.xml"
    assert (
        image_mosaic.meta_path
        == naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )
    assert image_mosaic.h5_path == naive_bdv_directory / "test_data_bdv.h5"
    assert (
        image_mosaic.meta_path
        == naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )
    assert image_mosaic.h5_file is not None
    assert len(image_mosaic.channel_names) == NUM_CHANNELS
    assert len(image_mosaic.tiles) == NUM_TILES
    assert len(image_mosaic.tile_names) == NUM_TILES
    assert image_mosaic.x_y_resolution == 4.08
    assert image_mosaic.z_resolution == 5.0
    assert image_mosaic.num_channels == NUM_CHANNELS


def test_write_big_stitcher_tile_config(image_mosaic, naive_bdv_directory):
    if (naive_bdv_directory / "test_data_bdv_tile_config.txt").exists():
        os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")

    image_mosaic.write_big_stitcher_tile_config(
        naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )

    assert (naive_bdv_directory / "test_data_bdv_tile_config.txt").exists()

    with open(naive_bdv_directory / "test_data_bdv_tile_config.txt", "r") as f:
        for idx, line in enumerate(f):
            assert line.strip() == EXPECTED_TILE_CONFIG[idx]


def test_data_for_napari(image_mosaic):
    data = image_mosaic.data_for_napari(0)

    assert len(data) == NUM_TILES

    for i in range(NUM_TILES):
        assert data[i][0].shape == TILE_SIZE
        assert (data[i][1] == EXPECTED_TILE_POSITIONS[i]).all()
