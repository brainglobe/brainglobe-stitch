import os
from pathlib import Path

import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic

NUM_TILES = 8
NUM_RESOLUTIONS = 5
NUM_CHANNELS = 2
NUM_OVERLAPS = 12
TILE_SIZE = (107, 128, 128)
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
EXPECTED_OVERLAP_COORDINATES = [
    [3, 120, 2],
    [6, 7, 118],
    [5, 123, 116],
    [6, 120, 118],
    [5, 123, 116],
    [3, 120, 2],
    [6, 7, 118],
    [5, 123, 116],
    [6, 120, 118],
    [5, 123, 116],
    [6, 123, 118],
    [6, 123, 118],
]
EXPECTED_OVERLAP_SIZE = [
    [106, 12, 126],
    [104, 125, 12],
    [105, 9, 14],
    [103, 15, 10],
    [104, 125, 12],
    [106, 12, 126],
    [104, 125, 12],
    [105, 9, 14],
    [103, 15, 10],
    [104, 125, 12],
    [106, 12, 126],
    [106, 12, 126],
]
EXPECTED_TILE_POSITIONS = [
    [3, 4, 2],
    [2, 120, 0],
    [3, 4, 2],
    [2, 120, 0],
    [6, 7, 118],
    [5, 123, 116],
    [6, 7, 118],
    [5, 123, 116],
]


@pytest.fixture(scope="module")
def image_mosaic(naive_bdv_directory):
    os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")
    image_mosaic = ImageMosaic(naive_bdv_directory)

    yield image_mosaic

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
    assert len(image_mosaic.intensity_adjusted) == NUM_RESOLUTIONS
    assert len(image_mosaic.overlaps_interpolated) == NUM_RESOLUTIONS


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


def test_stitch(mocker, image_mosaic, naive_bdv_directory):
    mock_completed_process = mocker.patch(
        "subprocess.CompletedProcess", autospec=True
    )
    mock_run_big_stitcher = mocker.patch(
        "brainglobe_stitch.image_mosaic.run_big_stitcher",
        return_value=mock_completed_process,
    )
    mock_completed_process.stdout = ""
    mock_completed_process.stderr = ""

    fiji_path = Path("/path/to/fiji")
    resolution_level = 2
    selected_channel = "567 nm"

    image_mosaic.stitch(fiji_path, resolution_level, selected_channel)

    mock_run_big_stitcher.assert_called_once()

    assert len(image_mosaic.overlaps) == NUM_OVERLAPS

    for idx, overlap in enumerate(image_mosaic.overlaps):
        assert (
            image_mosaic.overlaps[overlap].coordinates
            == EXPECTED_OVERLAP_COORDINATES[idx]
        ).all()
        assert (
            image_mosaic.overlaps[overlap].size[0]
            == EXPECTED_OVERLAP_SIZE[idx]
        ).all()


def test_data_for_napari(image_mosaic):
    data = image_mosaic.data_for_napari(0)

    assert len(data) == NUM_TILES

    for i in range(NUM_TILES):
        assert data[i][0].shape == TILE_SIZE
        assert (data[i][1] == EXPECTED_TILE_POSITIONS[i]).all()


def test_normalise_intensity(image_mosaic):
    image_mosaic.normalise_intensity(2)

    assert image_mosaic.intensity_adjusted[2]
