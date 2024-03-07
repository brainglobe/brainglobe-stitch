import os
from pathlib import Path

import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic

EXPECTED_TILE_CONFIG = [
    "dim=3",
    "00;;(0,0,0)",
    "01;;(0,115,0)",
    "02;;(0,0,0)",
    "03;;(0,115,0)",
    "04;;(115,0,0)",
    "05;;(115,115,0)",
    "06;;(115,0,0)",
    "07;;(115,115,0)",
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
    assert len(image_mosaic.channel_names) == 2
    assert len(image_mosaic.tiles) == 8
    assert len(image_mosaic.tile_names) == 8
    assert image_mosaic.x_y_resolution == 4.08
    assert image_mosaic.z_resolution == 5.0
    assert image_mosaic.num_channels == 2
    assert len(image_mosaic.intensity_adjusted) == 5
    assert len(image_mosaic.overlaps_interpolated) == 5


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

    assert len(image_mosaic.overlaps) == 12
