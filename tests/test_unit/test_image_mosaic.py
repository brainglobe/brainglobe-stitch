import os

import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic


@pytest.fixture(scope="module")
def image_mosaic(naive_bdv_directory):
    """
    Fixture for creating an ImageMosaic object for testing. A clean directory
    is created for this module using the naive_bdv_directory fixture. Tests
    using this fixture will modify the directory.

    The __del__ method is called at the end of the module to close any open h5
    files.

    Yields
    ------
    ImageMosaic
        An ImageMosaic object for testing.
    """
    os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")
    image_mosaic = ImageMosaic(naive_bdv_directory)

    yield image_mosaic

    # Explicit call to close open h5 files
    image_mosaic.__del__()


def test_image_mosaic_init(image_mosaic, naive_bdv_directory, test_constants):
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
    assert len(image_mosaic.channel_names) == test_constants["NUM_CHANNELS"]
    assert image_mosaic.channel_names == test_constants["CHANNELS"]
    assert len(image_mosaic.tiles) == test_constants["NUM_TILES"]
    assert len(image_mosaic.tile_names) == test_constants["NUM_TILES"]
    assert image_mosaic.x_y_resolution == test_constants["PIXEL_SIZE_XY"]
    assert image_mosaic.z_resolution == test_constants["PIXEL_SIZE_Z"]
    assert image_mosaic.num_channels == test_constants["NUM_CHANNELS"]


def test_write_big_stitcher_tile_config(
    image_mosaic, naive_bdv_directory, test_constants
):
    """
    Test the write_big_stitcher_tile_config method of the ImageMosaic class.
    The expected result is a file with the same contents as
    test_constants["EXPECTED_TILE_CONFIG"].
    """
    # Remove the test_data_bdv_tile_config.txt file if it exists
    if (naive_bdv_directory / "test_data_bdv_tile_config.txt").exists():
        os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")

    image_mosaic.write_big_stitcher_tile_config(
        naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )

    assert (naive_bdv_directory / "test_data_bdv_tile_config.txt").exists()

    with open(naive_bdv_directory / "test_data_bdv_tile_config.txt", "r") as f:
        for line, expected in zip(
            f.readlines(), test_constants["EXPECTED_TILE_CONFIG"]
        ):
            assert line.strip() == expected


def test_stitch(mocker, image_mosaic, naive_bdv_directory, test_constants):
    """
    Ensure that the stitch method calls run_big_stitcher with the correct
    arguments.
    """
    mock_completed_process = mocker.patch(
        "subprocess.CompletedProcess", autospec=True
    )
    mock_run_big_stitcher = mocker.patch(
        "brainglobe_stitch.image_mosaic.run_big_stitcher",
        return_value=mock_completed_process,
    )
    mock_completed_process.stdout = ""
    mock_completed_process.stderr = ""

    fiji_path = test_constants["MOCK_IMAGEJ_PATH"]
    resolution_level = 2
    selected_channel = test_constants["CHANNELS"][0]
    selected_channel_int = int(selected_channel.split()[0])
    downsample_z, downsample_y, downsample_x = tuple(
        image_mosaic.tiles[0].resolution_pyramid[resolution_level]
    )
    image_mosaic.stitch(fiji_path, resolution_level, selected_channel)

    mock_run_big_stitcher.assert_called_once_with(
        fiji_path,
        naive_bdv_directory / "test_data_bdv.xml",
        naive_bdv_directory / "test_data_bdv_tile_config.txt",
        False,
        selected_channel_int,
        downsample_x=downsample_x,
        downsample_y=downsample_y,
        downsample_z=downsample_z,
    )


def test_data_for_napari(image_mosaic, test_constants):
    """
    Checks the return of the data_for_napari method. Each element of the
    returned list should be a tuple containing the tile data and the expected
    position of the tile in the fused image. The expected results are stored
    in the dictionary returned by the test_constants fixture.
    """
    data = image_mosaic.data_for_napari(0)

    assert len(data) == test_constants["NUM_TILES"]

    for tile_data, expected_pos in zip(
        data, test_constants["EXPECTED_TILE_POSITIONS"]
    ):
        assert tile_data[0].shape == test_constants["TILE_SIZE"]
        assert (tile_data[1] == expected_pos).all()
