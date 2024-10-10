import os

import h5py
import numpy as np
import pytest
import zarr

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


@pytest.fixture(scope="module")
def fused_image(image_mosaic, test_constants):
    """
    Fixture for creating a fused image for testing. The fused image is created
    by iterating over the tiles in the image mosaic and placing them in
    reverse order of acquisition. This is used as a ground truth for testing
    the fusion functions.


    Parameters
    ----------
    image_mosaic
        ImageMosaic object loaded with test data.
    test_constants
        Dictionary containing constants for testing.

    Returns
    -------
    np.ndarray
        A 4D numpy array representing the fused image.
    """
    test_image = np.zeros(
        (image_mosaic.num_channels, *test_constants["EXPECTED_FUSED_SHAPE"]),
        dtype=np.int16,
    )
    z_size, y_size, x_size = test_constants["TILE_SIZE"]

    for tile in image_mosaic.tiles[-1::-1]:
        test_image[
            tile.channel_id,
            tile.position[0] : tile.position[0] + z_size,
            tile.position[1] : tile.position[1] + y_size,
            tile.position[2] : tile.position[2] + x_size,
        ] = tile.data_pyramid[0].compute()

    return test_image


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


def test_fuse_bdv_h5(image_mosaic, mocker, test_constants):
    mock_fuse_function = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_bdv_h5",
    )
    file_name = "fused.h5"

    image_mosaic.fuse(file_name)
    mock_fuse_function.assert_called_once_with(
        image_mosaic.xml_path.parent / file_name,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
    )


def test_fuse_zarr_file(image_mosaic, mocker, test_constants):
    file_name = "fused.zarr"

    mock_fuse_to_zarr = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr"
    )

    image_mosaic.fuse(file_name)

    mock_fuse_to_zarr.assert_called_once_with(
        image_mosaic.xml_path.parent / file_name,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )


@pytest.mark.parametrize(
    "pyramid_depth, num_channels",
    [(1, 1), (1, 2), (1, 5), (2, 1), (2, 2), (2, 5), (3, 1), (3, 2), (3, 5)],
)
def test_get_metadata_for_zarr(
    image_mosaic, pyramid_depth, num_channels, test_constants
):
    backup_num_channels = image_mosaic.num_channels
    image_mosaic.num_channels = num_channels

    metadata, axes = image_mosaic._generate_metadata_for_zarr(
        pyramid_depth,
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
    )

    if num_channels > 1:
        assert len(axes) == 4
        assert axes[0]["name"] == "c"
        assert axes[0]["type"] == "channel"
        axes.pop(0)
    else:
        assert len(axes) == 3

    expected_axes_names = ["z", "y", "x"]

    for idx, axes in enumerate(axes):
        assert axes["name"] == expected_axes_names[idx]
        assert axes["type"] == "space"
        assert axes["unit"] == "micrometer"

    for idx, transformation in enumerate(metadata):
        assert transformation[0]["type"] == "scale"
        expected_scale = [
            image_mosaic.z_resolution,
            image_mosaic.x_y_resolution * 2**idx,
            image_mosaic.x_y_resolution * 2**idx,
        ]
        if num_channels > 1:
            expected_scale.insert(0, 1)

        assert transformation[0]["scale"] == expected_scale

    image_mosaic.num_channels = backup_num_channels


def test_fuse_to_zarr(image_mosaic, test_constants, fused_image):
    pyramid_depth = 3

    output_file = image_mosaic.xml_path.parent / "fused.zarr"
    fused_image_shape = test_constants["EXPECTED_FUSED_SHAPE"]

    image_mosaic._fuse_to_zarr(
        output_file,
        fused_image_shape,
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        pyramid_depth,
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )

    assert output_file.exists()
    test_store = zarr.NestedDirectoryStore(str(output_file))
    root = zarr.group(store=test_store)

    assert root.attrs["multiscales"] is not None
    assert root.attrs["multiscales"][0]["axes"] is not None
    assert len(root.attrs["multiscales"][0]["datasets"]) == pyramid_depth
    assert root.attrs["omero"] is not None
    assert len(root.attrs["omero"]["channels"]) == image_mosaic.num_channels

    downsample_shape = [image_mosaic.num_channels, *fused_image_shape]
    assert root["0"].shape == tuple(downsample_shape)

    for i in range(1, pyramid_depth):
        downsample_shape[-2:] = [(x + 1) // 2 for x in downsample_shape[-2:]]
        assert root[str(i)].shape == tuple(downsample_shape)

    assert np.array_equal(np.array(root["0"]), fused_image)


def test_fuse_to_bdv_h5(image_mosaic, test_constants, fused_image):
    pyramid_depth = 3

    output_file = image_mosaic.xml_path.parent / "fused.h5"
    fused_image_shape = test_constants["EXPECTED_FUSED_SHAPE"]

    image_mosaic._fuse_to_bdv_h5(
        output_file,
        fused_image_shape,
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        pyramid_depth,
        test_constants["DEFAULT_CHUNK_SHAPE"],
    )

    assert output_file.exists()
    assert output_file.with_suffix(".xml").exists()

    expected_resolutions = np.ones((pyramid_depth, 3), dtype=np.int16)
    # Since one of the dimensions is less that the default chunk size of 128,
    # the expected subdivisions are simply the shape of the image at each
    # resolution level
    expected_subdivisions = np.array(
        [[246, 251, 113], [123, 126, 113], [62, 63, 113]], dtype=np.int16
    )

    for i in range(1, pyramid_depth):
        expected_resolutions[i, :-1] = 2**i

    with h5py.File(output_file, "r") as f:
        assert len(f["t00000"].keys()) == image_mosaic.num_channels
        # Extra group accounts for the t00000 group
        assert len(f.keys()) == image_mosaic.num_channels + 1

        for idx, tile_name in enumerate(f["t00000"].keys()):
            assert np.all(
                f[f"{tile_name}/resolutions"] == expected_resolutions
            )
            assert np.all(
                f[f"{tile_name}/subdivisions"] == expected_subdivisions
            )
            assert len(f[f"t00000/{tile_name}"].keys()) == pyramid_depth
            assert np.array_equal(
                f[f"t00000/{tile_name}/0/cells"], fused_image[idx, :, :, :]
            )
