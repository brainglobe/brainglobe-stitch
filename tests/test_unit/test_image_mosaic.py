import os
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import pytest
import zarr

from brainglobe_stitch.file_utils import get_slice_attributes, safe_find


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


def test_calculate_overlaps(image_mosaic, test_constants):
    """
    Ensure that the calculate_overlaps method calculates the overlaps
    correctly. The expected overlaps are stored in the dictionary returned by
    the test_constants fixture.
    """
    image_mosaic.calculate_overlaps()

    assert (
        len(image_mosaic.overlaps) == test_constants["EXPECTED_NUM_OVERLAPS"]
    )

    expected_overlap_coordinates = test_constants[
        "EXPECTED_OVERLAP_COORDINATES"
    ]
    expected_overlap_size = test_constants["EXPECTED_OVERLAP_SIZE"]

    for idx, overlap in enumerate(image_mosaic.overlaps):
        assert (
            image_mosaic.overlaps[overlap].coordinates
            == expected_overlap_coordinates[idx]
        ).all()
        assert (
            image_mosaic.overlaps[overlap].size[0]
            == expected_overlap_size[idx]
        ).all()


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


@pytest.mark.parametrize(
    "resolution_level",
    [0, 1],
)
def test_normalise_intensity(
    mocker, test_constants, image_mosaic, resolution_level
):
    def force_set_scale_factors(*args, **kwargs):
        image_mosaic.scale_factors = test_constants[
            "EXPECTED_INTENSITY_FACTORS"
        ]
        image_mosaic.intensity_adjusted[args[0]] = True

    mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.calculate_intensity_scale_factors",
        side_effect=force_set_scale_factors,
    )

    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    assert not image_mosaic.intensity_adjusted[resolution_level]
    image_mosaic.scale_factors = None

    image_mosaic.normalise_intensity(resolution_level)
    assert image_mosaic.intensity_adjusted[resolution_level]
    assert len(image_mosaic.scale_factors) == test_constants["NUM_TILES"]

    for i in range(test_constants["NUM_TILES"]):
        # Check that there each tile has the correct number of pending tasks
        # in the dask graph
        # Expect to have 4: 2 for loading the data, 2 for scaling the data
        # Since the resolution levels are less than 2, the scaling factors are
        # calculated on a different resolution level and then applied to the
        # current resolution level
        if test_constants["EXPECTED_INTENSITY_FACTORS"][i] != 1.0:
            assert (
                len(
                    image_mosaic.tiles[i]
                    .data_pyramid[resolution_level]
                    .dask.layers
                )
                == 4
            )


@pytest.mark.parametrize(
    "resolution_level",
    [2, 3, 4],
)
def test_normalise_intensity_done_with_factors(
    mocker, image_mosaic, resolution_level, test_constants
):
    def force_set_scale_factors(*args, **kwargs):
        image_mosaic.scale_factors = test_constants[
            "EXPECTED_INTENSITY_FACTORS"
        ]
        image_mosaic.intensity_adjusted[args[0]] = True

    mock_calc_intensity_factors = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.calculate_intensity_scale_factors",
        side_effect=force_set_scale_factors,
    )

    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    assert not image_mosaic.intensity_adjusted[resolution_level]
    image_mosaic.scale_factors = None

    image_mosaic.normalise_intensity(resolution_level)
    assert image_mosaic.intensity_adjusted[resolution_level]
    assert len(image_mosaic.scale_factors) == test_constants["NUM_TILES"]

    mock_calc_intensity_factors.assert_called_once_with(resolution_level, 80)

    # Check that no scale adjustment calculations are queued for the tiles
    # at the specified resolution level as the correction factors were
    # calculated based on this resolution level,
    # therefore no calculations queued.
    for i in range(test_constants["NUM_TILES"]):
        assert (
            len(
                image_mosaic.tiles[i]
                .data_pyramid[resolution_level]
                .dask.layers
            )
            == 2
        )


def test_calculate_intensity_scale_factors(image_mosaic, test_constants):
    resolution_level = 2
    percentile = 50
    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    image_mosaic.scale_factors = None

    image_mosaic.calculate_intensity_scale_factors(
        resolution_level, percentile
    )

    assert len(image_mosaic.scale_factors) == test_constants["NUM_TILES"]
    # Check the relative tolerance
    assert np.allclose(
        image_mosaic.scale_factors,
        test_constants["EXPECTED_INTENSITY_FACTORS"],
        rtol=1e-2,
    )


@pytest.mark.parametrize("resolution_level", [0, 1, 2, 3])
def test_interpolate_overlaps(
    image_mosaic, resolution_level, mocker, test_constants
):
    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    mock_linear_interpolation = mocker.patch(
        "brainglobe_stitch.tile.Overlap.linear_interpolation",
    )

    image_mosaic.interpolate_overlaps(resolution_level)

    assert image_mosaic.overlaps_interpolated[resolution_level]
    assert (
        mock_linear_interpolation.call_count
        == test_constants["EXPECTED_NUM_OVERLAPS"]
    )


def test_interpolate_overlaps_already_done(image_mosaic, mocker):
    resolution_level = 2
    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    image_mosaic.overlaps_interpolated[resolution_level] = True
    mock_linear_interpolation = mocker.patch(
        "brainglobe_stitch.tile.Overlap.linear_interpolation",
    )
    image_mosaic.interpolate_overlaps(resolution_level)

    assert image_mosaic.overlaps_interpolated[resolution_level]
    mock_linear_interpolation.assert_not_called()
    image_mosaic.reload_resolution_pyramid_level(resolution_level)


def test_fuse_invalid_file_type(image_mosaic):
    output_file = image_mosaic.xml_path.parent / "fused.txt"
    with pytest.raises(ValueError):
        image_mosaic.fuse(output_file)


def test_fuse_bdv_h5_defaults(image_mosaic, mocker, test_constants):
    mock_fuse_function = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_bdv_h5",
    )
    file_path = image_mosaic.xml_path.parent / "fused.h5"

    image_mosaic.fuse(file_path)
    mock_fuse_function.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
    )


def test_fuse_zarr_normalise_intensity(image_mosaic, mocker, test_constants):
    file_path = image_mosaic.xml_path.parent / "fused.zarr"

    mock_fuse_to_zarr = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr"
    )
    mock_normalise_intensity = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.normalise_intensity"
    )

    image_mosaic.fuse(file_path, normalise_intensity=True)

    mock_normalise_intensity.assert_called_once_with(0, 80)
    mock_fuse_to_zarr.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )


def test_fuse_zarr_normalise_already_done(
    image_mosaic, mocker, test_constants
):
    file_path = image_mosaic.xml_path.parent / "fused.zarr"

    mock_fuse_to_zarr = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr"
    )
    mock_normalise_intensity = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.normalise_intensity"
    )
    mock_reload_pyramid_level = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.reload_resolution_pyramid_level"
    )
    image_mosaic.intensity_adjusted[0] = True

    image_mosaic.fuse(file_path, normalise_intensity=False)

    mock_normalise_intensity.assert_not_called()
    mock_reload_pyramid_level.assert_called_once_with(0)
    mock_fuse_to_zarr.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )
    image_mosaic.intensity_adjusted[0] = False


def test_fuse_zarr_interpolate_already_done(
    image_mosaic, mocker, test_constants
):
    file_path = image_mosaic.xml_path.parent / "fused.zarr"

    mock_fuse_to_zarr = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr"
    )
    mock_interpolate_overlaps = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.interpolate_overlaps"
    )
    mock_reload_pyramid_level = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.reload_resolution_pyramid_level"
    )
    image_mosaic.overlaps_interpolated[0] = True

    image_mosaic.fuse(file_path, interpolate=False)

    mock_interpolate_overlaps.assert_not_called()
    mock_reload_pyramid_level.assert_called_once_with(0)
    mock_fuse_to_zarr.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )
    image_mosaic.overlaps_interpolated[0] = False


def test_fuse_zarr_interpolate(image_mosaic, mocker, test_constants):
    file_path = image_mosaic.xml_path.parent / "fused.zarr"

    mock_fuse_to_zarr = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr"
    )
    mock_interpolate_overlaps = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.interpolate_overlaps"
    )

    image_mosaic.fuse(file_path, interpolate=True)

    mock_interpolate_overlaps.assert_called_once_with(0)
    mock_fuse_to_zarr.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )


@pytest.mark.parametrize(
    "downscale_factors, chunk_shape, pyramid_depth",
    [((2, 2, 2), (64, 64, 64), 2), ((4, 4, 4), (32, 32, 32), 3)],
)
def test_fuse_bdv_h5_custom(
    image_mosaic,
    mocker,
    test_constants,
    downscale_factors,
    chunk_shape,
    pyramid_depth,
):
    mock_fuse_function = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_bdv_h5",
    )
    file_path = image_mosaic.xml_path.parent / "fused.h5"

    normalise_intensity = False
    interpolate = False
    image_mosaic.fuse(
        output_path=file_path,
        normalise_intensity=normalise_intensity,
        interpolate=interpolate,
        downscale_factors=downscale_factors,
        chunk_shape=chunk_shape,
        pyramid_depth=pyramid_depth,
    )
    mock_fuse_function.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        downscale_factors,
        pyramid_depth,
        chunk_shape,
    )


def test_fuse_zarr_file(image_mosaic, mocker, test_constants):
    file_path = image_mosaic.xml_path.parent / "fused.zarr"

    mock_fuse_to_zarr = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr"
    )

    image_mosaic.fuse(file_path)

    mock_fuse_to_zarr.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
        test_constants["DEFAULT_PYRAMID_DEPTH"],
        test_constants["DEFAULT_CHUNK_SHAPE"],
        test_constants["DEFAULT_COMPRESSION_METHOD"],
        test_constants["DEFAULT_COMPRESSION_LEVEL"],
    )


@pytest.mark.parametrize(
    "downscale_factors, chunk_shape, pyramid_depth, "
    "compression_method, compression_level",
    [
        ((2, 2, 2), (64, 64, 64), 2, "blosclz", 3),
        ((4, 4, 4), (32, 32, 32), 3, "lz4hc", 9),
    ],
)
def test_fuse_bdv_zarr_custom(
    image_mosaic,
    mocker,
    test_constants,
    downscale_factors,
    chunk_shape,
    pyramid_depth,
    compression_method,
    compression_level,
):
    mock_fuse_function = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic._fuse_to_zarr",
    )
    file_path = image_mosaic.xml_path.parent / "fused.zarr"

    normalise_intensity = False
    interpolate = False
    image_mosaic.fuse(
        output_path=file_path,
        normalise_intensity=normalise_intensity,
        interpolate=interpolate,
        downscale_factors=downscale_factors,
        chunk_shape=chunk_shape,
        pyramid_depth=pyramid_depth,
        compression_method=compression_method,
        compression_level=compression_level,
    )
    mock_fuse_function.assert_called_once_with(
        file_path,
        test_constants["EXPECTED_FUSED_SHAPE"],
        downscale_factors,
        pyramid_depth,
        chunk_shape,
        compression_method,
        compression_level,
    )


@pytest.mark.parametrize(
    "pyramid_depth, num_channels",
    [(1, 1), (1, 2), (1, 5), (2, 1), (2, 2), (2, 5), (3, 1), (3, 2), (3, 5)],
)
def test_generate_metadata_for_zarr(
    image_mosaic, pyramid_depth, num_channels, test_constants
):
    backup_num_channels = image_mosaic.num_channels
    image_mosaic.num_channels = num_channels

    metadata, axes = image_mosaic._generate_metadata_for_zarr(
        pyramid_depth,
        test_constants["DEFAULT_DOWNSAMPLE_FACTORS"],
    )

    assert len(axes) == 4
    assert axes[0]["name"] == "c"
    assert axes[0]["type"] == "channel"
    axes.pop(0)

    expected_axes_names = ["z", "y", "x"]

    for idx, axes in enumerate(axes):
        assert axes["name"] == expected_axes_names[idx]
        assert axes["type"] == "space"
        assert axes["unit"] == "micrometer"

    for idx, transformation in enumerate(metadata):
        assert transformation[0]["type"] == "scale"
        expected_scale = [
            1,
            image_mosaic.z_resolution,
            image_mosaic.x_y_resolution * 2**idx,
            image_mosaic.x_y_resolution * 2**idx,
        ]

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


def test_write_bdv_xml(image_mosaic, test_constants, tmp_path):
    output_path = tmp_path / "test_data_bdv.xml"
    hdf5_file_name = "test_data_bdv.h5"
    image_mosaic._write_bdv_xml(
        output_path,
        tmp_path / hdf5_file_name,
        test_constants["EXPECTED_FUSED_SHAPE"],
    )
    expected_tile_names = [
        f"s{i:02}" for i in range(image_mosaic.num_channels)
    ]
    assert output_path.exists()
    test_attributes = get_slice_attributes(output_path, expected_tile_names)

    assert len(test_attributes) == test_constants["NUM_CHANNELS"]

    xml_contents = ET.parse(output_path)
    root = xml_contents.getroot()

    hdf5_path = safe_find(root, ".//hdf5")

    assert hdf5_path is not None
    assert hdf5_path.text == hdf5_file_name

    view_setups = safe_find(root, ".//ViewSetup")
    flipped_shape = test_constants["EXPECTED_FUSED_SHAPE"][::-1]
    assert view_setups is not None
    assert view_setups[2].text == str(
        f"{flipped_shape[0]} {flipped_shape[1]} {flipped_shape[2]}"
    )
