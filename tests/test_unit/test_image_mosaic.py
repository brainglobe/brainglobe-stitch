import os
from pathlib import Path

import numpy as np
import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic

NUM_TILES = 8
NUM_RESOLUTIONS = 5
NUM_CHANNELS = 2
NUM_OVERLAPS = 12
TILE_SIZE = (107, 128, 128)
EXPECTED_FUSED_IMAGE_SHAPE = (113, 251, 246)
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
EXPECTED_INTENSITY_FACTORS = [
    1.0,
    0.9883720930232558,
    1.0,
    1.0545454545454545,
    0.5801054460122512,
    0.5267319002511789,
    1.074100378787879,
    1.083113545682018,
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


@pytest.mark.parametrize(
    "resolution_level",
    [0, 1],
)
def test_normalise_intensity(mocker, image_mosaic, resolution_level):
    def force_set_scale_factors(*args, **kwargs):
        image_mosaic.scale_factors = EXPECTED_INTENSITY_FACTORS
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
    assert len(image_mosaic.scale_factors) == NUM_TILES

    for i in range(NUM_TILES):
        if EXPECTED_INTENSITY_FACTORS[i] != 1.0:
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
    mocker, image_mosaic, resolution_level
):
    def force_set_scale_factors(*args, **kwargs):
        image_mosaic.scale_factors = EXPECTED_INTENSITY_FACTORS
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
    assert len(image_mosaic.scale_factors) == NUM_TILES

    mock_calc_intensity_factors.called_once_with(resolution_level, 50)

    for i in range(NUM_TILES):
        assert (
            len(
                image_mosaic.tiles[i]
                .data_pyramid[resolution_level]
                .dask.layers
            )
            == 2
        )


@pytest.mark.parametrize(
    "resolution_level",
    [0, 1, 2, 3, 4],
)
def test_normalise_intensity_already_adjusted(image_mosaic, resolution_level):
    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    image_mosaic.intensity_adjusted[resolution_level] = True
    image_mosaic.normalise_intensity(resolution_level)

    assert image_mosaic.intensity_adjusted[resolution_level]

    # Check that no scale adjustment calculations are queued for the tiles
    # at the specified resolution level
    for i in range(NUM_TILES):
        assert (
            len(
                image_mosaic.tiles[i]
                .data_pyramid[resolution_level]
                .dask.layers
            )
            == 2
        )


def test_calculate_intensity_scale_factors(image_mosaic):
    resolution_level = 2
    percentile = 50
    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    image_mosaic.scale_factors = None

    image_mosaic.calculate_intensity_scale_factors(
        resolution_level, percentile
    )

    assert len(image_mosaic.scale_factors) == NUM_TILES
    assert np.allclose(image_mosaic.scale_factors, EXPECTED_INTENSITY_FACTORS)


@pytest.mark.parametrize("resolution_level", [0, 1, 2, 3])
def test_interpolate_overlaps(image_mosaic, resolution_level, mocker):
    image_mosaic.reload_resolution_pyramid_level(resolution_level)
    mock_linear_interpolation = mocker.patch(
        "brainglobe_stitch.tile.Overlap.linear_interpolation",
    )

    image_mosaic.interpolate_overlaps(resolution_level)

    assert image_mosaic.overlaps_interpolated[resolution_level]
    assert mock_linear_interpolation.call_count == NUM_OVERLAPS


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


@pytest.mark.parametrize(
    "output_file_name, normalise_intensity, interpolate, fuse_function_name",
    [
        ("fused.zarr", False, False, "_fuse_to_zarr"),
        ("fused.zarr", True, False, "_fuse_to_zarr"),
        ("fused.zarr", False, True, "_fuse_to_zarr"),
        ("fused.zarr", True, True, "_fuse_to_zarr"),
        ("fused.h5", False, False, "_fuse_to_bdv_h5"),
        ("fused.h5", True, False, "_fuse_to_bdv_h5"),
        ("fused.h5", False, True, "_fuse_to_bdv_h5"),
        ("fused.h5", True, True, "_fuse_to_bdv_h5"),
    ],
)
def test_fuse(
    image_mosaic,
    mocker,
    output_file_name,
    normalise_intensity,
    interpolate,
    fuse_function_name,
):
    image_mosaic.reload_resolution_pyramid_level(0)

    mock_fuse_function = mocker.patch(
        f"brainglobe_stitch.image_mosaic.ImageMosaic.{fuse_function_name}",
    )
    mock_normalise_intensity = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.normalise_intensity",
    )
    mock_interpolate_overlaps = mocker.patch(
        "brainglobe_stitch.image_mosaic.ImageMosaic.interpolate_overlaps",
    )

    image_mosaic.fuse(output_file_name, normalise_intensity, interpolate)

    mock_fuse_function.assert_called_once_with(
        image_mosaic.xml_path.parent / output_file_name,
        EXPECTED_FUSED_IMAGE_SHAPE,
    )

    if normalise_intensity:
        mock_normalise_intensity.assert_called_once_with(0, 80)
    else:
        mock_normalise_intensity.assert_not_called()

    if interpolate:
        mock_interpolate_overlaps.assert_called_once_with(0)
    else:
        mock_interpolate_overlaps.assert_not_called()


@pytest.mark.parametrize(
    "pyramid_depth, num_channels",
    [(1, 1), (1, 2), (1, 5), (2, 1), (2, 2), (2, 5), (3, 1), (3, 2), (3, 5)],
)
def test_get_metadata_for_zarr(image_mosaic, pyramid_depth, num_channels):
    temp_num_channels = image_mosaic.num_channels
    image_mosaic.num_channels = num_channels

    metadata, axes = image_mosaic.get_metadata_for_zarr(pyramid_depth)

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

    image_mosaic.num_channels = temp_num_channels


def test_fuse_to_zarr(image_mosaic):
    image_mosaic.xml_path.parent / "fused.zarr"


def test_fuse_to_bdv_h5():
    pass
