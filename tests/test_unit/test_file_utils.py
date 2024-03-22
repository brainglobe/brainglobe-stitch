import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from brainglobe_stitch.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
    get_big_stitcher_transforms,
    get_slice_attributes,
    parse_mesospim_metadata,
)

TEMP_DIR = Path("./temp_directory")
NUM_RESOLUTIONS = 5
NUM_TILES = 8
CHANNELS = ["561 nm", "647 nm"]
PIXEL_SIZE_XY = 4.08
PIXEL_SIZE_Z = 5.0
EXPECTED_TRANSFORMS = np.array(
    [
        [2, 130, 4, 132, 3, 113],
        [0, 128, 120, 248, 2, 112],
        [2, 130, 4, 132, 3, 113],
        [0, 128, 120, 248, 2, 112],
        [118, 246, 7, 135, 6, 116],
        [116, 244, 123, 251, 5, 115],
        [118, 246, 7, 135, 6, 116],
        [116, 244, 123, 251, 5, 115],
    ]
)


@pytest.fixture
def bad_bdv_directory():
    bad_dir = Path("./bad_directory")
    bad_dir.mkdir()

    yield bad_dir

    shutil.rmtree(bad_dir)


def test_create_pyramid_bdv_h5(naive_bdv_directory, test_data_directory):
    h5_path = naive_bdv_directory / "test_data_bdv.h5"
    with h5py.File(h5_path, "r") as f:
        num_tiles = len(f["t00000"].keys())
        tile_names = f["t00000"].keys()

        for tile_name in tile_names:
            assert f[f"{tile_name}/resolutions"].shape[0] == 1
            assert f[f"{tile_name}/subdivisions"].shape[0] == 1
            assert len(f[f"t00000/{tile_name}"].keys()) == 1

    num_done = 1
    for progress in create_pyramid_bdv_h5(h5_path, yield_progress=True):
        assert progress == int(100 * num_done / num_tiles)
        num_done += 1

    with h5py.File(h5_path, "r") as f_out, h5py.File(
        test_data_directory / "test_data_bdv.h5", "r"
    ) as f_in:
        # Check that the number of groups/datasets in the parent is unchanged
        assert len(f_out.keys()) == len(f_in.keys())
        assert len(f_out["t00000"].keys()) == len(f_in["t00000"].keys())

        tile_names = f_in["t00000"].keys()

        for tile_name in tile_names:
            assert (
                f_out[f"{tile_name}/resolutions"].shape[0] == NUM_RESOLUTIONS
            )
            assert (
                f_out[f"{tile_name}/subdivisions"].shape[0] == NUM_RESOLUTIONS
            )
            assert len(f_out[f"t00000/{tile_name}"].keys()) == NUM_RESOLUTIONS


def test_parse_mesospim_metadata(naive_bdv_directory):
    meta_path = naive_bdv_directory / "test_data_bdv.h5_meta.txt"

    meta_data = parse_mesospim_metadata(meta_path)

    assert len(meta_data) == NUM_TILES
    for i in range(NUM_TILES):
        assert meta_data[i]["Laser"] == CHANNELS[i % 2]
        assert meta_data[i]["Pixelsize in um"] == PIXEL_SIZE_XY
        assert meta_data[i]["z_stepsize"] == PIXEL_SIZE_Z


def test_write_bdv_xml():
    pass


def test_check_mesospim_directory(naive_bdv_directory):
    xml_path, meta_path, h5_path = check_mesospim_directory(
        naive_bdv_directory
    )

    assert xml_path == naive_bdv_directory / "test_data_bdv.xml"
    assert meta_path == naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    assert h5_path == naive_bdv_directory / "test_data_bdv.h5"


@pytest.mark.parametrize(
    "file_names, error_message",
    [
        (
            ["test_data_bdv.xml", "test_data_bdv.h5_meta.txt"],
            "Expected 1 h5 file, found 0",
        ),
        (
            ["test_data_bdv.xml", "test_data_bdv.h5"],
            "Expected 1 h5_meta.txt file, found 0",
        ),
        (
            ["test_data_bdv.h5_meta.txt", "test_data_bdv.h5"],
            "Expected 1 xml file, found 0",
        ),
    ],
)
def test_check_mesospim_directory_missing_files(
    bad_bdv_directory, file_names, error_message
):
    for file_name in file_names:
        Path(bad_bdv_directory / file_name).touch()

    with pytest.raises(FileNotFoundError) as e:
        check_mesospim_directory(bad_bdv_directory)

        assert error_message in str(e)


@pytest.mark.parametrize(
    "file_names, error_message",
    [
        (
            ["a_bdv.xml", "a_bdv.h5_meta.txt", "a_bdv.h5", "b_bdv.xml"],
            "Expected 1 xml file, found 2",
        ),
        (
            [
                "a_bdv.xml",
                "a_bdv.h5_meta.txt",
                "a_bdv.h5",
                "b_bdv.h5_meta.txt",
            ],
            "Expected 1 h5_meta.txt file, found 2",
        ),
        (
            ["a_bdv.xml", "a_bdv.h5_meta.txt", "a_bdv.h5", "b_bdv.h5"],
            "Expected 1 h5 file, found 2",
        ),
    ],
)
def test_check_mesospim_directory_too_many_files(
    bad_bdv_directory, file_names, error_message
):
    for file_name in file_names:
        Path(bad_bdv_directory / file_name).touch()

    with pytest.raises(FileNotFoundError) as e:
        check_mesospim_directory(bad_bdv_directory)

        assert error_message in str(e)


def test_write_tiff():
    pass


def test_get_slice_attributes(naive_bdv_directory):
    xml_path = naive_bdv_directory / "test_data_bdv.xml"
    tile_names = [f"s{i:02}" for i in range(NUM_TILES)]

    slice_attributes = get_slice_attributes(xml_path, tile_names)

    assert len(slice_attributes) == NUM_TILES

    # The slices are arranged in a 2x2 grid with 2 channels
    # The tiles in the test data are arranged in columns per channel
    # Each column has its own illumination
    # e.g. s00, s01 are channel 0, tile 0 and 1, illumination 0
    #      s02, s03 are channel 1, tile 0 and 1, illumination 0
    #      s04, s05 are channel 0, tile 2 and 3, illumination 1
    #      s06, s07 are channel 1, tile 2 and 3, illumination 1
    for i in range(NUM_TILES):
        assert slice_attributes[tile_names[i]]["channel"] == str((i // 2) % 2)
        assert slice_attributes[tile_names[i]]["tile"] == str(
            i % 2 + (i // 4) * 2
        )
        assert slice_attributes[tile_names[i]]["illumination"] == str(i // 4)
        assert slice_attributes[tile_names[i]]["angle"] == "0"


def test_get_big_stitcher_transforms(naive_bdv_directory):
    xml_path = naive_bdv_directory / "test_data_bdv.xml"

    transforms = get_big_stitcher_transforms(xml_path, 128, 128, 110)

    assert np.equal(transforms, EXPECTED_TRANSFORMS).all()
