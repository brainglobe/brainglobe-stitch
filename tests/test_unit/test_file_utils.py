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


@pytest.fixture
def invalid_bdv_directory():
    """
    Fixture for creating an invalid directory for testing.
    The directory is created but empty. The directory is deleted after each
    test.

    Yields
    ------
    Path
        The invalid directory for testing.
    """
    bad_dir = Path("./bad_directory")
    bad_dir.mkdir()

    yield bad_dir

    shutil.rmtree(bad_dir)


def test_create_pyramid_bdv_h5(
    naive_bdv_directory, test_data_directory, test_constants
):
    """
    Check the create_pyramid_bdv_h5 function. The function should create a
    resolution pyramid of depth 5 for each tile in the h5 file. The function
    modifies the h5 file in place. The results are checked by comparing the
    modified h5 file to the expected h5 file, which is stored in the
    test_data_directory.
    """
    # Sanity check to ensure that the test h5 file doesn't contain any
    # resolutions or subdivisions, and no resolution pyramid.
    assert False
    h5_path = naive_bdv_directory / "test_data_bdv.h5"
    with h5py.File(h5_path, "r") as f:
        num_tiles = len(f["t00000"].keys())
        tile_names = f["t00000"].keys()

        for tile_name in tile_names:
            assert f[f"{tile_name}/resolutions"].shape[0] == 1
            assert f[f"{tile_name}/subdivisions"].shape[0] == 1
            assert len(f[f"t00000/{tile_name}"].keys()) == 1

    # Run the function and check that the correct value is yielded for each
    # iteration (percent complete)
    num_done = 1
    for progress in create_pyramid_bdv_h5(h5_path, yield_progress=True):
        assert progress == int(100 * num_done / num_tiles)
        num_done += 1

    with (
        h5py.File(h5_path, "r") as f_out,
        h5py.File(test_data_directory / "test_data_bdv.h5", "r") as f_in,
    ):
        # Check that the number of groups/datasets in the parent is unchanged
        assert len(f_out.keys()) == len(f_in.keys())
        assert len(f_out["t00000"].keys()) == len(f_in["t00000"].keys())

        tile_names = f_in["t00000"].keys()

        # Check that the resolutions and subdivisions have been added for
        # each tile, and that the resolution pyramid of correct depth has been
        # created for each tile.
        for tile_name in tile_names:
            assert (
                f_out[f"{tile_name}/resolutions"].shape[0]
                == test_constants["NUM_RESOLUTIONS"]
            )
            assert (
                f_out[f"{tile_name}/subdivisions"].shape[0]
                == test_constants["NUM_RESOLUTIONS"]
            )
            assert (
                len(f_out[f"t00000/{tile_name}"].keys())
                == test_constants["NUM_RESOLUTIONS"]
            )


def test_parse_mesospim_metadata(naive_bdv_directory, test_constants):
    """
    Check the parse_mesospim_metadata function. The function should parse the
    metadata stored in the h5_meta.txt file and return a list of dictionaries,
    one for each tile. The results are checked by comparing the metadata to the
    expected metadata, which is stored in the test_constants dictionary.
    """
    meta_path = naive_bdv_directory / "test_data_bdv.h5_meta.txt"

    meta_data = parse_mesospim_metadata(meta_path)

    assert len(meta_data) == test_constants["NUM_TILES"]
    # The tiles are alternating in channel names
    for i in range(test_constants["NUM_TILES"]):
        assert meta_data[i]["Laser"] == test_constants["CHANNELS"][i % 2]
        assert (
            meta_data[i]["Pixelsize in um"] == test_constants["PIXEL_SIZE_XY"]
        )
        assert meta_data[i]["z_stepsize"] == test_constants["PIXEL_SIZE_Z"]


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
    invalid_bdv_directory, file_names, error_message
):
    """
    Add the specified files to the invalid directory and check that the
    FileNotFoundError is raised with the correct error message.
    """
    for file_name in file_names:
        Path(invalid_bdv_directory / file_name).touch()

    with pytest.raises(FileNotFoundError) as e:
        check_mesospim_directory(invalid_bdv_directory)

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
    invalid_bdv_directory, file_names, error_message
):
    """
    Add the specified files to the invalid directory and check that the
    FileNotFoundError is raised with the correct error message.
    """
    for file_name in file_names:
        Path(invalid_bdv_directory / file_name).touch()

    with pytest.raises(FileNotFoundError) as e:
        check_mesospim_directory(invalid_bdv_directory)

        assert error_message in str(e)


def test_get_slice_attributes(naive_bdv_directory, test_constants):
    xml_path = naive_bdv_directory / "test_data_bdv.xml"
    tile_names = [f"s{i:02}" for i in range(test_constants["NUM_TILES"])]

    slice_attributes = get_slice_attributes(xml_path, tile_names)

    assert len(slice_attributes) == test_constants["NUM_TILES"]

    # The slices are arranged in a 2x2 grid with 2 channels
    # The tiles in the test data are arranged in columns per channel
    # Each column has its own illumination
    # e.g. s00, s01 are channel 0, tile 0 and 1, illumination 0
    #      s02, s03 are channel 1, tile 0 and 1, illumination 0
    #      s04, s05 are channel 0, tile 2 and 3, illumination 1
    #      s06, s07 are channel 1, tile 2 and 3, illumination 1
    for i in range(test_constants["NUM_TILES"]):
        assert slice_attributes[tile_names[i]]["channel"] == str((i // 2) % 2)
        assert slice_attributes[tile_names[i]]["tile"] == str(
            i % 2 + (i // 4) * 2
        )
        assert slice_attributes[tile_names[i]]["illumination"] == str(i // 4)
        assert slice_attributes[tile_names[i]]["angle"] == "0"


def test_get_big_stitcher_transforms(naive_bdv_directory, test_constants):
    xml_path = naive_bdv_directory / "test_data_bdv.xml"

    transforms = get_big_stitcher_transforms(xml_path)

    assert np.equal(
        transforms, test_constants["EXPECTED_TILE_POSITIONS"]
    ).all()
