from importlib.resources import files
from pathlib import Path

import pytest

from brainglobe_stitch.big_stitcher_bridge import (
    calculate_pairwise_links,
    filter_links,
    load_tile_config_file,
    optimise_globally,
    resolve_fiji_path,
    run_big_stitcher,
)


def test_run_big_stitcher_defaults(mocker, test_constants, tmp_path):
    """
    Test the run_big_stitcher function with default parameters. Mocks
    the subprocess.run function to check if the correct command is
    passed and to prevent the actual command from running.
    """
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )
    mock_big_stitcher_log = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.write_big_stitcher_log"
    )

    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]
    tile_config_path = tmp_path
    big_stitcher_log_path = Path.home() / "big_stitcher.log"

    run_big_stitcher(
        curr_imagej_path, xml_path, tile_config_path, big_stitcher_log_path
    )

    # Expect 3 calls to subprocess.run
    assert mock_subprocess_run.call_count == 4
    assert mock_big_stitcher_log.call_count == 4


def test_run_big_stitcher_defaults_no_tile_config(mocker, test_constants):
    """
    Test the run_big_stitcher function with default parameters. Mocks
    the subprocess.run function to check if the correct command is
    passed and to prevent the actual command from running.
    """
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )
    mock_big_stitcher_log = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.write_big_stitcher_log"
    )

    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]
    tile_config_path = test_constants["MOCK_TILE_CONFIG_PATH"]
    big_stitcher_log_path = Path.home() / "big_stitcher.log"

    run_big_stitcher(
        curr_imagej_path, xml_path, tile_config_path, big_stitcher_log_path
    )

    # Expect 3 calls to subprocess.run
    # load_tile_config won't be called as tile_config_path doesn't exist
    assert mock_subprocess_run.call_count == 3
    assert mock_big_stitcher_log.call_count == 3


def test_load_tile_config(mocker, test_constants, tmp_path):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )

    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]
    tile_config_path = test_constants["MOCK_TILE_CONFIG_PATH"]

    load_tile_config_file(curr_imagej_path, xml_path, tile_config_path)

    macro_path = files("brainglobe_stitch").joinpath(
        "bigstitcher_macros/load_tile_config.ijm"
    )

    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {tile_config_path}"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )


@pytest.mark.parametrize(
    "selected_channel, downsample_x, downsample_y, downsample_z",
    [
        ("All Channels", 4, 4, 4),
        ("All Channels", 4, 8, 16),
    ],
)
def test_calculate_pairwise_links_all_channels(
    mocker,
    test_constants,
    selected_channel,
    downsample_x,
    downsample_y,
    downsample_z,
):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )
    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]

    calculate_pairwise_links(
        curr_imagej_path,
        xml_path,
        selected_channel,
        downsample_x,
        downsample_y,
        downsample_z,
    )

    macro_path = files("brainglobe_stitch").joinpath(
        "bigstitcher_macros/calculate_pairwise_all_channel.ijm"
    )
    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {downsample_x} {downsample_y} {downsample_z}"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )


@pytest.mark.parametrize(
    "selected_channel, downsample_x, downsample_y, downsample_z",
    [
        ("488", 4, 4, 4),
        ("488", 4, 8, 16),
        ("488 nm", 4, 4, 4),
        ("488 nm", 4, 8, 16),
    ],
)
def test_calculate_pairwise_links_single_channel(
    mocker,
    test_constants,
    selected_channel,
    downsample_x,
    downsample_y,
    downsample_z,
):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )
    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]

    calculate_pairwise_links(
        curr_imagej_path,
        xml_path,
        selected_channel,
        downsample_x,
        downsample_y,
        downsample_z,
    )

    macro_path = files("brainglobe_stitch").joinpath(
        "bigstitcher_macros/calculate_pairwise_per_channel.ijm"
    )
    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    # Replace spaces with underscores
    selected_channel = selected_channel.replace(" ", "_")

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f"{xml_path} {selected_channel} "
        f"{downsample_x} {downsample_y} {downsample_z}"
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )


@pytest.mark.parametrize("min_r", [0, 1])
@pytest.mark.parametrize("max_r", [1, 2])
@pytest.mark.parametrize(
    "max_shift_x, max_shift_y, max_shift_z",
    [(100, 0, 0), (0, 100, 0), (0, 0, 100), (100, 100, 100)],
)
def test_filter_links(
    mocker, test_constants, min_r, max_r, max_shift_x, max_shift_y, max_shift_z
):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )
    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]

    filter_links(
        curr_imagej_path,
        xml_path,
        min_r,
        max_r,
        max_shift_x,
        max_shift_y,
        max_shift_z,
    )

    macro_path = files("brainglobe_stitch").joinpath(
        "bigstitcher_macros/filter_links.ijm"
    )
    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f"{xml_path} {min_r} {max_r} "
        f"{max_shift_x} {max_shift_y} {max_shift_z}"
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )


@pytest.mark.parametrize(
    "relative, absolute",
    [
        (1, 2),
        (2, 1),
        (2, 2),
    ],
)
def test_optimise_globally(mocker, test_constants, relative, absolute):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )
    curr_imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]

    optimise_globally(curr_imagej_path, xml_path, relative, absolute)

    macro_path = files("brainglobe_stitch").joinpath(
        "bigstitcher_macros/optimise_globally.ijm"
    )
    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {relative} {absolute}"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )


@pytest.mark.parametrize(
    "system",
    ["Linux", "Windows"],
)
def test_resolve_fiji_path(mocker, test_constants, system):
    mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.system", return_value=system
    )

    fiji_path = test_constants["MOCK_IMAGEJ_PATH"]

    resolved_path = resolve_fiji_path(fiji_path)

    assert resolved_path == fiji_path


def test_resolve_path_fiji_darwin(mocker, test_constants):
    mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.system", return_value="Darwin"
    )

    fiji_path = test_constants["MOCK_IMAGEJ_PATH"]

    resolved_path = resolve_fiji_path(fiji_path)

    assert resolved_path == fiji_path / "Contents/MacOS/ImageJ-macosx"
