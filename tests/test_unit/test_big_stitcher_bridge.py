from importlib.resources import files

import pytest

from brainglobe_stitch.big_stitcher_bridge import run_big_stitcher


def test_run_big_stitcher_defaults(mocker, test_constants):
    """
    Test the run_big_stitcher function with default parameters. Mocks
    the subprocess.run function to check if the correct command is
    passed and to prevent the actual command from running.
    """
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )

    imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]
    tile_config_path = test_constants["MOCK_TILE_CONFIG_PATH"]

    run_big_stitcher(imagej_path, xml_path, tile_config_path)

    # Expected path to the ImageJ macro
    # Should be in the root of the package
    macro_path = files("brainglobe_stitch") / "bigstitcher_macro.ijm"

    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {tile_config_path} 0 488 4 4 1"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )


@pytest.mark.parametrize(
    "all_channels, selected_channel, downsample_x, downsample_y, downsample_z",
    [
        (False, 488, 4, 4, 4),
        (True, 488, 4, 4, 4),
        (False, 488, 4, 8, 16),
        (True, 576, 4, 8, 16),
    ],
)
def test_run_big_stitcher(
    mocker,
    all_channels,
    selected_channel,
    downsample_x,
    downsample_y,
    downsample_z,
    test_constants,
):
    """
    Test the run_big_stitcher function with custom parameters. Mocks
    the subprocess.run function to check if the correct command is
    passed and to prevent the actual command from running.
    """
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )

    imagej_path = test_constants["MOCK_IMAGEJ_PATH"]
    xml_path = test_constants["MOCK_XML_PATH"]
    tile_config_path = test_constants["MOCK_TILE_CONFIG_PATH"]

    run_big_stitcher(
        imagej_path,
        xml_path,
        tile_config_path,
        all_channels=all_channels,
        selected_channel=selected_channel,
        downsample_x=downsample_x,
        downsample_y=downsample_y,
        downsample_z=downsample_z,
    )

    # Expected path to the ImageJ macro
    # Should be in the root of the package
    macro_path = files("brainglobe_stitch").joinpath("bigstitcher_macro.ijm")

    expected_imagej_path = test_constants["MOCK_IMAGEJ_EXEC_PATH"]

    command = (
        f"{expected_imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {tile_config_path} {int(all_channels)} '
        f"{selected_channel} "
        f'{downsample_x} {downsample_y} {downsample_z}"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )
