from importlib.resources import files
from pathlib import Path
from sys import platform

import pytest

from brainglobe_stitch.big_stitcher_bridge import run_big_stitcher

IMAGEJ_PATH_WINDOWS = Path("C:/Fiji.app/ImageJ-win64.exe")
XML_PATH_WINDOWS = Path("C:/stitching/Brain2/bdv.xml")
TILE_CONFIG_PATH_WINDOWS = Path("C:/stitching/Brain2/bdv_tile_config.txt")

IMAGEJ_PATH_MAC = Path("/Users/user/Fiji.app")
IMAGEJ_PATH_MAC_CHECK = Path(
    "/Users/user/Fiji.app/Contents/MacOS/ImageJ-macosx"
)
XML_PATH_MAC = Path("/Users/user/stitching/Brain2/bdv.xml")
TILE_CONFIG_PATH_MAC = Path("/Users/user/stitching/Brain2/bdv_tile_config.txt")


def test_run_big_stitcher_defaults(mocker):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )

    imagej_path = IMAGEJ_PATH_WINDOWS
    xml_path = XML_PATH_WINDOWS
    tile_config_path = TILE_CONFIG_PATH_WINDOWS

    if platform.startswith("darwin"):
        imagej_path = IMAGEJ_PATH_MAC
        xml_path = XML_PATH_MAC
        tile_config_path = TILE_CONFIG_PATH_MAC

    run_big_stitcher(imagej_path, xml_path, tile_config_path)

    macro_path = files("brainglobe_stitch").joinpath("bigstitcher_macro.ijm")

    if platform.startswith("darwin"):
        imagej_path = IMAGEJ_PATH_MAC_CHECK

    command = (
        f"{imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {tile_config_path} 0 488 4 4 4"'
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
):
    mock_subprocess_run = mocker.patch(
        "brainglobe_stitch.big_stitcher_bridge.subprocess.run"
    )

    imagej_path = IMAGEJ_PATH_WINDOWS
    xml_path = XML_PATH_WINDOWS
    tile_config_path = TILE_CONFIG_PATH_WINDOWS

    if platform.startswith("darwin"):
        imagej_path = IMAGEJ_PATH_MAC
        xml_path = XML_PATH_MAC
        tile_config_path = TILE_CONFIG_PATH_MAC

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

    macro_path = files("brainglobe_stitch").joinpath("bigstitcher_macro.ijm")

    if platform.startswith("darwin"):
        imagej_path = IMAGEJ_PATH_MAC_CHECK

    command = (
        f"{imagej_path} --ij2"
        f" --headless -macro {macro_path} "
        f'"{xml_path} {tile_config_path} {int(all_channels)} '
        f"{selected_channel} "
        f'{downsample_x} {downsample_y} {downsample_z}"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )