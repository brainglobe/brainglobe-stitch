from importlib.resources import files
from pathlib import Path

from mesospim_stitcher.big_stitcher_bridge import run_big_stitcher

IMAGEJ_PATH_WINDOWS = Path("C:/Fiji.app/ImageJ-win64.exe")
XML_PATH = Path("C:/stitching/Brain2/bdv.xml")
TILE_CONFIG_PATH_WINDOWS = Path("C:/stitching/Brain2/bdv_tile_config.txt")
IMAGEJ_PATH_MAC = Path("C:/Fiji.app/ImageJ-win64.exe")
DATA_DIRECTORY_MAC = Path("C:/stitching/Brain2")


def test_run_big_stitcher(mocker):
    mock_subprocess_run = mocker.patch(
        "mesospim_stitcher.big_stitcher_bridge.subprocess.run"
    )

    run_big_stitcher(
        IMAGEJ_PATH_WINDOWS,
        XML_PATH,
        TILE_CONFIG_PATH_WINDOWS,
        all_channels=False,
        selected_channel=488,
        downsample_x=4,
        downsample_y=4,
        downsample_z=4,
    )

    macro_path = files("mesospim_stitcher").joinpath("bigstitcher_macro.ijm")

    command = (
        f"{IMAGEJ_PATH_WINDOWS} --ij2"
        f" --headless -macro {macro_path} "
        f'"{XML_PATH} {TILE_CONFIG_PATH_WINDOWS} 0 488 4 4 4"'
    )

    mock_subprocess_run.assert_called_with(
        command, capture_output=True, text=True, check=True, shell=True
    )
