import subprocess
from pathlib import Path
from sys import platform


def run_big_stitcher(
    imagej_path: Path,
    xml_path: Path,
    tile_config_path: Path,
    all_channels: bool = False,
    selected_channel: int = 488,
    downsample_x: int = 4,
    downsample_y: int = 4,
    downsample_z: int = 4,
):
    stitch_macro_path = Path(__file__).resolve().parent / "stitch_macro.ijm"

    if platform.startswith("darwin"):
        imagej_path = imagej_path / "Contents/MacOS/ImageJ-macosx"

    command = (
        f"{imagej_path} --ij2"
        f" --headless -macro {stitch_macro_path} "
        f'"{xml_path} {tile_config_path} {int(all_channels)}'
        f' {selected_channel} {downsample_x} {downsample_y} {downsample_z}"'
    )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True, shell=True
    )

    return result
