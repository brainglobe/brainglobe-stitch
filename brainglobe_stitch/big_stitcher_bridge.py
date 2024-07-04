import subprocess
from pathlib import Path
from platform import system


def run_big_stitcher(
    imagej_path: Path,
    xml_path: Path,
    tile_config_path: Path,
    all_channels: bool = False,
    selected_channel: int = 488,
    downsample_x: int = 4,
    downsample_y: int = 4,
    downsample_z: int = 1,
) -> subprocess.CompletedProcess:
    """
    Run the BigStitcher ImageJ macro in headless mode. Output is captured
    and returned as part of the subprocess.CompletedProcess.

    Parameters
    ----------
    imagej_path: Path
        The path to the ImageJ executable.
    xml_path: Path
        The path to the BigDataViewer XML file.
    tile_config_path: Path
        The path to the BigStitcher tile configuration file.
    all_channels: bool, optional
        Whether to stitch based on all channels (default False).
    selected_channel: int, optional
        The channel on which to base the stitching (default 488).
    downsample_x: int, optional
        The downsample factor in the x-dimension for the stitching (default 4).
    downsample_y: int, optional
        The downsample factor in the y-dimension for the stitching (default 4).
    downsample_z: int, optional
        The downsample factor in the z-dimension for the stitching (default 1).

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess returns a non-zero exit status.
    """
    stitch_macro_path = (
        Path(__file__).resolve().parent / "bigstitcher_macro.ijm"
    )

    platform = system()

    if platform.startswith("Darwin"):
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
