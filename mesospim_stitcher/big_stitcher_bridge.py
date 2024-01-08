import subprocess
from pathlib import Path


def run_big_stitcher(
    imageJ_path: Path,
    xml_path: Path,
    tile_config_path: Path,
    all_channels: bool = False,
    selected_channel: int = 488,
    downsample_x: int = 4,
    downsample_y: int = 4,
    downsample_z: int = 4,
):
    stitch_macro_path = Path(__file__).resolve().parent / "stitch_macro.ijm"
    command = (
        f"{imageJ_path} --ij2"
        f" --headless -macro {stitch_macro_path} "
        f'"{xml_path} {tile_config_path} {int(all_channels)}'
        f' {selected_channel} {downsample_x} {downsample_y} {downsample_z}"'
    )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True
    )

    return result
