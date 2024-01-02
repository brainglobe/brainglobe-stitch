import subprocess
from pathlib import Path

XML_PATH = (
    "D:/TiledDataset/2.5x_tile/"
    "2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv3.xml"
)
TILE_CONFIG_PATH = (
    "D:/TiledDataset/"
    "2.5x_tile/"
    "2.5x_tile_igor_rightonly_Mag2.5x"
    "_ch488_ch561_ch647_bdv_tile_config.txt"
)


def run_big_stitcher(
    imageJ_path: Path, xml_path: Path, tile_config_path: Path
):
    stitch_macro_path = Path(__file__).resolve().parent / "stitch_macro.ijm"
    command = (
        f"{imageJ_path} --ij2"
        f" --headless -macro {stitch_macro_path} "
        f'"{xml_path} {tile_config_path}"'
    )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True
    )

    return result


if __name__ == "__main__":
    result = run_big_stitcher(
        Path("C:/Users/Igor/Documents/Fiji.app/ImageJ-win64.exe"),
        Path(XML_PATH),
        Path(TILE_CONFIG_PATH),
    )

    with open("big_stitcher_output.txt", "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
