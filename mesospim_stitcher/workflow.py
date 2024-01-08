from pathlib import Path

import numpy as np

from mesospim_stitcher.big_stitcher_bridge import run_big_stitcher
from mesospim_stitcher.file_utils import (
    create_pyramid_bdv_h5,
    write_big_stitcher_tile_config,
)
from mesospim_stitcher.fuse import fuse_image

XML_PATH = (
    "D:/TiledDataset/2.5x_tile/"
    "2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv3.xml"
)
H5_PATH = (
    "D:/TiledDataset/2.5x_tile/"
    "2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv.h5"
)
META_PATH = (
    "D:/TiledDataset/2.5x_tile/"
    "2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv.h5_meta.txt"
)
TILE_CONFIG_PATH = (
    "D:/TiledDataset/"
    "2.5x_tile/"
    "2.5x_tile_igor_rightonly_Mag2.5x"
    "_ch488_ch561_ch647_bdv_tile_config.txt"
)

DOWNSAMPLE_ARRAY = np.array(
    [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
)
SUBDIVISION_ARRAY = np.array(
    [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
)

create_pyramid_bdv_h5(Path(H5_PATH), DOWNSAMPLE_ARRAY, SUBDIVISION_ARRAY)

tile_metadata = write_big_stitcher_tile_config(Path(TILE_CONFIG_PATH))

result_big_stitcher = run_big_stitcher(
    Path("C:/Users/Igor/Documents/Fiji.app/ImageJ-win64.exe"),
    Path(XML_PATH),
    Path(TILE_CONFIG_PATH),
)

with open("big_stitcher_output.txt", "w") as f:
    f.write(result_big_stitcher.stdout)
    f.write(result_big_stitcher.stderr)

fuse_image(
    Path(XML_PATH),
    Path(H5_PATH),
    Path(
        "D:/TiledDataset/2.5x_tile/2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv_fused.h5"
    ),
)
