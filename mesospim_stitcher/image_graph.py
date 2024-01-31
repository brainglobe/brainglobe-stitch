from pathlib import Path
from typing import List, Tuple

import dask.array as da
import h5py
import numpy as np
import numpy.typing as npt
from rich.progress import Progress

from mesospim_stitcher.big_stitcher_bridge import run_big_stitcher
from mesospim_stitcher.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
    get_slice_attributes,
    parse_mesospim_metadata,
    write_big_stitcher_tile_config,
)
from mesospim_stitcher.tile import Tile

DOWNSAMPLE_ARRAY = np.array(
    [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
)
SUBDIVISION_ARRAY = np.array(
    [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
)


class ImageGraph:
    def __init__(self, directory: Path):
        self.directory: Path = directory
        self.xml_path: Path | None = None
        self.meta_path: Path | None = None
        self.h5_path: Path | None = None
        self.tile_config_path: Path | None = None
        self.h5_file: h5py.File | None = None
        self.tiles: List[Tile] = []

        self.load_mesospim_directory()

    def load_mesospim_directory(self) -> None:
        try:
            (
                self.xml_path,
                self.meta_path,
                self.h5_path,
            ) = check_mesospim_directory(self.directory)
        except FileNotFoundError:
            print("Invalid mesoSPIM directory")

        assert self.xml_path is not None
        assert self.meta_path is not None
        assert self.h5_path is not None

        self.h5_file = h5py.File(self.h5_path, "r")

        if len(self.h5_file["t00000/s00"].keys()) <= 1:
            print("Resolution pyramid not found.")
            self.h5_file.close()
            print("Creating resolution pyramid...")

            with Progress() as progress:
                task = progress.add_task(
                    "Creating resolution pyramid...", total=100
                )

                assert self.h5_path is not None

                for update in create_pyramid_bdv_h5(
                    self.h5_path,
                    DOWNSAMPLE_ARRAY,
                    SUBDIVISION_ARRAY,
                    yield_progress=True,
                ):
                    progress.update(task, advance=update)

            self.h5_file = h5py.File(self.h5_path, "r")

        self.tile_config_path = Path(
            str(self.xml_path)[:-4] + "_tile_config.txt"
        )

        if not self.tile_config_path.exists():
            tile_metadata = write_big_stitcher_tile_config(
                self.meta_path, self.h5_path
            )
        else:
            tile_metadata = parse_mesospim_metadata(self.meta_path)

        channel_names = []
        idx = 0
        while tile_metadata[idx]["Laser"] not in channel_names:
            channel_names.append(tile_metadata[idx]["Laser"])
            idx += 1

        tile_group = self.h5_file["t00000"]
        tile_names = list(tile_group.keys())
        slice_attributes = get_slice_attributes(self.xml_path, tile_names)

        self.tiles = []
        for idx, tile_name in enumerate(tile_names):
            tile = Tile(tile_name, idx, slice_attributes[tile_name])
            tile.channel_name = channel_names[tile.channel_id]
            self.tiles.append(tile)
            tile_data = []

            for pyramid_level in tile_group[tile_name].keys():
                tile_data.append(
                    da.from_array(
                        tile_group[f"{tile_name}/{pyramid_level}/cells"],
                    )
                )

            tile.data_pyramid = tile_data
            tile.resolution_pyramid = np.array(
                self.h5_file[f"{tile_name}/resolutions"]
            )

        with open(self.tile_config_path, "r") as f:
            # Skip header
            f.readline()

            for line, tile in zip(f.readlines(), self.tiles):
                split_line = line.split(";")[-1].strip("()\n").split(",")
                # BigStitcher uses x,y,z order
                # Switch to z,y,x order
                translation = [
                    int(split_line[2]),
                    int(split_line[1]),
                    int(split_line[0]),
                ]
                tile.position = translation

    def stitch(
        self,
        imagej_path: Path,
        resolution_level: int,
        selected_channel: str,
    ):
        all_channels = len(selected_channel) == 0
        channel_int = -1

        if not all_channels:
            try:
                channel_int = int(selected_channel.split()[0])
            except ValueError as e:
                e.add_note("Invalid channel name.")
                raise

        downsample_x, downsample_y, downsample_z = self.tiles[
            0
        ].resolution_pyramid[resolution_level]

        assert self.xml_path is not None
        assert self.tile_config_path is not None

        run_big_stitcher(
            imagej_path,
            self.xml_path,
            self.tile_config_path,
            all_channels,
            channel_int,
            downsample_x=downsample_x,
            downsample_y=downsample_y,
            downsample_z=downsample_z,
        )

    def data_for_napari(
        self, resolution_level: int = 0
    ) -> List[Tuple[da.Array, npt.NDArray]]:
        data = []
        for tile in self.tiles:
            scaled_tile = tile.data_pyramid[resolution_level]
            scaled_translation = (
                np.array(tile.position)
                // tile.resolution_pyramid[resolution_level]
            )
            data.append((scaled_tile, scaled_translation))

        return data
