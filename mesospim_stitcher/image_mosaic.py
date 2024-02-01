from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple

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
from mesospim_stitcher.fuse import get_big_stitcher_transforms
from mesospim_stitcher.tile import Overlap, Tile

DOWNSAMPLE_ARRAY = np.array(
    [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
)
SUBDIVISION_ARRAY = np.array(
    [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
)


class ImageMosaic:
    def __init__(self, directory: Path):
        self.directory: Path = directory
        self.xml_path: Path | None = None
        self.meta_path: Path | None = None
        self.h5_path: Path | None = None
        self.tile_config_path: Path | None = None
        self.h5_file: h5py.File | None = None
        self.tiles: List[Tile] = []
        self.overlaps: Dict[Tuple[int, int], Overlap] = {}

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

        result = run_big_stitcher(
            imagej_path,
            self.xml_path,
            self.tile_config_path,
            all_channels,
            channel_int,
            downsample_x=downsample_x,
            downsample_y=downsample_y,
            downsample_z=downsample_z,
        )

        big_stitcher_output_path = self.directory / "big_stitcher_output.txt"

        with open(big_stitcher_output_path, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        print(result.stdout)

        # Wait for the BigStitcher to write XML file
        # Need to find a better way to do this
        sleep(1)

        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape

        stitcher_translations = get_big_stitcher_transforms(
            self.xml_path, z_size, y_size, x_size
        )

        for tile in self.tiles:
            # BigStitcher uses x,y,z order, switch to z,y,x order
            stitched_position = [
                stitcher_translations[tile.id][4],
                stitcher_translations[tile.id][2],
                stitcher_translations[tile.id][0],
            ]
            tile.stitched_position = stitched_position
            tile.position = stitched_position

        self.calculate_overlaps()

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

    def calculate_overlaps(self):
        self.overlaps = {}
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape

        for tile_i in self.tiles[:-1]:
            position_i = tile_i.position
            for tile_j in self.tiles[tile_i.id + 1 :]:
                position_j = tile_j.position

                if (
                    (position_i[1] + y_size > position_j[1])
                    and (position_i[2] + x_size > position_j[2])
                    and (tile_i.tile_id != tile_j.tile_id)
                    and (tile_i.channel_id == tile_j.channel_id)
                ):
                    starts = np.array(
                        [max(position_i[i], position_j[i]) for i in range(3)]
                    )
                    ends = np.add(
                        [min(position_i[i], position_j[i]) for i in range(3)],
                        [z_size, y_size, x_size],
                    )
                    size = np.subtract(ends, starts)

                    self.overlaps[(tile_i.id, tile_j.id)] = Overlap(
                        starts, size, tile_i, tile_j
                    )
                    tile_i.neighbours.append(tile_j.id)

    def normalise_intensity(
        self, percentile: int = 80, resolution_level: int = 0
    ) -> npt.NDArray:
        num_tiles = len(self.tiles)
        scale_factors = np.ones((num_tiles, num_tiles))

        for tile_i in self.tiles:
            for neighbour_id in tile_i.neighbours:
                tile_j = self.tiles[neighbour_id]
                overlap = self.overlaps[(tile_i.id, tile_j.id)]

                scaled_coordinates = overlap.local_coords[resolution_level]
                scaled_size = overlap.size[resolution_level]

                i_overlap = tile_i.data_pyramid[resolution_level][
                    scaled_coordinates[0][0] : scaled_coordinates[0][0]
                    + scaled_size[0],
                    scaled_coordinates[0][1] : scaled_coordinates[0][1]
                    + scaled_size[1],
                    scaled_coordinates[0][2] : scaled_coordinates[0][2]
                    + scaled_size[2],
                ]

                j_overlap = tile_j.data_pyramid[resolution_level][
                    scaled_coordinates[1][0] : scaled_coordinates[1][0]
                    + scaled_size[0],
                    scaled_coordinates[1][1] : scaled_coordinates[1][1]
                    + scaled_size[1],
                    scaled_coordinates[1][2] : scaled_coordinates[1][2]
                    + scaled_size[2],
                ]

                median_i = np.percentile(i_overlap.ravel(), percentile)
                median_j = np.percentile(j_overlap.ravel(), percentile)

                curr_scale_factor = (median_i / median_j).compute()
                scale_factors[tile_i.id][tile_j.id] = curr_scale_factor[0]

                del i_overlap
                del j_overlap
                del median_i
                del median_j

                tile_j.data_pyramid[resolution_level] = np.multiply(
                    tile_j.data_pyramid[resolution_level],
                    curr_scale_factor,
                    dtype=np.float16,
                ).astype(np.uint16)

        return scale_factors
