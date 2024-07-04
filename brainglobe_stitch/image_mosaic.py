from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple

import dask.array as da
import h5py
import numpy as np
import numpy.typing as npt
from rich.progress import Progress

from brainglobe_stitch.big_stitcher_bridge import run_big_stitcher
from brainglobe_stitch.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
    get_big_stitcher_transforms,
    get_slice_attributes,
    parse_mesospim_metadata,
)
from brainglobe_stitch.tile import Tile


class ImageMosaic:
    """
    Class to represent an image as a collection of tiles.

    Attributes
    ----------
    directory : Path
        The directory containing the image data.
    xml_path : Optional[Path]
        The path to the Big Data Viewer XML file.
    meta_path : Optional[Path]
        The path to the mesoSPIM metadata file.
    h5_path : Optional[Path]
        The path to the Big Data Viewer h5 file containing the raw data.
    tile_config_path : Optional[Path]
        The path to the BigStitcher tile configuration file.
    h5_file : Optional[h5py.File]
        An open h5py file object for the raw data.
    channel_names : List[str]
        The names of the channels in the image as strings.
    tiles : List[Tile]
        The tiles in the image.
    tile_names : List[str]
        The names of the image tiles from BigDataViewer.
    x_y_resolution : float
        The resolution of the image in the x and y dimensions
        in micrometers per pixel.
    z_resolution : float
        The resolution of the image in the z dimension
        in micrometers per pixel.
    num_channels : int
        The number of channels in the image.
    """

    def __init__(self, directory: Path):
        self.directory: Path = directory
        self.xml_path: Optional[Path] = None
        self.meta_path: Optional[Path] = None
        self.h5_path: Optional[Path] = None
        self.tile_config_path: Optional[Path] = None
        self.h5_file: Optional[h5py.File] = None
        self.channel_names: List[str] = []
        self.tiles: List[Tile] = []
        self.tile_names: List[str] = []
        self.tile_metadata: List[Dict] = []
        self.x_y_resolution: float = 4.0  # um per pixel
        self.z_resolution: float = 5.0  # um per pixel
        self.num_channels: int = 1

        self.load_mesospim_directory()

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def data_for_napari(
        self, resolution_level: int = 0
    ) -> List[Tuple[da.Array, npt.NDArray]]:
        """
        Return data for visualisation in napari.

        Parameters
        ----------
        resolution_level: int
            The resolution level to get the data for.

        Returns
        -------
        List[Tuple[da.Array, npt.NDArray]]
            A list of tuples containing the data and the translation for each
            tile scaled to the selected resolution.
        """
        data = []
        for tile in self.tiles:
            scaled_tile = tile.data_pyramid[resolution_level]
            scaled_translation = (
                np.array(tile.position)
                // tile.resolution_pyramid[resolution_level]
            )
            data.append((scaled_tile, scaled_translation))

        return data

    def load_mesospim_directory(self) -> None:
        """
        Load the mesoSPIM directory and its data into the ImageMosaic.
        """
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

        # Check if resolution pyramid exists
        # s00 should have more than 1 key if the resolution pyramid exists
        # Each key in ["t00000/s00"] corresponds to a resolution level
        if len(self.h5_file["t00000/s00"].keys()) <= 1:
            print("Resolution pyramid not found.")
            # Close the file as it's open as read only
            self.h5_file.close()
            print("Creating resolution pyramid.")

            # Create resolution pyramid
            with Progress() as progress:
                task = progress.add_task("Downsampling...", total=100)

                for update in create_pyramid_bdv_h5(
                    self.h5_path,
                    yield_progress=True,
                ):
                    progress.update(task, advance=update)

            # Reopen the file
            self.h5_file = h5py.File(self.h5_path, "r")

        self.tile_config_path = Path(
            str(self.xml_path)[:-4] + "_tile_config.txt"
        )

        self.tile_metadata = parse_mesospim_metadata(self.meta_path)

        self.channel_names = []
        idx = 0
        while self.tile_metadata[idx]["Laser"] not in self.channel_names:
            self.channel_names.append(self.tile_metadata[idx]["Laser"])
            idx += 1

        self.x_y_resolution = self.tile_metadata[0]["Pixelsize in um"]
        self.z_resolution = self.tile_metadata[0]["z_stepsize"]
        self.num_channels = len(self.channel_names)

        # Each tile is a group under "t00000"
        # Ordered in increasing order based on acquisition
        # Names aren't always contiguous, e.g. s00, s01, s04, s05 is valid
        tile_group = self.h5_file["t00000"]
        self.tile_names = list(tile_group.keys())
        slice_attributes = get_slice_attributes(self.xml_path, self.tile_names)

        self.tiles = []
        for idx, tile_name in enumerate(self.tile_names):
            tile = Tile(tile_name, idx, slice_attributes[tile_name])
            tile.channel_name = self.channel_names[tile.channel_id]
            self.tiles.append(tile)
            tile_data = []

            for pyramid_level in tile_group[tile_name].keys():
                tile_data.append(
                    da.from_array(
                        tile_group[f"{tile_name}/{pyramid_level}/cells"],
                    )
                )

            tile.data_pyramid = tile_data

            # Add the scaling factor for each resolution level of the pyramid.
            resolutions = self.h5_file[f"{tile_name}/resolutions"]
            tile.resolution_pyramid = np.ones(
                (len(resolutions), 3), dtype=np.int16
            )

            # Switch to z,y,x order from x,y,z order
            for i, resolution in enumerate(resolutions):
                tile.resolution_pyramid[i] = resolution[-1::-1]

        # Don't rewrite the tile config file if it already exists
        # These will be used as the initial tile positions
        # This file will be passed to BigStitcher
        if not self.tile_config_path.exists():
            print("Tile positions not found. Writing tile config file.")
            self.write_big_stitcher_tile_config(self.meta_path)

        # Read the tile config file to get the initial tile positions
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

    def write_big_stitcher_tile_config(self, meta_file_name: Path) -> None:
        """
        Write the BigStitcher tile configuration file
        (placement for each tile based on stage coordinates).

        Parameters
        ----------
        meta_file_name: Path
            The path to the mesoSPIM metadata file.
        """
        # Remove .h5_meta.txt from the file name
        output_file = str(meta_file_name)[:-12] + "_tile_config.txt"

        tile_xy_locations = []
        for i in range(0, len(self.tiles), self.num_channels):
            curr_tile_dict = self.tile_metadata[i]

            # Get the x and y positions in pixels
            x = round(
                curr_tile_dict["x_pos"] / curr_tile_dict["Pixelsize in um"]
            )
            y = round(
                curr_tile_dict["y_pos"] / curr_tile_dict["Pixelsize in um"]
            )

            tile_xy_locations.append((x, y))

        # Calculate relative pixel positions for each tile
        # The first tile is at (0,0)
        relative_locations = [(0, 0)]
        for abs_tuple in tile_xy_locations[1:]:
            rel_tuple = (
                abs(abs_tuple[0] - tile_xy_locations[0][0]),
                abs(abs_tuple[1] - tile_xy_locations[0][1]),
            )
            relative_locations.append(rel_tuple)

        # Write the tile config file based on what BigStitcher expects
        with open(output_file, "w") as f:
            f.write("dim=3\n")
            for tile, tile_name in zip(self.tiles, self.tile_names):
                f.write(
                    f"{tile_name[1:]};;"
                    f"({relative_locations[tile.tile_id][0]},"
                    f"{relative_locations[tile.tile_id][1]},0)\n"
                )

        return

    def stitch(
        self,
        fiji_path: Path,
        resolution_level: int,
        selected_channel: str,
    ) -> None:
        """
        Stitch the tiles in the image using BigStitcher.

        Parameters
        ----------
        fiji_path: Path
            The path to the Fiji application.
        resolution_level: int
            The resolution level to stitch the tiles at.
        selected_channel: str
            The name of the channel to stitch.
        """

        # If selected_channel is an empty string then stitch based on
        # all channels
        all_channels = len(selected_channel) == 0
        channel_int = -1

        # Extract the wavelength from the channel name
        if not all_channels:
            try:
                channel_int = int(selected_channel.split()[0])
            except ValueError:
                print("Invalid channel name.")
                raise

        # Extract the downsample factors for the selected resolution level
        downsample_z, downsample_y, downsample_x = self.tiles[
            0
        ].resolution_pyramid[resolution_level]

        assert self.xml_path is not None
        assert self.tile_config_path is not None

        result = run_big_stitcher(
            fiji_path,
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

        # Print the output of BigStitcher to the command line
        print(result.stdout)

        # Wait for the BigStitcher to write XML file
        # Need to find a better way to do this
        sleep(1)

        self.read_big_stitcher_transforms()

    def read_big_stitcher_transforms(self) -> None:
        """
        Read the BigStitcher transforms from the XML file and update the tile
        positions accordingly.
        """
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape
        assert self.xml_path is not None
        stitched_translations = get_big_stitcher_transforms(
            self.xml_path, z_size, y_size, x_size
        )
        for tile in self.tiles:
            # BigStitcher uses x,y,z order, switch to z,y,x order
            stitched_position = [
                stitched_translations[tile.id][4],
                stitched_translations[tile.id][2],
                stitched_translations[tile.id][0],
            ]
            tile.position = stitched_position
