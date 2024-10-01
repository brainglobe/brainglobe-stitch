from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple

import dask.array as da
import h5py
import numpy as np
import numpy.typing as npt
import zarr
from numcodecs import Blosc
from ome_zarr.dask_utils import downscale_nearest
from ome_zarr.writer import write_multiscales_metadata
from rich.progress import Progress

from brainglobe_stitch.big_stitcher_bridge import run_big_stitcher
from brainglobe_stitch.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
    get_big_stitcher_transforms,
    get_slice_attributes,
    parse_mesospim_metadata,
    write_bdv_xml,
)
from brainglobe_stitch.tile import Tile
from brainglobe_stitch.utils import calculate_thresholds


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
        resolution_level : int
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
        meta_file_name : Path
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
        fiji_path : Path
            The path to the Fiji application.
        resolution_level : int
            The resolution level to stitch the tiles at.
        selected_channel : str
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
                raise ValueError("Invalid channel name.")

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
        assert self.xml_path is not None
        stitched_translations = get_big_stitcher_transforms(self.xml_path)
        for tile in self.tiles:
            stitched_position = stitched_translations[tile.id]
            tile.position = stitched_position

    def fuse(
        self,
        output_file_name: str,
        downscale_factors: Tuple[int, int, int] = (1, 2, 2),
        chunk_shape: Tuple[int, int, int] = (128, 128, 128),
        pyramid_depth: int = 5,
    ) -> None:
        """
        Fuse the tiles into a single image and save it to the output file.

        Parameters
        ----------
        output_file_name: str
            The name of the output file, suffix dictates the output file type.
            Accepts .zarr and .h5 extensions.
        downscale_factors: Tuple[int, int, int], default: (1, 2, 2)
            The factors to downscale the image by in the z, y, x dimensions.
        chunk_shape: Tuple[int, ...], default: (128, 128, 128)
            The shape of the chunks in the zarr file.
        pyramid_depth: int, default: 5
            The depth of the resolution pyramid.
        """
        output_path = self.directory / output_file_name

        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape
        # Calculate the shape of the fused image
        fused_image_shape: Tuple[int, int, int] = (
            max([tile.position[0] for tile in self.tiles]) + z_size,
            max([tile.position[1] for tile in self.tiles]) + y_size,
            max([tile.position[2] for tile in self.tiles]) + x_size,
        )

        if output_path.suffix == ".zarr":
            compression_method = "zstd"
            compression_level = 3
            self._fuse_to_zarr(
                output_path,
                fused_image_shape,
                downscale_factors,
                pyramid_depth,
                chunk_shape,
                compression_method,
                compression_level,
            )
        elif output_path.suffix == ".h5":
            self._fuse_to_bdv_h5(
                output_path,
                fused_image_shape,
                downscale_factors,
                pyramid_depth,
                chunk_shape,
            )
        else:
            raise ValueError(
                "Invalid output file type. "
                "Currently, .zarr and .h5 are supported."
            )

    def _fuse_to_zarr(
        self,
        output_path: Path,
        fused_image_shape: Tuple[int, ...],
        downscale_factors: Tuple[int, int, int],
        pyramid_depth: int,
        chunk_shape: Tuple[int, ...],
        compression_method: str,
        compression_level: int,
    ) -> None:
        """
        Fuse the tiles in the ImageMosaic into a single image and save it as a
        zarr file.

        Parameters
        ----------
        output_path: Path
            The path of the output file.
        fused_image_shape: Tuple[int, ...]
            The shape of the fused image.
        downscale_factors: Tuple[int, int, int]
            The factors to downscale the image by in the z, y, x dimensions.
        pyramid_depth: int,
            The depth of the resolution pyramid.
        chunk_shape: Tuple[int, ...],
            The shape of the chunks in the zarr file.
        compression_method: str, default: "zstd"
            The compression algorithm to use.
        compression_level: int, default: 3
            The compression level to use.
        """
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape

        transformation_metadata, axes_metadata = (
            self._generate_metadata_for_zarr(pyramid_depth, downscale_factors)
        )

        fused_image_shape = (self.num_channels, *fused_image_shape)
        chunk_shape = (self.num_channels, *chunk_shape)

        store = zarr.NestedDirectoryStore(str(output_path))
        root = zarr.group(store=store)
        compressor = Blosc(
            cname=compression_method,
            clevel=compression_level,
            shuffle=Blosc.SHUFFLE,
        )

        fused_image_store = root.create(
            "0",
            shape=fused_image_shape,
            chunks=chunk_shape,
            dtype="i2",
            compressor=compressor,
        )

        # Place the tiles in reverse order of acquisition
        for tile in self.tiles[-1::-1]:
            fused_image_store[
                tile.channel_id,
                tile.position[0] : tile.position[0] + z_size,
                tile.position[1] : tile.position[1] + y_size,
                tile.position[2] : tile.position[2] + x_size,
            ] = tile.data_pyramid[0].compute()

            print(f"Done tile {tile.id}")

        for i in range(1, len(transformation_metadata)):
            prev_resolution = da.from_zarr(
                root[str(i - 1)], chunks=chunk_shape
            )

            factors = (1, *downscale_factors)
            downsampled_image = downscale_nearest(prev_resolution, factors)

            downsampled_shape = downsampled_image.shape
            downsampled_store = root.require_dataset(
                f"{i}",
                shape=downsampled_shape,
                chunks=chunk_shape,
                dtype="i2",
                compressor=compressor,
            )
            downsampled_image.to_zarr(downsampled_store)

            print(f"Done resolution {i}")

        # Create the datasets containing the correct scaling transforms
        datasets = []
        for i, transform in enumerate(transformation_metadata):
            datasets.append(
                {"path": f"{i}", "coordinateTransformations": transform}
            )

        write_multiscales_metadata(
            group=root,
            datasets=datasets,
            axes=axes_metadata,
        )

        channel_thresholds = calculate_thresholds(self.tiles)
        # Create the channels attribute
        channels = []
        for i, name in enumerate(self.channel_names):
            channels.append(
                {
                    "active": True,
                    "color": "FFFFFF",
                    "label": name,
                    "window": {
                        "start": 0,
                        "end": channel_thresholds[name],
                        "min": 0,
                        "max": 65535,
                    },
                }
            )

        root.attrs["omero"] = {"channels": channels}

    def _fuse_to_bdv_h5(
        self,
        output_path: Path,
        fused_image_shape: Tuple[int, int, int],
        downscale_factors: Tuple[int, int, int],
        pyramid_depth,
        chunk_shape: Tuple[int, int, int],
    ) -> None:
        """
        Fuse the tiles in the ImageMosaic into a single image and save it as a
        Big Data Viewer h5 file.

        Parameters
        ----------
        output_path: Path
            The path of the output file.
        fused_image_shape: Tuple[int, ...]
            The shape of the fused image.
        downscale_factors: Tuple[int, int, int],
            The factors to downscale the image by in the z, y, x dimensions.
        pyramid_depth: int,
            The depth of the resolution pyramid.
        chunk_shape: Tuple[int, int, int],
            The shape of the chunks in the h5 file.
        """
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape
        output_file = h5py.File(output_path, mode="w")

        # Metadata is in x, y, z order for Big Data Viewer
        downscale_factors_array = np.array(downscale_factors)
        resolutions = np.ones((pyramid_depth, 3), dtype=np.int16)

        for i in range(1, pyramid_depth):
            resolutions[i, :] = downscale_factors_array[-1::-1] ** i

        channel_ds_list: List[List[h5py.Dataset]] = []
        for i in range(self.num_channels):
            # Write the resolutions and subdivisions for each channel
            temp_chunk_shape = chunk_shape
            output_file.require_dataset(
                f"s{i:02}/resolutions",
                data=resolutions,
                dtype="i2",
                shape=resolutions.shape,
            )
            output_file.create_dataset(
                f"s{i:02}/subdivisions",
                dtype="i2",
                shape=resolutions.shape,
            )

            if np.any(np.array(fused_image_shape) < temp_chunk_shape):
                temp_chunk_shape = fused_image_shape

            # Create the datasets for each resolution level
            ds_list: List[h5py.Dataset] = []
            ds = output_file.require_dataset(
                f"t00000/s{i:02}/0/cells",
                shape=fused_image_shape,
                chunks=temp_chunk_shape,
                dtype="i2",
            )
            ds_list.append(ds)
            # Set the chunk shape for the first resolution level
            # Flip the shape to match the x, y, z order
            output_file[f"s{i:02}/subdivisions"][0] = temp_chunk_shape[::-1]
            new_shape = fused_image_shape

            for j in range(1, len(resolutions)):
                new_shape = calculate_downsampled_image_shape(
                    new_shape, downscale_factors
                )

                if np.any(np.array(new_shape) < temp_chunk_shape):
                    temp_chunk_shape = new_shape

                down_ds = output_file.require_dataset(
                    f"t00000/s{i:02}/{j}/cells",
                    shape=new_shape,
                    chunks=temp_chunk_shape,
                    dtype="i2",
                )

                ds_list.append(down_ds)
                # Set the chunk shape for the other resolution levels
                # Flip the shape to match the x, y, z order
                output_file[f"s{i:02}/subdivisions"][j] = temp_chunk_shape[
                    ::-1
                ]

            channel_ds_list.append(ds_list)

        # Place the tiles in reverse order of acquisition
        for tile in self.tiles[-1::-1]:
            current_tile_data = tile.data_pyramid[0].compute()
            channel_ds_list[tile.channel_id][0][
                tile.position[0] : tile.position[0] + z_size,
                tile.position[1] : tile.position[1] + y_size,
                tile.position[2] : tile.position[2] + x_size,
            ] = current_tile_data

            # Use a simple downsample for the other resolutions
            for i in range(1, len(resolutions)):
                scaled_position = tile.position // resolutions[i, -1::-1]
                scaled_size = (
                    (z_size + (resolutions[i][2] > 1)) // resolutions[i][2],
                    (y_size + (resolutions[i][1] > 1)) // resolutions[i][1],
                    (x_size + (resolutions[i][0] > 1)) // resolutions[i][0],
                )
                channel_ds_list[tile.channel_id][i][
                    scaled_position[0] : scaled_position[0] + scaled_size[0],
                    scaled_position[1] : scaled_position[1] + scaled_size[1],
                    scaled_position[2] : scaled_position[2] + scaled_size[2],
                ] = current_tile_data[
                    :: resolutions[i][2],
                    :: resolutions[i][1],
                    :: resolutions[i][0],
                ]

            print(f"Done tile {tile.id}")

        assert self.xml_path is not None

        write_bdv_xml(
            output_path.with_suffix(".xml"),
            self.xml_path,
            output_path,
            fused_image_shape,
            self.num_channels,
        )

        output_file.close()

    def _generate_metadata_for_zarr(
        self,
        pyramid_depth: int,
        downscale_factors: Tuple[int, int, int],
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """
        Prepare the metadata for a zarr file. The metadata conforms to the
        OME-Zarr specification (https://ngff.openmicroscopy.org/latest/).

        Parameters
        ----------
        pyramid_depth: int
            The depth of the resolution pyramid.
        downscale_factors: Tuple[int, int, int]
            The factors to downscale the image by in the z, y, x dimensions.

        Returns
        -------
        Tuple[List[List[Dict]], List[Dict]]
            A tuple with the coordinate transformations and axes metadata.
        """
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        coordinate_transformations = []
        for i in range(pyramid_depth):
            coordinate_transformations.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            self.z_resolution * downscale_factors[0] ** i,
                            self.x_y_resolution * downscale_factors[1] ** i,
                            self.x_y_resolution * downscale_factors[2] ** i,
                        ],
                    }
                ]
            )

        return coordinate_transformations, axes


def calculate_downsampled_image_shape(
    image_shape: Tuple[int, int, int], downscale_factors: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    new_shape = (
        (image_shape[0] + (downscale_factors[0] > 1)) // downscale_factors[0],
        (image_shape[1] + (downscale_factors[1] > 1)) // downscale_factors[1],
        (image_shape[2] + (downscale_factors[2] > 1)) // downscale_factors[2],
    )

    return new_shape
