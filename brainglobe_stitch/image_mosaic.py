import copy
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

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
    safe_find,
)
from brainglobe_stitch.tile import Overlap, Tile


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
    tile_metadata : List[Dict]
        The metadata for each tile.
    overlaps : Dict[Tuple[int, int], Overlap]
        A dictionary of tile pairs and their overlaps.
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
        self.overlaps: Dict[Tuple[int, int], Overlap] = {}
        self.x_y_resolution: float = 4.0  # um per pixel
        self.z_resolution: float = 5.0  # um per pixel
        self.num_channels: int = 1

        self.load_mesospim_directory()

        self.scale_factors: Optional[npt.NDArray] = None
        self.intensity_adjusted: List[bool] = [False] * len(
            self.tiles[0].resolution_pyramid
        )
        self.overlaps_interpolated: List[bool] = [False] * len(
            self.tiles[0].resolution_pyramid
        )

    def __del__(self):
        if self.h5_file:
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

        self.calculate_overlaps()

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

        self.calculate_overlaps()

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

    def reload_resolution_pyramid_level(self, resolution_level: int) -> None:
        """
        Reload the data for a given resolution level.

        Parameters
        ----------
        resolution_level: int
            The resolution level to reload the data for.
        """
        if self.h5_file:
            for tile in self.tiles:
                tile.data_pyramid[resolution_level] = da.from_array(
                    self.h5_file[
                        f"t00000/{tile.name}/{resolution_level}/cells"
                    ]
                )

            self.intensity_adjusted[resolution_level] = False
            self.overlaps_interpolated[resolution_level] = False

    def calculate_overlaps(self) -> None:
        """
        Calculate the overlaps between the tiles in the ImageMosaic.
        """
        self.overlaps = {}
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape

        for tile_i in self.tiles[:-1]:
            position_i = tile_i.position
            tile_i.neighbours = []
            for tile_j in self.tiles[tile_i.id + 1 :]:
                position_j = tile_j.position

                # Check for overlap in the x and y dimensions
                # and that the tiles do not have the same tile_id
                # and that the tiles are from the same channel
                if (
                    (
                        position_i[1] + y_size > position_j[1]
                        and position_i[1] < position_j[1] + y_size
                    )
                    and (
                        position_i[2] + x_size > position_j[2]
                        and position_i[2] < position_j[2] + x_size
                    )
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
        self, resolution_level: int = 0, percentile: int = 80
    ) -> None:
        """
        Normalise the intensity of the image at a given resolution level.

        Parameters
        ----------
        resolution_level: int
            The resolution level to normalise the intensity at.
        percentile: int
            The percentile based on which the normalisation is done.
        """
        # Calculate scale factors on at least resolution level 2
        # The tiles are adjusted as the scale factors are calculated
        self.calculate_intensity_scale_factors(
            max(resolution_level, 2), percentile
        )

        # Won't adjust the intensity factors if it was done during the scale
        # factor calculation
        if self.intensity_adjusted[resolution_level]:
            return

        assert self.scale_factors is not None

        # Adjust the intensity of each tile based on the scale factors
        for tile in self.tiles:
            if self.scale_factors[tile.id] != 1.0:
                tile.data_pyramid[resolution_level] = da.multiply(
                    tile.data_pyramid[resolution_level],
                    self.scale_factors[tile.id],
                ).astype(tile.data_pyramid[resolution_level].dtype)

        self.intensity_adjusted[resolution_level] = True

    def calculate_intensity_scale_factors(
        self, resolution_level: int, percentile: int
    ):
        """
        Calculate the scale factors for normalising the intensity of the image.

        Parameters
        ----------
        resolution_level: int
            The resolution level to calculate the scale factors at.
        percentile: int
            The percentile based on which the normalisation is done.
        """
        num_tiles = len(self.tiles)
        scale_factors = np.ones((num_tiles, num_tiles))

        if self.intensity_adjusted[resolution_level]:
            self.reload_resolution_pyramid_level(resolution_level)
            self.intensity_adjusted[resolution_level] = False

        for tile_i in self.tiles:
            # Iterate through the neighbours of each tile
            print(f"Calculating scale factors for tile {tile_i.id}")
            for neighbour_id in tile_i.neighbours:
                tile_j = self.tiles[neighbour_id]
                overlap = self.overlaps[(tile_i.id, tile_j.id)]

                # Extract the overlapping data from both tiles
                i_overlap, j_overlap = overlap.extract_tile_overlaps(
                    resolution_level
                )

                # Calculate the percentile intensity of the overlapping data
                median_i = da.percentile(i_overlap.ravel(), percentile)
                median_j = da.percentile(j_overlap.ravel(), percentile)

                curr_scale_factor = (median_i / median_j).compute()
                scale_factors[tile_i.id][tile_j.id] = curr_scale_factor[0]

                # Adjust the tile intensity based on the scale factor
                tile_j.data_pyramid[resolution_level] = da.multiply(
                    tile_j.data_pyramid[resolution_level],
                    curr_scale_factor,
                ).astype(tile_j.data_pyramid[resolution_level].dtype)

        self.intensity_adjusted[resolution_level] = True
        # Calculate the product of the scale factors for each tile's neighbours
        # The product is the final scale factor for that tile
        self.scale_factors = np.prod(scale_factors, axis=0)

        return

    def interpolate_overlaps(self, resolution_level: int) -> None:
        """
        Interpolate the overlaps between the tiles at a given resolution level.

        Parameters
        ----------
        resolution_level: int
            The resolution level to interpolate the overlaps at.
        """
        tile_shape = self.tiles[0].data_pyramid[resolution_level].shape

        if self.overlaps_interpolated[resolution_level]:
            print("Overlaps already interpolated at this resolution scale.")
            return

        for tile_i in self.tiles[:-1]:
            # Iterate through each neighbour
            for neighbour_id in tile_i.neighbours:
                tile_j = self.tiles[neighbour_id]
                overlap = self.overlaps[(tile_i.id, tile_j.id)]

                overlap.linear_interpolation(resolution_level, tile_shape)

                print(
                    f"Done interpolating tile {tile_i.id}" f" and {tile_j.id}"
                )

        self.overlaps_interpolated[resolution_level] = True

    def fuse(
        self,
        output_path: Path,
        normalise_intensity: bool = False,
        normalise_intensity_percentile: int = 80,
        interpolate: bool = False,
        downscale_factors: Tuple[int, int, int] = (1, 2, 2),
        chunk_shape: Tuple[int, int, int] = (128, 128, 128),
        pyramid_depth: int = 5,
        compression_method: str = "zstd",
        compression_level: int = 6,
    ) -> None:
        """
        Fuse the tiles into a single image and save it to the output file.

        Parameters
        ----------
        output_path: Path
            The name of the output file, suffix dictates the output file type.
            Accepts .zarr and .h5 extensions.#
        normalise_intensity: bool, default: False
            Normalise the intensity differences between tiles.
        normalise_intensity_percentile: int, default: 80
            The brightness percentile to use for normalising intensity.
        interpolate: bool, default: False
            Interpolate the overlaps between tiles.
        downscale_factors: Tuple[int, int, int], default: (1, 2, 2)
            The factors to downscale the image by in the z, y, x dimensions.
        chunk_shape: Tuple[int, ...], default: (128, 128, 128)
            The shape of the chunks in the zarr file.
        pyramid_depth: int, default: 5
            The depth of the resolution pyramid.
        compression_method: str, default: "zstd"
            The compression algorithm to use (only used for zarr).
        compression_level: int, default: 6
            The compression level to use (only used for zarr).
        """
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape
        # Calculate the shape of the fused image
        fused_image_shape: Tuple[int, int, int] = (
            max([tile.position[0] for tile in self.tiles]) + z_size,
            max([tile.position[1] for tile in self.tiles]) + y_size,
            max([tile.position[2] for tile in self.tiles]) + x_size,
        )

        if normalise_intensity:
            self.normalise_intensity(0, normalise_intensity_percentile)
        elif self.intensity_adjusted[0]:
            self.reload_resolution_pyramid_level(0)

        if interpolate:
            self.interpolate_overlaps(0)
        elif self.overlaps_interpolated[0]:
            self.reload_resolution_pyramid_level(0)

        if output_path.suffix == ".zarr":
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

        print(f"Fused image saved to {output_path}")

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

        channel_thresholds = self.calculate_contrast_max()
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
                        "end": channel_thresholds[name] * 1.5,
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
        pyramid_depth: int,
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

        self._write_bdv_xml(
            output_path.with_suffix(".xml"),
            output_path,
            fused_image_shape,
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

        Generates the coordinate transformations and axes metadata to be
        written to the fused zarr file.
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

    def _write_bdv_xml(
        self,
        output_xml_path: Path,
        hdf5_path: Path,
        image_size: Tuple[int, ...],
    ) -> None:
        """
        Write a Big Data Viewer (BDV) XML file.

        Parameters
        ----------
        output_xml_path: Path
            The path to the output BDV XML file.
        hdf5_path:
            The path to the output HDF5 file.
        image_size:
            The size of the image in pixels.
        """
        if self.xml_path is None:
            raise ValueError("No input XML file provided.")

        input_tree = ET.parse(self.xml_path)
        input_root = input_tree.getroot()
        base_path = safe_find(input_root, ".//BasePath")

        root = ET.Element("SpimData", version="0.2")
        root.append(base_path)

        sequence_desc = ET.SubElement(root, "SequenceDescription")

        image_loader = safe_find(input_root, ".//ImageLoader")
        hdf5_path_node = safe_find(image_loader, ".//hdf5")
        # Replace the hdf5 path with the new relative path
        hdf5_path_node.text = str(hdf5_path.name)
        sequence_desc.append(image_loader)

        view_setup = safe_find(input_root, ".//ViewSetup")
        # Replace the size of the image with the new size
        # The image shape is in z,y,x order,
        # metadata needs to be in x,y,z order
        view_setup[2].text = f"{image_size[2]} {image_size[1]} {image_size[0]}"

        view_setups = ET.SubElement(sequence_desc, "ViewSetups")
        view_setups.append(view_setup)

        # Add the view setups for the other channels
        for i in range(1, self.num_channels):
            view_setup_copy = copy.deepcopy(view_setup)
            view_setup_copy[0].text = f"{i}"
            view_setup_copy[1].text = f"setup {i}"
            view_setup_copy[4][1].text = f"{i}"
            view_setups.append(view_setup_copy)

        attributes_illumination = safe_find(
            input_root, ".//Attributes[@name='illumination']"
        )
        view_setups.append(attributes_illumination)

        attributes_channel = safe_find(
            input_root, ".//Attributes[@name='channel']"
        )
        view_setups.append(attributes_channel)

        attributes_tiles = ET.SubElement(
            view_setups, "Attributes", name="tile"
        )
        tile = safe_find(input_root, ".//Tile/[id='0']")
        attributes_tiles.append(tile)

        attributes_angles = safe_find(
            input_root, ".//Attributes[@name='angle']"
        )
        view_setups.append(attributes_angles)

        timepoints = safe_find(input_root, ".//Timepoints")
        sequence_desc.append(timepoints)

        # Missing views are not necessary for the BDV XML
        # May not be present in all BDV XML files
        try:
            missing_views = safe_find(input_root, ".//MissingViews")
            sequence_desc.append(missing_views)
        except ValueError as e:
            print(e)

        view_registrations = ET.SubElement(root, "ViewRegistrations")

        # Write the calibrations for each channel
        # Allows BDV to convert pixel coordinates to physical coordinates
        for i in range(self.num_channels):
            view_registration = ET.SubElement(
                view_registrations,
                "ViewRegistration",
                attrib={"timepoint": "0", "setup": f"{i}"},
            )
            calibration = safe_find(
                input_root, ".//ViewTransform/[Name='calibration']"
            )
            view_registration.append(calibration)

        tree = ET.ElementTree(root)
        # Add a two space indentation to the file
        ET.indent(tree, space="  ")
        tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    def calculate_contrast_max(
        self, pyramid_level: int = 3
    ) -> Dict[str, float]:
        """
        Calculate the appropriate contrast max for each channel.

        The 99th percentile of the middle slice of each tile is calculated.
        The maximum of these values is taken as the threshold for each channel.

        Parameters
        ----------
        pyramid_level: int
            The pyramid level at which the contrast max is to be calculated.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the channel names as keys
            and the contrast maxes as values.
        """
        middle_slice_index = (
            self.tiles[0].data_pyramid[pyramid_level].shape[0] // 2
        )
        thresholds: Dict[str, List[float]] = {}

        for tile in self.tiles:
            tile_data = tile.data_pyramid[pyramid_level]
            curr_threshold = np.percentile(
                tile_data[middle_slice_index].ravel(), 99
            ).compute()[0]
            threshold_list = thresholds.get(tile.channel_name, [])
            threshold_list.append(curr_threshold)
            thresholds[tile.channel_name] = threshold_list

        final_thresholds: Dict[str, float] = {
            channel: np.max(thresholds.get(channel))
            for channel, threshold_value in thresholds.items()
        }

        return final_thresholds


def calculate_downsampled_image_shape(
    image_shape: Tuple[int, int, int], downscale_factors: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    new_shape = (
        (image_shape[0] + (downscale_factors[0] > 1)) // downscale_factors[0],
        (image_shape[1] + (downscale_factors[1] > 1)) // downscale_factors[1],
        (image_shape[2] + (downscale_factors[2] > 1)) // downscale_factors[2],
    )

    return new_shape
