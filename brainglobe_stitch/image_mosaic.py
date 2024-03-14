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
from brainglobe_stitch.tile import Overlap, Tile


class ImageMosaic:
    """
    Class to represent an image as a collection of tiles.

    Attributes
    ----------
    directory : Path
        The directory containing the image data.
    xml_path : Path | None
        The path to the Big Data Viewer XML file.
    meta_path : Path | None
        The path to the mesoSPIM metadata file.
    h5_path : Path | None
        The path to the Big Data Viewer h5 file containing the raw data.
    tile_config_path : Path | None
        The path to the BigStitcher tile configuration file.
    h5_file : h5py.File | None
        An open h5py file object for the raw data.
    channel_names : List[str]
        The names of the channels in the image as strings.
    tiles : List[Tile]
        The tiles in the image.
    tile_names : List[str]
        The names of the image tiles from BigDataViewer.
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
    scale_factors : npt.NDArray | None
        The scale factors for normalising the intensity of the image.
    intensity_adjusted : List[bool]
        A list of booleans indicating whether the intensity has been adjusted
        at each resolution level.
    overlaps_interpolated : List[bool]
        A list of booleans indicating whether the overlaps have been
        interpolated at each resolution level.
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
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

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
            tile.resolution_pyramid = np.array(
                self.h5_file[f"{tile_name}/resolutions"]
            )

        # Don't rewrite the tile config if it already exists
        # Need to read in stage coordinates if not writing the tile config
        # These will be used as the initial tile positions
        if not self.tile_config_path.exists():
            self.write_big_stitcher_tile_config(self.meta_path)
        else:
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
        try:
            self.read_big_stitcher_transforms()
        except AssertionError:
            print("BigStitcher transforms not found.")

        self.calculate_overlaps()

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
        print("Tile positions not found. Writing tile config file.")
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
                # Save the relative locations for each tile
                # BigStitcher uses x,y,z order, switch to z,y,x order
                tile.position = [
                    0,
                    relative_locations[tile.tile_id][1],
                    relative_locations[tile.tile_id][0],
                ]

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
        downsample_x, downsample_y, downsample_z = self.tiles[
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

        print(result.stdout)

        # Wait for the BigStitcher to write XML file
        # Need to find a better way to do this
        sleep(1)

        self.read_big_stitcher_transforms()

        self.calculate_overlaps()

    def read_big_stitcher_transforms(self):
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
            tile.position = stitched_position

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
        self, resolution_level: int = 0, percentile: int = 50
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
        if self.intensity_adjusted[resolution_level]:
            print("Intensity already adjusted at this resolution scale.")
            return

        if self.scale_factors is None:
            # Calculate scale factors on at least resolution level 2
            # The tiles are adjusted as the scale factors are calculated
            self.calculate_intensity_scale_factors(
                max(resolution_level, 2), percentile
            )

            if self.intensity_adjusted[resolution_level]:
                return

        assert self.scale_factors is not None

        # Adjust the intensity of each tile based on the scale factors
        for tile in self.tiles:
            if self.scale_factors[tile.id] != 1.0:
                tile.data_pyramid[resolution_level] = np.multiply(
                    tile.data_pyramid[resolution_level],
                    self.scale_factors[tile.id],
                    dtype=np.float16,
                ).astype(np.uint16)

        self.intensity_adjusted[resolution_level] = True

    def calculate_intensity_scale_factors(
        self, resolution_level: int = 2, percentile: int = 50
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

        for tile_i in self.tiles:
            # Iterate through the neighbours of each tile
            for neighbour_id in tile_i.neighbours:
                tile_j = self.tiles[neighbour_id]
                overlap = self.overlaps[(tile_i.id, tile_j.id)]

                # Extract the overlapping data from both tiles
                i_overlap, j_overlap = overlap.extract_tile_overlaps(
                    resolution_level
                )

                # Calculate the percentile intensity of the overlapping data
                median_i = np.percentile(i_overlap.ravel(), percentile)
                median_j = np.percentile(j_overlap.ravel(), percentile)

                curr_scale_factor = (median_i / median_j).compute()
                scale_factors[tile_i.id][tile_j.id] = curr_scale_factor[0]

                # Adjust the tile intensity based on the scale factor
                tile_j.data_pyramid[resolution_level] = np.multiply(
                    tile_j.data_pyramid[resolution_level],
                    curr_scale_factor,
                    dtype=np.float16,
                ).astype(np.uint16)

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
        output_file_name: str = "fused.zarr",
        normalise_intensity: bool = False,
        interpolate: bool = False,
    ) -> None:
        """
        Fuse the tiles into a single image and save it to the output file.

        Parameters
        ----------
        output_file_name: str
            The name of the output file, suffix dictates the output file type.
        normalise_intensity: bool
            Whether to normalise the intensity of the image before fusing.
        interpolate: bool
            Whether to interpolate the overlaps before fusing.
        """
        output_path = self.directory / output_file_name

        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape
        # Calculate the shape of the fused image
        fused_image_shape: Tuple[int, ...] = (
            max([tile.position[0] for tile in self.tiles]) + z_size,
            max([tile.position[1] for tile in self.tiles]) + y_size,
            max([tile.position[2] for tile in self.tiles]) + x_size,
        )

        if normalise_intensity:
            self.normalise_intensity(0, 80)

        if interpolate:
            self.interpolate_overlaps(0)

        if output_path.suffix == ".zarr":
            self._fuse_to_zarr(output_path, fused_image_shape)
        elif output_path.suffix == ".h5":
            self._fuse_to_bdv_h5(output_path, fused_image_shape)

    def _fuse_to_zarr(
        self,
        output_path: Path,
        fused_image_shape: Tuple[int, ...],
        pyramid_depth: int = 6,
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
        """
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape

        # Default chunk shape is (256, 256, 256) for the highest resolution
        chunk_shape: Tuple[int, ...] = (256, 256, 256)

        transformation_metadata, axes_metadata = self.get_metadata_for_zarr(
            pyramid_depth=pyramid_depth
        )

        if self.num_channels > 1:
            fused_image_shape = (self.num_channels, *fused_image_shape)
            chunk_shape = (self.num_channels, *chunk_shape)

        store = zarr.NestedDirectoryStore(str(output_path))
        root = zarr.group(store=store)
        compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)

        fused_image_store = root.create(
            "0",
            shape=fused_image_shape,
            chunks=chunk_shape,
            dtype="i2",
            compressor=compressor,
        )

        # Provide a way to set the max number of threads for compression
        # blosc.set_nthreads(24)

        # Place the tiles in reverse order of acquisition
        for tile in self.tiles[-1::-1]:
            if self.num_channels > 1:
                fused_image_store[
                    tile.channel_id,
                    tile.position[0] : tile.position[0] + z_size,
                    tile.position[1] : tile.position[1] + y_size,
                    tile.position[2] : tile.position[2] + x_size,
                ] = tile.data_pyramid[0].compute()
            else:
                fused_image_store[
                    tile.position[0] : tile.position[0] + z_size,
                    tile.position[1] : tile.position[1] + y_size,
                    tile.position[2] : tile.position[2] + x_size,
                ] = tile.data_pyramid[0].compute()

            print(f"Done tile {tile.id}")

        for i in range(1, len(transformation_metadata)):
            prev_resolution = da.from_zarr(
                root[str(i - 1)], chunks=chunk_shape
            )

            if self.num_channels > 1:
                downsampled_image = downscale_nearest(
                    prev_resolution, (1, 1, 2, 2)
                )
            else:
                downsampled_image = downscale_nearest(
                    prev_resolution, (1, 2, 2)
                )

            # Downsampled chunk shape is half of the previous resolution
            # down to 32
            chunk_shape = tuple([s // 2 if s > 32 else s for s in chunk_shape])
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

        possible_channel_colors = [
            "00FF00",
            "FF0000",
            "0000FF",
            "FFFF00",
            "00FFFF",
            "FF00FF",
        ]
        # Create the channels attribute
        channels = []
        for i in range(self.num_channels):
            channels.append(
                {
                    "active": True,
                    "color": possible_channel_colors[i],
                    "name": f"ch{i + 1}",
                    "window": {
                        "start": 0,
                        "end": 4000,
                        "min": 0,
                        "max": 65535,
                    },
                }
            )

        root.attrs["omero"] = {"channels": channels}

    def _fuse_to_bdv_h5(
        self,
        output_path: Path,
        fused_image_shape: Tuple[int, ...],
        pyramid_depth: int = 6,
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
        """
        z_size, y_size, x_size = self.tiles[0].data_pyramid[0].shape
        output_file = h5py.File(output_path, mode="w")

        # Everything is in x, y, z order for Big Data Viewer
        subdivision = np.array([32, 32, 16], dtype=np.int16)
        subdivisions = np.tile(subdivision, (pyramid_depth, 1))

        # Only downsampling in the x and y dimensions
        resolutions = np.ones((pyramid_depth, 3), dtype=np.int16)

        for i in range(1, pyramid_depth):
            resolutions[i, :-1] = 2**i

        channel_ds_list: List[List[h5py.Dataset]] = []
        for i in range(self.num_channels):
            # Write the resolutions and subdivisions for each channel
            output_file.require_dataset(
                f"s{i:02}/resolutions",
                data=resolutions,
                dtype="i2",
                shape=resolutions.shape,
            )
            output_file.require_dataset(
                f"s{i:02}/subdivisions",
                data=subdivisions,
                dtype="i2",
                shape=subdivisions.shape,
            )

            chunk_shape: Tuple[int, ...] = (256, 256, 256)

            if (np.array(fused_image_shape) < 256).any():
                chunk_shape = fused_image_shape

            # Create the datasets for each resolution level
            ds_list: List[h5py.Dataset] = []
            ds = output_file.require_dataset(
                f"t00000/s{i:02}/0/cells",
                shape=fused_image_shape,
                chunks=chunk_shape,
                dtype="i2",
            )
            ds_list.append(ds)

            for j in range(1, len(resolutions)):
                new_shape = (
                    fused_image_shape[0],
                    (fused_image_shape[1] + 1) // 2**j,
                    (fused_image_shape[2] + 1) // 2**j,
                )

                chunk_shape = tuple(
                    [s // 2 if s > 32 else s for s in chunk_shape]
                )
                down_ds = output_file.require_dataset(
                    f"t00000/s{i:02}/{j}/cells",
                    shape=new_shape,
                    chunks=chunk_shape,
                    dtype="i2",
                )

                ds_list.append(down_ds)

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
                    z_size // resolutions[i][2],
                    (y_size + 1) // resolutions[i][1],
                    (x_size + 1) // resolutions[i][0],
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

    def get_metadata_for_zarr(
        self, pyramid_depth: int = 6
    ) -> Tuple[List[List[Dict]], List[Dict]]:
        """
        Prepare the metadata for a zarr file.

        Parameters
        ----------
        pyramid_depth: int
            The depth of the resolution pyramid.

        Returns
        -------
        Tuple[List[List[Dict]], List[Dict]]
            A tuple with the coordinate transformations and axes metadata.
        """
        axes = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        # Assumes z is never downsampled
        coordinate_transformations = []
        for i in range(pyramid_depth):
            coordinate_transformations.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            self.z_resolution,
                            self.x_y_resolution * 2**i,
                            self.x_y_resolution * 2**i,
                        ],
                    }
                ]
            )

        # Add metadata for channel axis
        if self.num_channels > 1:
            axes = [
                {"name": "c", "type": "channel"},
                *axes,
            ]

            for transform in coordinate_transformations:
                transform[0]["scale"] = [1.0, *transform[0]["scale"]]

        return coordinate_transformations, axes
