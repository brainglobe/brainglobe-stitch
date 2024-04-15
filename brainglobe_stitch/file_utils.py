import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np

HEADERS = [
    "[POSITION]",
    "[ETL PARAMETERS]",
    "[GALVO PARAMETERS]",
    "[CAMERA PARAMETERS]",
]


def create_pyramid_bdv_h5(
    input_file: Path,
    yield_progress: bool = False,
):
    """
    Create a resolution pyramid for a Big Data Viewer HDF5 file.

    Parameters
    ----------
    input_file: Path
        The path to the input HDF5 file.
    yield_progress: bool, optional
        Whether to yield progress.
    """
    resolutions_array = np.array(
        [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
    )

    subdivisions_array = np.array(
        [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
    )

    with h5py.File(input_file, "r+") as f:
        data_group = f["t00000"]
        num_done = 0
        num_slices = len(data_group)

        for curr_slice in data_group:
            # Delete the old resolutions and subdivisions datasets
            del f[f"{curr_slice}/resolutions"]
            del f[f"{curr_slice}/subdivisions"]
            f[f"{curr_slice}/resolutions"] = resolutions_array
            f[f"{curr_slice}/subdivisions"] = subdivisions_array

            grp: h5py.Group = f[f"t00000/{curr_slice}"]
            for i in range(1, resolutions_array.shape[0]):
                downsampling_factors = (
                    resolutions_array[i] // resolutions_array[i - 1]
                )
                prev_resolution = grp[f"{i - 1}/cells"]
                # Add 1 to account for odd dimensions
                grp.require_dataset(
                    f"{i}/cells",
                    dtype=prev_resolution.dtype,
                    shape=(
                        (prev_resolution.shape[0] + 1)
                        // downsampling_factors[2],
                        (prev_resolution.shape[1] + 1)
                        // downsampling_factors[1],
                        (prev_resolution.shape[2] + 1)
                        // downsampling_factors[0],
                    ),
                )
                grp[f"{i}/cells"][...] = prev_resolution[
                    :: downsampling_factors[2],
                    :: downsampling_factors[1],
                    :: downsampling_factors[0],
                ]

            num_done += 1

            if yield_progress:
                yield int(100 * num_done / num_slices)


def parse_mesospim_metadata(
    meta_file_name: Path,
) -> List[Dict]:
    """
    Parse the metadata from a mesoSPIM .h5_meta.txt file.

    Parameters
    ----------
    meta_file_name: Path
        The path to the h5_meta.txt file.

    Returns
    -------
    List[Dict]
        A list of dictionaries containing the metadata for each tile.
    """
    tile_metadata = []
    with open(meta_file_name, "r") as f:
        lines = f.readlines()
        curr_tile_metadata: Dict[str, Union[str, int, float]] = {}

        for line in lines[3:]:
            line = line.strip()
            # Tile metadata is separated by a line starting with [CFG]
            if line.startswith("[CFG"):
                tile_metadata.append(curr_tile_metadata)
                curr_tile_metadata = {}
            # Skip the headers
            elif line in HEADERS:
                continue
            # Skip empty lines
            elif not line:
                continue
            else:
                split_line = line.split("]")
                value = split_line[1].strip()
                # Check if the value is an int or a float
                # If it is neither, store it as a string
                if value.isdigit():
                    curr_tile_metadata[split_line[0][1:]] = int(value)
                else:
                    try:
                        curr_tile_metadata[split_line[0][1:]] = float(value)
                    except ValueError:
                        curr_tile_metadata[split_line[0][1:]] = value

    tile_metadata.append(curr_tile_metadata)
    return tile_metadata


def check_mesospim_directory(
    mesospim_directory: Path,
) -> Tuple[Path, Path, Path]:
    """
    Check that the mesoSPIM directory contains the expected files.

    Parameters
    ----------
    mesospim_directory: Path
        The path to the mesoSPIM directory.

    Returns
    -------
    Tuple[Path, Path, Path]
        The paths to the bdv.xml, h5_meta.txt, and bdv.h5 files.
    """
    # List all files in the directory that do not start with a period
    # But end in the correct file extension
    xml_path = list(mesospim_directory.glob("[!.]*bdv.xml"))
    meta_path = list(mesospim_directory.glob("[!.]*h5_meta.txt"))
    h5_path = list(mesospim_directory.glob("[!.]*bdv.h5"))

    # Check that there is exactly one file of each type
    if len(xml_path) != 1:
        raise FileNotFoundError(
            f"Expected 1 bdv.xml file, found {len(xml_path)}"
        )

    if len(meta_path) != 1:
        raise FileNotFoundError(
            f"Expected 1 h5_meta.txt file, found {len(meta_path)}"
        )

    if len(h5_path) != 1:
        raise FileNotFoundError(f"Expected 1 h5 file, found {len(h5_path)}")

    return xml_path[0], meta_path[0], h5_path[0]


def get_slice_attributes(
    xml_path: Path, tile_names: List[str]
) -> Dict[str, Dict]:
    """
    Get the slice attributes from a Big Data Viewer XML file. Attributes
    include the illumination id, channel id, and tile id, and angle id.

    Parameters
    ----------
    xml_path: Path
        The path to the XML file.
    tile_names: List[str]
        The names of the tiles.

    Returns
    -------
    Dict[str, Dict]
        A dictionary containing the slice attributes for each tile.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    view_setups = root.findall(".//ViewSetup//attributes")

    slice_attributes = {}
    for child, name in zip(view_setups, tile_names):
        sub_dict = {}
        for sub_child in child:
            sub_dict[sub_child.tag] = sub_child.text

        slice_attributes[name] = sub_dict

    return slice_attributes


def get_big_stitcher_transforms(
    xml_path: Path, x_size: int, y_size: int, z_size: int
) -> List[List[int]]:
    """
    Get the translations for each tile from a Big Data Viewer XML file.
    The translations are calculated by BigStitcher.

    Parameters
    ----------
    xml_path: Path
        The path to the Big Data Viewer XML file.
    x_size: int
        The size of the image in the x-dimension.
    y_size: int
        The size of the image in the y-dimension.
    z_size: int
        The size of the image in the z-dimension.

    Returns
    -------
    List[List[int]]
        A list of translations for each tile.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stitch_transforms = root.findall(
        ".//ViewTransform/[Name='Stitching Transform']/affine"
    )
    assert (
        stitch_transforms is not None
    ), "No stitching transforms found in XML file"

    # Stitching Transforms are if aligning to grid is done manually
    # Translation from Tile Configuration is if aligned automatically
    grid_transforms = root.findall(
        ".//ViewTransform/[Name='Translation from Tile Configuration']/affine"
    )
    if len(grid_transforms) == 0:
        grid_transforms = root.findall(
            ".//ViewTransform/[Name='Translation to Regular Grid']/affine"
        )
    assert grid_transforms is not None, "No grid transforms found in XML file"

    z_scale_str = root.find(".//ViewTransform/[Name='calibration']/affine")
    assert z_scale_str is not None, "No z scale found in XML file"
    assert z_scale_str.text is not None, "No z scale found in XML file"
    z_scale = float(z_scale_str.text.split()[-2])
    deltas = []
    grids = []
    for i in range(len(stitch_transforms)):
        delta_nums_text = stitch_transforms[i].text
        grid_nums_text = grid_transforms[i].text

        assert (
            delta_nums_text is not None
        ), "Error reading stitch transform from XML file"
        assert (
            grid_nums_text is not None
        ), "Error reading grid transform from XML file"

        delta_nums = delta_nums_text.split()
        grid_nums = grid_nums_text.split()

        curr_delta = [
            round(float(delta_nums[3])),
            round(float(delta_nums[7])),
            round(float(delta_nums[11]) / z_scale),
        ]
        curr_grid = [
            round(float(grid_nums[3])),
            round(float(grid_nums[7])),
            round(float(grid_nums[11]) / z_scale),
        ]
        deltas.append(curr_delta)
        grids.append(curr_grid)

    # Calculate the minimum value for the grid transform for each dimension
    min_grid = [min([grid[i] for grid in grids]) for i in range(3)]
    # Normalise the grid transforms by subtracting the minimum value
    grids = [[grid[i] - min_grid[i] for i in range(3)] for grid in grids]
    # Calculate the maximum delta (from BigStitcher) for each dimension
    max_delta = [max([abs(delta[i]) for delta in deltas]) for i in range(3)]

    translations = []

    # Calculate the start and end coordinates for each tile such that the
    # first tile is at 0,0,0
    for i in range(len(deltas)):
        curr_delta = deltas[i]
        curr_grid = grids[i]

        x_start = curr_grid[0] + curr_delta[0] + max_delta[0]
        x_end = x_start + x_size
        y_start = curr_grid[1] + curr_delta[1] + max_delta[1]
        y_end = y_start + y_size
        z_start = curr_grid[2] + curr_delta[2] + max_delta[2]
        z_end = z_start + z_size

        translations.append([x_start, x_end, y_start, y_end, z_start, z_end])

    return translations
