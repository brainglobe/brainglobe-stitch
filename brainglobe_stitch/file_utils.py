import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt

HEADERS = [
    "[POSITION]",
    "[ETL PARAMETERS]",
    "[GALVO PARAMETERS]",
    "[CAMERA PARAMETERS]",
]


def create_pyramid_bdv_h5(
    input_file: Path,
    yield_progress: bool = False,
    resolutions_array: npt.NDArray = np.array(
        [[1, 1, 1], [2, 2, 1], [4, 4, 1], [8, 8, 1], [16, 16, 1]]
    ),
    subdivisions_array: npt.NDArray = np.array(
        [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
    ),
):
    """
    Create a resolution pyramid for a Big Data Viewer HDF5 file. The function
    assumes no pyramid exists and creates a new one in place. By default,
    the function creates a 5 level pyramid with downsampling factors of 1, 2,
    4, 8, and 16 in x and y, with no downsampling in z. Deletes the old
    resolutions and subdivisions datasets and creates new ones.


    Parameters
    ----------
    input_file : Path
        The path to the input HDF5 file.
    yield_progress : bool, optional
        Whether to yield progress. If True, the function will yield the
        progress as a percentage.
    resolutions_array : npt.NDArray, optional
        The downsampling factors to use for each resolution level.
        This is a 2D array where each row represents a resolution level and the
        columns represent the downsampling factors for x, y, and z.
    subdivisions_array : npt.NDArray, optional
        The size of the blocks at each resolution level.
        This is a 2D array where each row represents a resolution level and the
        columns represent the size of the blocks for x, y, and z.
    """
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
                    # Data is stored in z,y,x, but the downsampling
                    # factors are in x,y,z, so need to reverse
                    # Adding 1 allows to account for dimensions of odd size,
                    # Only add 1 if the downsampling factor is greater than 1
                    shape=(
                        (
                            prev_resolution.shape[0]
                            + (downsampling_factors[2] > 1)
                        )
                        // downsampling_factors[2],
                        (
                            prev_resolution.shape[1]
                            + (downsampling_factors[1] > 1)
                        )
                        // downsampling_factors[1],
                        (
                            prev_resolution.shape[2]
                            + (downsampling_factors[0] > 1)
                        )
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
    meta_file_name : Path
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
    mesospim_directory : Path
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
    xml_path : Path
        The path to the XML file.
    tile_names : List[str]
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


def get_big_stitcher_transforms(xml_path: Path) -> npt.NDArray:
    """
    Get the translations for each tile from a Big Data Viewer XML file.
    The translations are calculated by BigStitcher.

    Parameters
    ----------
    xml_path : Path
        The path to the Big Data Viewer XML file.

    Returns
    -------
    npt.NDArray
        A numpy array of shape (num_tiles, num_dim) with the translations.
        Each row corresponds to a tile and each column to a dimension.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    stitch_transforms = safe_find_all(
        root, ".//ViewTransform/[Name='Stitching Transform']/affine"
    )

    # Stitching Transforms are there if aligning to grid is done manually
    # Translation from Tile Configurations are there if aligned automatically
    grid_transforms = safe_find_all(
        root,
        ".//ViewTransform/[Name='Translation from Tile Configuration']/affine",
    )
    if len(grid_transforms) == 0:
        grid_transforms = safe_find_all(
            root,
            ".//ViewTransform/[Name='Translation to Regular Grid']/affine",
        )

    z_scale_str = safe_find(
        root, ".//ViewTransform/[Name='calibration']/affine"
    )

    if not z_scale_str.text:
        raise ValueError("No z scale found in XML")

    z_scale = float(z_scale_str.text.split()[-2])

    deltas = np.ones((len(stitch_transforms), 3))
    grids = np.ones((len(grid_transforms), 3))
    for i in range(len(stitch_transforms)):
        delta_nums_text = stitch_transforms[i].text
        grid_nums_text = grid_transforms[i].text

        if not delta_nums_text or not grid_nums_text:
            raise ValueError("No translation values found in XML")

        delta_nums = delta_nums_text.split()
        grid_nums = grid_nums_text.split()

        # Extract the translation values from the transform.
        # Swap the order of the axis (x,y,z) to (z,y,x).
        # The input values are a flattened 4x4 matrix where
        # the translation values in the last column.
        deltas[i] = np.array(delta_nums[11:2:-4])
        grids[i] = np.array(grid_nums[11:2:-4])

    # Divide the z translation by the z scale
    deltas[:, 0] /= z_scale
    grids[:, 0] /= z_scale

    # Round the translations to the nearest integer
    grids = grids.round().astype(np.int32)
    deltas = deltas.round().astype(np.int32)

    # Normalise the grid transforms by subtracting the minimum value
    norm_grids = grids - grids.min(axis=0)
    # Calculate the maximum delta (from BigStitcher) for each dimension
    max_delta = np.absolute(deltas).max(axis=0)

    # Calculate the start and end coordinates for each tile such that the
    # first tile is at 0,0,0 and provide enough padding to account for the
    # transforms from BigStitcher
    translations = norm_grids + deltas + max_delta

    return translations


def safe_find_all(root: ET.Element, query: str) -> List[ET.Element]:
    """
    Find all elements matching a query in an ElementTree root. If no
    elements are found, return an empty list.

    Parameters
    ----------
    root : ET.Element
        The root of the ElementTree.
    query : str
        The query to search for.

    Returns
    -------
    List[ET.Element]
        A list of elements matching the query.
    """
    elements = root.findall(query)
    if elements is None:
        return []

    return elements


def safe_find(root: ET.Element, query: str) -> ET.Element:
    """
    Find the first element matching a query in an ElementTree root.
    Raise a ValueError if no element found.

    Parameters
    ----------
    root : ET.Element
        The root of the ElementTree.
    query : str
        The query to search for.

    Returns
    -------
    ET.Element
        The element matching the query or None.

    Raises
    ------
    ValueError
        If no element is found.
    """
    element = root.find(query)
    if element is None or element.text is None:
        raise ValueError(f"No element found for query {query}")

    return element
