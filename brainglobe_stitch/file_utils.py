import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Union

import dask.array as da
import h5py
import numpy as np
import zarr
from tifffile import imwrite

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
    yield_progress:
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


def write_bdv_xml(
    output_xml_path: Path,
    input_xml_path: Path,
    hdf5_path: Path,
    image_size: Tuple[int, ...],
    num_channels: int,
) -> None:
    """
    Write a Big Data Viewer (BDV) XML file.

    Parameters
    ----------
    output_xml_path: Path
        The path to the output BDV XML file.
    input_xml_path: Path
        The path to the input BDV XML file.
    hdf5_path:
        The path to the output HDF5 file.
    image_size:
        The size of the image in pixels.
    num_channels:
        The number of channels in the image.
    """
    input_tree = ET.parse(input_xml_path)
    input_root = input_tree.getroot()

    base_path = input_root.find(".//BasePath")

    root = ET.Element("SpimData", version="0.2")

    assert base_path is not None, "No BasePath tag found in the input XML file"
    root.append(base_path)

    sequence_desc = ET.SubElement(root, "SequenceDescription")
    image_loader = input_root.find(".//ImageLoader")
    assert (
        image_loader is not None
    ), "No ImageLoader tag found in the input XML file"

    hdf5_path_node = image_loader.find(".//hdf5")
    assert (
        hdf5_path_node is not None
    ), "No hdf5 tag found in the input XML file"
    # Replace the hdf5 path with the new relative path
    hdf5_path_node.text = str(hdf5_path.name)
    sequence_desc.append(image_loader)

    view_setup = input_root.find(".//ViewSetup")
    assert (
        view_setup is not None
    ), "No ViewSetup tag found in the input XML file"
    # Replace the size of the image with the new size
    view_setup[2].text = f"{image_size[2]} {image_size[1]} {image_size[0]}"

    view_setups = ET.SubElement(sequence_desc, "ViewSetups")
    view_setups.append(view_setup)

    # Add the view setups for the other channels
    for i in range(1, num_channels):
        view_setup_copy = copy.deepcopy(view_setup)
        view_setup_copy[0].text = f"{i}"
        view_setup_copy[1].text = f"setup {i}"
        view_setup_copy[4][1].text = f"{i}"
        view_setups.append(view_setup_copy)

    attributes_illumination = input_root.find(
        ".//Attributes[@name='illumination']"
    )
    assert (
        attributes_illumination is not None
    ), "No illumination attributes found in the input XML file"
    view_setups.append(attributes_illumination)

    attributes_channel = input_root.find(".//Attributes[@name='channel']")
    assert (
        attributes_channel is not None
    ), "No channel attributes found in the input XML file"
    view_setups.append(attributes_channel)

    attributes_tiles = ET.SubElement(view_setups, "Attributes", name="tile")
    tile = input_root.find(".//Tile/[id='0']")
    assert tile is not None, "No Tile tag found in the input XML file"
    attributes_tiles.append(tile)

    attributes_angles = input_root.find(".//Attributes[@name='angle']")
    assert (
        attributes_angles is not None
    ), "No angle attributes found in the input XML file"
    view_setups.append(attributes_angles)

    timepoints = input_root.find(".//Timepoints")
    assert (
        timepoints is not None
    ), "No Timepoints tag found in the input XML file"
    sequence_desc.append(timepoints)

    missing_views = input_root.find(".//MissingViews")
    assert (
        missing_views is not None
    ), "No MissingViews tag found in the input XML file"
    sequence_desc.append(missing_views)

    view_registrations = ET.SubElement(root, "ViewRegistrations")

    # Write the calibrations for each channel
    # Allows BDV to convert pixel coordinates to physical coordinates
    for i in range(num_channels):
        view_registration = ET.SubElement(
            view_registrations,
            "ViewRegistration",
            attrib={"timepoint": "0", "setup": f"{i}"},
        )
        calibration = input_root.find(".//ViewTransform/[Name='calibration']")
        assert (
            calibration is not None
        ), "No calibration tag found in the input XML file"
        view_registration.append(calibration)

    tree = ET.ElementTree(root)
    # Add a two space indentation to the file
    ET.indent(tree, space="  ")
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    return


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


def write_tiff(
    source_file: Path, output_file: Path, resolution_level: int = 2
) -> None:
    """
    Write one resolution level of a zarr file to a TIFF file.

    Parameters
    ----------
    source_file: Path
        The path to the source zarr file.
    output_file: Path
        The path to the output TIFF file.
    resolution_level: int
        The resolution level to write to the TIFF file.
    """
    if source_file.suffix == ".zarr":
        store = zarr.NestedDirectoryStore(str(source_file))
        root = zarr.group(store=store)
        # Extract the correct resolution level
        data = root[str(resolution_level)]
        # Extract the scale metadata
        scale = root.attrs["multiscales"][0]["datasets"][resolution_level][
            "coordinateTransformations"
        ][0]["scale"][1:]
        # Swap the axes to match the expected order from CZYX to ZCYX
        adjusted_array = da.swapaxes(data, 0, 1)

        imwrite(
            output_file,
            adjusted_array,
            imagej=True,
            resolution=(1 / scale[1], 1 / scale[2]),
            metadata={
                "spacing": scale[0],
                "unit": "um",
                "axes": "ZCYX",
            },
        )


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

    min_grid = [min([grid[i] for grid in grids]) for i in range(3)]
    grids = [[grid[i] - min_grid[i] for i in range(3)] for grid in grids]
    max_delta = [max([abs(delta[i]) for delta in deltas]) for i in range(3)]

    translations = []

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
