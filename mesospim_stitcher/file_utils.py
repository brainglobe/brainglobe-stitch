import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import dask.array as da
import h5py
import numpy.typing as npt
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
    resolutions_array: npt.NDArray,
    subdivisions_array: npt.NDArray,
    yield_progress: bool = False,
):
    with h5py.File(input_file, "r+") as f:
        data_group = f["t00000"]
        num_done = 0
        num_slices = len(data_group)

        for curr_slice in data_group:
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


def parse_mesospim_metadata(meta_file_name: Path):
    tile_metadata = []
    with open(meta_file_name, "r") as f:
        lines = f.readlines()
        curr_tile_metadata: Dict[str, str | int | float] = {}

        for line in lines[3:]:
            line = line.strip()
            if line.startswith("[CFG"):
                tile_metadata.append(curr_tile_metadata)
                curr_tile_metadata = {}
            elif line in HEADERS:
                continue
            elif not line:
                continue
            else:
                split_line = line.split("]")
                value = split_line[1].strip()
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
    image_size: tuple,
):
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
    hdf5_path_node.text = str(hdf5_path.name)
    sequence_desc.append(image_loader)

    view_setup = input_root.find(".//ViewSetup")
    assert (
        view_setup is not None
    ), "No ViewSetup tag found in the input XML file"
    view_setup[2].text = f"{image_size[2]} {image_size[1]} {image_size[0]}"

    view_setups = ET.SubElement(sequence_desc, "ViewSetups")
    view_setups.append(view_setup)

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
    missing_views = input_root.find(".//MissingViews")
    assert (
        missing_views is not None
    ), "No MissingViews tag found in the input XML file"

    sequence_desc.append(timepoints)
    sequence_desc.append(missing_views)

    view_registrations = ET.SubElement(root, "ViewRegistrations")
    view_registration = ET.SubElement(
        view_registrations,
        "ViewRegistration",
        attrib={"timepoint": "0", "setup": "0"},
    )
    calibration = input_root.find(".//ViewTransform/[Name='calibration']")
    assert (
        calibration is not None
    ), "No calibration tag found in the input XML file"
    view_registration.append(calibration)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    return


def check_mesospim_directory(
    mesospim_directory: Path,
) -> tuple[Path, Path, Path]:
    xml_path = list(mesospim_directory.glob("*bdv.xml"))
    meta_path = list(mesospim_directory.glob("*h5_meta.txt"))
    h5_path = list(mesospim_directory.glob("*.h5"))

    if len(xml_path) != 1:
        raise FileNotFoundError(
            "Expected 1 bdv.xml file, found {len(xml_path)}"
        )

    if len(meta_path) != 1:
        raise FileNotFoundError(
            "Expected 1 h5_meta.txt file, found {len(meta_path)}"
        )

    if len(h5_path) != 1:
        raise FileNotFoundError("Expected 1 h5 file, found {len(h5_path)}")

    return xml_path[0], meta_path[0], h5_path[0]


def write_tiff(
    source_file: Path, output_file: Path, resolution_level: int = 2
):
    if source_file.suffix == ".zarr":
        store = zarr.NestedDirectoryStore(str(source_file))
        root = zarr.group(store=store)
        data = root[str(resolution_level)]
        scale = root.attrs["multiscales"][0]["datasets"][resolution_level][
            "coordinateTransformations"
        ][0]["scale"][1:]
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
