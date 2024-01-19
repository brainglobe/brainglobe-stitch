import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

HEADERS = [
    "[POSITION]",
    "[ETL PARAMETERS]",
    "[GALVO PARAMETERS]",
    "[CAMERA PARAMETERS]",
]


def create_pyramid_bdv_h5(
    input_file: Path, resolutions_array: np.array, subdivisions_array: np.array
):
    with h5py.File(input_file, "r+") as f:
        data_group = f["t00000"]

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
                grp.create_dataset(
                    f"{i}/cells",
                    data=prev_resolution[
                        :: downsampling_factors[2],
                        :: downsampling_factors[1],
                        :: downsampling_factors[0],
                    ],
                )

    return


def write_big_stitcher_tile_config(meta_file_name: Path) -> list[dict]:
    tile_metadata = parse_mesospim_metadata(meta_file_name)

    output_file = str(meta_file_name)[:-12] + "_tile_config.txt"

    first_channel = tile_metadata[0]["Laser"]
    num_channels = 1

    for metadata in tile_metadata[1:]:
        if metadata["Laser"] == first_channel:
            break
        else:
            num_channels += 1

    num_tiles = len(tile_metadata) // num_channels
    tile_xy_locations = []

    for i in range(0, len(tile_metadata), num_channels):
        curr_tile_dict = tile_metadata[i]

        x = round(curr_tile_dict["x_pos"] / curr_tile_dict["Pixelsize in um"])
        y = round(curr_tile_dict["y_pos"] / curr_tile_dict["Pixelsize in um"])

        tile_xy_locations.append((x, y))

    relative_locations = [(0, 0)]

    for abs_tuple in tile_xy_locations[1:]:
        rel_tuple = (
            abs(abs_tuple[0] - tile_xy_locations[0][0]),
            abs(abs_tuple[1] - tile_xy_locations[0][1]),
        )
        relative_locations.append(rel_tuple)

    with open(output_file, "w") as f:
        f.write("dim=3\n")
        for i in range(len(tile_metadata)):
            f.write(
                f"{i};;"
                f"({relative_locations[i%num_tiles][0]},"
                f"{relative_locations[i%num_tiles][1]},0)\n"
            )

    return tile_metadata


def parse_mesospim_metadata(meta_file_name: Path):
    tile_metadata = []
    with open(meta_file_name, "r") as f:
        lines = f.readlines()
        curr_tile_metadata: dict[str, str | int | float] = {}

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

    generated_by = input_root.find(".//generatedBy")
    base_path = input_root.find(".//BasePath")

    root = ET.Element("SpimData", version="0.2")
    assert (
        generated_by is not None
    ), "No generatedBy tag found in the input XML file"
    assert base_path is not None, "No BasePath tag found in the input XML file"
    root.append(generated_by)
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
    hdf5_path_node.text = str(hdf5_path)
    sequence_desc.append(image_loader)

    view_setup = input_root.find(".//ViewSetup")
    assert (
        view_setup is not None
    ), "No ViewSetup tag found in the input XML file"
    view_setup[3].text = f"{image_size[2]} {image_size[1]} {image_size[0]}"

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
