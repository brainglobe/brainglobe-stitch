import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import rmtree

import dask.array as da
import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image


def fuse_image(
    xml_path: Path, input_path: Path, output_path: Path, overwrite: bool = True
):
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

    input_file = h5py.File(input_path, "r")
    group = input_file["t00000"]
    tile_names = list(group.keys())
    max_delta = [max([abs(delta[i]) for delta in deltas]) for i in range(3)]

    tile = da.from_array(group[f"{tile_names[0]}/0/cells"])
    z_size = tile.shape[0]
    x_y_size = tile.shape[1]

    translations = []

    for i in range(len(deltas)):
        curr_delta = deltas[i]
        curr_grid = grids[i]

        x_start = curr_grid[0] + curr_delta[0] + max_delta[0]
        x_end = x_start + x_y_size
        y_start = curr_grid[1] + curr_delta[1] + max_delta[1]
        y_end = y_start + x_y_size
        z_start = curr_grid[2] + curr_delta[2] + max_delta[2]
        z_end = z_start + z_size

        translations.append([x_start, x_end, y_start, y_end, z_start, z_end])

    # new_image = da.zeros(
    #     (
    #         max(translation[5] for translation in translations),
    #         max(translation[3] for translation in translations),
    #         max(translation[1] for translation in translations),
    #     ),
    #     dtype="int16",
    # )

    fused_image_shape = (
        max(translation[5] for translation in translations),
        max(translation[3] for translation in translations),
        max(translation[1] for translation in translations),
    )

    num_tiles = len(tile_names)

    output_file = h5py.File(output_path, mode="w", compression="lzf")
    ds = output_file.require_dataset(
        "t00000/s00/0/cells", shape=fused_image_shape, dtype="i2"
    )

    square_root_cpu = 4

    x_y_split_size = x_y_size // square_root_cpu

    x_y_borders = [0]

    for j in range(1, square_root_cpu):
        x_y_borders.append(x_y_borders[j - 1] + x_y_split_size)

    x_y_borders.append(x_y_size)

    for i in range(num_tiles - 1, -1, -1):
        # for rank in range(square_root_cpu ** 2):
        #     x_tile_s = x_y_borders[rank % square_root_cpu]
        #     x_tile_e = x_y_borders[rank % square_root_cpu + 1]
        #     y_tile_s = x_y_borders[rank // square_root_cpu]
        #     y_tile_e = x_y_borders[(rank // square_root_cpu + 1)]
        #     curr_tile = group[f"{tile_names[i]}/0/cells"]
        #     [:, y_tile_s:y_tile_e, x_tile_s:x_tile_e]
        #
        #     x_s, x_e, y_s, y_e, z_s, z_e = translations[i]
        #     x_e = x_s + x_tile_e
        #     x_s = x_s + x_tile_s
        #     y_e = y_s + y_tile_e
        #     y_s = y_s + y_tile_s
        #
        #     with ds.collective:
        #         ds[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile
        #
        #     print(f"Done tile {tile_names[i]} part {rank}")

        x_s, x_e, y_s, y_e, z_s, z_e = translations[i]

        curr_tile = group[f"{tile_names[i]}/0/cells"]

        # new_image[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile
        ds[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile

    output_file.close()
    input_file.close()

    write_bdv_xml(Path("testing.xml"), xml_path, output_path, ds.shape)

    # try:
    #     write_ome_zarr(output_path, new_image, overwrite)
    #     # write_hdf5(output_path, new_image, overwrite)
    # except Exception as e:
    #     raise e
    # finally:
    #     input_file.close()


def write_ome_zarr(output_path: Path, image: da, overwrite: bool):
    if output_path.exists() and overwrite:
        rmtree(output_path)
    else:
        try:
            os.makedirs(output_path)
        except OSError:
            raise OSError(f"Output path {output_path} already exists")

    store = parse_url(output_path, mode="w").store
    root = zarr.group(store=store)
    compressor = Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE)
    write_image(
        image=image,
        group=root,
        axes=[
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        coordinate_transformations=[
            [{"type": "scale", "scale": [5.0, 2.6, 2.6]}],
            [{"type": "scale", "scale": [5.0, 5.2, 5.2]}],
            [{"type": "scale", "scale": [5.0, 10.4, 10.4]}],
            [{"type": "scale", "scale": [10.0, 20.8, 20.8]}],
            [{"type": "scale", "scale": [10.0, 41.6, 41.6]}],
        ],
        storage_options=dict(
            chunks=(4, image.shape[1], image.shape[2]),
            compressor=compressor,
        ),
    )

    root.attrs["omero"] = {
        "channels": [
            {
                "window": {"start": 0, "end": 1200, "min": 0, "max": 65535},
                "label": "stitched",
                "active": True,
            }
        ]
    }


def write_hdf5(output_path: Path, image: da, overwrite: bool):
    subdivisions = np.array(
        [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
    )
    resolutions = np.array(
        [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
    )

    # rank = MPI.COMM_WORLD.rank
    #
    # print(f"Rank {rank} starting to write")
    f = h5py.File(output_path, "w")

    f.create_dataset(
        "s00/subdivisions",
        data=subdivisions,
        shape=subdivisions.shape,
        dtype="uint16",
    )
    f.create_dataset(
        "s00/resolutions",
        data=resolutions,
        shape=resolutions.shape,
        dtype="uint16",
    )

    data_group: h5py.Group = f.create_group("t00000/s00")

    data_group.create_dataset(
        "0/cells",
        data=image,
        chunks=(16, 32, 32),
        dtype="uint16",
        shape=image.shape,
    )

    # image_shape = image.shape
    # x_split_size = image_shape[2] // 4
    # y_split_size = image_shape[1] // 3
    #
    # x_borders = [
    #     0,
    #     x_split_size,
    #     x_split_size * 2,
    #     x_split_size * 3,
    #     image_shape[2],
    # ]
    # y_borders = [0, y_split_size, y_split_size * 2, image_shape[1]]
    #
    # x_start = x_borders[rank % 4]
    # x_end = x_borders[rank % 4 + 1]
    # y_start = y_borders[rank // 4]
    # y_end = y_borders[(rank // 4 + 1)]
    #
    # orig_image[:, y_start:y_end, x_start:x_end] = image[
    #     :, y_start:y_end, x_start:x_end
    # ]
    #
    # for i in range(1, resolutions.shape[0]):
    #     prev_resolution = data_group[f"{i-1}/cells"][
    #         :, y_start:y_end, x_start:x_end
    #     ]
    #     data_group.require_dataset(
    #         f"{i}/cells",
    #         data=prev_resolution[::2, ::2, ::2],
    #         chunks=(16, 32, 32),
    #         compression="gzip",
    #         dtype="uint16",
    #         shape=prev_resolution[::2, ::2, ::2].shape,
    #     )

    f.close()


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

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)

    return


if __name__ == "__main__":
    xml_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/"
        "One_Channel/2.5x_tile_igor_rightonly_Mag2.5x_"
        "ch488_ch561_ch647_bdv.xml"
    )
    input_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/One_Channel/test.h5"
    )
    output_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/One_Channel/test_out.zarr"
    )

    fuse_image(xml_path, input_path, output_path)
