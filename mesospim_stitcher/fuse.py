import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import rmtree

import dask.array as da
import h5py
import numpy as np
import zarr
from mpi4py import MPI
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
    tiles = [da.from_array(group[f"{child}/0/cells"]) for child in group]

    z_size = tiles[0].shape[0]
    x_y_size = tiles[0].shape[1]
    max_delta = [max([abs(delta[i]) for delta in deltas]) for i in range(3)]

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

    new_image = da.zeros(
        (
            max(translation[5] for translation in translations),
            max(translation[3] for translation in translations),
            max(translation[1] for translation in translations),
        ),
        dtype="uint16",
    )

    for i in range(len(tiles) - 1, -1, -1):
        curr_tile = tiles[i]
        x_s, x_e, y_s, y_e, z_s, z_e = translations[i]
        new_image[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile

    try:
        # write_ome_zarr(output_path, new_image, overwrite)
        write_hdf5(output_path, new_image, overwrite)
    except Exception as e:
        raise e
    finally:
        input_file.close()


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
            chunks=(2, image.shape[1], image.shape[2]),
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

    rank = MPI.COMM_WORLD.rank
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

    orig_image = data_group.create_dataset(
        "0/cells",
        data=image,
        chunks=(16, 32, 32),
        dtype="uint16",
        shape=image.shape,
    )

    image_shape = image.shape
    x_split_size = image_shape[2] // 4
    y_split_size = image_shape[1] // 3

    x_borders = [
        0,
        x_split_size,
        x_split_size * 2,
        x_split_size * 3,
        image_shape[2],
    ]
    y_borders = [0, y_split_size, y_split_size * 2, image_shape[1]]

    x_start = x_borders[rank % 4]
    x_end = x_borders[rank % 4 + 1]
    y_start = y_borders[rank // 4]
    y_end = y_borders[(rank // 4 + 1)]

    orig_image[:, y_start:y_end, x_start:x_end] = image[
        :, y_start:y_end, x_start:x_end
    ]

    for i in range(1, resolutions.shape[0]):
        prev_resolution = data_group[f"{i-1}/cells"][
            :, y_start:y_end, x_start:x_end
        ]
        data_group.require_dataset(
            f"{i}/cells",
            data=prev_resolution[::2, ::2, ::2],
            chunks=(16, 32, 32),
            compression="gzip",
            dtype="uint16",
            shape=prev_resolution[::2, ::2, ::2].shape,
        )

    f.close()

    # print(f"Rank {rank} finished writing")


if __name__ == "__main__":
    xml_path = Path(
        "/mnt/Data/TiledDataset/2.5x_tile/"
        "One_Channel/2.5x_tile_igor_rightonly_Mag2.5x_"
        "ch488_ch561_ch647_bdv.xml"
    )
    input_path = Path("/mnt/Data/TiledDataset/2.5x_tile/One_Channel/test.h5")
    output_path = Path(
        "/mnt/Data/TiledDataset/2.5x_tile/One_Channel/test_out.h5"
    )

    fuse_image(xml_path, input_path, output_path)
