import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import rmtree

import dask.array as da
import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from ome_zarr.dask_utils import downscale_nearest
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_multiscales_metadata

from mesospim_stitcher.file_utils import write_bdv_xml


def fuse_image(xml_path: Path, input_path: Path, output_path: Path):
    input_file = h5py.File(input_path, "r")
    group = input_file["t00000"]
    tile_names = list(group.keys())

    tile = group[f"{tile_names[0]}/0/cells"]
    z_size, y_size, x_size = tile.shape

    tile_positions = get_big_stitcher_transforms(
        xml_path, x_size, y_size, z_size
    )

    fused_image_shape = (
        max(tile_position[5] for tile_position in tile_positions),
        max(tile_position[3] for tile_position in tile_positions),
        max(tile_position[1] for tile_position in tile_positions),
    )

    if output_path.suffix == ".zarr":
        fuse_to_zarr(
            fused_image_shape, group, output_path, tile_names, tile_positions
        )
    elif output_path.suffix == ".h5":
        fuse_to_bdv_h5(
            fused_image_shape,
            group,
            output_path,
            tile_names,
            tile_positions,
            xml_path,
        )

    input_file.close()


def get_big_stitcher_transforms(xml_path, x_size, y_size, z_size):
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


def fuse_to_zarr(
    fused_image_shape: tuple[int, int, int],
    group: h5py.Group,
    output_path: Path,
    tile_names: list,
    translations: list,
):
    num_tiles = len(tile_names)

    tiles = []

    for child in group:
        curr_tile = group[f"{child}/0/cells"]
        tiles.append(da.from_array(curr_tile))

    # fused_image = da.zeros(fused_image_shape, dtype="int16")
    store = zarr.NestedDirectoryStore(str(output_path))
    root = zarr.group(store=store)
    fused_image_store = root.create(
        "0",
        shape=fused_image_shape,
        chunks=(4, 2048, 2048),
        dtype="i2",
        compressor=Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE),
    )

    coordinate_transformations = [
        [{"type": "scale", "scale": [5.0, 2.6, 2.6]}],
        [{"type": "scale", "scale": [5.0, 5.2, 5.2]}],
        [{"type": "scale", "scale": [5.0, 10.4, 10.4]}],
        [{"type": "scale", "scale": [5.0, 20.8, 20.8]}],
        [{"type": "scale", "scale": [5.0, 41.6, 41.6]}],
    ]

    for i in range(num_tiles - 1, -1, -1):
        x_s, x_e, y_s, y_e, z_s, z_e = translations[i]
        curr_tile = tiles[i]
        fused_image_store[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile

        print("Done tile " + str(i))

    for i in range(1, len(coordinate_transformations)):
        prev_resolution = da.from_zarr(root[f"{i - 1}"])
        downsampled_image = downscale_nearest(prev_resolution, (1, 2, 2))
        downsampled_shape = downsampled_image.shape
        downsampled_store = root.require_dataset(
            f"{i}",
            shape=downsampled_shape,
            chunks=(4, (2048 // 2**i), (2048 // 2**i)),
            dtype="i2",
            compressor=Blosc(cname="zstd", clevel=4, shuffle=Blosc.SHUFFLE),
        )
        downsampled_image.to_zarr(downsampled_store)

    datasets = []

    for i, transform in enumerate(coordinate_transformations):
        datasets.append(
            {"path": f"{i}", "coordinateTransformations": transform}
        )

    write_multiscales_metadata(
        group=root,
        datasets=datasets,
        axes=[
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
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

    # write_ome_zarr(output_path, fused_image, overwrite=True)


def fuse_to_bdv_h5(
    fused_image_shape: tuple[int, int, int],
    group: h5py.Group,
    output_path: Path,
    tile_names: list,
    translations: list,
    xml_path: Path,
):
    num_tiles = len(tile_names)

    output_file = h5py.File(output_path, mode="w")
    ds = output_file.require_dataset(
        "t00000/s00/0/cells",
        shape=fused_image_shape,
        dtype="i2",
    )
    subdivisions = np.array(
        [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]],
        dtype=np.int16,
    )
    resolutions = np.array(
        [[1, 1, 1], [2, 2, 1], [4, 4, 1], [8, 8, 1]],
        dtype=np.int16,
    )
    output_file.require_dataset(
        "s00/resolutions",
        data=resolutions,
        dtype="i2",
        shape=resolutions.shape,
    )
    output_file.require_dataset(
        "s00/subdivisions",
        data=subdivisions,
        dtype="i2",
        shape=subdivisions.shape,
    )
    ds_list = [ds]
    for i in range(1, resolutions.shape[0]):
        ds_list.append(
            output_file.require_dataset(
                f"t00000/s00/{i}/cells",
                shape=(
                    fused_image_shape[0] // resolutions[i][2],
                    fused_image_shape[1] // resolutions[i][1],
                    fused_image_shape[2] // resolutions[i][0],
                ),
                dtype="i2",
            )
        )

    for i in range(num_tiles - 1, -1, -1):
        x_s, x_e, y_s, y_e, z_s, z_e = translations[i]
        curr_tile = group[f"{tile_names[i]}/0/cells"]
        ds[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile

        for j in range(1, resolutions.shape[0]):
            x_s_down = x_s // resolutions[j][0]
            x_e_down = x_e // resolutions[j][0]
            y_s_down = y_s // resolutions[j][1]
            y_e_down = y_e // resolutions[j][1]
            z_s_down = z_s // resolutions[j][2]
            z_e_down = z_e // resolutions[j][2]
            ds_list[j][
                z_s_down:z_e_down, y_s_down:y_e_down, x_s_down:x_e_down
            ] = curr_tile[
                :: resolutions[j][2],
                :: resolutions[j][1],
                :: resolutions[j][0],
            ]

    write_bdv_xml(Path("testing.xml"), xml_path, output_path, ds.shape)
    output_file.close()


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


if __name__ == "__main__":
    output_xml_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/" "One_Channel/test.xml"
    )

    xml_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/"
        "One_Channel/2.5x_tile_igor_rightonly_Mag2.5x_"
        "ch488_ch561_ch647_bdv.xml"
    )

    input_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/One_Channel/test.h5"
    )

    output_path = Path(
        "/home/igor/NIU-dev/stitching_dataset/One_Channel/test_out.h5"
    )

    fuse_image(xml_path, input_path, output_path)
