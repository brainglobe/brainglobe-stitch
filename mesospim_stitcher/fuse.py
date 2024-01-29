import os
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Tuple

import dask.array as da
import h5py
import numpy as np
import zarr
from numcodecs import Blosc, blosc
from ome_zarr.dask_utils import downscale_nearest
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_multiscales_metadata

from mesospim_stitcher.file_utils import get_slice_attributes, write_bdv_xml
from mesospim_stitcher.tile import Tile


def fuse_image(
    xml_path: Path,
    input_path: Path,
    output_path: Path,
    tile_metadata: List[Dict],
    intensity_scale_factors: List[float],
    num_channels: int = 1,
    yield_progress: bool = False,
):
    input_file = h5py.File(input_path, "r")
    group = input_file["t00000"]
    tile_names = list(group.keys())

    tile = group[f"{tile_names[0]}/0/cells"]
    z_size, y_size, x_size = tile.shape

    tile_positions = get_big_stitcher_transforms(
        xml_path, x_size, y_size, z_size
    )

    slice_attributes = get_slice_attributes(xml_path, tile_names)

    fused_image_shape = (
        max(tile_position[5] for tile_position in tile_positions),
        max(tile_position[3] for tile_position in tile_positions),
        max(tile_position[1] for tile_position in tile_positions),
    )

    num_tiles = len(tile_names)

    if output_path.suffix == ".zarr":
        fuse_to_zarr(
            fused_image_shape,
            group,
            output_path,
            tile_names,
            num_tiles,
            tile_positions,
            tile_metadata,
            slice_attributes,
            intensity_scale_factors,
            num_channels,
            yield_progress,
        )
    elif output_path.suffix == ".h5":
        fuse_to_bdv_h5(
            fused_image_shape,
            group,
            output_path,
            tile_names,
            num_tiles,
            tile_positions,
            slice_attributes,
            xml_path,
            num_channels,
        )

    input_file.close()

    print("Done")


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
    fused_image_shape: Tuple[int, ...],
    group: h5py.Group,
    output_path: Path,
    tile_names: List[str],
    num_tiles: int,
    translations: List[List[int]],
    tile_metadata: List[Dict],
    slice_attributes: Dict[str, Dict[str, str]],
    intensity_scale_factors: List[float],
    num_channels: int,
    yield_progress: bool,
):
    print("Fusing to zarr")

    chunk_shape: Tuple[int, ...] = (64, 128, 128)
    tiles = []

    for child in group:
        curr_tile = group[f"{child}/0/cells"]
        tiles.append(da.from_array(curr_tile, chunks=chunk_shape))

    x_y_resolution = tile_metadata[0]["Pixelsize in um"]
    z_resolution = tile_metadata[0]["z_stepsize"]
    coordinate_transformations = [
        [
            {
                "type": "scale",
                "scale": [z_resolution, x_y_resolution, x_y_resolution],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 2,
                    x_y_resolution * 2,
                ],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 4,
                    x_y_resolution * 4,
                ],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 8,
                    x_y_resolution * 8,
                ],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 16,
                    x_y_resolution * 16,
                ],
            }
        ],
    ]
    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    if num_channels > 1:
        fused_image_shape = (num_channels,) + fused_image_shape
        chunk_shape = (1,) + chunk_shape

        for transform in coordinate_transformations:
            transform[0]["scale"] = [1.0, *transform[0]["scale"]]

        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

    # overlaps = calculate_overlaps(translations)
    # adjust_contrast(intensity_scale_factors, tiles)
    # interpolate_overlaps(
    #     overlaps, tiles, slice_attributes, tile_names, translations
    # )

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
    blosc.set_nthreads(24)

    # total_work = num_tiles + len(coordinate_transformations)

    for i in range(num_tiles - 1, -1, -1):
        x_s, x_e, y_s, y_e, z_s, z_e = translations[i]
        curr_tile = tiles[i]
        channel_idx = int(slice_attributes[tile_names[i]]["channel"])
        # if intensity_scale_factors[i] != 1.0:
        #     curr_tile = da.multiply(
        #         curr_tile, intensity_scale_factors[i], dtype=np.float16
        #     ).astype(np.int16)

        if num_channels > 1:
            fused_image_store[
                channel_idx, z_s:z_e, y_s:y_e, x_s:x_e
            ] = curr_tile.compute()
        else:
            fused_image_store[z_s:z_e, y_s:y_e, x_s:x_e] = curr_tile.compute()

        print(f"Done tile {i}")

        # if yield_progress:
        #     yield int(100 * (num_tiles - i) / total_work)

    # new_image.store(fused_image_store)

    for i in range(1, len(coordinate_transformations)):
        prev_resolution = da.from_zarr(root[f"{i - 1}"])

        if num_channels > 1:
            downsampled_image = downscale_nearest(
                prev_resolution, (1, 1, 2, 2)
            )
            chunk_shape = (1, 32, 64, 64)
        else:
            downsampled_image = downscale_nearest(prev_resolution, (1, 2, 2))
            chunk_shape = (32, 64, 64)

        downsampled_shape = downsampled_image.shape
        downsampled_store = root.require_dataset(
            f"{i}",
            shape=downsampled_shape,
            chunks=chunk_shape,
            dtype="i2",
            compressor=compressor,
        )
        downsampled_image.to_zarr(downsampled_store)

        # if yield_progress:
        #     yield int(100 * (num_tiles + i) / total_work)

    datasets = []

    for i, transform in enumerate(coordinate_transformations):
        datasets.append(
            {"path": f"{i}", "coordinateTransformations": transform}
        )

    write_multiscales_metadata(
        group=root,
        datasets=datasets,
        axes=axes,
    )

    possible_channel_colors = [
        "00FF00",
        "FF0000",
        "0000FF",
        "FFFF00",
        "00FFFF",
        "FF00FF",
    ]
    channels = []
    for i in range(num_channels):
        channels.append(
            {
                "active": True,
                "color": possible_channel_colors[i],
                "name": f"ch{i+1}",
                "window": {"start": 0, "end": 4000, "min": 0, "max": 65535},
            }
        )

    root.attrs["omero"] = {"channels": channels}


def fuse_to_bdv_h5(
    fused_image_shape: Tuple[int, int, int],
    group: h5py.Group,
    output_path: Path,
    tile_names: List[str],
    num_tiles: int,
    translations: List[List[int]],
    slice_attributes: Dict[str, Dict[str, str]],
    xml_path: Path,
    num_channels: int = 1,
):
    output_file = h5py.File(output_path, mode="w")
    ds = output_file.require_dataset(
        "t00000/s00/0/cells",
        shape=fused_image_shape,
        dtype="i2",
    )

    subdivisions = np.array(
        [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]],
        dtype=np.int16,
    )
    resolutions = np.array(
        [[1, 1, 1], [2, 2, 1], [4, 4, 1], [8, 8, 1], [16, 16, 1]],
        dtype=np.int16,
    )

    for i in range(num_channels):
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

    write_bdv_xml(Path("testing.xml"), xml_path, output_path, ds[0].shape)
    output_file.close()


def write_ome_zarr(
    output_path: Path,
    image: da,
    tile_metadata: List[Dict],
    num_channels: int,
    overwrite: bool = False,
):
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

    x_y_resolution = tile_metadata[0]["Pixelsize in um"]
    z_resolution = tile_metadata[0]["z_stepsize"]
    coordinate_transformations = [
        [
            {
                "type": "scale",
                "scale": [z_resolution, x_y_resolution, x_y_resolution],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 2,
                    x_y_resolution * 2,
                ],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 4,
                    x_y_resolution * 4,
                ],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 8,
                    x_y_resolution * 8,
                ],
            }
        ],
        [
            {
                "type": "scale",
                "scale": [
                    z_resolution,
                    x_y_resolution * 16,
                    x_y_resolution * 16,
                ],
            }
        ],
    ]

    chunk_shape: Tuple[int, ...] = (2, 2048, 2048)
    axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    if len(image.shape) == 4:
        chunk_shape = (image.shape[0], 2, 2048, 2048)
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

    write_image(
        image=image,
        group=root,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=dict(
            chunks=chunk_shape,
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


def calculate_overlaps(translations):
    overlaps = {}

    for i in range(len(translations) - 1):
        curr_translation = translations[i]
        for j in range(i + 1, len(translations)):
            next_translation = translations[j]

            if (curr_translation[1] > next_translation[0]) and (
                curr_translation[3] > next_translation[2]
            ):
                x_overlap_s = max(curr_translation[0], next_translation[0])
                x_overlap_e = min(curr_translation[1], next_translation[1])
                y_overlap_s = max(curr_translation[2], next_translation[2])
                y_overlap_e = min(curr_translation[3], next_translation[3])
                z_overlap_s = max(curr_translation[4], next_translation[4])
                z_overlap_e = min(curr_translation[5], next_translation[5])

                overlaps[(i, j)] = (
                    x_overlap_s,
                    x_overlap_e,
                    y_overlap_s,
                    y_overlap_e,
                    z_overlap_s,
                    z_overlap_e,
                )

    return overlaps


def calculate_scale_factors(
    overlaps: Dict[Tuple[int, int], Tuple[int, ...]],
    tile_objects: List[Tile],
    percentile,
):
    num_tiles = len(tile_objects)
    scale_factors = np.ones((num_tiles, num_tiles))

    for curr_tile in tile_objects[:-1]:
        for next_tile in tile_objects[curr_tile.id + 1 :]:
            if ((curr_tile.id, next_tile.id) in overlaps) and (
                curr_tile.channel_id == next_tile.channel_id
            ):
                i_overlap, j_overlap, _, _ = extract_overlap_data(
                    overlaps[(curr_tile.id, next_tile.id)],
                    curr_tile,
                    next_tile,
                    full_res=False,
                )

                median_i = np.percentile(i_overlap.ravel(), percentile)
                median_j = np.percentile(j_overlap.ravel(), percentile)

                curr_scale_factor = (median_i / median_j).compute()
                scale_factors[curr_tile.id][next_tile.id] = curr_scale_factor[
                    0
                ]

                del i_overlap
                del j_overlap
                del median_i
                del median_j

                next_tile.downsampled_data = np.multiply(
                    next_tile.downsampled_data,
                    curr_scale_factor,
                    dtype=np.float16,
                ).astype(np.int16)

    return scale_factors, tile_objects


def adjust_contrast(intensity_scale_factors, images):
    num_tiles = len(images)

    for i in range(num_tiles):
        if intensity_scale_factors[i] != 1.0:
            images[i] = da.multiply(
                images[i], intensity_scale_factors[i], dtype=np.float16
            ).astype(np.int16)


def interpolate_overlaps(
    overlaps, tile_objects: List[Tile], full_res: bool = False
):
    assert tile_objects[0].data is not None, "No data found in tile objects"
    if full_res:
        z_size, y_size, x_size = tile_objects[0].data.shape
    else:
        z_size, y_size, x_size = tile_objects[0].downsampled_data.shape

    for curr_tile in tile_objects[:-1]:
        for next_tile in tile_objects[curr_tile.id + 1 :]:
            if ((curr_tile.id, next_tile.id) in overlaps) and (
                curr_tile.channel_id == next_tile.channel_id
            ):
                (
                    i_overlap,
                    j_overlap,
                    i_indices,
                    j_indices,
                ) = extract_overlap_data(
                    overlaps[(curr_tile.id, next_tile.id)],
                    curr_tile,
                    next_tile,
                    full_res=full_res,
                )

                x_overlap_size = i_overlap.shape[2]
                y_overlap_size = i_overlap.shape[1]

                if (
                    x_overlap_size / x_size < 0.2
                    and y_overlap_size / y_size < 0.2
                ):
                    # Skip the small diagonal overlaps
                    continue

                elif x_overlap_size / x_size < 0.2:
                    x_lin = np.linspace(1, 0, x_overlap_size)

                    # 1 in the first column,
                    # linearly decreasing to 0 in the last column
                    yx_grid = np.tile(x_lin, (y_overlap_size, 1))

                    if (
                        curr_tile.stitched_position[2]
                        < next_tile.stitched_position[2]
                    ):
                        decreasing_image = i_overlap
                        increasing_image = j_overlap
                    else:
                        decreasing_image = j_overlap
                        increasing_image = i_overlap
                else:
                    y_lin = np.linspace(1, 0, y_overlap_size)

                    # 1 in the first row,
                    # linearly decreasing to 0 in the last row
                    yx_grid = np.tile(y_lin, (x_overlap_size, 1)).T

                    if (
                        curr_tile.stitched_position[1]
                        < next_tile.stitched_position[1]
                    ):
                        decreasing_image = i_overlap
                        increasing_image = j_overlap
                    else:
                        decreasing_image = j_overlap
                        increasing_image = i_overlap

                interp = (
                    np.multiply(
                        decreasing_image.compute(),
                        yx_grid,
                        dtype=np.float16,
                    )
                    + np.multiply(
                        increasing_image.compute(),
                        1 - yx_grid,
                        dtype=np.float16,
                    )
                ).astype(np.int16)

                i_z_s, i_z_e, i_y_s, i_y_e, i_x_s, i_x_e = i_indices
                j_z_s, j_z_e, j_y_s, j_y_e, j_x_s, j_x_e = j_indices

                if full_res:
                    curr_tile.data[
                        i_z_s:i_z_e, i_y_s:i_y_e, i_x_s:i_x_e
                    ] = interp
                    next_tile.data[
                        j_z_s:j_z_e, j_y_s:j_y_e, j_x_s:j_x_e
                    ] = interp
                else:
                    curr_tile.downsampled_data[
                        i_z_s:i_z_e, i_y_s:i_y_e, i_x_s:i_x_e
                    ] = interp
                    next_tile.downsampled_data[
                        j_z_s:j_z_e, j_y_s:j_y_e, j_x_s:j_x_e
                    ] = interp

                print(
                    f"Done interpolating tile {curr_tile.id}"
                    f" and {next_tile.id}"
                )


def calculate_image_stats(image: da) -> Tuple[float, float, float, float]:
    num_pixels = image.shape[0] * image.shape[1] * image.shape[2]
    raveled_image = image.ravel()
    top_min = raveled_image.topk(-int(num_pixels * 0.00001)).compute()
    top_max = raveled_image.topk(int(num_pixels * 0.00001)).compute()
    true_min = top_min[0]
    clean_min = top_min[-1]
    true_max = top_max[0]
    clean_max = top_max[-1]

    return clean_min, clean_max, true_min, true_max


def extract_overlap_data(
    overlap: Tuple[int, ...],
    tile_i: Tile,
    tile_j: Tile,
    full_res: bool = False,
) -> Tuple[da.Array, da.Array, Tuple[int, ...], Tuple[int, ...]]:
    (
        x_overlap_s,
        x_overlap_e,
        y_overlap_s,
        y_overlap_e,
        z_overlap_s,
        z_overlap_e,
    ) = overlap
    x_overlap = x_overlap_e - x_overlap_s
    y_overlap = y_overlap_e - y_overlap_s
    z_overlap = z_overlap_e - z_overlap_s

    if full_res:
        data_i = tile_i.data
        data_j = tile_j.data
    else:
        data_i = tile_i.downsampled_data
        data_j = tile_j.downsampled_data

    z_size, y_size, x_size = data_i.shape

    if tile_i.stitched_position[2] < tile_j.stitched_position[2]:
        i_x_s = x_size - x_overlap
        i_x_e = x_size
        j_x_s = 0
        j_x_e = x_overlap
    else:
        i_x_s = 0
        i_x_e = x_overlap
        j_x_s = x_size - x_overlap
        j_x_e = x_size

    if tile_i.stitched_position[1] < tile_j.stitched_position[1]:
        i_y_s = y_size - y_overlap
        i_y_e = y_size
        j_y_s = 0
        j_y_e = y_overlap
    else:
        i_y_s = 0
        i_y_e = y_overlap
        j_y_s = y_size - y_overlap
        j_y_e = y_size

    if tile_i.stitched_position[0] < tile_j.stitched_position[0]:
        i_z_s = z_size - z_overlap
        i_z_e = z_size
        j_z_s = 0
        j_z_e = z_overlap
    else:
        i_z_s = 0
        i_z_e = z_overlap
        j_z_s = z_size - z_overlap
        j_z_e = z_size

    i_overlap = data_i[i_z_s:i_z_e, i_y_s:i_y_e, i_x_s:i_x_e]
    j_overlap = data_j[j_z_s:j_z_e, j_y_s:j_y_e, j_x_s:j_x_e]

    i_indices = (i_z_s, i_z_e, i_y_s, i_y_e, i_x_s, i_x_e)
    j_indices = (j_z_s, j_z_e, j_y_s, j_y_e, j_x_s, j_x_e)

    return i_overlap, j_overlap, i_indices, j_indices
