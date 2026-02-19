from pathlib import Path
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from dask_image.ndinterp import affine_transform as da_affine_transform

from brainglobe_stitch.image_mosaic import (
    ImageMosaic,
    blend_along_axis,
)
from brainglobe_stitch.tile import Tile


def register_shutters(
    image_mosaic_left: ImageMosaic,
    image_mosaic_right: ImageMosaic,
    pyramid_level: int,
    resolution: npt.NDArray,
) -> Dict[int, Tuple[Tile, Tile, sitk.Transform]]:
    """
    Register the images acquired from the left and right shutters from a
    light-sheet microscope when each side is contained in a separate ImageMosaic
    object.

    Parameters
    ----------
    image_mosaic_left : ImageMosaic
        The image mosaic object containing the left shutter tiles.
    image_mosaic_right : ImageMosaic
        The image mosaic object containing the right shutter tiles.
    pyramid_level : int
        The pyramid level to use for registration.
    resolution : npt.NDArray
        The resolution of the images in the format [z, y, x].

    Returns
    -------
    Dict[int, Tuple[Tile, Tile, sitk.Transform]]
        A dictionary containing the channel ID as the key and a tuple
        containing the left and right tiles and the transform as the value.
    """
    l_r_pairs: Dict[int, List[Tile]] = {}
    for tile in image_mosaic_left.tiles + image_mosaic_right.tiles:
        l_r_pair = l_r_pairs.get(tile.channel_id, [])
        l_r_pair.append(tile)
        l_r_pairs[tile.channel_id] = l_r_pair

    out_dict: Dict[int, Tuple[Tile, Tile, sitk.Transform]] = {}
    for channel_id in l_r_pairs.keys():
        # Assumes same channel_id in both which should be the case
        channel_name = image_mosaic_left.channel_names[channel_id]
        curr_tiles = l_r_pairs[channel_id]
        print(
            f"Registering downsampled left and right images "
            f"for channel {channel_name}"
        )
        right_tile_ind = 0 if curr_tiles[0].illumination_name == "Right" else 1
        left_tile_ind = 1 - right_tile_ind

        right_tile = curr_tiles[right_tile_ind]
        left_tile = curr_tiles[left_tile_ind]

        fixed = sitk.GetImageFromArray(
            right_tile.data_pyramid[pyramid_level].compute().astype(np.float32)
        )
        moving = sitk.GetImageFromArray(
            left_tile.data_pyramid[pyramid_level].compute().astype(np.float32)
        )

        fixed.SetSpacing(tuple(resolution[::-1]))
        moving.SetSpacing(tuple(resolution[::-1]))

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=24
        )
        registration_method.SetOptimizerAsRegularStepGradientDescent(
            1.0, 0.001, 200
        )
        registration_method.SetInitialTransform(
            sitk.TranslationTransform(fixed.GetDimension())
        )
        registration_method.SetInterpolator(sitk.sitkLinear)

        out_transform = registration_method.Execute(fixed, moving)

        print("-------")
        print(out_transform)
        print(
            f"Optimizer stop condition: "
            f"{registration_method.GetOptimizerStopConditionDescription()}"
        )
        print(f" Iteration: {registration_method.GetOptimizerIteration()}")
        print(f" Metric value: {registration_method.GetMetricValue()}")
        print("-------")

        out_dict[channel_id] = (left_tile, right_tile, out_transform)

    return out_dict


def l_r_fuse(
    left_path: Path,
    right_path: Path,
    pyramid_level: int = 2,
    order: int = 0,
    output_path: Path | None = None,
) -> ImageMosaic:
    """
    Fuse the images acquired from the left and right shutters of a light-sheet
    microscope when each side is contained in a separate hdf5 file.
    object.

    Parameters
    ----------
    left_path : Path
        The path to the stack illuminated by the left sheet.
    right_path : Path
        The path to the stack illuminated by the right sheet.
    pyramid_level : int, optional
        The pyramid level to use for registration, by default 2.
    order: int, optional
        The order of the interpolation to use when transforming the
        left and right images following registration, by default 0.
    save_result: bool, optional
        Whether to save the fused image to disk, by default False.
    output_path: Path | None, optional
        The path to save the fused image, None if data should not be saved, by
        default None.

    Returns
    -------
    ImageMosaic
        The image mosaic object containing the fused image.
    """

    image_mosaic_left = ImageMosaic(left_path)
    image_mosaic_right = ImageMosaic(right_path)

    resolution = (
        np.array(
            [
                image_mosaic_left.z_resolution,
                image_mosaic_left.x_y_resolution,
                image_mosaic_left.x_y_resolution,
            ]
        )
        * image_mosaic_left.tiles[0].resolution_pyramid[pyramid_level]
    )

    transforms = register_shutters(
        image_mosaic_left, image_mosaic_right, pyramid_level, resolution
    )

    for channel_id, (left_tile, right_tile, transform) in transforms.items():
        print(
            f"Applying transform for channel "
            f"{image_mosaic_left.channel_names[channel_id]}"
        )
        # SimpleITK returns the transform as x, y, z. in physical space
        # Transform back to z, y, x and scale to pixel coordinates
        descaled_offset = transform.GetParameters()[::-1] / resolution

        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = descaled_offset
        left_tile.data_pyramid[0] = da_affine_transform(
            left_tile.data_pyramid[0],
            transform_matrix,
            order,
            output_chunks=left_tile.data_pyramid[0].chunksize,
        )

        left_tile.data_pyramid[0] = blend_along_axis(
            right_tile.data_pyramid[0], left_tile.data_pyramid[0], axis=2
        )

        if output_path is not None:
            da.to_hdf5(output_path, "/lr_fused", left_tile.data_pyramid[0])

        image_mosaic_right.tiles.remove(right_tile)
        image_mosaic_right.tile_names.remove(right_tile.name)

    return image_mosaic_left
