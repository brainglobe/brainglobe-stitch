from pathlib import Path
from typing import Dict, List, Tuple

import dask.array as da
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from dask_image.ndinterp import affine_transform as da_affine_transform

from brainglobe_stitch.image_mosaic import ImageMosaic
from brainglobe_stitch.tile import Tile


def register_shutters(
    image_mosaic: ImageMosaic,
    pyramid_level: int,
    resolution: npt.NDArray,
) -> Dict[int, Tuple[Tile, Tile, sitk.Transform]]:
    """
    Register the images acquired from the left and right shutters from a
    light-sheet microscope.

    Parameters
    ----------
    image_mosaic : ImageMosaic
        The image mosaic object containing the tiles to register.
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
    for tile in image_mosaic.tiles:
        l_r_pair = l_r_pairs.get(tile.channel_id, [])
        l_r_pair.append(tile)
        l_r_pairs[tile.channel_id] = l_r_pair

    out_dict: Dict[int, Tuple[Tile, Tile, sitk.Transform]] = {}
    for channel_id in l_r_pairs.keys():
        channel_name = image_mosaic.channel_names[channel_id]
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


def blend_along_axis(
    im1,
    im2,
    axis=0,
    blending_center_coord=None,
    pixel_width=None,
    weight_at_pixel_width=0.95,
    weight_threshold=1e-3,
) -> da.Array:
    """Sigmoidal blending of two arrays of equal shape along an axis

    Parameters
    ----------
    im1 : ndarray
        input array with values to keep at starting coordinates
    im2 : ndarray
        input array with values to keep at ending coordinates
    axis : int
        axis along which to blend the two input arrays
    blending_center_coord : float, optional
        coordinate representing the blending center.
        If None, the center along the axis is used.
    pixel_width : float, optional
        width of the blending function in pixels.
        If None, 5% of the array's extent along the axis is used.
    weight_at_pixel_width : float, optional
        weight value at distance `pixel_width` to `blending_center_coord`.
    weight_threshold : float, optional
        below this weight threshold the resulting array is just a copy of
        the input array with the highest weight. This is faster and more
        memory efficient.

    Returns
    -------
    blended_array : da.Array
        blended result of same shape as input arrays

    Notes
    -----
    This function could require less memory by blending plane by plane
    along the axis.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> im = np.random.randint(0, 1000, [7, 9, 9])
    >>> im = ndimage.zoom(im, [21]*3, order=1)
    >>> degrad = np.exp(-0.01 * np.arange(len(im)))
    >>> im_l = (im.T * degrad).T.astype(np.uint16)
    >>> im_r = (im.T * degrad[::-1]).T.astype(np.uint16)
    >>> blend_along_axis(im_l, im_r, axis=0, pixel_width=5)

    Author - Marvin Albert
    Code taken from: https://gist.github.com/m-albert/bb4fa38436760c4e6d171239fb3e16c4
    """

    # use center coordinate along blending axis
    if blending_center_coord is None:
        blending_center_coord = (im1.shape[axis] - 1) / 2

    # use 10% of extent along blending axis
    if pixel_width is None:
        pixel_width = im1.shape[axis] / 20

    shape = im1.shape

    # define sigmoidal blending function
    a = (
        -np.log((1 - weight_at_pixel_width) / weight_at_pixel_width)
        / pixel_width
    )
    sigmoid = 1 / (
        1 + np.exp(-a * (np.arange(shape[axis]) - blending_center_coord))
    )
    sigmoid = sigmoid.astype(np.float32)

    # swap array axes such that blending axis is last one
    im1 = da.swapaxes(im1, -1, axis)
    im2 = da.swapaxes(im2, -1, axis)

    # initialise output array
    out = da.zeros_like(im1, chunks=(1, 2048, 2048))

    # define sub threshold regions
    mask1 = sigmoid < weight_threshold
    mask2 = sigmoid > (1 - weight_threshold)
    maskb = ~(mask1 ^ mask2)
    # copy input arrays in sub threshold regions
    out[..., mask1] = im1[..., mask1]
    out[..., mask2] = im2[..., mask2]

    # blend
    out[..., maskb] = (1 - sigmoid[maskb]) * im1[..., maskb] + sigmoid[
        maskb
    ] * im2[..., maskb]

    # rearrange array
    out = da.swapaxes(out, -1, axis)

    return out


def l_r_fuse(
    image_mosaic: ImageMosaic,
    pyramid_level: int = 2,
    order: int = 0,
) -> ImageMosaic:
    """
    Fuse the images acquired from the left and right shutters of a light-sheet
    microscope.

    Parameters
    ----------
    image_mosaic : ImageMosaic
        The image mosaic object containing the tiles to fuse.
    pyramid_level : int, optional
        The pyramid level to use for registration, by default 2.
    order: int, optional
        The order of the interpolation to use when transforming the
        left and right images following registration, by default 0.

    Returns
    -------
    ImageMosaic
        The image mosaic object containing the fused image.
    """
    resolution = (
        np.array(
            [
                image_mosaic.z_resolution,
                image_mosaic.x_y_resolution,
                image_mosaic.x_y_resolution,
            ]
        )
        * image_mosaic.tiles[0].resolution_pyramid[pyramid_level]
    )

    transforms = register_shutters(image_mosaic, pyramid_level, resolution)

    for channel_id, (left_tile, right_tile, transform) in transforms.items():
        print(
            f"Applying transform for channel "
            f"{image_mosaic.channel_names[channel_id]}"
        )
        # SimpleITK returns the transform as x, y, z. in physical space
        # Transform back to z, y, x and scale to pixel coordinates
        descaled_offset = transform.GetParameters()[::-1] / resolution

        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = descaled_offset
        left_tile.data_pyramid[0] = da_affine_transform(
            left_tile.data_pyramid[0], transform_matrix, order
        )

        left_tile.data_pyramid[0] = blend_along_axis(
            right_tile.data_pyramid[0], left_tile.data_pyramid[0], axis=2
        )

        image_mosaic.tiles.remove(right_tile)
        image_mosaic.tile_names.remove(right_tile.name)

    return image_mosaic


if __name__ == "__main__":
    fused_file_name = "fused.zarr"
    working_dir = Path("/mnt/Data/Phillip/")
    output_path = working_dir / fused_file_name
    image_mosaic_in = ImageMosaic(working_dir)

    fused_mosaic = l_r_fuse(image_mosaic_in)

    fused_mosaic.fuse(output_path)
