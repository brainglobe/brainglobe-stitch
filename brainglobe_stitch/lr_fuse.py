from pathlib import Path

import dask.array as da
import napari
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from brainglobe_stitch.image_mosaic import ImageMosaic
from brainglobe_stitch.tile import Tile


def register_shutters(
    fixed_tile: Tile,
    moving_tile: Tile,
    pyramid_level: int,
    resolution: npt.NDArray,
) -> sitk.Transform:
    print("Registering downsampled left and right images")

    fixed = sitk.GetImageFromArray(
        fixed_tile.data_pyramid[pyramid_level].compute().astype(np.float32)
    )
    moving = sitk.GetImageFromArray(
        moving_tile.data_pyramid[pyramid_level].compute().astype(np.float32)
    )

    fixed.SetSpacing(tuple(resolution[::-1]))
    moving.SetSpacing(tuple(resolution[::-1]))

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=24
    )
    # registration_method.SetMetricSamplingPercentage(0.25, sitk.sitkWallClock)
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
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

    return out_transform


def blend_along_axis(
    im1,
    im2,
    axis=0,
    blending_center_coord=None,
    pixel_width=None,
    weight_at_pixel_width=0.95,
    weight_threshold=1e-3,
):
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
    blended_array : ndarray
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
    image_mosaic: ImageMosaic, pyramid_level: int = 2
) -> sitk.Transform:
    resolution = (
        np.array(
            [
                image_mosaic.z_resolution,
                image_mosaic.x_y_resolution,
                image_mosaic.x_y_resolution,
            ]
        )
        * image_mosaic.tiles[3].resolution_pyramid[pyramid_level]
    )

    transform = register_shutters(
        image_mosaic.tiles[1], image_mosaic.tiles[3], pyramid_level, resolution
    )

    return transform


if __name__ == "__main__":
    working_dir = Path("/mnt/Data/Phillip/")
    image_mosaic_in = ImageMosaic(working_dir)

    transform_out = l_r_fuse(image_mosaic_in)

    preview_level = 0
    blended = blend_along_axis(
        image_mosaic_in.tiles[1].data_pyramid[preview_level],
        image_mosaic_in.tiles[3].data_pyramid[preview_level],
        axis=2,
        weight_threshold=1e-4,
    )

    viewer = napari.Viewer()
    scale = (
        np.array(
            [
                image_mosaic_in.z_resolution,
                image_mosaic_in.x_y_resolution,
                image_mosaic_in.x_y_resolution,
            ]
        )
        * image_mosaic_in.tiles[3].resolution_pyramid[preview_level]
    )

    viewer.add_image(
        image_mosaic_in.tiles[3].data_pyramid[preview_level],
        contrast_limits=[0, 3000],
        scale=scale,
    )
    viewer.add_image(
        image_mosaic_in.tiles[1].data_pyramid[preview_level],
        contrast_limits=[0, 3000],
        scale=scale,
    )
    viewer.add_image(blended, contrast_limits=[0, 3000], scale=scale)
    viewer.show(block=True)
