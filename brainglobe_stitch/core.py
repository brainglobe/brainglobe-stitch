from pathlib import Path

from brainglobe_stitch.image_mosaic import ImageMosaic


def load(directory: Path) -> ImageMosaic:
    return ImageMosaic(directory)


def stitch(
    image_mosaic: ImageMosaic,
    imagej_path: Path,
    resolution_level: int = 2,
    selected_channel: str = "",
) -> None:
    image_mosaic.stitch(imagej_path, resolution_level, selected_channel)


def normalise_intensity(
    image_mosaic: ImageMosaic, resolution_level: int = 2, percentile: int = 50
) -> None:
    image_mosaic.normalise_intensity(resolution_level, percentile)


def interpolate_overlaps(
    image_mosaic: ImageMosaic, resolution_level: int = 2
) -> None:
    image_mosaic.interpolate_overlaps(resolution_level)


def fuse(
    image_mosaic: ImageMosaic,
    output_file_name: str,
    normalise_intensity: bool = False,
    interpolate: bool = False,
) -> None:
    image_mosaic.fuse(output_file_name, normalise_intensity, interpolate)