from pathlib import Path

from mesospim_stitcher.image_mosaic import ImageMosaic


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


if __name__ == "__main__":
    data_directory = Path("C:/Users/Igor/Documents/NIU-dev/stitching/Brain2")
    data_graph = load(data_directory)

    resolution_level = 2

    stitch(
        data_graph,
        Path("C:/Users/Igor/Documents/Fiji.app/ImageJ-win64.exe"),
        selected_channel="561 nm",
    )

    normalise_intensity(data_graph, resolution_level, 50)

    data_graph.fuse("test.zarr", True)
