from pathlib import Path

import napari

from mesospim_stitcher.image_mosaic import ImageMosaic


def load(directory: Path) -> ImageMosaic:
    return ImageMosaic(directory)


def stitch(
    graph: ImageMosaic,
    imagej_path: Path,
    resolution_level: int = 2,
    selected_channel: str = "",
) -> None:
    graph.stitch(imagej_path, resolution_level, selected_channel)


if __name__ == "__main__":
    data_directory = Path("C:/Users/Igor/Documents/NIU-dev/stitching/Brain2")
    data_graph = load(data_directory)

    stitch(
        data_graph,
        Path("C:/Users/Igor/Documents/Fiji.app/ImageJ-win64.exe"),
        selected_channel="561 nm",
    )

    data_graph.normalise_intensity(80, 2)

    tiles = data_graph.data_for_napari(2)

    viewer = napari.Viewer()

    for tile in tiles:
        image = viewer.add_image(tile[0])
        image.translate = tile[1]

    napari.run()
