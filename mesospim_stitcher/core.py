from pathlib import Path

from mesospim_stitcher.image_graph import ImageGraph


def load(directory: Path) -> ImageGraph:
    return ImageGraph(directory)


def stitch(
    graph: ImageGraph,
    imagej_path: Path,
    resolution_level: int = 2,
    selected_channel: str = "",
) -> None:
    graph.stitch(imagej_path, resolution_level, selected_channel)


if __name__ == "__main__":
    data_directory = Path("C:/Users/Igor/Documents/NIU-dev/stitching/Brain2")
    graph = load(data_directory)

    stitch(
        graph,
        Path("C:/Users/Igor/Documents/Fiji.app/ImageJ-win64.exe"),
        selected_channel="561 nm",
    )

    print(graph.data_for_napari(1))
