from pathlib import Path

from mesospim_stitcher.image_graph import ImageGraph


def load(directory: Path) -> ImageGraph:
    return ImageGraph(directory)
