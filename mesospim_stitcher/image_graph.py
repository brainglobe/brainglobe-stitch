from pathlib import Path

import h5py
import numpy as np
from rich.progress import Progress

from mesospim_stitcher.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
)

DOWNSAMPLE_ARRAY = np.array(
    [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
)
SUBDIVISION_ARRAY = np.array(
    [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
)


class ImageGraph:
    def __init__(self, directory: Path):
        self.directory: Path = directory
        self.xml_path: Path | None = None
        self.meta_path: Path | None = None
        self.h5_path: Path | None = None
        self.h5_file: h5py.File | None = None

        self.load_mesospim_directory()

    def load_mesospim_directory(self) -> None:
        try:
            (
                self.xml_path,
                self.meta_path,
                self.h5_path,
            ) = check_mesospim_directory(self.directory)
        except FileNotFoundError:
            print("Invalid mesoSPIM directory")

        self.h5_file = h5py.File(self.h5_path, "r")

        if len(self.h5_file["t00000/s00"].keys()) <= 1:
            print("Resolution pyramid not found.")
            self.h5_file.close()
            print("Creating resolution pyramid...")

            with Progress() as progress:
                task = progress.add_task(
                    "Creating resolution pyramid...", total=100
                )

                assert self.h5_path is not None

                for update in create_pyramid_bdv_h5(
                    self.h5_path,
                    DOWNSAMPLE_ARRAY,
                    SUBDIVISION_ARRAY,
                    yield_progress=True,
                ):
                    progress.update(task, advance=update)

            self.h5_file = h5py.File(self.h5_path, "r")
