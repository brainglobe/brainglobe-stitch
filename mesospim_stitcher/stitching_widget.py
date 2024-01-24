from pathlib import Path
from time import sleep
from typing import Dict, List, Tuple

import dask.array as da
import h5py
import napari.layers
import numpy as np
from brainglobe_utils.qtpy.logo import header_widget
from napari.qt.threading import create_worker
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mesospim_stitcher.big_stitcher_bridge import run_big_stitcher
from mesospim_stitcher.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
    parse_mesospim_metadata,
    write_big_stitcher_tile_config,
)
from mesospim_stitcher.fuse import fuse_image, get_big_stitcher_transforms

DOWNSAMPLE_ARRAY = np.array(
    [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
)

SUBDIVISION_ARRAY = np.array(
    [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
)


class StitchingWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.progress_bar = QProgressBar(self)
        self._viewer = napari_viewer
        self.xml_path: Path | None = None
        self.meta_path: Path | None = None
        self.h5_path: Path | None = None
        self.tile_config_path: Path | None = None
        self.imagej_path: Path | None = None
        self.h5_file: h5py.File | None = None
        self.tiles: List[da] = []
        self.tile_layers: List[napari.layers.Image] = []
        self.tile_metadata: List[Dict] = []
        self.num_channels: int = 1
        self.original_image_shape: Tuple[int, int, int] | None = None
        self.resolution_to_display: int = 2

        self.setLayout(QVBoxLayout())

        self.header = header_widget(
            "BrainGlobe_Stitcher", "Stitching", "stitching.html"
        )

        self.default_directory = Path.home()
        self.working_directory = self.default_directory

        self.select_mesospim_directory = QWidget()
        self.select_mesospim_directory.setLayout(QHBoxLayout())

        self.mesospim_directory_text_field = QLineEdit()
        self.mesospim_directory_text_field.setText(str(self.default_directory))
        self.mesospim_directory_text_field.editingFinished.connect(
            self._on_mesospim_directory_text_edited
        )
        self.select_mesospim_directory.layout().addWidget(
            self.mesospim_directory_text_field
        )

        self.open_file_dialog = QPushButton("Browse")
        self.open_file_dialog.clicked.connect(
            self._on_open_file_dialog_clicked
        )
        self.select_mesospim_directory.layout().addWidget(
            self.open_file_dialog
        )

        self.layout().addWidget(self.header)
        self.layout().addWidget(QLabel("Select mesospim directory:"))
        self.layout().addWidget(self.select_mesospim_directory)

        self.create_pyramid_button = QPushButton("Create resolution pyramid")
        self.create_pyramid_button.clicked.connect(
            self._on_create_pyramid_button_clicked
        )
        self.create_pyramid_button.setEnabled(False)

        self.layout().addWidget(self.create_pyramid_button)

        self.add_tiles_button = QPushButton("Add tiles to viewer")
        self.add_tiles_button.clicked.connect(
            self._on_add_tiles_button_clicked
        )
        self.add_tiles_button.setEnabled(False)
        self.layout().addWidget(self.add_tiles_button)

        self.select_imagej_path = QWidget()
        self.select_imagej_path.setLayout(QHBoxLayout())

        self.imagej_path_text_field = QLineEdit()
        self.imagej_path_text_field.setText(str(self.default_directory))
        self.imagej_path_text_field.editingFinished.connect(
            self._on_imagej_path_text_edited
        )
        self.select_imagej_path.layout().addWidget(self.imagej_path_text_field)

        self.open_file_dialog_imagej = QPushButton("Browse")
        self.open_file_dialog_imagej.clicked.connect(
            self._on_open_file_dialog_imagej_clicked
        )
        self.select_imagej_path.layout().addWidget(
            self.open_file_dialog_imagej
        )

        self.layout().addWidget(QLabel("Select ImageJ path:"))
        self.layout().addWidget(self.select_imagej_path)

        self.fuse_channel_dropdown = QComboBox(parent=self)
        self.layout().addWidget(self.fuse_channel_dropdown)

        self.stitch_button = QPushButton("Stitch")
        self.stitch_button.clicked.connect(self._on_stitch_button_clicked)
        self.stitch_button.setEnabled(False)
        self.layout().addWidget(self.stitch_button)

        self.fuse_button = QPushButton("Fuse")
        self.fuse_button.clicked.connect(self._on_fuse_button_clicked)
        self.fuse_button.setEnabled(False)
        self.layout().addWidget(self.fuse_button)

        self.layout().addWidget(self.progress_bar)

    def _on_open_file_dialog_clicked(self):
        self.working_directory = Path(
            QFileDialog.getExistingDirectory(
                self, "Select mesoSPIM directory", str(self.default_directory)
            )
        )
        self.mesospim_directory_text_field.setText(str(self.working_directory))
        self.check_and_load_mesospim_directory()

    def _on_mesospim_directory_text_edited(self):
        self.working_directory = Path(
            self.mesospim_directory_text_field.text()
        )
        self.check_and_load_mesospim_directory()

    def _on_open_file_dialog_imagej_clicked(self):
        self.imagej_path = Path(
            QFileDialog.getOpenFileName(
                self, "Select ImageJ Path", str(self.default_directory)
            )[0]
        )
        self.imagej_path_text_field.setText(str(self.imagej_path))
        self.check_imagej_path()

    def _on_imagej_path_text_edited(self):
        self.imagej_path = Path(self.imagej_path_text_field.text())
        self.check_imagej_path()

    def _on_create_pyramid_button_clicked(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)

        worker = create_worker(
            create_pyramid_bdv_h5,
            self.h5_path,
            DOWNSAMPLE_ARRAY,
            SUBDIVISION_ARRAY,
            yield_progress=True,
        )
        worker.yielded.connect(self.progress_bar.setValue)
        worker.finished.connect(self.progress_bar.reset)
        worker.start()

    def _on_add_tiles_button_clicked(self):
        self.tile_config_path = Path(
            str(self.xml_path)[:-4] + "_tile_config.txt"
        )

        if not self.tile_config_path.exists():
            self.tile_metadata = write_big_stitcher_tile_config(
                self.meta_path, self.h5_path
            )
        else:
            self.tile_metadata = parse_mesospim_metadata(self.meta_path)

        translations: List[tuple[int, int, int]] = []
        with open(self.tile_config_path, "r") as f:
            # Skip header
            f.readline()
            for line in f.readlines():
                split_line = line.split(";")[-1].strip("()\n").split(",")
                translations.append(
                    (
                        int(split_line[2]),
                        int(split_line[1]),
                        int(split_line[0]),
                    )
                )

        channel_names = []
        idx = 0
        while self.tile_metadata[idx]["Laser"] not in channel_names:
            channel_names.append(self.tile_metadata[idx]["Laser"])
            idx += 1

        self.fuse_channel_dropdown.clear()
        self.fuse_channel_dropdown.addItems(channel_names)
        self.num_channels = len(channel_names)

        self.h5_file = h5py.File(self.h5_path, "r")
        self.original_image_shape = self.h5_file["t00000/s00/0/cells"].shape
        self.tiles = []
        tile_group = self.h5_file["t00000"]

        for idx, child in enumerate(tile_group):
            try:
                curr_tile = da.from_array(
                    tile_group[f"{child}/{self.resolution_to_display}/cells"]
                )
            except KeyError:
                show_warning("Resolution pyramid not found")

                return

            self.tiles.append(curr_tile)
            tile = self._viewer.add_image(
                curr_tile,
                blending="translucent",
                opacity=0.6,
                contrast_limits=[0, 2000],
                multiscale=False,
                name=child,
            )

            self.tile_layers.append(tile)
            tile.translate = (
                translations[idx]
                // DOWNSAMPLE_ARRAY[self.resolution_to_display]
            )

    def _on_stitch_button_clicked(self):
        try:
            channel_int = int(
                self.fuse_channel_dropdown.currentText().split()[0]
            )
        except ValueError:
            show_warning("Invalid channel name")

            return

        # Use a worker to run the stitching in a separate thread
        results = run_big_stitcher(
            self.imagej_path,
            self.xml_path,
            self.tile_config_path,
            all_channels=False,
            selected_channel=channel_int,
            downsample_x=4,
            downsample_y=4,
            downsample_z=4,
        )

        with open("big_stitcher_output.txt", "w") as f:
            f.write(results.stdout)
            f.write(results.stderr)

        # Wait for the BigStitcher to write XML file
        # Need to find a better way to do this
        sleep(1)

        z_size, y_size, x_size = self.original_image_shape
        self.translations = get_big_stitcher_transforms(
            self.xml_path, x_size, y_size, z_size
        )
        self.translations = [
            translation[-2::-2] // DOWNSAMPLE_ARRAY[self.resolution_to_display]
            for translation in self.translations
        ]

        for tile, translation in zip(self.tile_layers, self.translations):
            tile.translate = translation

        self.fuse_button.setEnabled(True)

        return

    def _on_fuse_button_clicked(self):
        contrast_limits = np.array(
            [tile.contrast_limits[1] for tile in self.tile_layers],
            dtype=np.int16,
        )
        fuse_image(
            self.xml_path,
            self.h5_path,
            self.h5_path.with_suffix(".zarr"),
            self.tile_metadata,
            contrast_limits,
            self.num_channels,
            yield_progress=False,
        )
        return

    def check_and_load_mesospim_directory(self):
        try:
            (
                self.xml_path,
                self.meta_path,
                self.h5_path,
            ) = check_mesospim_directory(self.working_directory)
            with h5py.File(self.h5_path, "r") as f:
                if len(f["t00000/s00"].keys()) <= 1:
                    show_warning("Resolution pyramid not found")
                    self.create_pyramid_button.setEnabled(True)

            self.add_tiles_button.setEnabled(True)
        except FileNotFoundError:
            show_warning("mesoSPIM directory not valid")

    def check_imagej_path(self):
        if self.imagej_path.exists():
            self.stitch_button.setEnabled(True)
        else:
            show_warning("ImageJ path not valid")
