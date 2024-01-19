from pathlib import Path

import dask.array as da
import h5py
from brainglobe_utils.qtpy.logo import header_widget
from napari.utils.notifications import show_warning
from napari.viewer import Viewer
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mesospim_stitcher.file_utils import check_mesospim_directory


class StitchingWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._viewer = napari_viewer
        self.xml_path = None
        self.meta_path = None
        self.h5_path = None
        self.h5_file = None
        self.tiles: da = []

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

        self.add_tiles_button = QPushButton("Add tiles")
        self.add_tiles_button.clicked.connect(
            self._on_add_tiles_button_clicked
        )
        self.add_tiles_button.setEnabled(False)
        self.layout().addWidget(self.add_tiles_button)

    def _on_open_file_dialog_clicked(self):
        self.working_directory = Path(
            QFileDialog.getExistingDirectory(
                self, "Select directory", str(self.default_directory)
            )
        )
        self.mesospim_directory_text_field.setText(str(self.working_directory))
        self.check_and_load_mesospim_directory()

    def _on_mesospim_directory_text_edited(self):
        self.working_directory = Path(
            self.mesospim_directory_text_field.text()
        )
        self.check_and_load_mesospim_directory()

    def _on_add_tiles_button_clicked(self):
        self.tiles = []
        tile_group = self.h5_file["t00000"]

        for child in tile_group:
            curr_tile = da.from_array(tile_group[f"{child}/0/cells"])
            self.tiles.append(curr_tile)
            print("Adding tile to napari")
            self._viewer.add_image(
                curr_tile,
                contrast_limits=[0, 1500],
                multiscale=False,
                name=child,
            )

    def check_and_load_mesospim_directory(self):
        try:
            (
                self.xml_path,
                self.meta_path,
                self.h5_path,
            ) = check_mesospim_directory(self.working_directory)
            self.h5_file = h5py.File(self.h5_path, "r")
            self.add_tiles_button.setEnabled(True)
        except FileNotFoundError:
            show_warning("mesoSPIM directory not valid")
