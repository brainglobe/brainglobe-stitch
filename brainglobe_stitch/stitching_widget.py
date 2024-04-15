from pathlib import Path
from typing import List, Optional, Tuple

import dask.array as da
import napari
import numpy.typing as npt
from brainglobe_utils.qtpy.logo import header_widget
from napari import Viewer
from napari.qt.threading import create_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from brainglobe_stitch.file_utils import create_pyramid_bdv_h5
from brainglobe_stitch.image_mosaic import ImageMosaic


def add_tiles_from_mosaic(
    napari_data: List[Tuple[da.Array, npt.NDArray]], tile_names: List[str]
):
    """
    Add tiles to the napari viewer from the ImageMosaic.

    Parameters
    ----------
    napari_data: List[Tuple[da.Array, npt.NDArray]]
        The data and position for each tile in the mosaic.
    tile_names: List[str]
        The list of tile names.
    """

    for data, tile_name in zip(napari_data, tile_names):
        tile_data, tile_position = data
        tile_layer = napari.layers.Image(
            tile_data.compute(),
            name=tile_name,
            blending="translucent",
            contrast_limits=[0, 4000],
            multiscale=False,
        )
        tile_layer.translate = tile_position

        yield tile_layer


class StitchingWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._viewer = napari_viewer
        self.image_mosaic: Optional[ImageMosaic] = None
        self.tile_layers: List[napari.layers.Image] = []
        self.resolution_to_display: int = 3

        self.setLayout(QVBoxLayout())

        self.header = header_widget(
            package_name="brainglobe-stitch",
            package_tagline="Stitching mesoSPIM data",
        )
        self.layout().addWidget(self.header)

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

    def _on_open_file_dialog_clicked(self):
        self.working_directory = Path(
            QFileDialog.getExistingDirectory(
                self, "Select mesoSPIM directory", str(self.default_directory)
            )
        )
        # Add the text to the mesospim directory text field
        self.mesospim_directory_text_field.setText(str(self.working_directory))
        self.check_and_load_mesospim_directory()

    def _on_mesospim_directory_text_edited(self):
        self.working_directory = Path(
            self.mesospim_directory_text_field.text()
        )
        self.check_and_load_mesospim_directory()

    def _on_create_pyramid_button_clicked(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)

        worker = create_worker(
            create_pyramid_bdv_h5,
            self.h5_path,
            yield_progress=True,
        )
        worker.yielded.connect(self.progress_bar.setValue)
        worker.finished.connect(self.progress_bar.reset)
        worker.start()

        self.create_pyramid_button.setEnabled(False)
        self.add_tiles_button.setEnabled(True)

    def _on_add_tiles_button_clicked(self):
        self.image_mosaic = ImageMosaic(self.working_directory)

        self.fuse_channel_dropdown.clear()
        self.fuse_channel_dropdown.addItems(self.image_mosaic.channel_names)

        napari_data = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        worker = create_worker(
            add_tiles_from_mosaic, napari_data, self.image_mosaic.tile_names
        )
        worker.yielded.connect(self._set_tile_layers)
        worker.start()

    def _set_tile_layers(self, tile_layer: napari.layers.Image):
        tile_layer = self._viewer.add_layer(tile_layer)
        self.tile_layers.append(tile_layer)
