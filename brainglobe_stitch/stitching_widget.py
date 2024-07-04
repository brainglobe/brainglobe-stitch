from pathlib import Path
from typing import List, Optional, Tuple

import dask.array as da
import h5py
import napari
import numpy.typing as npt
from brainglobe_utils.qtpy.logo import header_widget
from napari import Viewer
from napari.qt.threading import create_worker
from napari.utils.notifications import show_warning
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

from brainglobe_stitch.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
)
from brainglobe_stitch.image_mosaic import ImageMosaic


def add_tiles_from_mosaic(
    napari_data: List[Tuple[da.Array, npt.NDArray]], tile_names: List[str]
):
    """
    Add tiles to the napari viewer from the ImageMosaic.

    Parameters
    ------------
    napari_data : List[Tuple[da.Array, npt.NDArray]]
        The data and position for each tile in the mosaic.
    tile_names : List[str]
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
    """
    napari widget for stitching large tiled 3d images.

    Parameters
    ------------
    napari_viewer : napari.Viewer
        The napari viewer to add the widget to.

    Attributes
    ----------
    progress_bar : QProgressBar
        The progress bar for the widget, reused for multiple function.
    image_mosaic : Optional[ImageMosaic]
        The ImageMosaic object representing the data that will be stitched.
    tile_layers : List[napari.layers.Image]
        The list of napari layers containing the tiles.
    resolution_to_display : int
        The resolution level of the pyramid to display in napari.
    header : QWidget
        The header widget for the StitchingWidget.
    default_directory : Path
        The default directory for the widget (home directory by default).
    working_directory : Path
        The working directory for the widget.
    select_mesospim_directory : QWidget
        The widget for selecting the mesoSPIM directory.
    mesospim_directory_text_field : QLineEdit
        The text field for the mesoSPIM directory.
    open_file_dialog : QPushButton
        The button for opening the file dialog.
    create_pyramid_button : QPushButton
        The button for creating the resolution pyramid.
    add_tiles_button : QPushButton
        The button for adding the tiles to the viewer.
    """

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._viewer = napari_viewer
        self.progress_bar = QProgressBar(self)
        self.image_mosaic: Optional[ImageMosaic] = None
        self.imagej_path: Optional[Path] = None
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

    def _on_open_file_dialog_clicked(self):
        """
        Open a file dialog to select the mesoSPIM directory.
        """
        self.working_directory = Path(
            QFileDialog.getExistingDirectory(
                self, "Select mesoSPIM directory", str(self.default_directory)
            )
        )
        # Add the text to the mesospim directory text field
        self.mesospim_directory_text_field.setText(str(self.working_directory))
        self.check_and_load_mesospim_directory()

    def _on_mesospim_directory_text_edited(self):
        """
        Update the working directory when the text field is edited.
        """
        self.working_directory = Path(
            self.mesospim_directory_text_field.text()
        )
        self.check_and_load_mesospim_directory()

    def _on_create_pyramid_button_clicked(self):
        """
        Create the resolution pyramid for the input mesoSPIM h5 data.
        """
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
        """
        Add the tiles from the mesoSPIM h5 file to the viewer.
        """
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
        """
        Add the tile layer to the viewer and store it in the tile_layers list.

        Parameters
        ----------
        tile_layer : napari.layers.Image
        """
        tile_layer = self._viewer.add_layer(tile_layer)
        self.tile_layers.append(tile_layer)

    def check_and_load_mesospim_directory(self):
        """
        Check if the selected directory is a valid mesoSPIM directory,
        if valid load the h5 file and check if the resolution pyramid
        is present. If not present, enable the create pyramid button.
        Otherwise, enable the add tiles button.
        """
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
                else:
                    self.add_tiles_button.setEnabled(True)
        except FileNotFoundError:
            show_warning("mesoSPIM directory not valid")

    def _on_open_file_dialog_imagej_clicked(self):
        """
        Open a file dialog to select the FIJI path.
        """
        self.imagej_path = Path(
            QFileDialog.getOpenFileName(
                self, "Select FIJI Path", str(self.default_directory)
            )[0]
        )
        self.imagej_path_text_field.setText(str(self.imagej_path))
        self.check_imagej_path()

    def _on_imagej_path_text_edited(self):
        """
        Update the FIJI path when the text field is edited.
        """
        self.imagej_path = Path(self.imagej_path_text_field.text())
        self.check_imagej_path()

    def _on_stitch_button_clicked(self):
        """
        Stitch the tiles in the viewer using BigStitcher.
        """
        self.image_mosaic.stitch(
            self.imagej_path,
            resolution_level=2,
            selected_channel=self.fuse_channel_dropdown.currentText(),
        )

        napari_data = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        self.update_tiles_from_mosaic(napari_data)

    def check_imagej_path(self):
        """
        Check if the selected ImageJ path is valid. If valid, enable the
        stitch button. Otherwise, show a warning.
        """
        if self.imagej_path.exists():
            self.stitch_button.setEnabled(True)
        else:
            show_warning("ImageJ path not valid")

    def update_tiles_from_mosaic(
        self, napari_data: List[Tuple[da.Array, npt.NDArray]]
    ):
        """
        Update the data stored in the napari viewer for each tile based on
        the ImageMosaic.

        Parameters
        ----------
        napari_data: List[Tuple[da.Array, npt.NDArray]]
            The data and position for each tile in the mosaic.
        """
        for data, tile_layer in zip(napari_data, self.tile_layers):
            tile_data, tile_position = data
            tile_layer.data = tile_data.compute()
            tile_layer.translate = tile_position
