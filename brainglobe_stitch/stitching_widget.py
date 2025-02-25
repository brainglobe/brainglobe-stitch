import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask.array as da
import h5py
import napari
import numpy as np
import numpy.typing as npt
from brainglobe_utils.qtpy.logo import header_widget
from napari import Viewer
from napari.qt.threading import create_worker
from napari.utils.notifications import show_info, show_warning
from qt_niu.dialog import display_info, display_warning
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
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
    napari_data: List[Tuple[da.Array, npt.NDArray]], image_mosaic: ImageMosaic
) -> Generator[napari.layers.Image, None, None]:
    """
    Add tiles to the napari viewer from the ImageMosaic.

    Parameters
    ------------
    napari_data : List[Tuple[da.Array, npt.NDArray]]
        The data and position for each tile in the mosaic.
    image_mosaic : ImageMosaic
        The ImageMosaic object containing the data for the tiles.
    """
    final_thresholds: Dict[str, float] = image_mosaic.calculate_contrast_max()

    for data, tile_name, tile in zip(
        napari_data, image_mosaic.tile_names, image_mosaic.tiles
    ):
        channel_name = tile.channel_name
        tile_data, tile_position = data
        tile_layer = napari.layers.Image(
            tile_data.compute(),
            name=tile_name,
            blending="translucent",
            contrast_limits=[0, final_thresholds[channel_name] * 1.5],
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
    imagej_path : Optional[Path]
        The path to the ImageJ executable.
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
        The button for opening the file dialog for the mesoSPIM directory.
    create_pyramid_button : QPushButton
        The button for creating the resolution pyramid.
    add_tiles_button : QPushButton
        The button for adding the tiles to the viewer.
    select_imagej_path : QWidget
        The widget for selecting the ImageJ path.
    imagej_path_text_field : QLineEdit
        The text field for the ImageJ path.
    open_file_dialog_imagej : QPushButton
        The button for opening the file dialog for the ImageJ path.#
    fuse_channel_dropdown : QComboBox
        The dropdown for selecting the channel to fuse.
    stitch_button : QPushButton
        The button for stitching the tiles.
    fuse_option_widget : QWidget
        The widget for the fuse options.
    output_file_name_field : QLineEdit
        The text field for the output file name.
    fuse_button : QPushButton
        The button for fusing the stitched tiles.
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

        self.layout().addWidget(QLabel("Path to ImageJ executable:"))
        self.layout().addWidget(self.select_imagej_path)

        self.fuse_channel_dropdown = QComboBox(parent=self)
        self.layout().addWidget(self.fuse_channel_dropdown)

        self.stitch_button = QPushButton("Stitch")
        self.stitch_button.clicked.connect(self._on_stitch_button_clicked)
        self.stitch_button.setEnabled(False)
        self.layout().addWidget(self.stitch_button)

        self.fuse_option_widget = QWidget()
        self.fuse_option_widget.setLayout(QFormLayout())
        self.output_file_name_field = QLineEdit()
        self.fuse_option_widget.layout().addRow(
            "Output file name:", self.output_file_name_field
        )

        self.layout().addWidget(self.fuse_option_widget)

        self.fuse_button = QPushButton("Fuse")
        self.fuse_button.setEnabled(False)
        self.fuse_button.clicked.connect(self._on_fuse_button_clicked)
        self.layout().addWidget(self.fuse_button)

        self.layout().addWidget(self.progress_bar)

    def _on_open_file_dialog_clicked(self) -> None:
        """
        Open a file dialog to select the mesoSPIM directory.
        """
        working_directory_str = QFileDialog.getExistingDirectory(
            self, "Select mesoSPIM directory", str(self.default_directory)
        )
        # A blank string is returned if the user cancels the dialog
        if not working_directory_str:
            return

        self.working_directory = Path(working_directory_str)
        # Add the text to the mesospim directory text field
        self.mesospim_directory_text_field.setText(str(self.working_directory))
        self.check_and_load_mesospim_directory()

    def _on_mesospim_directory_text_edited(self) -> None:
        """
        Update the working directory when the text field is edited.
        """
        self.working_directory = Path(
            self.mesospim_directory_text_field.text()
        )
        self.check_and_load_mesospim_directory()

    def _on_create_pyramid_button_clicked(self) -> None:
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

    def _on_add_tiles_button_clicked(self) -> None:
        """
        Add the tiles from the mesoSPIM h5 file to the viewer.
        """
        self.image_mosaic = ImageMosaic(self.working_directory)

        self.fuse_channel_dropdown.clear()
        self.fuse_channel_dropdown.addItems(self.image_mosaic.channel_names)

        min_size_display = np.array((256, 256, 256))
        self.resolution_to_display = 0

        for tile_data in self.image_mosaic.tiles[0].data_pyramid:
            if np.any(tile_data.shape <= min_size_display):
                break
            self.resolution_to_display += 1

        napari_data = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        worker = create_worker(
            add_tiles_from_mosaic, napari_data, self.image_mosaic
        )
        worker.yielded.connect(self._set_tile_layers)
        worker.start()

    def _set_tile_layers(self, tile_layer: napari.layers.Image) -> None:
        """
        Add the tile layer to the viewer and store it in the tile_layers list.

        Parameters
        ----------
        tile_layer : napari.layers.Image
        """
        tile_layer = self._viewer.add_layer(tile_layer)
        self.tile_layers.append(tile_layer)

    def check_and_load_mesospim_directory(self) -> None:
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
                    error_message = "Resolution pyramid not found"
                    show_warning(error_message)
                    display_info(self, "Warning", error_message)
                    self.create_pyramid_button.setEnabled(True)
                else:
                    self.add_tiles_button.setEnabled(True)
        except FileNotFoundError:
            error_message = "mesoSPIM directory not valid"
            show_warning(error_message)
            display_info(self, "Warning", error_message)

    def _on_open_file_dialog_imagej_clicked(self) -> None:
        """
        Open a file dialog to select the FIJI path.
        """
        imagej_path_str = QFileDialog.getOpenFileName(
            self, "Select FIJI Path", str(self.default_directory)
        )[0]
        # A blank string is returned if the user cancels the dialog
        if not imagej_path_str:
            return

        self.imagej_path = Path(imagej_path_str)
        self.imagej_path_text_field.setText(str(self.imagej_path))
        self.check_imagej_path()

    def _on_imagej_path_text_edited(self) -> None:
        """
        Update the FIJI path when the text field is edited.
        """
        self.imagej_path = Path(self.imagej_path_text_field.text())
        self.check_imagej_path()

    def _on_stitch_button_clicked(self) -> None:
        """
        Stitch the tiles in the viewer using BigStitcher.
        """
        if self.image_mosaic is None:
            error_message = "Open a mesoSPIM directory prior to stitching"
            show_warning(error_message)
            display_info(self, "Warning", error_message)
            return

        if not self.imagej_path:
            error_message = "Select the ImageJ path prior to stitching"
            show_warning(error_message)
            display_info(self, "Warning", error_message)
            return

        self.image_mosaic.stitch(
            self.imagej_path,
            resolution_level=2,
            selected_channel=self.fuse_channel_dropdown.currentText(),
        )

        show_info("Stitching complete")

        napari_data = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        self.update_tiles_from_mosaic(napari_data)
        self.fuse_button.setEnabled(True)

    def _on_fuse_button_clicked(self) -> None:
        if not self.output_file_name_field.text():
            error_message = "Output file name not specified"
            show_warning(error_message)
            display_info(self, "Warning", error_message)
            return

        if self.image_mosaic is None:
            error_message = "Open a mesoSPIM directory prior to stitching"
            show_warning(error_message)
            display_info(self, "Warning", error_message)
            return

        path = self.working_directory / self.output_file_name_field.text()
        valid_extensions = [".zarr", ".h5"]

        if path.suffix not in valid_extensions:
            error_message = (
                f"Output file name should end with "
                f"{', '.join(valid_extensions)}"
            )
            show_warning(error_message)
            display_info(self, "Warning", error_message)
            return

        if path.exists():
            error_message = (
                f"Output file {path} already exists. Replace existing file?"
            )
            if display_warning(self, "Warning", error_message):
                (
                    shutil.rmtree(path)
                    if path.suffix == ".zarr"
                    else path.unlink()
                )
            else:
                show_warning(
                    "Output file already exists. "
                    "Please choose a different name."
                )
                return

        self.image_mosaic.fuse(
            self.output_file_name_field.text(),
        )

        show_info("Fusing complete")
        display_info(self, "Info", f"Fused image saved to {path}")

    def check_imagej_path(self) -> None:
        """
        Check if the selected ImageJ path is valid. If valid, enable the
        stitch button. Otherwise, show a warning.
        """
        if self.imagej_path and self.imagej_path.exists():
            self.stitch_button.setEnabled(True)
        else:
            error_message = (
                "ImageJ path not valid. "
                "Please select a valid path to the imageJ executable."
            )
            show_warning(error_message)
            display_info(self, "Warning", error_message)

    def update_tiles_from_mosaic(
        self, napari_data: List[Tuple[da.Array, npt.NDArray]]
    ) -> None:
        """
        Update the data stored in the napari viewer for each tile based on
        the ImageMosaic.

        Parameters
        ----------
        napari_data : List[Tuple[da.Array, npt.NDArray]]
            The data and position for each tile in the mosaic.
        """
        for data, tile_layer in zip(napari_data, self.tile_layers):
            tile_data, tile_position = data
            tile_layer.data = tile_data.compute()
            tile_layer.translate = tile_position
