from pathlib import Path
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
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from mesospim_stitcher.core import (
    fuse,
    interpolate_overlaps,
    load,
    normalise_intensity,
    stitch,
)
from mesospim_stitcher.file_utils import (
    check_mesospim_directory,
    create_pyramid_bdv_h5,
)
from mesospim_stitcher.image_mosaic import ImageMosaic
from mesospim_stitcher.tile import Tile

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
        self.image_mosaic: ImageMosaic | None = None
        self.xml_path: Path | None = None
        self.meta_path: Path | None = None
        self.h5_path: Path | None = None
        self.tile_config_path: Path | None = None
        self.imagej_path: Path | None = None
        self.h5_file: h5py.File | None = None
        self.slice_attributes: Dict[str, Dict] = {}
        self.intensity_scale_factors: List[float] = []
        self.tile_objects: List[Tile] = []
        self.tiles: List[da] = []
        self.tile_layers: List[napari.layers.Image] = []
        self.tile_names: List[str] = []
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

        self.adjust_intensity_button = QPushButton("Adjust Intensity")
        self.adjust_intensity_button.clicked.connect(
            self._on_adjust_intensity_button_clicked
        )
        self.adjust_intensity_button.setEnabled(False)
        self.layout().addWidget(self.adjust_intensity_button)

        self.interpolate_button = QPushButton("Interpolate")
        self.interpolate_button.clicked.connect(
            self._on_interpolation_button_clicked
        )
        self.interpolate_button.setEnabled(False)
        self.layout().addWidget(self.interpolate_button)

        self.adjust_intensity_collapsible = QCollapsible(
            "Intensity Adjustment Options"
        )
        self.adjust_intensity_menu = QWidget()
        self.adjust_intensity_menu.setLayout(
            QFormLayout(parent=self.adjust_intensity_menu)
        )

        self.percentile_field = QSpinBox(parent=self.adjust_intensity_menu)
        self.percentile_field.setRange(0, 100)
        self.percentile_field.setValue(80)
        self.adjust_intensity_menu.layout().addRow(
            "Percentile", self.percentile_field
        )

        self.adjust_intensity_collapsible.setContent(
            self.adjust_intensity_menu
        )

        self.layout().addWidget(self.adjust_intensity_collapsible)
        self.adjust_intensity_collapsible.collapse(animate=False)

        self.fuse_option_widget = QWidget()
        self.fuse_option_widget.setLayout(QFormLayout())
        self.normalise_intensity_toggle = QCheckBox()
        self.interpolate_toggle = QCheckBox()

        self.fuse_option_widget.layout().addRow(
            "Normalise intensity:", self.normalise_intensity_toggle
        )
        self.fuse_option_widget.layout().addRow(
            "Interpolate overlaps:", self.interpolate_toggle
        )

        self.layout().addWidget(self.fuse_option_widget)

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
        # Need to run in a separate worker thread
        self.image_mosaic = load(self.working_directory)

        self.fuse_channel_dropdown.clear()
        self.fuse_channel_dropdown.addItems(self.image_mosaic.channel_names)

        napari_data = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        self.add_tiles_from_mosaic(napari_data)

    def _on_stitch_button_clicked(self):
        stitch(
            self.image_mosaic,
            self.imagej_path,
            resolution_level=2,
            selected_channel=self.fuse_channel_dropdown.currentText(),
        )

        napari_data = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        self.update_tiles_from_mosaic(napari_data)

        self.fuse_button.setEnabled(True)
        self.adjust_intensity_button.setEnabled(True)
        self.interpolate_button.setEnabled(True)

        return

    def _on_adjust_intensity_button_clicked(self):
        normalise_intensity(
            self.image_mosaic,
            resolution_level=self.resolution_to_display,
            percentile=self.percentile_field.value(),
        )

        data_for_napari = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        self.update_tiles_from_mosaic(data_for_napari)

        return

    def _on_interpolation_button_clicked(self):
        interpolate_overlaps(self.image_mosaic, self.resolution_to_display)

        data_for_napari = self.image_mosaic.data_for_napari(
            self.resolution_to_display
        )

        self.update_tiles_from_mosaic(data_for_napari)

        return

    def _on_fuse_button_clicked(self):
        fuse(
            self.image_mosaic,
            "fused.zarr",
            self.normalise_intensity_toggle.isChecked(),
            self.interpolate_toggle.isChecked(),
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

    def add_tiles_from_mosaic(self, napari_data):
        for data, tile_name in zip(napari_data, self.image_mosaic.tile_names):
            tile_data, tile_position = data
            tile_layer = self._viewer.add_image(
                tile_data,
                blending="translucent",
                contrast_limits=[0, 4000],
                multiscale=False,
                name=tile_name,
            )

            self.tile_layers.append(tile_layer)
            tile_layer.translate = tile_position

    def update_tiles_from_mosaic(self, napari_data):
        for data, tile_layer in zip(napari_data, self.tile_layers):
            tile_data, tile_position = data
            tile_layer.data = tile_data
            tile_layer.translate = tile_position

    # def hideEvent(self, a0, QHideEvent=None):
    #     super().hideEvent(a0)
    #     if self.h5_file:
    #         self.h5_file.close()
