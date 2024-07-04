from pathlib import Path

import dask.array as da
import napari.layers
import numpy as np
import pytest

import brainglobe_stitch
from brainglobe_stitch.image_mosaic import ImageMosaic
from brainglobe_stitch.stitching_widget import (
    StitchingWidget,
    add_tiles_from_mosaic,
)


def test_add_tiles_from_mosaic():
    num_tiles = 4

    test_data = []
    for i in range(num_tiles):
        test_data.append((da.ones((10, 10, 10)), np.array([i, i, i])))

    tile_names = [f"s{i:02}" for i in range(num_tiles)]

    for data, tile in zip(
        test_data, add_tiles_from_mosaic(test_data, tile_names)
    ):
        assert isinstance(tile, napari.layers.Image)
        assert (tile.data == data[0]).all()
        assert (tile.translate == data[1]).all()


def test_stitching_widget_init(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)

    assert stitching_widget._viewer == viewer
    assert stitching_widget.image_mosaic is None
    assert len(stitching_widget.tile_layers) == 0
    assert stitching_widget.resolution_to_display == 3


def test_on_open_file_dialog_clicked(make_napari_viewer_proxy, mocker):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    test_dir = str(Path.home() / "test_dir")
    mocker.patch(
        "brainglobe_stitch.stitching_widget.QFileDialog.getExistingDirectory",
        return_value=test_dir,
    )
    mocker.patch(
        "brainglobe_stitch.stitching_widget.StitchingWidget.check_and_load_mesospim_directory",
    )

    stitching_widget._on_open_file_dialog_clicked()

    assert stitching_widget.mesospim_directory_text_field.text() == test_dir
    assert stitching_widget.working_directory == Path(test_dir)


def test_on_mesospim_directory_text_edited(make_napari_viewer_proxy, mocker):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    test_dir = str(Path.home() / "test_dir")
    mocker.patch(
        "brainglobe_stitch.stitching_widget.StitchingWidget.check_and_load_mesospim_directory",
    )

    stitching_widget.mesospim_directory_text_field.setText(test_dir)

    stitching_widget._on_mesospim_directory_text_edited()

    assert stitching_widget.working_directory == Path(test_dir)


def test_on_create_pyramid_button_clicked(make_napari_viewer_proxy, mocker):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.h5_path = Path.home() / "test_path"
    mock_create_worker = mocker.patch(
        "brainglobe_stitch.stitching_widget.create_worker",
        autospec=True,
    )

    stitching_widget._on_create_pyramid_button_clicked()

    mock_create_worker.assert_called_once_with(
        brainglobe_stitch.file_utils.create_pyramid_bdv_h5,
        stitching_widget.h5_path,
        yield_progress=True,
    )

    assert not stitching_widget.create_pyramid_button.isEnabled()
    assert stitching_widget.add_tiles_button.isEnabled()


def test_on_add_tiles_button_clicked(
    make_napari_viewer_proxy, naive_bdv_directory, mocker
):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.working_directory = naive_bdv_directory

    mock_create_worker = mocker.patch(
        "brainglobe_stitch.stitching_widget.create_worker",
        autospec=True,
    )

    stitching_widget._on_add_tiles_button_clicked()

    mock_create_worker.assert_called_once()


@pytest.mark.parametrize("num_layers", [1, 2, 5])
def test_set_tile_layers_multiple(make_napari_viewer_proxy, num_layers):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    test_data = da.ones((10, 10, 10))

    test_layers = []

    for i in range(num_layers):
        test_layer = napari.layers.Image(data=test_data)
        stitching_widget._set_tile_layers(test_layer)
        test_layers.append(test_layer)

    assert len(stitching_widget.tile_layers) == num_layers
    for i in range(num_layers):
        assert stitching_widget.tile_layers[i] == test_layers[i]


def test_check_and_load_mesospim_directory(
    make_napari_viewer_proxy, naive_bdv_directory
):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.working_directory = naive_bdv_directory

    stitching_widget.check_and_load_mesospim_directory()

    assert stitching_widget.h5_path == naive_bdv_directory / "test_data_bdv.h5"
    assert (
        stitching_widget.xml_path == naive_bdv_directory / "test_data_bdv.xml"
    )
    assert (
        stitching_widget.meta_path
        == naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )
    assert stitching_widget.add_tiles_button.isEnabled()


def test_check_and_load_mesospim_directory_no_pyramid(
    make_napari_viewer_proxy, bdv_directory_function_level, mocker
):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.working_directory = bdv_directory_function_level

    mock_show_warning = mocker.patch(
        "brainglobe_stitch.stitching_widget.show_warning"
    )

    stitching_widget.check_and_load_mesospim_directory()

    mock_show_warning.assert_called_once_with("Resolution pyramid not found")
    assert not stitching_widget.add_tiles_button.isEnabled()
    assert stitching_widget.create_pyramid_button.isEnabled()


@pytest.mark.parametrize(
    "file_to_remove",
    ["test_data_bdv.h5", "test_data_bdv.xml", "test_data_bdv.h5_meta.txt"],
)
def test_check_and_load_mesospim_directory_missing_files(
    make_napari_viewer_proxy,
    bdv_directory_function_level,
    mocker,
    file_to_remove,
):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.working_directory = bdv_directory_function_level
    error_message = "mesoSPIM directory not valid"

    mock_show_warning = mocker.patch(
        "brainglobe_stitch.stitching_widget.show_warning"
    )
    (bdv_directory_function_level / file_to_remove).unlink()

    stitching_widget.check_and_load_mesospim_directory()

    mock_show_warning.assert_called_once_with(error_message)


def test_on_open_file_dialog_imagej_clicked(make_napari_viewer_proxy, mocker):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    imagej_dir = str(Path.home() / "imageJ")
    mocker.patch(
        "brainglobe_stitch.stitching_widget.QFileDialog.getOpenFileName",
        return_value=(imagej_dir, ""),
    )
    mocker.patch(
        "brainglobe_stitch.stitching_widget.StitchingWidget.check_imagej_path",
    )

    stitching_widget._on_open_file_dialog_imagej_clicked()

    assert stitching_widget.imagej_path_text_field.text() == imagej_dir
    assert stitching_widget.imagej_path == Path(imagej_dir)


def test_on_imagej_path_text_edited(make_napari_viewer_proxy, mocker):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    imagej_dir = str(Path.home() / "imageJ")
    mocker.patch(
        "brainglobe_stitch.stitching_widget.StitchingWidget.check_imagej_path",
    )

    stitching_widget.imagej_path_text_field.setText(imagej_dir)

    stitching_widget._on_imagej_path_text_edited()

    assert stitching_widget.imagej_path == Path(imagej_dir)


def test_on_stitch_button_clicked(
    make_napari_viewer_proxy, naive_bdv_directory, mocker
):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)

    stitching_widget.image_mosaic = ImageMosaic(naive_bdv_directory)

    mock_stitch_function = mocker.patch(
        "brainglobe_stitch.stitching_widget.ImageMosaic.stitch",
        autospec=True,
    )

    stitching_widget._on_stitch_button_clicked()

    mock_stitch_function.assert_called_once_with(
        stitching_widget.image_mosaic,
        stitching_widget.imagej_path,
        resolution_level=2,
        selected_channel="",
    )


def test_check_imagej_path_valid(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.imagej_path = Path.home() / "imageJ"
    stitching_widget.imagej_path.touch(exist_ok=True)
    stitching_widget.check_imagej_path()
    # Clean up before assertions to make sure nothing is left behind
    # regardless of test outcome
    stitching_widget.imagej_path.unlink()

    assert stitching_widget.stitch_button.isEnabled()


def test_check_imagej_path_invalid(make_napari_viewer_proxy, mocker):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    stitching_widget.imagej_path = Path.home() / "imageJ"

    mock_show_warning = mocker.patch(
        "brainglobe_stitch.stitching_widget.show_warning"
    )
    error_message = "ImageJ path not valid"

    stitching_widget.check_imagej_path()

    mock_show_warning.assert_called_once_with(error_message)


def test_update_tiles_from_mosaic(
    make_napari_viewer_proxy, naive_bdv_directory
):
    viewer = make_napari_viewer_proxy()
    stitching_widget = StitchingWidget(viewer)
    num_tiles = 4
    initial_data = []
    test_data = []

    for i in range(num_tiles):
        initial_data.append((da.zeros((10, 10, 10)), np.array([0, 0, 0])))
        test_data.append((da.ones((10, 10, 10)) + i, np.array([i, i, i])))

    for tile in add_tiles_from_mosaic(
        initial_data, [f"s{i:02}" for i in range(num_tiles)]
    ):
        stitching_widget.tile_layers.append(tile)

    stitching_widget.update_tiles_from_mosaic(test_data)

    for tile, test_data in zip(stitching_widget.tile_layers, test_data):
        assert (tile.data == test_data[0]).all()
        assert (tile.translate == test_data[1]).all()
