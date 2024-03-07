import os

import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic


@pytest.fixture(scope="class")
def image_mosaic(naive_bdv_directory):
    os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")
    image_mosaic = ImageMosaic(naive_bdv_directory)

    yield image_mosaic

    del image_mosaic


def test_image_mosaic_init(image_mosaic, naive_bdv_directory):
    image_mosaic = image_mosaic
    assert image_mosaic.xml_path == naive_bdv_directory / "test_data_bdv.xml"
    assert (
        image_mosaic.meta_path
        == naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )
    assert (
        image_mosaic.h5_path
        == naive_bdv_directory / "test_data_original_bdv.h5"
    )
    assert (
        image_mosaic.meta_path
        == naive_bdv_directory / "test_data_bdv.h5_meta.txt"
    )
    assert image_mosaic.h5_file is not None
    assert len(image_mosaic.channel_names) == 2
    assert len(image_mosaic.tiles) == 8
    assert len(image_mosaic.tile_names) == 8
    assert image_mosaic.x_y_resolution == 4.08
    assert image_mosaic.z_resolution == 5.0
    assert image_mosaic.num_channels == 2
    assert len(image_mosaic.intensity_adjusted) == 5
    assert len(image_mosaic.overlaps_interpolated) == 5
