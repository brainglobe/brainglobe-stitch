import os

import pytest

from brainglobe_stitch.image_mosaic import ImageMosaic


@pytest.fixture(scope="class")
def image_mosaic(naive_bdv_directory):
    os.remove(naive_bdv_directory / "test_data_bdv_tile_config.txt")
    image_mosaic = ImageMosaic(naive_bdv_directory)

    return image_mosaic


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
