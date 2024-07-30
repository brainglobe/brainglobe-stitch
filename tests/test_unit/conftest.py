import platform
import shutil
from pathlib import Path

import pooch
import pytest

TEMP_DIR = Path.home() / "temp_test_directory"
TEST_DATA_URL = "https://gin.g-node.org/IgorTatarnikov/brainglobe-stitch-test/raw/master/brainglobe-stitch/brainglobe-stitch-test-data.zip"


## BACKUP CONSTANTS
NUM_TILES = 8
NUM_RESOLUTIONS = 5
NUM_CHANNELS = 2
TILE_SIZE = (107, 128, 128)

EXPECTED_TILE_CONFIG = [
    "dim=3",
    "00;;(0,0,0)",
    "01;;(0,115,0)",
    "04;;(0,0,0)",
    "05;;(0,115,0)",
    "10;;(115,0,0)",
    "11;;(115,115,0)",
    "14;;(115,0,0)",
    "15;;(115,115,0)",
]

EXPECTED_TILE_POSITIONS = [
    [3, 4, 2],
    [2, 120, 0],
    [3, 4, 2],
    [2, 120, 0],
    [6, 7, 118],
    [5, 123, 116],
    [6, 7, 118],
    [5, 123, 116],
]


@pytest.fixture(scope="session", autouse=True)
def download_test_data():
    TEMP_DIR.mkdir(exist_ok=True)
    pooch.retrieve(
        TEST_DATA_URL,
        known_hash="9437cb05566f03cd78cd905da1cd939dd1bb439837f96e32c6fb818b9b010a6e",
        processor=pooch.Unzip(extract_dir=str(TEMP_DIR)),
    )

    yield TEMP_DIR

    shutil.rmtree(TEMP_DIR)


@pytest.fixture(scope="session")
def test_data_directory():
    yield TEMP_DIR


@pytest.fixture(scope="module")
def naive_bdv_directory():
    test_dir = Path.home() / "test_directory"

    shutil.copytree(
        TEMP_DIR,
        test_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*_interpolated_bdv.h5"),
    )
    # Create UNIX style hidden files that should be ignored
    (test_dir / ".test_data_bdv.h5").touch()
    (test_dir / ".test_data_bdv.h5_meta.txt").touch()
    (test_dir / ".test_data_bdv.h5_meta.txt").touch()

    yield test_dir

    shutil.rmtree(test_dir)


@pytest.fixture
def bdv_directory_function_level():
    test_dir = Path.home() / "quick_test_directory"

    shutil.copytree(
        TEMP_DIR,
        test_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*_interpolated_bdv.h5"),
    )

    yield test_dir

    shutil.rmtree(test_dir)


@pytest.fixture(scope="module")
def imagej_path():
    if platform.system() == "Windows":
        return Path.home() / "Fiji.app/ImageJ-win64.exe"
    elif platform.system() == "Darwin":
        return Path.home() / "Fiji.app"
    else:
        return Path.home() / "Fiji.app/ImageJ-linux64"


@pytest.fixture(scope="module")
def test_constants(imagej_path):
    # The tiles lie in one z-plane and are arranged in a 2x2 grid.
    # There are 2 channels.
    # Each tile is 128x128x107 pixels (x, y, z).
    # The tiles overlap by 10% in x and y (13 pixels).
    # The tiles are arranged in the following pattern:
    # channel 0   | channel 1
    # 00 10          | 04 05
    # 01 11           | 14 15

    # EXPECTED_TILE_CONFIG is based on test_data_bdv.xml
    # The tile positions are in pixels in x, y, z order

    # EXPECTED_TILE_POSITIONS are based on the stitch transforms found
    # in test_data_bdv.xml
    # The tile positions are in pixels in z, y, x order
    constants_dict = {
        "NUM_TILES": 8,
        "NUM_CHANNELS": 2,
        "NUM_RESOLUTIONS": 5,
        "TILE_SIZE": (107, 128, 128),
        "EXPECTED_TILE_CONFIG": [
            "dim=3",
            "00;;(0,0,0)",
            "01;;(0,115,0)",
            "04;;(0,0,0)",
            "05;;(0,115,0)",
            "10;;(115,0,0)",
            "11;;(115,115,0)",
            "14;;(115,0,0)",
            "15;;(115,115,0)",
        ],
        "EXPECTED_TILE_POSITIONS": [
            [3, 4, 2],
            [2, 120, 0],
            [3, 4, 2],
            [2, 120, 0],
            [6, 7, 118],
            [5, 123, 116],
            [6, 7, 118],
            [5, 123, 116],
        ],
        "CHANNELS": ["561 nm", "647 nm"],
        "PIXEL_SIZE_XY": 4.08,
        "PIXEL_SIZE_Z": 5.0,
        "MOCK_IMAGEJ_PATH": imagej_path,
        "MOCK_XML_PATH": Path.home() / "stitching/Brain2/bdv.xml",
        "MOCK_TILE_CONFIG_PATH": Path.home()
        / "stitching/Brain2/bdv_tile_config.txt",
    }

    return constants_dict
