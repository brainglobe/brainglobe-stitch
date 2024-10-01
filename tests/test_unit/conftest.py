import shutil
from pathlib import Path
from platform import system

import pooch
import pytest

TEMP_DIR = Path.home() / "temp_test_directory"
TEST_DATA_URL = "https://gin.g-node.org/IgorTatarnikov/brainglobe-stitch-test/raw/master/brainglobe-stitch/brainglobe-stitch-test-data.zip"


@pytest.fixture(scope="session", autouse=True)
def download_test_data():
    """
    Downloads the test data and extracts it to a temporary directory.
    This fixture is session-scoped and automatically run once.

    Yields
    ------
    Path
        The path to the temporary directory.
    """
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
    """
    Returns the path to the clean test data directory.

    Yields
    ------
    Path
        The path to the clean test data directory.
    """
    yield TEMP_DIR


@pytest.fixture(scope="module")
def naive_bdv_directory():
    """
    Creates a temporary directory and copies the test data to it. This allows
    tests to modify the directory without affecting the original test data.

    The temporary directory is cleaned up after the tests are run.

    Yields
    ------
    Path
        The path to the temporary directory.
    """
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
    """
    Creates a temporary directory and copies the test data to it. This allows
    tests to modify the directory without affecting the original test data.

    This fixture is function-scoped.

    The temporary directory is cleaned up after the tests are run.

    Yields
    ------
    Path
        The path to the temporary directory.
    """
    test_dir = Path.home() / "quick_test_directory"

    shutil.copytree(
        TEMP_DIR,
        test_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("*_interpolated_bdv.h5"),
    )

    yield test_dir

    shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def imagej_path():
    """
    Returns the path to a mock ImageJ executable based on the operating
    system. This is used to mimic the behavior of the QFileDialog in macOS,
    it returns a path to the "Fiji.app" directory instead of the executable.

    Returns
    -------
    Path
        The path to the mock ImageJ executable.
    """
    if system() == "Windows":
        return Path.home() / "Fiji.app/ImageJ-win64.exe"
    elif system() == "Darwin":
        return Path.home() / "Fiji.app"
    else:
        return Path.home() / "Fiji.app/ImageJ-linux64"


@pytest.fixture(scope="module")
def test_constants(imagej_path):
    """
    Provides a dictionary of constants that's used in the tests.
    Contains metadata about the test data and the expected results.

    The tiles lie in one z-plane and are arranged in a 2x2 grid.
    There are 2 channels.
    Each tile is 128x128x107 pixels (x, y, z).
    The tiles overlap by 10% in x and y (13 pixels).
    The tiles are arranged in the following pattern:
    channel 0   | channel 1
     00 10          | 04 05
     01 11           | 14 15

    EXPECTED_TILE_CONFIG is based on test_data_bdv.xml
    The tile positions are in pixels in x, y, z order

    EXPECTED_TILE_POSITIONS are based on the stitch transforms found
    in test_data_bdv.xml
    The tile positions are in pixels in z, y, x order

    Parameters
    ----------
    imagej_path : Path
        The path to the mock ImageJ executable.

    Returns
    -------
    Dict
        A dictionary containing the constants.
    """
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
        "EXPECTED_FUSED_SHAPE": (113, 251, 246),
        "CHANNELS": ["561 nm", "647 nm"],
        "PIXEL_SIZE_XY": 4.08,
        "PIXEL_SIZE_Z": 5.0,
        "MOCK_IMAGEJ_PATH": imagej_path,
        # The file dialogue on macOS has a different behaviour
        # The selected file path is to the "Fiji.app" directory
        # The ImageJ executable is in "Fiji.app/Contents/MacOS/ImageJ-macosx"
        "MOCK_IMAGEJ_EXEC_PATH": (
            imagej_path / "Contents/MacOS/ImageJ-macosx"
            if system() == "Darwin"
            else imagej_path
        ),
        "MOCK_XML_PATH": Path.home() / "stitching/Brain2/bdv.xml",
        "MOCK_TILE_CONFIG_PATH": Path.home()
        / "stitching/Brain2/bdv_tile_config.txt",
        "DEFAULT_PYRAMID_DEPTH": 5,
        "DEFAULT_DOWNSAMPLE_FACTORS": (1, 2, 2),
        "DEFAULT_CHUNK_SHAPE": (128, 128, 128),
        "DEFAULT_COMPRESSION_METHOD": "zstd",
        "DEFAULT_COMPRESSION_LEVEL": 3,
    }

    return constants_dict
