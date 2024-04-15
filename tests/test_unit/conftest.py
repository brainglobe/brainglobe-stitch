import shutil
from pathlib import Path

import pooch
import pytest

TEMP_DIR = Path.home() / "temp_test_directory"
TEST_DATA_URL = "https://gin.g-node.org/IgorTatarnikov/brainglobe-stitch-test/raw/master/brainglobe-stitch/brainglobe-stitch-test-data.zip"


@pytest.fixture(scope="session", autouse=True)
def download_test_data():
    TEMP_DIR.mkdir(exist_ok=True)
    pooch.retrieve(
        TEST_DATA_URL,
        known_hash=None,
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
