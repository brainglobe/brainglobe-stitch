import shutil
from pathlib import Path

import pytest

TEMP_DIR = Path("./temp_directory")


@pytest.fixture(scope="module")
def naive_bdv_directory():
    test_dir = Path("./test_directory")

    shutil.copytree(
        TEMP_DIR,
        test_dir,
        dirs_exist_ok=True,
    )
    # Create UNIX style hidden files that should be ignored
    (test_dir / ".test_data_bdv.h5").touch()
    (test_dir / ".test_data_bdv.h5_meta.txt").touch()
    (test_dir / ".test_data_bdv.h5_meta.txt").touch()

    yield test_dir

    shutil.rmtree(test_dir)
