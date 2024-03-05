import shutil
from pathlib import Path

import h5py
import pytest

from brainglobe_stitch.file_utils import create_pyramid_bdv_h5

TEMP_DIR = Path("./temp_directory")
NUM_RESOLUTIONS = 5
NUM_SLICES = 8


@pytest.fixture
def copy_naive_bdv_directory():
    test_dir = Path("./test_directory")

    shutil.copytree(TEMP_DIR, test_dir, dirs_exist_ok=True)

    yield test_dir

    shutil.rmtree(test_dir)


def test_create_pyramid_bdv_h5(copy_naive_bdv_directory):
    h5_path = copy_naive_bdv_directory / "test_data_original_bdv.h5"
    with h5py.File(h5_path, "r") as f:
        num_tiles = len(f["t00000"].keys())
        tile_names = f["t00000"].keys()

        for tile_name in tile_names:
            assert f[f"{tile_name}/resolutions"].shape[0] == 1
            assert f[f"{tile_name}/subdivisions"].shape[0] == 1
            assert len(f[f"t00000/{tile_name}"].keys()) == 1

    num_done = 1
    for progress in create_pyramid_bdv_h5(h5_path, yield_progress=True):
        assert progress == int(100 * num_done / num_tiles)
        num_done += 1

    with h5py.File(h5_path, "r") as f_out, h5py.File(
        TEMP_DIR / "test_data_original_bdv.h5", "r"
    ) as f_in:
        # Check that the number of groups/datasets in the parent is unchanged
        assert len(f_out.keys()) == len(f_in.keys())
        assert len(f_out["t00000"].keys()) == len(f_in["t00000"].keys())

        tile_names = f_in["t00000"].keys()

        for tile_name in tile_names:
            assert (
                f_out[f"{tile_name}/resolutions"].shape[0] == NUM_RESOLUTIONS
            )
            assert (
                f_out[f"{tile_name}/subdivisions"].shape[0] == NUM_RESOLUTIONS
            )
            assert len(f_out[f"t00000/{tile_name}"].keys()) == NUM_RESOLUTIONS
