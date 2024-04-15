from pathlib import Path

import h5py
import numpy as np

HEADERS = [
    "[POSITION]",
    "[ETL PARAMETERS]",
    "[GALVO PARAMETERS]",
    "[CAMERA PARAMETERS]",
]


def create_pyramid_bdv_h5(
    input_file: Path,
    yield_progress: bool = False,
):
    """
    Create a resolution pyramid for a Big Data Viewer HDF5 file.

    Parameters
    ----------
    input_file: Path
        The path to the input HDF5 file.
    yield_progress: bool, optional
        Whether to yield progress.
    """
    resolutions_array = np.array(
        [[1, 1, 1], [2, 2, 2], [4, 4, 4], [8, 8, 8], [16, 16, 16]]
    )

    subdivisions_array = np.array(
        [[32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16], [32, 32, 16]]
    )

    with h5py.File(input_file, "r+") as f:
        data_group = f["t00000"]
        num_done = 0
        num_slices = len(data_group)

        for curr_slice in data_group:
            # Delete the old resolutions and subdivisions datasets
            del f[f"{curr_slice}/resolutions"]
            del f[f"{curr_slice}/subdivisions"]
            f[f"{curr_slice}/resolutions"] = resolutions_array
            f[f"{curr_slice}/subdivisions"] = subdivisions_array

            grp: h5py.Group = f[f"t00000/{curr_slice}"]
            for i in range(1, resolutions_array.shape[0]):
                downsampling_factors = (
                    resolutions_array[i] // resolutions_array[i - 1]
                )
                prev_resolution = grp[f"{i - 1}/cells"]
                # Add 1 to account for odd dimensions
                grp.require_dataset(
                    f"{i}/cells",
                    dtype=prev_resolution.dtype,
                    shape=(
                        (prev_resolution.shape[0] + 1)
                        // downsampling_factors[2],
                        (prev_resolution.shape[1] + 1)
                        // downsampling_factors[1],
                        (prev_resolution.shape[2] + 1)
                        // downsampling_factors[0],
                    ),
                )
                grp[f"{i}/cells"][...] = prev_resolution[
                    :: downsampling_factors[2],
                    :: downsampling_factors[1],
                    :: downsampling_factors[0],
                ]

            num_done += 1

            if yield_progress:
                yield int(100 * num_done / num_slices)
