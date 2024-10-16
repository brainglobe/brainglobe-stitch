from typing import List, Tuple

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest

from brainglobe_stitch.tile import Overlap, Tile

TILE_DATA_HIGH = 100


def generate_tile(tile_data: da.array):
    resolution_array = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]])

    image_pyramid = [tile_data]

    for i in range(1, len(resolution_array)):
        image_pyramid.append(
            tile_data[
                :: resolution_array[i][0],
                :: resolution_array[i][1],
                :: resolution_array[i][2],
            ]
        )

    return image_pyramid, resolution_array


def generate_tile_data_random() -> Tuple[List[da.Array], npt.NDArray]:
    image_data = da.random.randint(0, 5000, (256, 256, 256), dtype=np.int16)

    return generate_tile(image_data)


def generate_tile_data_high() -> Tuple[List[da.Array], npt.NDArray]:
    image_data = da.ones((256, 256, 256), dtype=np.int16) * TILE_DATA_HIGH

    return generate_tile(image_data)


def generate_tile_data_low() -> Tuple[List[da.Array], npt.NDArray]:
    image_data = da.zeros((256, 256, 256), dtype=np.int16)

    return generate_tile(image_data)


@pytest.fixture
def generate_overlap():
    tile_i = Tile(
        "test_1", 0, {"channel": 0, "tile": 0, "illumination": 0, "angle": 0}
    )
    tile_j = Tile(
        "test_2", 1, {"channel": 0, "tile": 1, "illumination": 0, "angle": 0}
    )

    # Set the position such that there is an overlap of size [246, 26, 251]
    tile_i.position = np.array([10, 0, 10])
    tile_j.position = np.array([0, 230, 15])

    (
        tile_i.data_pyramid,
        tile_i.resolution_pyramid,
    ) = generate_tile_data_random()
    (
        tile_j.data_pyramid,
        tile_j.resolution_pyramid,
    ) = generate_tile_data_random()

    overlap_coordinates = np.array(
        [max(tile_i.position[i], tile_j.position[i]) for i in range(3)]
    )

    overlap_size = (
        tile_i.data_pyramid[0].shape
        - overlap_coordinates
        + np.array(
            [min(tile_i.position[i], tile_j.position[i]) for i in range(3)]
        )
    )

    return Overlap(overlap_coordinates, overlap_size, tile_i, tile_j)


def test_tile_init():
    tile_id = 0
    attribute_tile_id = 0
    channel_id = 0
    illumination_id = 0
    angle = 234.5

    attributes = {
        "channel": channel_id,
        "tile": attribute_tile_id,
        "illumination": illumination_id,
        "angle": angle,
    }

    tile = Tile("test", tile_id, attributes)

    assert tile.name == "test"
    assert tile.tile_id == tile_id
    assert tile.position == [0, 0, 0]
    assert len(tile.neighbours) == 0
    assert len(tile.data_pyramid) == 0
    assert len(tile.resolution_pyramid) == 0
    assert tile.channel_id == channel_id
    assert tile.channel_name is None
    assert tile.illumination_id == illumination_id
    assert tile.angle == angle


def test_overlap_init(generate_overlap):
    overlap = generate_overlap

    tile_j = overlap.tiles[1]

    assert overlap.coordinates.shape == (3,)
    assert overlap.size[0].shape == (3,)
    assert len(overlap.size) == len(tile_j.resolution_pyramid)
    assert len(overlap.local_coordinates) == len(tile_j.resolution_pyramid)


def test_get_local_coordinates(generate_overlap):
    overlap = generate_overlap
    tile_i, tile_j = overlap.tiles

    expected_i_position = overlap.coordinates - tile_i.position
    expected_j_position = overlap.coordinates - tile_j.position

    assert (overlap.local_coordinates[0][0] == expected_i_position).all()
    assert (overlap.local_coordinates[0][1] == expected_j_position).all()


def test_extract_tile_overlaps(generate_overlap):
    overlap = generate_overlap

    local_coord = overlap.local_coordinates[0]

    overlaps = overlap.extract_tile_overlaps(0)

    for i in range(len(local_coord)):
        assert overlaps[i].shape == tuple(overlap.size[0])
        assert (
            overlaps[i]
            == overlap.tiles[i].data_pyramid[0][
                local_coord[i][0] : local_coord[i][0] + overlap.size[0][0],
                local_coord[i][1] : local_coord[i][1] + overlap.size[0][1],
                local_coord[i][2] : local_coord[i][2] + overlap.size[0][2],
            ]
        ).all()


def test_replace_overlap_data(generate_overlap):
    overlap = generate_overlap

    new_data = da.zeros(overlap.size[0], dtype=np.int16)
    overlap.replace_overlap_data(0, new_data)
    res_level = 0

    for i in range(len(overlap.tiles)):
        assert (
            overlap.tiles[i].data_pyramid[res_level][
                overlap.local_coordinates[res_level][i][
                    0
                ] : overlap.local_coordinates[res_level][i][0]
                + overlap.size[res_level][0],
                overlap.local_coordinates[res_level][i][
                    1
                ] : overlap.local_coordinates[res_level][i][1]
                + overlap.size[res_level][1],
                overlap.local_coordinates[res_level][i][
                    2
                ] : overlap.local_coordinates[res_level][i][2]
                + overlap.size[res_level][2],
            ]
            == 0
        ).all()


@pytest.mark.parametrize(
    "resolution_level, tile_positions",
    [
        (0, [[10, 0, 10], [0, 245, 15]]),
        (0, [[10, 10, 0], [0, 15, 245]]),
        (0, [[0, 245, 15], [10, 0, 10]]),
        (0, [[0, 15, 245], [10, 10, 0]]),
        (1, [[10, 0, 10], [0, 245, 15]]),
        (1, [[10, 10, 0], [0, 15, 245]]),
        (1, [[0, 245, 15], [10, 0, 10]]),
        (1, [[0, 15, 245], [10, 10, 0]]),
        (2, [[10, 0, 10], [0, 245, 15]]),
        (2, [[10, 10, 0], [0, 15, 245]]),
        (2, [[0, 245, 15], [10, 0, 10]]),
        (2, [[0, 15, 245], [10, 10, 0]]),
    ],
)
def test_linear_interpolation(resolution_level, tile_positions):
    tile_zero = Tile(
        "test_1", 0, {"channel": 0, "tile": 0, "illumination": 0, "angle": 0}
    )
    tile_one = Tile(
        "test_2", 1, {"channel": 0, "tile": 1, "illumination": 0, "angle": 0}
    )

    # Set the position such that there is an overlap of size [246, 9, 251]
    tile_zero.position = np.array(tile_positions[0])
    tile_one.position = np.array(tile_positions[1])

    (
        tile_zero.data_pyramid,
        tile_zero.resolution_pyramid,
    ) = generate_tile_data_low()
    (
        tile_one.data_pyramid,
        tile_one.resolution_pyramid,
    ) = generate_tile_data_high()

    overlap_coordinates = np.array(
        [max(tile_zero.position[i], tile_one.position[i]) for i in range(3)]
    )

    overlap_size = (
        tile_zero.data_pyramid[0].shape
        - overlap_coordinates
        + np.array(
            [
                min(tile_zero.position[i], tile_one.position[i])
                for i in range(3)
            ]
        )
    )

    overlap = Overlap(overlap_coordinates, overlap_size, tile_zero, tile_one)

    overlap.linear_interpolation(
        resolution_level, tile_one.data_pyramid[resolution_level].shape
    )

    overlap_i, overlap_j = overlap.extract_tile_overlaps(resolution_level)

    assert (overlap_i == overlap_j).all()
    assert overlap_i.mean() == TILE_DATA_HIGH / 2
    assert overlap_j.mean() == TILE_DATA_HIGH / 2


@pytest.mark.parametrize(
    "resolution_level, tile_positions",
    [
        (0, [[10, 0, 0], [0, 245, 245]]),
        (1, [[10, 0, 0], [0, 245, 245]]),
        (2, [[10, 0, 0], [0, 245, 245]]),
    ],
)
def test_linear_interpolation_diagonal(resolution_level, tile_positions):
    tile_zero = Tile(
        "test_1", 0, {"channel": 0, "tile": 0, "illumination": 0, "angle": 0}
    )
    tile_one = Tile(
        "test_2", 1, {"channel": 0, "tile": 1, "illumination": 0, "angle": 0}
    )

    # Set the position such that there is an overlap of size [246, 9, 251]
    tile_zero.position = np.array(tile_positions[0])
    tile_one.position = np.array(tile_positions[1])

    (
        tile_zero.data_pyramid,
        tile_zero.resolution_pyramid,
    ) = generate_tile_data_low()
    (
        tile_one.data_pyramid,
        tile_one.resolution_pyramid,
    ) = generate_tile_data_high()

    overlap_coordinates = np.array(
        [max(tile_zero.position[i], tile_one.position[i]) for i in range(3)]
    )

    overlap_size = (
        tile_zero.data_pyramid[0].shape
        - overlap_coordinates
        + np.array(
            [
                min(tile_zero.position[i], tile_one.position[i])
                for i in range(3)
            ]
        )
    )

    overlap = Overlap(overlap_coordinates, overlap_size, tile_zero, tile_one)

    overlap.linear_interpolation(
        resolution_level, tile_one.data_pyramid[resolution_level].shape
    )

    overlap_i, overlap_j = overlap.extract_tile_overlaps(resolution_level)

    assert (overlap_i != overlap_j).all()
    assert overlap_i.mean() == 0
    assert overlap_j.mean() == TILE_DATA_HIGH
