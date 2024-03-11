import dask.array as da
import numpy as np
import pytest

from brainglobe_stitch.tile import Overlap, Tile


@pytest.fixture
def generate_tile_data():
    image_data = da.random.randint(0, 5000, (256, 256, 256), dtype=np.int16)
    resolution_array = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]])

    image_pyramid = [image_data]

    for i in range(1, len(resolution_array)):
        image_pyramid.append(
            image_data[
                :: resolution_array[i][0],
                :: resolution_array[i][1],
                :: resolution_array[i][2],
            ]
        )

    return image_pyramid, resolution_array


@pytest.fixture
def generate_overlap(generate_tile_data):
    tile_i = Tile(
        "test_1", 0, {"channel": 0, "tile": 0, "illumination": 0, "angle": 0}
    )
    tile_j = Tile(
        "test_2", 1, {"channel": 0, "tile": 1, "illumination": 0, "angle": 0}
    )

    # Set the position such that there is an overlap of size [246, 26, 251]
    tile_i.position = np.array([10, 0, 10])
    tile_j.position = np.array([0, 230, 15])

    tile_i.data_pyramid, tile_i.resolution_pyramid = generate_tile_data
    tile_j.data_pyramid, tile_j.resolution_pyramid = generate_tile_data

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