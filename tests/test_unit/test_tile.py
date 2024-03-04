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


def test_overlap_init(generate_tile_data):
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
    overlap_size = np.array([246, 26, 251])

    overlap = Overlap(overlap_coordinates, overlap_size, tile_i, tile_j)

    assert overlap.coordinates.shape == (3,)
    assert overlap.size[0].shape == (3,)
    assert len(overlap.size) == len(tile_j.resolution_pyramid)
    assert len(overlap.local_coordinates) == len(tile_j.resolution_pyramid)


def test_get_local_coordinates(generate_tile_data):
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

    overlap = Overlap(overlap_coordinates, overlap_size, tile_i, tile_j)

    expected_i_position = overlap_coordinates - tile_i.position
    expected_j_position = overlap_coordinates - tile_j.position

    assert (overlap.local_coordinates[0][0] == expected_i_position).all()
    assert (overlap.local_coordinates[0][1] == expected_j_position).all()


def test_extract_tile_overlaps(generate_tile_data):
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

    overlap = Overlap(overlap_coordinates, overlap_size, tile_i, tile_j)

    i_local = overlap.local_coordinates[0][0]
    j_local = overlap.local_coordinates[0][1]

    i_overlap, j_overlap = overlap.extract_tile_overlaps(0)

    assert i_overlap.shape == tuple(overlap_size)
    assert j_overlap.shape == tuple(overlap_size)
    assert (
        i_overlap
        == tile_i.data_pyramid[0][
            i_local[0] : i_local[0] + overlap_size[0],
            i_local[1] : i_local[1] + overlap_size[1],
            i_local[2] : i_local[2] + overlap_size[2],
        ]
    ).all()
    assert (
        j_overlap
        == tile_j.data_pyramid[0][
            j_local[0] : j_local[0] + overlap_size[0],
            j_local[1] : j_local[1] + overlap_size[1],
            j_local[2] : j_local[2] + overlap_size[2],
        ]
    ).all()
