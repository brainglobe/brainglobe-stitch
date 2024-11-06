from brainglobe_stitch.tile import Tile


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
    assert tile.channel_name == ""
    assert tile.illumination_id == illumination_id
    assert tile.angle == angle
