from typing import Dict, List

import numpy as np

from brainglobe_stitch.tile import Tile


def calculate_thresholds(
    tiles: List[Tile], pyramid_level: int = 3
) -> Dict[str, float]:
    middle_slice_index = tiles[0].data_pyramid[pyramid_level].shape[0] // 2
    thresholds: Dict[str, List[float]] = {}

    for tile in tiles:
        tile_data = tile.data_pyramid[pyramid_level]
        curr_threshold = np.percentile(
            tile_data[middle_slice_index].ravel(), 99
        ).compute()[0]
        threshold_list = thresholds.get(tile.channel_name, [])
        threshold_list.append(curr_threshold)
        thresholds[tile.channel_name] = threshold_list

    final_thresholds: Dict[str, float] = dict(
        (channel, np.max(thresholds.get(channel))) for channel in thresholds
    )

    return final_thresholds
