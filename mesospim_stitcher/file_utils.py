from pathlib import Path

HEADERS = [
    "[POSITION]",
    "[ETL PARAMETERS]",
    "[GALVO PARAMETERS]",
    "[CAMERA PARAMETERS]",
]


def write_big_stitcher_tile_config(meta_file_name: Path) -> list[dict]:
    tile_metadata = parse_mesospim_metadata(meta_file_name)

    output_file = str(meta_file_name)[:-12] + "_tile_config.txt"

    first_channel = tile_metadata[0]["Laser"]
    num_channels = 1

    for metadata in tile_metadata[1:]:
        if metadata["Laser"] == first_channel:
            break
        else:
            num_channels += 1

    num_tiles = len(tile_metadata) // num_channels
    tile_xy_locations = []

    for i in range(0, len(tile_metadata), num_channels):
        curr_tile_dict = tile_metadata[i]

        x = round(curr_tile_dict["x_pos"] / curr_tile_dict["Pixelsize in um"])
        y = round(curr_tile_dict["y_pos"] / curr_tile_dict["Pixelsize in um"])

        tile_xy_locations.append((x, y))

    relative_locations = [(0, 0)]

    for abs_tuple in tile_xy_locations[1:]:
        rel_tuple = (
            abs(abs_tuple[0] - tile_xy_locations[0][0]),
            abs(abs_tuple[1] - tile_xy_locations[0][1]),
        )
        relative_locations.append(rel_tuple)

    with open(output_file, "w") as f:
        f.write("dim=3\n")
        for i in range(len(tile_metadata)):
            f.write(
                f"{i};;"
                f"({relative_locations[i%num_tiles][0]},"
                f"{relative_locations[i%num_tiles][1]},0)\n"
            )

    return tile_metadata


def parse_mesospim_metadata(meta_file_name: Path):
    tile_metadata = []
    with open(meta_file_name, "r") as f:
        lines = f.readlines()
        curr_tile_metadata: dict[str, str | int | float] = {}

        for line in lines[3:]:
            line = line.strip()
            if line.startswith("[CFG"):
                tile_metadata.append(curr_tile_metadata)
                curr_tile_metadata = {}
            elif line in HEADERS:
                continue
            elif not line:
                continue
            else:
                split_line = line.split("]")
                value = split_line[1].strip()
                if value.isdigit():
                    curr_tile_metadata[split_line[0][1:]] = int(value)
                else:
                    try:
                        curr_tile_metadata[split_line[0][1:]] = float(value)
                    except ValueError:
                        curr_tile_metadata[split_line[0][1:]] = value

    tile_metadata.append(curr_tile_metadata)
    return tile_metadata
