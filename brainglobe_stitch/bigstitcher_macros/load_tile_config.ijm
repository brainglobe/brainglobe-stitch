args = getArgument();
args = split(args, " ");
xml_path = args[0];
tile_path = args[1];

print("Loading TileConfiguration from " + tile_path)
run(
    "Load TileConfiguration from File...",
    "browse=" + xml_path +
    " select=" + xml_path +
    " tileconfiguration=" + tile_path +
    " use_pixel_units keep_metadata_rotation"
);
