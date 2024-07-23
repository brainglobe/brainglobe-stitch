args = getArgument();
args = split(args, " ");
xml_path = args[0];
tile_path = args[1];
all_channels = args[2];
selected_channel_combo = args[3];
down_sampleX = args[4];
down_sampleY = args[5];
down_sampleZ = args[6];

selected_channel_array = split(selected_channel_combo, "_");
selected_channel = selected_channel_array[0];

for (i = 1; i < selected_channel_array.length; i++)
    selected_channel = selected_channel + " " + selected_channel_array[i];


print("Stitching " + xml_path);
print("Loading TileConfiguration from " + tile_path)
run(
    "Load TileConfiguration from File...",
    "browse=" + xml_path +
    " select=" + xml_path +
    " tileconfiguration=" + tile_path +
    " use_pixel_units keep_metadata_rotation"
);

print("Calculating pairwise shifts");
if (all_channels == 1) {
    run(
        "Calculate pairwise shifts ...",
        "select=" + xml_path +
        " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
        " process_tile=[All tiles] process_timepoint=[All Timepoints]" +
        " method=[Phase Correlation] channels=[Average Channels]" +
        " downsample_in_x=" + down_sampleX +
        " downsample_in_y=" + down_sampleY +
        " downsample_in_z=" + down_sampleZ
    );
} else {
    run(
        "Calculate pairwise shifts ...",
        "select=" + xml_path +
        " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
        " process_tile=[All tiles] process_timepoint=[All Timepoints]" +
        " method=[Phase Correlation] channels=[use Channel " + selected_channel + "]" +
        " downsample_in_x=" + down_sampleX +
        " downsample_in_y=" + down_sampleY +
        " downsample_in_z=" + down_sampleZ
    );
}

print("Optimizing globally and applying shifts");
run(
    "Optimize globally and apply shifts ...",
    "select=" + xml_path +
    " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
    " process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500" +
    " global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links]" +
    " fix_group_0-0,"
);

print("Done");
eval("script", "System.exit(0);");
