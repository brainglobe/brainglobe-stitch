args = getArgument();
args = split(args, " ");
xmlPath = args[0];
tilePath = args[1];
allChannels = args[2];
selectedChannel = args[3];
downSampleX = args[4];
downSampleY = args[5];
downSampleZ = args[6];

print("Stitching " + xmlPath);
print("Loading TileConfiguration from " + tilePath)
run(
    "Load TileConfiguration from File...",
    "browse=" + xmlPath +
    " select=" + xmlPath +
    " tileconfiguration=" + tilePath +
    " use_pixel_units keep_metadata_rotation"
);

print("Calculating pairwise shifts");
if (allChannels == 1) {
    run(
        "Calculate pairwise shifts ...",
        "select=" + xmlPath +
        " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
        " process_tile=[All tiles] process_timepoint=[All Timepoints]" +
        " method=[Phase Correlation] channels=[Average Channels]" +
        " downsample_in_x=" + downSampleX +
        " downsample_in_y=" + downSampleY +
        " downsample_in_z=" + downSampleZ
    );
} else {
    run(
        "Calculate pairwise shifts ...",
        "select=" + xmlPath +
        " process_angle=[All angles] process_channel=[Single channel (Select from List)] process_illumination=[All illuminations]" +
        " process_tile=[All tiles] process_timepoint=[All Timepoints] processing_channel=[channel " + selectedChannel +" nm]" +
        " method=[Phase Correlation] channels=[Average Channels]" +
        " downsample_in_x=" + downSampleX +
        " downsample_in_y=" + downSampleY +
        " downsample_in_z=" + downSampleZ
    );
}

run(
    "Calculate pairwise shifts ...",
    "select=" + xmlPath +
    " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
    " process_tile=[All tiles] process_timepoint=[All Timepoints]" +
    " method=[Phase Correlation] channels=[Average Channels]" +
    " downsample_in_x=4 downsample_in_y=4 downsample_in_z=4"
);

print("Optimizing globally and applying shifts")
run(
    "Optimize globally and apply shifts ...",
    "select=" + xmlPath +
    " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
    " process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500" +
    " global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links]" +
    " fix_group_0-0,"
);

print("Done");
eval("script", "System.exit(0);");
