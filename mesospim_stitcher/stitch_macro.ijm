args = getArgument();
args = split(args, " ");
xmlPath = args[0];
tilePath = args[1]

print("Stitching " + xmlPath);
print("Loading TileConfiguration from " + tilePath)
run("Load TileConfiguration from File...",
    "browse=" + xmlPath +
    " select=" + xmlPath +
    " tileconfiguration=" + tilePath +
    " use_pixel_units keep_metadata_rotation");

print("Calculating pairwise shifts");
run("Calculate pairwise shifts ...",
    "select=" + xmlPath +
    " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
    " process_tile=[All tiles] process_timepoint=[All Timepoints]" +
    " method=[Phase Correlation] channels=[Average Channels]" +
    " downsample_in_x=4 downsample_in_y=4 downsample_in_z=4");

print("Optimizing globally and applying shifts")
run("Optimize globally and apply shifts ...",
    "select=" + xmlPath +
    " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
    " process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500" +
    " global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links]" +
    " fix_group_0-0,");

print("Done");
eval("script", "System.exit(0);");
