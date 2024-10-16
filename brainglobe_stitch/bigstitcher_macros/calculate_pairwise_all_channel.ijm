args = getArgument();
args = split(args, " ");
xml_path = args[0];
down_sampleX = args[1];
down_sampleY = args[2];
down_sampleZ = args[3];

print("Calculating pairwise shifts");
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

print("Done");
eval("script", "System.exit(0);");
