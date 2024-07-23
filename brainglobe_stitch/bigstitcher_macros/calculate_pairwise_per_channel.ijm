args = getArgument();
args = split(args, " ");
xml_path = args[0];
selected_channel = args[1];
down_sampleX = args[2];
down_sampleY = args[3];
down_sampleZ = args[4];

print("Calculating pairwise shifts");

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

print("Done");
eval("script", "System.exit(0);");
