args = getArgument();
args = split(args, " ");
xml_path = args[0];
selected_channel_combo = args[1];
down_sampleX = args[2];
down_sampleY = args[3];
down_sampleZ = args[4];

selected_channel_array = split(selected_channel_combo, "_");
selected_channel = selected_channel_array[0];

for (i = 1; i < selected_channel_array.length; i++)
    selected_channel = selected_channel + " " + selected_channel_array[i];

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
