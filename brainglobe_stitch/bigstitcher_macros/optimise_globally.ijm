args = getArgument();
args = split(args, " ");
xml_path = args[0];
relative = args[1];
absolute = args[2];

print("Optimizing globally and applying shifts");
run(
    "Optimize globally and apply shifts ...",
    "select=" + xml_path +
    " process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations]" +
    " process_tile=[All tiles] process_timepoint=[All Timepoints] relative=" + relative + " absolute=" + absolute +
    " global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links]" +
    " fix_group_0-0,"
);

print("Done");
eval("script", "System.exit(0);");
