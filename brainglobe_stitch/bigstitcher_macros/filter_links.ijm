args = getArgument();
args = split(args, " ");
xml_path = args[0];
min_r = args[1];
max_r = args[2];
max_shift_in_x = args[3];
max_shift_in_y = args[4];
max_shift_in_z = args[5];

print("Filtering pairwise shifts");
run("Filter pairwise shifts ...",
    "browse=" + xml_path +
    " select=" + xml_path +
    " filter_by_link_quality" +
    " min_r=" + min_r +
    " max_r=" + max_r +
    " filter_by_shift_in_each_dimension" +
    " max_shift_in_x=" + max_shift_in_x +
    " max_shift_in_y=" + max_shift_in_y +
    " max_shift_in_z=" + max_shift_in_z +
    " max_displacement=0"
);

print("Done");
eval("script", "System.exit(0);");
