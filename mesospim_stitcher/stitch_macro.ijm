run("Load TileConfiguration from File...",
    "browse=D:/TiledDataset/2.5x_tile/2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv3.xml " +
    "select=D:/TiledDataset/2.5x_tile//2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv3.xml " +
    "tileconfiguration=D:/TiledDataset/2.5x_tile//2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv_tile_config.txt " +
    "use_pixel_units keep_metadata_rotation");

run("Calculate pairwise shifts ...",
    "select=D:/TiledDataset/2.5x_tile//2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv3.xml " +
    "process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] " +
    "process_tile=[All tiles] process_timepoint=[All Timepoints] " +
    "method=[Phase Correlation] channels=[Average Channels] " +
    "downsample_in_x=4 downsample_in_y=4 downsample_in_z=4");

run("Optimize globally and apply shifts ...",
    "select=D:/TiledDataset/2.5x_tile//2.5x_tile_igor_rightonly_Mag2.5x_ch488_ch561_ch647_bdv3.xml " +
    "process_angle=[All angles] process_channel=[All channels] process_illumination=[All illuminations] " +
    "process_tile=[All tiles] process_timepoint=[All Timepoints] relative=2.500 absolute=3.500 " +
    "global_optimization_strategy=[Two-Round using Metadata to align unconnected Tiles and iterative dropping of bad links] " +
    "fix_group_0-0,");
