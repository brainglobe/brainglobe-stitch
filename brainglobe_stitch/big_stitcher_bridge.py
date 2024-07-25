import subprocess
from pathlib import Path
from platform import system


def run_big_stitcher(
    fiji_path: Path,
    xml_path: Path,
    tile_config_path: Path,
    big_stitcher_log: Path,
    selected_channel: str = "488 nm",
    downsample_x: int = 4,
    downsample_y: int = 4,
    downsample_z: int = 1,
    min_r: float = 0.7,
    max_r: float = 1.0,
    max_shift_x: float = 100.0,
    max_shift_y: float = 100.0,
    max_shift_z: float = 100.0,
) -> None:
    """
    Run the BigStitcher ImageJ macro in headless mode. Output is captured
    and returned as part of the subprocess.CompletedProcess.

    Parameters
    ----------
    fiji_path: Path
        The path to the ImageJ executable.
    xml_path: Path
        The path to the BigDataViewer XML file.
    tile_config_path: Path
        The path to the BigStitcher tile configuration file.
    all_channels: bool
        Whether to stitch based on all channels.
    selected_channel: int
        The channel on which to base the stitching.
    downsample_x: int
        The downsample factor in the x-dimension for the stitching.
    downsample_y: int
        The downsample factor in the y-dimension for the stitching.
    downsample_z: int
        The downsample factor in the z-dimension for the stitching.
    min_r: float
        The minimum correlation coefficient for a link to be accepted.
        Default is 0.7.
    max_r: float
        The maximum correlation coefficient for a link to be accepted.
        Default is 1.0.
    max_shift_x: float
        The maximum shift in the x-dimension for a link to be accepted.
        Default is 100.0.
    max_shift_y: float
        The maximum shift in the y-dimension for a link to be accepted.
        Default is 100.0.
    max_shift_z: float
        The maximum shift in the z-dimension for a link to be accepted.
        Default is 100.0.

    Returns
    -------
    None

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess returns a non-zero exit status.
    """
    result = calculate_pairwise_links(
        fiji_path,
        xml_path,
        selected_channel,
        downsample_x,
        downsample_y,
        downsample_z,
    )
    write_big_stitcher_log(
        result, big_stitcher_log, "Calculating pairwise links"
    )

    result = filter_links(
        fiji_path,
        xml_path,
        min_r,
        max_r,
        max_shift_x,
        max_shift_y,
        max_shift_z,
    )
    write_big_stitcher_log(result, big_stitcher_log, "Filtering links")

    result = optimise_globally(fiji_path, xml_path)
    write_big_stitcher_log(
        result, big_stitcher_log, "Optimising links globally"
    )

    return


def load_tile_config_file(
    fiji_path: Path,
    xml_path: Path,
    tile_config_path: Path,
) -> subprocess.CompletedProcess:
    """
    Load the tile configuration file into the BigStitcher plugin
    in ImageJ.

    Parameters
    ----------
    fiji_path: Path
        The path to the FIJI executable.
    xml_path: Path
        The path to the BigDataViewer XML file.
    tile_config_path: Path
        The path to the tile configuration file.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess returns a non-zero exit status.
    """
    macro_directory = Path(__file__).resolve().parent / "bigstitcher_macros"

    fiji_path = resolve_fiji_path(fiji_path)

    stitch_macro_path = macro_directory / "load_tile_config.ijm"

    command = (
        f"{fiji_path} --ij2"
        f" --headless -macro {stitch_macro_path} "
        f'"{xml_path} {tile_config_path}"'
    )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True, shell=True
    )

    return result


def calculate_pairwise_links(
    fiji_path: Path,
    xml_path: Path,
    selected_channel: str = "488 nm",
    downsample_x: int = 4,
    downsample_y: int = 4,
    downsample_z: int = 1,
):
    """
    Calculate pairwise links between tiles using the BigStitcher plugin.

    Parameters
    ----------
    fiji_path: Path
        The path to the FIJI executable.
    xml_path: Path
        The path to the BigDataViewer XML file.
    selected_channel: str
        The channel on which to base the stitching.
    downsample_x: int
        The downsample factor in the x-dimension for the stitching.
    downsample_y: int
        The downsample factor in the y-dimension for the stitching.
    downsample_z: int
        The downsample factor in the z-dimension for the stitching.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess returns a non-zero exit status.
    """
    macro_directory = Path(__file__).resolve().parent / "bigstitcher_macros"

    fiji_path = resolve_fiji_path(fiji_path)

    if selected_channel == "All channels":
        stitch_macro_path = (
            macro_directory / "calculate_pairwise_all_channel.ijm"
        )
        command = (
            f"{fiji_path} --ij2"
            f" --headless -macro {stitch_macro_path} "
            f'"{xml_path} {downsample_x} {downsample_y} {downsample_z}"'
        )
    else:
        stitch_macro_path = (
            macro_directory / "calculate_pairwise_per_channel.ijm"
        )

        # Replace spaces with underscores to avoid issues with passing
        # command line arguments to the macro
        if len(selected_channel.split(" ")) > 1:
            selected_channel = selected_channel.replace(" ", "_")

        command = (
            f"{fiji_path} --ij2"
            f" --headless -macro {stitch_macro_path} "
            f'"{xml_path} {selected_channel} '
            f'{downsample_x} {downsample_y} {downsample_z}"'
        )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True, shell=True
    )

    return result


def filter_links(
    fiji_path: Path,
    xml_path: Path,
    min_r: float = 0.7,
    max_r: float = 1.0,
    max_shift_x: float = 100.0,
    max_shift_y: float = 100.0,
    max_shift_z: float = 100.0,
):
    """
    Filter pairwise links between tiles using the BigStitcher plugin in FIJI.

    Parameters
    ----------
    fiji_path: Path
        The path to the FIJI executable.
    xml_path: Path
        The path to the BigDataViewer XML file.
    min_r: float
        The minimum correlation coefficient for a link to be accepted.
    max_r: float
        The maximum Pearson coefficient for a link to be accepted.
    max_shift_x: float
        The maximum shift in the x-dimension for a link to be accepted.
    max_shift_y: float
        The maximum shift in the y-dimension for a link to be accepted.
    max_shift_z: float
        The maximum shift in the z-dimension for a link to be accepted.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess returns a non-zero exit status.
    """
    macro_directory = Path(__file__).resolve().parent / "bigstitcher_macros"

    fiji_path = resolve_fiji_path(fiji_path)

    stitch_macro_path = macro_directory / "filter_links.ijm"

    command = (
        f"{fiji_path} --ij2"
        f" --headless -macro {stitch_macro_path} "
        f'"{xml_path} {min_r} {max_r} '
        f'{max_shift_x} {max_shift_y} {max_shift_z}"'
    )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True, shell=True
    )

    return result


def optimise_globally(
    fiji_path: Path,
    xml_path: Path,
    relative: float = 2.5,
    absolute: float = 3.5,
):
    """
    Globally optimise the pairwise links between tiles using the BigStitcher
    plugin in FIJI.

    Parameters
    ----------
    fiji_path: Path
        The path to the FIJI executable.
    xml_path: Path
        The path to the BigDataViewer XML file.
    relative: float
        The relative threshold for the optimisation.
    absolute: float
        The absolute threshold for the optimisation.

    Returns
    -------
    subprocess.CompletedProcess
        The result of the subprocess run.

    Raises
    ------
    subprocess.CalledProcessError
        If the subprocess returns a non-zero exit status.
    """
    macro_directory = Path(__file__).resolve().parent / "bigstitcher_macros"

    fiji_path = resolve_fiji_path(fiji_path)

    stitch_macro_path = macro_directory / "optimise_globally.ijm"

    command = (
        f"{fiji_path} --ij2"
        f" --headless -macro {stitch_macro_path} "
        f'"{xml_path} {relative} {absolute}"'
    )

    result = subprocess.run(
        command, capture_output=True, text=True, check=True, shell=True
    )

    return result


def resolve_fiji_path(fiji_path: Path) -> Path:
    """
    Resolve the path to the Fiji executable. This is necessary because
    the naive selection of the Fiji app in macOS does not return a path
    to the executable.

    Parameters
    ----------
    fiji_path: Path
        The path to the Fiji executable.

    Returns
    -------
    Path
        The resolved path to the Fiji executable.
    """
    platform = system()
    if platform.startswith("Darwin"):
        fiji_path = fiji_path / "Contents/MacOS/ImageJ-macosx"

    return fiji_path


def write_big_stitcher_log(
    result: subprocess.CompletedProcess,
    output_file: Path,
    task_name: str,
):
    """
    Write the output of a BigStitcher subprocess to a log file.

    Parameters
    ----------
    result: subprocess.CompletedProcess
        The result of the BigStitcher subprocess.
    output_file: Path
        The path to the output log file.
    task_name: str
        The name of the task being run.
    """
    with open(output_file, "w+") as f:
        f.write(f"{task_name}\n")
        f.write(f"Command: {result.args}\n")
        f.write(f"Return code: {result.returncode}\n")
        f.write(f"Output: {result.stdout}\n")
        f.write(f"Error: {result.stderr}\n")

    print(task_name)
    print(f"Command: {result.args}")
    print(f"Return code: {result.returncode}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")

    return
