# brainglobe-stitch

Stitching tiled 3D light-sheet data in napari

----------------------------------

A [napari] plugin for stitching tiled 3D acquisitions from a [mesoSPIM] light-sheet microscope.
The plugin utilises [BigStitcher] to align the tiles and napari to visualise the stitched data.

## Installation

We strongly recommend to use a virtual environment manager (like `conda` or `venv`). The installation instructions below
will not specify the Qt backend for napari, and you will therefore need to install that separately. Please see the
[`napari` installation instructions](https://napari.org/stable/tutorials/fundamentals/installation.html) for further advice on this.

To install latest development version :

    pip install git+https://github.com/brainglobe/brainglobe-stitch.git

This plugin requires Fiji to be installed on your system. You can download Fiji [here](https://imagej.net/Fiji/Downloads).

The BigStitcher plugin must be installed in Fiji. Please follow the instructions [here](https://imagej.net/plugins/bigstitcher/#download).


## License

Distributed under the terms of the [BSD-3] license,
"brainglobe-stitch" is free and open source software

## Seeking help or contributing
We are always happy to help users of our tools, and welcome any contributions. If you would like to get in contact with us for any reason, please see the [contact page of our website](https://brainglobe.info/contact.html).


[napari]: https://napari.org
[mesoSPIM]: https://www.mesospim.org/
[BigStitcher]: https://imagej.net/BigStitcher
