from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mesospim_stitcher")
except PackageNotFoundError:
    # package is not installed
    pass
