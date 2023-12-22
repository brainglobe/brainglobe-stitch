from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mesoSPIM-stitcher")
except PackageNotFoundError:
    # package is not installed
    pass
