from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brainglobe_stitch")
except PackageNotFoundError:
    # package is not installed
    pass
