'''
This files is the shared mapping between camera PixelFormat strings (GenICam/pypylon
naming) and the numpy dtype each one is stored as.
'''
# Imports
import numpy as np

# Constants
# Numpy doesn't support 10 and 12-bit containers, so we store those as uint16 and leave the top 6 (4) bits unused.
PIXEL_FORMAT_DTYPES: dict[str, np.dtype] = {
    "Mono8": np.dtype(np.uint8),
    "Mono10": np.dtype(np.uint16),
    "Mono12": np.dtype(np.uint16),
    "Mono16": np.dtype(np.uint16),
}

# Classes

# Functions
def dtype_for_pixel_format(pixel_format: str) -> np.dtype:

    """
    Looks up the numpy dtype a given camera PixelFormat is stored as.
    Raises ValueError if pixel_format isn't one of the supported formats.
    """
    try:
        return PIXEL_FORMAT_DTYPES[pixel_format]
    except KeyError:
        raise ValueError(
            f"unknown pixel_format {pixel_format!r}; "
            f"must be one of {list(PIXEL_FORMAT_DTYPES)}"
        ) from None

#if __name__ == "__main__":
    