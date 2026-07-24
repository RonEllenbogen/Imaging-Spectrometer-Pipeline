'''
This files is the shared mapping between camera PixelFormat strings (GenICam/pypylon
naming) and the numpy dtype each one is stored as.
'''
# Imports
import numpy as np

# Constants
# Numpy doesn't support 10 and 12-bit containers, so we store those as uint16 and leave the top 6 (4) bits unused.
PIXEL_FORMAT_INFO: dict[str, tuple[np.dtype, int]] = {
    "Mono8":  (np.dtype(np.uint8),  255),
    "Mono10": (np.dtype(np.uint16), 1023),
    "Mono12": (np.dtype(np.uint16), 4095),
    "Mono16": (np.dtype(np.uint16), 65535),
}

# Classes

# Functions
def dtype_for_pixel_format(pixel_format: str) -> np.dtype:

    '''
    Look up the numpy dtype a given camera PixelFormat is stored as.

    Parameters
    ----------
    pixel_format
        A GenICam/pypylon PixelFormat string, e.g. "Mono8" or "Mono12".
        Must be a key present in PIXEL_FORMAT_INFO.

    Returns
    -------
    np.dtype
        The numpy dtype used to store frames of this pixel format. Note
        this is the storage container, not necessarily the format's true
        bit depth -- e.g. "Mono12" returns np.uint16, since numpy has no
        native 12-bit integer type.
    '''

    return _lookup(pixel_format)[0]


def max_value_for_pixel_format(pixel_format: str) -> int:

    '''
    Look up the true maximum pixel value a given camera PixelFormat allows.

    Parameters
    ----------
    pixel_format
        A GenICam/pypylon PixelFormat string, e.g. "Mono8" or "Mono12".
        Must be a key present in PIXEL_FORMAT_INFO.

    Returns
    -------
    int
        The true maximum value a pixel can take under this format, e.g.
        4095 for "Mono12". This is distinct from the storage dtype's own
        maximum (np.iinfo(dtype).max) -- for Mono10/Mono12, the dtype is
        shared (uint16) but the true ceiling is lower than the container
        allows, and this is the value that should be used when clipping
        or validating pixel data, not the container's ceiling.
    '''

    return _lookup(pixel_format)[1]


def _lookup(pixel_format: str) -> tuple[np.dtype, int]:

    '''
    Shared lookup used internally by dtype_for_pixel_format() and
    max_value_for_pixel_format(), so both stay consistent with a single
    source of truth (PIXEL_FORMAT_INFO) rather than duplicating the
    dict-access-and-error-handling logic in each.

    Parameters
    ----------
    pixel_format
        A GenICam/pypylon PixelFormat string. Must be a key present in
        PIXEL_FORMAT_INFO.

    Returns
    -------
    tuple[np.dtype, int]
        A (dtype, true_max) pair for the given pixel format.
    '''

    try:
        return PIXEL_FORMAT_INFO[pixel_format]
    except KeyError:
        raise ValueError(
            f"unknown pixel_format {pixel_format!r}; must be one of {list(PIXEL_FORMAT_INFO)}"
        ) from None

#if __name__ == "__main__":
    