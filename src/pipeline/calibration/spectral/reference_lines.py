"""
Loads known spectral-line wavelengths for a chosen reference lamp from
data/reference/oriel_spectral_calibration_lamps.csv (a lab-wide table
covering every lamp available, not just the one this project uses) --
for line_matching.py's peak-to-reference-line matching search.

Uses Python's built-in csv module rather than pandas -- pandas isn't a
dependency anywhere else in this codebase, and a plain 3-column
(wavelength_nm, lamp, model_no) file doesn't need it.
"""

# Imports

import csv
from pathlib import Path

import numpy as np

# Constants

DEFAULT_REFERENCE_LINES_PATH = Path("data/reference/oriel_spectral_calibration_lamps.csv")

# Argon: this project's chosen reference lamp (see docs/project_state.md).
ARGON_LAMP_NAME = "Ar"

# Curated window, not the full Argon line list -- the CSV's Ar lines fall
# into two disjoint clusters (355-434nm and 641-843nm, nothing in
# between), and the CSV carries no intensity/strength data to predict
# which lines will actually be bright enough to detect. Chosen to be
# roughly symmetric about the 800nm central wavelength (Ti:Sapphire):
# 842.46nm is the largest-wavelength Ar line available, so 751.46nm
# (equally spaced below 800nm as 842.46nm is above the true midpoint of
# this window) was picked as the lower bound. Both endpoints are real
# lines in the CSV, included inclusively.
ARGON_MIN_WAVELENGTH_NM = 751.46
ARGON_MAX_WAVELENGTH_NM = 842.46

# Classes

# Functions


def load_reference_lines(
    lamp: str,
    wavelength_min_nm: float,
    wavelength_max_nm: float,
    path: str | Path = DEFAULT_REFERENCE_LINES_PATH,
) -> np.ndarray:

    '''
    Reads path's lamp reference-line table and returns the sorted,
    ascending wavelengths (nm) for lamp within
    [wavelength_min_nm, wavelength_max_nm] inclusive.

    Parameters
    ----------
    lamp
        Lamp name exactly as it appears in the CSV's "lamp" column (e.g.
        ARGON_LAMP_NAME) -- an exact string match, not a substring/regex
        one, since some lamp names in this table share prefixes with
        marker suffixes (e.g. "Hg*", "Ne dagger").
    wavelength_min_nm, wavelength_max_nm
        Inclusive wavelength range to keep.
    path
        CSV path. Defaults to DEFAULT_REFERENCE_LINES_PATH (relative to
        the repo root -- this codebase already requires running from
        there, see configs/default.yaml's own loading convention).

    Returns
    -------
    np.ndarray
        Sorted ascending wavelengths, in nanometres. Empty if no rows
        match lamp/range -- not an error, since an over-narrow range is
        a caller mistake to discover, not this function's to guess at.

    Raises
    ------
    FileNotFoundError
        If path doesn't exist.
    '''

    wavelengths = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["lamp"] != lamp:
                continue
            wavelength_nm = float(row["wavelength_nm"])
            if wavelength_min_nm <= wavelength_nm <= wavelength_max_nm:
                wavelengths.append(wavelength_nm)

    return np.sort(np.array(wavelengths, dtype=np.float64))


__all__ = [
    "load_reference_lines",
    "DEFAULT_REFERENCE_LINES_PATH",
    "ARGON_LAMP_NAME",
    "ARGON_MIN_WAVELENGTH_NM",
    "ARGON_MAX_WAVELENGTH_NM",
]
