"""
Generic save/load for one calibration artifact (one or more named arrays
plus the dataclass record describing how it was built) as a single .npz
file. np.savez only -- never pickle -- so loading an artifact built by a
different process, machine, or older version of this codebase can never
execute arbitrary code; only known numeric/string fields on the record
dataclass are ever read back.

Lives in shared/ (not sensor/) because every calibration subpackage
(sensor/, spectral/, spatial/) needs exactly this pattern -- one or more
arrays, one small metadata record, save it, load it back later in a
different process. Arrays are named (a dict) rather than a single
positional array so a multi-array artifact -- e.g. spectral/'s fit result,
which needs coefficients and coefficient_sigma at minimum -- fits the same
mechanism sensor/'s single-array artifacts (baseline, flat field,
bad-pixel mask) already use, without a second save/load path. Per-
artifact-type save_*()/load_*() wrappers (e.g. sensor/baseline.py's
save_baseline()/load_baseline()) exist for a discoverable, type-specific
API and to do artifact-specific things like logging a loaded record's age
-- this module only provides the shared mechanism they're built on.
"""

# Imports

import dataclasses
from pathlib import Path
from typing import TypeVar

import numpy as np

# Constants

_RECORD_FIELD_PREFIX = "record__"

_RecordT = TypeVar("_RecordT")

# Classes

# Functions


def _with_npz_suffix(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def save_artifact(path: str | Path, arrays: dict[str, np.ndarray], record) -> None:

    '''
    Saves one calibration artifact to a single .npz file, overwriting
    whatever was already there. Creates path's parent directory if it
    doesn't exist yet.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    arrays
        The artifact's array data, keyed by name (e.g. {"baseline": ...},
        or {"coefficients": ..., "coefficient_sigma": ...} for a
        multi-array artifact). Names must not start with "record__" --
        that prefix is reserved for record fields.
    record
        An instance of any frozen dataclass of simple (numeric/string)
        fields describing the artifact's build-time metadata (e.g.
        CalibrationRecord) -- every field is stored alongside arrays in
        the same archive.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any key in arrays starts with "record__".
    '''

    for key in arrays:
        if key.startswith(_RECORD_FIELD_PREFIX):
            raise ValueError(f"array key {key!r} must not start with {_RECORD_FIELD_PREFIX!r}")

    path = _with_npz_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record_fields = {
        f"{_RECORD_FIELD_PREFIX}{field.name}": getattr(record, field.name)
        for field in dataclasses.fields(record)
    }
    np.savez(path, **arrays, **record_fields)


def load_artifact(path: str | Path, record_cls: type[_RecordT]) -> tuple[dict[str, np.ndarray], _RecordT]:

    '''
    Loads one calibration artifact previously written by save_artifact().

    Parameters
    ----------
    path
        The file to load. A ".npz" suffix is added if not already
        present, matching save_artifact()'s own normalization.
    record_cls
        The dataclass type to reconstruct the record as -- must be
        field-compatible with whatever was passed to save_artifact()
        (ordinarily the same type). Reconstruction goes through
        record_cls's own constructor, so its usual validation
        (__post_init__) still runs against the loaded values.

    Returns
    -------
    tuple[dict[str, np.ndarray], _RecordT]
        The saved arrays, keyed by the same names passed to
        save_artifact(), and the reconstructed record.

    Raises
    ------
    FileNotFoundError
        If path doesn't exist -- propagated directly from np.load().
    '''

    path = _with_npz_suffix(path)

    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if not key.startswith(_RECORD_FIELD_PREFIX)}
        record_kwargs = {
            key[len(_RECORD_FIELD_PREFIX):]: data[key].item()
            for key in data.files
            if key.startswith(_RECORD_FIELD_PREFIX)
        }

    return arrays, record_cls(**record_kwargs)


__all__ = ["save_artifact", "load_artifact"]
