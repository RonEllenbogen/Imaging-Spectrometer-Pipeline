"""
Generic save/load for one calibration artifact (an array plus the
dataclass record describing how it was built) as a single .npz file.
np.savez only -- never pickle -- so loading an artifact built by a
different process, machine, or older version of this codebase can never
execute arbitrary code; only known numeric/string fields on the record
dataclass are ever read back.

Lives in shared/ (not sensor/) because every calibration subpackage
(sensor/, eventually spectral/, spatial/) needs exactly this pattern --
one array, one small metadata record, save it, load it back later in a
different process. Per-artifact-type save_*()/load_*() wrappers (e.g.
sensor/baseline.py's save_baseline()/load_baseline()) exist for a
discoverable, type-specific API and to do artifact-specific things like
logging a loaded record's age -- this module only provides the shared
mechanism they're built on.
"""

# Imports

import dataclasses
from pathlib import Path
from typing import TypeVar

import numpy as np

# Constants

_ARRAY_KEY = "array"
_RECORD_FIELD_PREFIX = "record__"

_RecordT = TypeVar("_RecordT")

# Classes

# Functions


def _with_npz_suffix(path: str | Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def save_artifact(path: str | Path, array: np.ndarray, record) -> None:

    '''
    Saves one calibration artifact to a single .npz file, overwriting
    whatever was already there. Creates path's parent directory if it
    doesn't exist yet.

    Parameters
    ----------
    path
        Destination file. A ".npz" suffix is added if not already
        present.
    array
        The artifact itself (baseline, flat field, bad-pixel mask, ...).
    record
        An instance of any frozen dataclass of simple (numeric/string)
        fields describing the artifact's build-time metadata (e.g.
        CalibrationRecord) -- every field is stored alongside array in
        the same archive.

    Returns
    -------
    None
    '''

    path = _with_npz_suffix(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    record_fields = {
        f"{_RECORD_FIELD_PREFIX}{field.name}": getattr(record, field.name)
        for field in dataclasses.fields(record)
    }
    np.savez(path, **{_ARRAY_KEY: array}, **record_fields)


def load_artifact(path: str | Path, record_cls: type[_RecordT]) -> tuple[np.ndarray, _RecordT]:

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
    tuple[np.ndarray, _RecordT]

    Raises
    ------
    FileNotFoundError
        If path doesn't exist -- propagated directly from np.load().
    '''

    path = _with_npz_suffix(path)

    with np.load(path) as data:
        array = data[_ARRAY_KEY]
        record_kwargs = {
            key[len(_RECORD_FIELD_PREFIX):]: data[key].item()
            for key in data.files
            if key.startswith(_RECORD_FIELD_PREFIX)
        }

    return array, record_cls(**record_kwargs)


__all__ = ["save_artifact", "load_artifact"]
