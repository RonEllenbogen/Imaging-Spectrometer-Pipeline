"""
This file contains various helper functions used throughout the project
"""

# Imports

from pathlib import Path
import yaml

# Constants

# Classes

# Functions

def load_config(config_path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    with open(config_path, "r") as f:
        return yaml.safe_load(f)