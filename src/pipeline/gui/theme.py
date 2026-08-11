"""
Shared visual constants for gui/ -- dark palette colors, spacing, and the
bundled UI font. Every gui/ widget module should import COLOR_*/SPACING_*/
MARGIN_DEFAULT and call load_bundled_font() from here rather than
hardcoding its own colors or fonts, so the whole GUI reads as one visual
system instead of a patchwork of screens styled independently.

Names and palette values here match what gui/calibration_screen.py and
gui/calibration_dialogs.py already import (built, screenshotted, and
reviewed before this module's font-loading logic was corrected below) --
kept stable deliberately, since changing them would mean re-touching
already-verified widget code for no visual benefit. COLOR_PLOT_* additions
are new, for gui/main_window.py's live-updating plot, which has no
precedent in this repo to match.

The bundled font is Latin Modern Roman (OFL-licensed, freely
redistributable -- see assets/fonts/README.md), not "Inter" -- corrected
from a first-draft placeholder that referenced a font file that was never
actually present. load_bundled_font() degrades gracefully to a generic
serif if the files are ever missing, so gui/ stays importable and runnable
regardless.
"""

# Imports

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase

# Constants

# -- Palette --------------------------------------------------------------
# Base surfaces, darkest to lightest.
COLOR_BACKGROUND = "#1c1c22"
COLOR_SURFACE = "#26262e"
COLOR_SURFACE_ALT = "#302f39"
COLOR_BORDER = "#3f3f4a"

# Text.
COLOR_TEXT_PRIMARY = "#e8e8ec"
COLOR_TEXT_SECONDARY = "#9a9aa6"
COLOR_TEXT_DISABLED = "#5c5c66"

# Accents. COLOR_ACCENT marks the primary "acquire from camera" action
# color, shared by baseline/flat-field/conversion-gain (all camera-driven).
# COLOR_ACCENT_ALT is reserved for spatial calibration specifically, since
# it has no camera interaction at all and is visually distinguished from
# the other four flows rather than forced into the same acquire -> build
# -> save visual language.
COLOR_ACCENT = "#5b8cff"
COLOR_ACCENT_ALT = "#c084fc"

# Status colors.
COLOR_SUCCESS = "#4ade80"
COLOR_WARNING = "#fbbf24"
COLOR_ERROR = "#f87171"

# Plot-specific colors, for gui/main_window.py's live-updating scatter/
# heatmap/strip-chart -- no equivalent need existed in calibration_screen.py,
# which has no plots.
COLOR_PLOT_GRID = "#3a3a44"
COLOR_PLOT_DATA = "#5b8cff"
COLOR_PLOT_FIT = "#ff9d5c"

# -- Spacing (px) -----------------------------------------------------------
SPACING_XS = 4
SPACING_SMALL = 8
SPACING_MEDIUM = 16
SPACING_LARGE = 24
SPACING_XL = 32

MARGIN_DEFAULT = 16

# -- Font --
# Bundled rather than assumed-installed, so rendering doesn't depend on
# whatever happens to be on the host machine.
FONT_FAMILY_NAME = "Latin Modern Roman"
_ASSETS_FONT_DIR = Path(__file__).parent / "assets" / "fonts"
_FONT_FILES = [
    _ASSETS_FONT_DIR / "LatinModernRoman-Regular.otf",
    _ASSETS_FONT_DIR / "LatinModernRoman-Bold.otf",
]

# Classes

# Functions


def group_box_stylesheet() -> str:

    '''
    Explicit background/text-color styling for a QGroupBox -- a
    gui/ screen's own setStyleSheet() doesn't reliably cascade into
    QGroupBox/QComboBox on every platform, leaving them a visibly
    different color from the rest of the page. Shared across screens
    (originally private copies in live_view.py) so every group box in
    the GUI stays visually identical without each screen re-deriving
    the same two-line stylesheet.
    '''

    return f"QGroupBox {{ background-color: {COLOR_BACKGROUND}; color: {COLOR_TEXT_PRIMARY}; }}"


def combo_box_stylesheet() -> str:

    '''See group_box_stylesheet() -- same rationale, for QComboBox.'''

    return f"QComboBox {{ background-color: {COLOR_BACKGROUND}; color: {COLOR_TEXT_PRIMARY}; }}"


def load_bundled_font(point_size: int = 10, *, bold: bool = False) -> QFont:

    '''
    Loads the bundled Latin Modern Roman font (registering it with Qt's
    font database on first use) and returns a QFont set to it, for plot
    axes/labels and data displays throughout gui/.

    Parameters
    ----------
    point_size
        Font point size to apply.
    bold
        Whether to request the bold weight.

    Returns
    -------
    QFont
        Set to FONT_FAMILY_NAME if the bundled font file(s) loaded
        successfully, otherwise a generic serif -- never raises just
        because the font file is missing.
    '''

    loaded_any = False
    for font_file in _FONT_FILES:
        if font_file.exists() and QFontDatabase.addApplicationFont(str(font_file)) != -1:
            loaded_any = True

    family = FONT_FAMILY_NAME if loaded_any else "Serif"
    font = QFont(family, point_size)
    font.setBold(bold)
    return font


__all__ = [
    "COLOR_BACKGROUND", "COLOR_SURFACE", "COLOR_SURFACE_ALT", "COLOR_BORDER",
    "COLOR_TEXT_PRIMARY", "COLOR_TEXT_SECONDARY", "COLOR_TEXT_DISABLED",
    "COLOR_ACCENT", "COLOR_ACCENT_ALT",
    "COLOR_SUCCESS", "COLOR_WARNING", "COLOR_ERROR",
    "COLOR_PLOT_GRID", "COLOR_PLOT_DATA", "COLOR_PLOT_FIT",
    "SPACING_XS", "SPACING_SMALL", "SPACING_MEDIUM", "SPACING_LARGE", "SPACING_XL",
    "MARGIN_DEFAULT",
    "FONT_FAMILY_NAME", "load_bundled_font",
    "group_box_stylesheet", "combo_box_stylesheet",
]
