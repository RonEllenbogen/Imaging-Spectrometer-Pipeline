"""
Shared visual constants for gui/ -- dark palette colors, spacing, and the
bundled UI font. Every gui/ widget module should import COLOR_*/SPACING_*/
MARGIN_DEFAULT and call load_bundled_font() from here rather than
hardcoding its own colors or fonts, so the whole GUI reads as one visual
system instead of a patchwork of screens styled independently.

PLACEHOLDER STATUS: this module did not exist yet when gui/calibration_screen.py
was built (neither did a bundled font asset under assets/, nor a GUI design
section in docs/project_state.md) -- both were expected prior art per this
module's originating task brief but were absent from the repo at build time.
This is a minimal, deliberately conservative implementation added to unblock
that work: a reasonable dark palette plus a load_bundled_font() that degrades
gracefully to a system sans-serif since no bundled .ttf exists under assets/
yet. Treat every constant here as provisional -- safe to retune once a real
design pass (and an actual bundled font file) happens, but keep the *names*
stable, since gui/ widget modules import them by name.
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
# it has no camera interaction at all and the task brief calls for
# visually distinguishing it from the other four flows rather than
# forcing it into the same acquire -> build -> save visual language.
COLOR_ACCENT = "#5b8cff"
COLOR_ACCENT_ALT = "#c084fc"

# Status colors.
COLOR_SUCCESS = "#4ade80"
COLOR_WARNING = "#fbbf24"
COLOR_ERROR = "#f87171"

# -- Spacing (px) -----------------------------------------------------------
SPACING_XS = 4
SPACING_SMALL = 8
SPACING_MEDIUM = 16
SPACING_LARGE = 24
SPACING_XL = 32

MARGIN_DEFAULT = 16

# Font family requested from the bundled font file, if/when one exists
# under assets/fonts/. Falls back to a system sans-serif via Qt's normal
# family-substitution behavior when no such file is found.
_BUNDLED_FONT_FAMILY = "Inter"
_ASSETS_FONT_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"

# Classes

# Functions


def load_bundled_font(point_size: int = 10, *, bold: bool = False) -> QFont:

    '''
    Load the GUI's bundled font, registering it with Qt's font database on
    first use so every widget constructing a QFont("Inter", ...) resolves
    to the same bundled typeface rather than an arbitrary system default.

    No .ttf currently exists under assets/fonts/ (see module docstring) --
    until one is added, this degrades to requesting the "Inter" family by
    name anyway, which Qt silently substitutes with a platform sans-serif
    if unavailable. Callers don't need to change once a real font file is
    added; only this function does.

    Parameters
    ----------
    point_size
        Font size in points.
    bold
        Whether to request the bold weight.

    Returns
    -------
    QFont
        Configured font, ready to hand to a widget's setFont() or a
        QApplication as the default application font.
    '''

    if _ASSETS_FONT_DIR.is_dir():
        for font_file in sorted(_ASSETS_FONT_DIR.glob("*.ttf")):
            QFontDatabase.addApplicationFont(str(font_file))

    font = QFont(_BUNDLED_FONT_FAMILY, point_size)
    font.setBold(bold)
    return font


__all__ = [
    "COLOR_BACKGROUND", "COLOR_SURFACE", "COLOR_SURFACE_ALT", "COLOR_BORDER",
    "COLOR_TEXT_PRIMARY", "COLOR_TEXT_SECONDARY", "COLOR_TEXT_DISABLED",
    "COLOR_ACCENT", "COLOR_ACCENT_ALT",
    "COLOR_SUCCESS", "COLOR_WARNING", "COLOR_ERROR",
    "SPACING_XS", "SPACING_SMALL", "SPACING_MEDIUM", "SPACING_LARGE", "SPACING_XL",
    "MARGIN_DEFAULT",
    "load_bundled_font",
]
