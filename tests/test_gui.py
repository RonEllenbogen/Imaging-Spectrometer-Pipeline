"""
Smoke tests for the gui/ calibration screen -- widget-layer only (does
the screen launch, do each of the five type-flows' widgets exist), no
hardware and no display required (QT_QPA_PLATFORM=offscreen).

PySide6/pytest-qt are local-only dependencies (see CLAUDE.md -- pyproject.toml/
requirements.txt are deliberately left empty), so every test here is skipped
outright, rather than failing collection, when PySide6 isn't installed --
that keeps `python3 -m pytest` green in any environment that hasn't pip
installed the GUI extras.

PHASE 1 (VISUAL SKELETON) NOTE: calibration_screen.py/calibration_dialogs.py
have no real camera or calibration package call wired in yet (see their
module docstrings), so these tests only check structure/layout, not
behavior -- e.g. that CreatePage exposes exactly the four enabled type
cards plus a disabled spectral card, not that clicking "Configure..."
actually acquires anything. A follow-up pass adds tests for the automatic
bad-pixel-map chaining and error-dialog paths once those are wired, using
mocked calibration calls rather than a real camera.
"""

# Imports

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="PySide6 is a local-only GUI dependency")
pytest.importorskip("pytestqt", reason="pytest-qt is a local-only GUI dependency")

from pipeline.gui.calibration_dialogs import (  # noqa: E402
    BaselineDialog,
    ConversionGainDialog,
    FlatFieldDialog,
    SpatialCalibrationDialog,
)
from pipeline.gui.calibration_screen import (  # noqa: E402
    CalibrationScreen,
    SPECTRAL_UNAVAILABLE_NOTE,
    _LOAD_ROWS,
)

# Constants

# Classes

# Functions


def test_calibration_screen_launches(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    assert screen.welcome_page is not None
    assert screen.load_page is not None
    assert screen.create_page is not None
    assert screen.get_calibration_bundle() is None


def test_welcome_page_navigates_to_load_and_create(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)

    screen.welcome_page.load_requested.emit()
    assert screen._stack.currentWidget() is screen.load_page

    screen.welcome_page.create_requested.emit()
    assert screen._stack.currentWidget() is screen.create_page


def test_load_page_lists_five_artifact_rows(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    assert len(_LOAD_ROWS) == 5
    assert not screen.load_page.continue_button.isEnabled()


def test_create_page_has_four_enabled_type_cards_and_no_bad_pixel_option(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    create_page = screen.create_page

    for card in (
        create_page.baseline_card,
        create_page.flat_field_card,
        create_page.conversion_gain_card,
        create_page.spatial_card,
    ):
        assert card.action_button.isEnabled()

    assert not create_page.spectral_card.action_button.isEnabled()
    assert not hasattr(create_page, "bad_pixel_card")


def test_spectral_card_shows_unavailable_note(qtbot):
    screen = CalibrationScreen()
    qtbot.addWidget(screen)
    assert screen.create_page.spectral_card.description_label.text() == SPECTRAL_UNAVAILABLE_NOTE


def test_baseline_dialog_has_n_frames_and_gain_fields(qtbot):
    dialog = BaselineDialog()
    qtbot.addWidget(dialog)
    assert dialog.n_frames_spin.value() > 0
    assert dialog.gain_db_spin is not None
    assert dialog.start_button is not None


def test_flat_field_dialog_two_phase_sequence(qtbot):
    dialog = FlatFieldDialog()
    qtbot.addWidget(dialog)

    assert dialog._phase == dialog.PHASE_DARK
    assert "Block the beam" in dialog.instruction_label.text()

    dialog._advance_phase()
    assert dialog._phase == dialog.PHASE_ILLUMINATED
    assert "uniform illumination" in dialog.instruction_label.text()

    dialog._advance_phase()
    assert dialog._phase == dialog.PHASE_FINISHING
    assert "automatically" in dialog.status_label.text()


def test_conversion_gain_dialog_has_all_required_fields(qtbot):
    dialog = ConversionGainDialog()
    qtbot.addWidget(dialog)
    assert dialog.exposure_min_spin is not None
    assert dialog.exposure_max_spin is not None
    assert dialog.n_levels_spin is not None
    assert dialog.n_frames_per_level_spin is not None
    assert dialog.gain_db_spin is not None


def test_spatial_dialog_defaults_to_given_scale_factor(qtbot):
    dialog = SpatialCalibrationDialog(1.5)
    qtbot.addWidget(dialog)
    assert dialog.scale_factor_spin.value() == pytest.approx(1.5)
