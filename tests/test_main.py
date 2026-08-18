'''
Smoke test for main.py -- the program's entry point (`python -m
pipeline.main`), which just builds the QApplication and shows MainWindow
(see main.py's own docstring for why gui/app.py's own `__main__` block is
kept alongside this rather than replaced). No camera/calibration I/O
happens at MainWindow construction time (see tests/test_gui.py), so this
needs no SyntheticBackend/CalibrationSet fixtures -- only an offscreen Qt
platform, same "gate, don't require" pattern every other gui/ test file
in this repo uses for the PySide6/pyqtgraph/pytest-qt local-only
dependencies.
'''

# Imports

import os

import pytest

pytest.importorskip("PySide6", reason="PySide6 is a local-only GUI dependency")
pytest.importorskip("pytestqt", reason="pytest-qt is a local-only GUI dependency")

# Must be set before any QApplication is constructed -- pytest-qt creates
# one lazily the first time a test requests the qtbot fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import pipeline.main as main_module  # noqa: E402
from pipeline.gui.app import DEFAULT_HEIGHT, DEFAULT_WIDTH  # noqa: E402

# Constants

# Classes

# Functions


def test_main_builds_and_shows_main_window(qtbot, monkeypatch):
    '''
    main() must build a QApplication, construct/show a MainWindow sized
    and titled per gui/app.py's own __main__ block, and hand control to
    the event loop -- app.exec() is mocked to a no-op (it blocks for real
    input otherwise) so this test can inspect the window afterward instead
    of hanging. main() holds the only Python reference to the window it
    builds and doesn't return it, so MainWindow itself is patched to
    capture the live instance for inspection here -- otherwise it's
    garbage-collected (and its underlying C++ QWidget destroyed) the
    moment main() returns.
    '''
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)

    created = []
    real_main_window = main_module.MainWindow

    def _capturing_main_window(*args, **kwargs):
        window = real_main_window(*args, **kwargs)
        created.append(window)
        return window

    monkeypatch.setattr(main_module, "MainWindow", _capturing_main_window)

    main_module.main()

    assert len(created) == 1
    window = created[0]
    qtbot.addWidget(window)

    assert window.windowTitle() == "Imaging Spectrometer Pipeline"
    assert window.width() == DEFAULT_WIDTH
    assert window.height() == DEFAULT_HEIGHT
    assert window.isVisible()

    window.close()
