'''
Top-level program entry point: launches the full GUI application
(pipeline.gui.app.MainWindow), so the program is run via
`python -m pipeline.main` instead of directly via `python -m
pipeline.gui.app`. gui/app.py keeps its own `__main__` block as well --
see its module docstring for why that independent launchability is
intentional (the same pattern every other gui/ screen already follows,
e.g. scripts/demo_live_view.py) -- this module is just the one the user
guide points people to as the actual program entry point.
'''

# Imports

from PySide6.QtWidgets import QApplication

from pipeline.gui.app import DEFAULT_HEIGHT, DEFAULT_WIDTH, MainWindow

# Constants

# Classes

# Functions


def main() -> None:

    '''Builds the QApplication and MainWindow, shows the window, and runs
    the Qt event loop until the window is closed.'''

    app = QApplication.instance() or QApplication([])

    window = MainWindow()
    window.setWindowTitle("Imaging Spectrometer Pipeline")
    window.resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
    window.show()

    app.exec()


if __name__ == "__main__":
    main()
