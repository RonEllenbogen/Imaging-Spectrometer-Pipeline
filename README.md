# Imaging-Spectrometer-Pipeline

## Overview

Ultrashort laser systems are susceptible to distortions that require careful monitoring. One such distortion is ***spatial chirp***, a correlation between optical frequency and transverse position across the laser beam.

As part of a summer research project in the ***Laser Plasma Accelerator Group (LPAG)*** at the University of Oxford, an imaging spectrometer was designed and built to detect and quantify spatial chirp in the group's Ti:sapphire laser system.

This repository contains the Python application developed for the instrument. The software acquires images from a camera, preprocesses and analyses the data to estimate spatial dispersion, and presents the results in real time through a graphical user interface.

---

## Objectives

- Acquire live data from the Imaging Spectrometer's camera
- Preprocess and clean the images
- Execute computations to estimate spatial dispersion
- Display results in a responsive GUI

---

## Current Status

The repository structure has been initialised and the `README.md` is being written.

---

## Project Structure

```text
Imaging-Spectrometer-Pipeline/
├── README.md                 # Project overview and documentation
├── .gitignore                # Files and folders ignored by Git
├── LICENSE                   # Project license
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata and build configuration
├── .env.example              # Example environment configuration
│
├── docs/                     # Project documentation
│   ├── architecture.md
│   └── notes.md
│
├── configs/                  # Configuration files
│   └── default.yaml
│
├── data/                     # Data used during development
│   ├── raw/                  # Unprocessed camera images
│   ├── interim/              # Intermediate processing outputs
│   └── processed/            # Final processed data
│
├── assets/                   # Images and other static resources
│   └── images/
│
├── src/                      # Source code
│   └── pipeline/
│       ├── acquisition/      # Camera interface and image acquisition
│       ├── preprocessing/    # Image cleaning and preprocessing
│       ├── analysis/        # Spatial chirp estimation and computations
│       ├── gui/              # Graphical user interface
│       ├── utils/            # Shared utility functions
│       └── main.py           # Application entry point
│
├── tests/                    # Unit and integration tests
│
└── scripts/                  # Stand-alone utility scripts
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/RonEllenbogen/Imaging-Spectrometer-Pipeline.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Implementation in progress.

---

## Roadmap

- [ ] **Hardware bring-up** — camera connectivity verified via pylon Viewer; minimal single-frame grab script
- [ ] **Acquisition** (`src/pipeline/acquisition/`) — threaded Basler/pypylon interface for live frame grabbing
- [ ] **Preprocessing** (`src/pipeline/preprocessing/`) — dark-frame subtraction, ROI cropping, wavelength calibration
- [ ] **Analysis** (`src/pipeline/analysis/`) — per-column centroid + uncertainty, weighted linear fit for spatial chirp ζ
- [ ] **Validation** (`tests/`) — synthetic-data checks (injected-ζ recovery, null case) and validation against real calibration data (`data/raw` → `data/processed`)
- [ ] **Headless integration** (`scripts/`) — capture-and-analyze CLI, no GUI
- [ ] **GUI** (`src/pipeline/gui/`) — live camera display with threaded acquisition
- [ ] **Full integration** (`src/pipeline/main.py`) — GUI wired to acquisition + analysis pipeline
- [ ] **Documentation & polish** (`docs/`) — architecture write-up, edge-case handling, results export

---

## License

MIT License

---

## Acknowledgements

This project was developed as part of a summer research project in the Laser Plasma Accelerator Group (LPAG) at the University of Oxford.

Project supervisor: Dr Benjamin Greenwood.