# Imaging-Spectrometer-Pipeline

## Overview

Ultrashort laser systems are susceptible to distortions that require careful monitoring. One such distortion is *spatial chirp*, a correlation between optical frequency and transverse position across the laser beam.

As part of a summer research project in the *Laser Plasma Accelerator Group (LPAG)* at the University of Oxford, an imaging spectrometer was designed and built to detect and quantify spatial chirp in the group's Ti:sapphire laser system.

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

## Acknowledgements

This project was developed as part of a summer research project in the Laser Plasma Accelerator Group (LPAG) at the University of Oxford.

Project supervisor: Dr Benjamin Greenwood.