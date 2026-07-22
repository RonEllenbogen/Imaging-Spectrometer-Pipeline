This document contains notes from the development of the project

## **Hardware bring-up** — camera connectivity verified via pylon Viewer; minimal single-frame grab script

Local pylon camera details:

    Vendor:
    Basler

    Model Name:
    a2A1920-51gmBAS

    Device User ID:
    ImagingSpecCam

    Serial Number:
    41948281

    MAC Address:
    00:30:53:38:97:FF

    IP Configuration:
    Static IP

    IP Address:
    170.254.1.105

    Subnet Mask:
    255.255.0.0

    Gateway:
    0.0.0.0

Image format:

    shape:
    (1200, 1920)

    dtype:
    uint8


## **Acquisition** (`src/pipeline/acquisition/`) — threaded Basler/pypylon interface for live frame grabbing

## **Preprocessing** (`src/pipeline/preprocessing/`) — dark-frame subtraction, ROI cropping, wavelength calibration

## **Analysis** (`src/pipeline/analysis/`) — per-column centroid + uncertainty, weighted linear fit for spatial chirp ζ

## **Validation** (`tests/`) — synthetic-data checks (injected-ζ recovery, null case) and validation against real calibration data (`data/raw` → `data/processed`)

## **Headless integration** (`scripts/`) — capture-and-analyze CLI, no GUI

## **GUI** (`src/pipeline/gui/`) — live camera display with threaded acquisition

## **Full integration** (`src/pipeline/main.py`) — GUI wired to acquisition + analysis pipeline

## **Documentation & polish** (`docs/`) — architecture write-up, edge-case handling, results export
