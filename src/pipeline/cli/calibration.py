"""
Headless sensor calibration CLI.

Wires calibration/sensor/ workflow functions to a real CameraStream built
from configs/default.yaml plus a required --gain-db argument. Requires a
connected Basler camera for subcommands that acquire frames.
"""

# Imports

import argparse
import logging
from pathlib import Path

from pipeline.acquisition import CameraStream
from pipeline.analysis import SensorNoiseModel
from pipeline.calibration.sensor import (
    build_bad_pixel_map,
    capture_dark_frames,
    capture_illuminated_frames,
    finish_flat_field_calibration,
    load_baseline,
    load_conversion_gain,
    load_flat_field,
    run_baseline_calibration,
    run_conversion_gain_calibration,
    save_bad_pixel_map,
)
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
DEFAULT_ARTIFACT_DIR = Path("calibration_artifacts")
DEFAULT_BASELINE_FILENAME = "baseline.npz"
DEFAULT_FLAT_FIELD_FILENAME = "flat_field.npz"
DEFAULT_BAD_PIXEL_MAP_FILENAME = "bad_pixel_map.npz"
DEFAULT_CONVERSION_GAIN_FILENAME = "conversion_gain.npz"
DEFAULT_N_FRAMES = 50

# Classes

# Functions


def build_camera_stream(gain_db: float, *, config_path: Path | None = None) -> CameraStream:

    '''
    Build a CameraStream from default config and a caller-supplied gain.

    Does not start or stop the stream — callers own stream lifecycle.

    Parameters
    ----------
    gain_db
        Sensor gain in decibels (not stored in the YAML config).
    config_path
        YAML config file. Defaults to repo-root configs/default.yaml.

    Returns
    -------
    CameraStream
        Stream configured for the real PylonBackend (backend=None).
    '''

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    camera = config["camera"]
    return CameraStream(
        exposure_us=camera["exposure_time"],
        gain_db=gain_db,
        pixel_format=camera["pixel_format"],
        timeout_ms=camera["timeout"],
        serial_number=camera["serial_number"],
    )


def resolve_artifact_path(
    output_dir: str | Path | None,
    path: str | Path | None,
    default_filename: str,
) -> Path:

    '''
    Resolve an artifact output path from CLI --output-dir and --path flags.

    Parameters
    ----------
    output_dir
        Base directory from --output-dir, or None for DEFAULT_ARTIFACT_DIR.
    path
        Explicit path from --path, or None for the default filename under base.
    default_filename
        Filename used when path is omitted.

    Returns
    -------
    Path
        Resolved artifact path.
    '''

    base = Path(output_dir) if output_dir is not None else DEFAULT_ARTIFACT_DIR
    if path is None:
        return base / default_filename
    path = Path(path)
    if path.is_absolute():
        return path
    if output_dir is not None:
        return base / path
    return path


def _add_output_dir(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help=f"Directory for artifacts (default: {DEFAULT_ARTIFACT_DIR}).",
    )


def _add_output_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--path",
        default=None,
        help="Output artifact path (see --output-dir for relative-path rules).",
    )
    _add_output_dir(subparser)


def _add_gain_db(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--gain-db",
        type=float,
        required=True,
        dest="gain_db",
        help="Camera gain in decibels.",
    )


def _cmd_baseline(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(args.output_dir, args.path, DEFAULT_BASELINE_FILENAME)
    stream = build_camera_stream(args.gain_db)
    stream.start()
    try:
        run_baseline_calibration(stream, args.n_frames, path)
    finally:
        if stream.is_running:
            stream.stop()


def _cmd_flat_field(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(args.output_dir, args.path, DEFAULT_FLAT_FIELD_FILENAME)
    stream = build_camera_stream(args.gain_db)
    stream.start()
    try:
        input("Block the beam, then press Enter to capture dark frames...")
        dark = capture_dark_frames(stream, args.n_frames)
        input(
            "Apply uniform illumination to the sensor, then press Enter "
            "to capture illuminated frames..."
        )
        illuminated = capture_illuminated_frames(stream, args.n_frames)
        finish_flat_field_calibration(illuminated, dark, path)
    finally:
        if stream.is_running:
            stream.stop()


def _cmd_bad_pixel_map(args: argparse.Namespace) -> None:
    if args.flat_field is None:
        flat_field_path = resolve_artifact_path(
            args.output_dir, None, DEFAULT_FLAT_FIELD_FILENAME
        )
    else:
        flat_field_path = Path(args.flat_field)
    output_path = resolve_artifact_path(
        args.output_dir, args.path, DEFAULT_BAD_PIXEL_MAP_FILENAME
    )
    flat_field, flat_field_record = load_flat_field(flat_field_path)
    mask, record = build_bad_pixel_map(flat_field, flat_field_record)
    save_bad_pixel_map(output_path, mask, record)


def _cmd_conversion_gain(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(
        args.output_dir, args.path, DEFAULT_CONVERSION_GAIN_FILENAME
    )
    stream = build_camera_stream(args.gain_db)
    stream.start()
    try:
        run_conversion_gain_calibration(
            stream,
            args.exposure_min_us,
            args.exposure_max_us,
            args.n_levels,
            args.n_frames_per_level,
            path,
        )
    finally:
        if stream.is_running:
            stream.stop()


def _cmd_noise_model(args: argparse.Namespace) -> None:
    if args.baseline is None:
        baseline_path = resolve_artifact_path(
            args.output_dir, None, DEFAULT_BASELINE_FILENAME
        )
    else:
        baseline_path = Path(args.baseline)
    if args.conversion_gain is None:
        conversion_gain_path = resolve_artifact_path(
            args.output_dir, None, DEFAULT_CONVERSION_GAIN_FILENAME
        )
    else:
        conversion_gain_path = Path(args.conversion_gain)
    baseline_result, _ = load_baseline(baseline_path)
    conversion_gain_result, _ = load_conversion_gain(conversion_gain_path)
    model = SensorNoiseModel(
        gain_e_per_adu=conversion_gain_result.gain_e_per_adu,
        background_sigma=baseline_result.background_sigma,
    )
    print(repr(model))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=(
            "Headless sensor calibration CLI. Subcommands that acquire frames "
            "require a connected Basler camera."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="Acquire background frames and save a baseline artifact.",
    )
    _add_gain_db(baseline_parser)
    _add_output_args(baseline_parser)
    baseline_parser.add_argument(
        "--n-frames",
        type=int,
        default=DEFAULT_N_FRAMES,
        dest="n_frames",
        help=f"Number of background frames to average (default: {DEFAULT_N_FRAMES}).",
    )
    baseline_parser.set_defaults(func=_cmd_baseline)

    flat_field_parser = subparsers.add_parser(
        "flat-field",
        help="Two-phase flat-field calibration with interactive setup prompts.",
    )
    _add_gain_db(flat_field_parser)
    _add_output_args(flat_field_parser)
    flat_field_parser.add_argument(
        "--n-frames",
        type=int,
        default=DEFAULT_N_FRAMES,
        dest="n_frames",
        help=f"Number of frames per phase (default: {DEFAULT_N_FRAMES}).",
    )
    flat_field_parser.set_defaults(func=_cmd_flat_field)

    bad_pixel_parser = subparsers.add_parser(
        "bad-pixel-map",
        help="Build a bad-pixel map from an existing flat-field artifact.",
    )
    _add_output_args(bad_pixel_parser)
    bad_pixel_parser.add_argument(
        "--flat-field",
        default=None,
        dest="flat_field",
        help=(
            f"Flat-field artifact to derive the map from "
            f"(default: {DEFAULT_ARTIFACT_DIR}/{DEFAULT_FLAT_FIELD_FILENAME})."
        ),
    )
    bad_pixel_parser.set_defaults(func=_cmd_bad_pixel_map)

    conversion_gain_parser = subparsers.add_parser(
        "conversion-gain",
        help="Sweep exposure time and save a conversion-gain artifact.",
    )
    _add_gain_db(conversion_gain_parser)
    _add_output_args(conversion_gain_parser)
    conversion_gain_parser.add_argument(
        "--exposure-min-us",
        type=float,
        required=True,
        dest="exposure_min_us",
        help="Minimum exposure time for the sweep, in microseconds.",
    )
    conversion_gain_parser.add_argument(
        "--exposure-max-us",
        type=float,
        required=True,
        dest="exposure_max_us",
        help="Maximum exposure time for the sweep, in microseconds.",
    )
    conversion_gain_parser.add_argument(
        "--n-levels",
        type=int,
        required=True,
        dest="n_levels",
        help="Number of evenly-spaced exposure levels to sample.",
    )
    conversion_gain_parser.add_argument(
        "--n-frames-per-level",
        type=int,
        required=True,
        dest="n_frames_per_level",
        help="Number of frames to capture at each exposure level.",
    )
    conversion_gain_parser.set_defaults(func=_cmd_conversion_gain)

    noise_model_parser = subparsers.add_parser(
        "noise-model",
        help="Print a SensorNoiseModel built from saved calibration artifacts.",
    )
    _add_output_dir(noise_model_parser)
    noise_model_parser.add_argument(
        "--baseline",
        default=None,
        help=(
            f"Baseline artifact (default: "
            f"{DEFAULT_ARTIFACT_DIR}/{DEFAULT_BASELINE_FILENAME})."
        ),
    )
    noise_model_parser.add_argument(
        "--conversion-gain",
        default=None,
        dest="conversion_gain",
        help=(
            f"Conversion-gain artifact (default: "
            f"{DEFAULT_ARTIFACT_DIR}/{DEFAULT_CONVERSION_GAIN_FILENAME})."
        ),
    )
    noise_model_parser.set_defaults(func=_cmd_noise_model)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
