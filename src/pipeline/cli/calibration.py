"""
Headless sensor calibration CLI.

Wires calibration/sensor/ workflow functions to a real CameraStream built
from configs/default.yaml plus a required --gain-db argument. Requires a
connected Basler camera for subcommands that acquire frames.
"""

# Imports

import argparse
import logging
import time
from pathlib import Path

from pipeline.acquisition import CameraStream
from pipeline.analysis import SensorNoiseModel
from pipeline.calibration.sensor import (
    build_bad_pixel_map,
    capture_dark_frames,
    capture_illuminated_frames,
    finish_flat_field_calibration,
    load_baseline,
    load_bad_pixel_map,
    load_conversion_gain,
    load_flat_field,
    run_baseline_calibration,
    run_conversion_gain_calibration,
    save_bad_pixel_map,
)
from pipeline.calibration.shared import CalibrationRecord
from pipeline.calibration.spatial import (
    ScaleFactorPositionCalibration,
    load_scale_factor,
    save_scale_factor,
)
from pipeline.calibration.spectral import (
    build_manual_spectral_calibration,
    run_spectral_calibration,
    save_spectral_calibration,
)
from pipeline.preprocessing import CalibrationSet
from pipeline.utils.helpers import load_config

# Constants

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "default.yaml"
DEFAULT_ARTIFACT_DIR = Path("calibration_artifacts")
DEFAULT_BASELINE_FILENAME = "baseline.npz"
DEFAULT_FLAT_FIELD_FILENAME = "flat_field.npz"
DEFAULT_BAD_PIXEL_MAP_FILENAME = "bad_pixel_map.npz"
DEFAULT_CONVERSION_GAIN_FILENAME = "conversion_gain.npz"
# Not exported from calibration/spatial/io.py -- unlike the four artifact
# types above, the scale factor previously had no CLI-facing filename at
# all (see docs/project_state.md's GUI review notes). Named here to match
# its siblings' convention; gui/calibration_screen.py's own
# _DEFAULT_SCALE_FACTOR_FILENAME constant predates this and is left as-is.
DEFAULT_SCALE_FACTOR_FILENAME = "scale_factor.npz"
DEFAULT_SPECTRAL_FILENAME = "spectral.npz"
DEFAULT_N_FRAMES = 50
DEFAULT_SPECTRAL_DEGREE = 1

# A manual spectral entry captures no frame at all, but CalibrationRecord
# requires a positive exposure_us and a gain_db regardless (see
# calibration/shared/metadata.py) -- these are "not applicable"
# placeholders, same convention as source_frame_count=1 for a manual
# entry (see build_manual_spectral_calibration()'s own docstring).
MANUAL_SPECTRAL_EXPOSURE_US = 1.0
MANUAL_SPECTRAL_GAIN_DB = 0.0

# Classes

# Functions


def build_camera_stream(
    gain_db: float,
    *,
    config_path: Path | None = None,
    exposure_us: float | None = None,
    auto_exposure: bool = False,
) -> CameraStream:

    '''
    Build a CameraStream from default config and a caller-supplied gain.

    Does not start or stop the stream — callers own stream lifecycle.

    Parameters
    ----------
    gain_db
        Sensor gain in decibels (not stored in the YAML config).
    config_path
        YAML config file. Defaults to repo-root configs/default.yaml.
    exposure_us
        Exposure time in microseconds for this session. Required unless
        auto_exposure is True -- there is no implicit config-file default;
        every caller (a CLI flag or a GUI field) must supply an explicit
        value. Ignored when auto_exposure is True -- still passed through
        to CameraStream, which ignores it identically (see CameraStream's
        own docstring) -- so it may be omitted in that case.
    auto_exposure
        If True, the real camera runs a one-time ExposureAuto convergence
        instead of using exposure_us -- see PylonBackend. The converged
        value becomes CameraStream.exposure_us once start() returns (used
        to tag every captured frame/CalibrationRecord correctly).

    Returns
    -------
    CameraStream
        Stream configured for the real PylonBackend (backend=None).

    Raises
    ------
    ValueError
        If auto_exposure is False and exposure_us is None.
    '''

    if not auto_exposure and exposure_us is None:
        raise ValueError(
            "exposure_us is required when auto_exposure is False -- pass "
            "--exposure-us (CLI) or enter an exposure time (GUI); there is "
            "no implicit config-file default."
        )

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    config = load_config(config_path)
    camera = config["camera"]
    return CameraStream(
        exposure_us=exposure_us if exposure_us is not None else 0.0,
        gain_db=gain_db,
        pixel_format=camera["pixel_format"],
        timeout_ms=camera["timeout"],
        serial_number=camera["serial_number"],
        auto_exposure=auto_exposure,
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


def _add_exposure_args(subparser: argparse.ArgumentParser) -> None:

    '''
    --auto-exposure and --exposure-us, mutually exclusive and one of the
    two required -- there is no implicit config-file default (see
    build_camera_stream()), so the caller must explicitly choose fixed or
    auto exposure every time.
    '''

    group = subparser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--auto-exposure",
        action="store_true",
        dest="auto_exposure",
        help="Run one-time auto-exposure convergence instead of a fixed exposure time.",
    )
    group.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        dest="exposure_us",
        help="Exposure time in microseconds (required unless --auto-exposure is passed).",
    )


def _cmd_baseline(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(args.output_dir, args.path, DEFAULT_BASELINE_FILENAME)
    stream = build_camera_stream(
        args.gain_db, exposure_us=args.exposure_us, auto_exposure=args.auto_exposure
    )
    stream.start()
    try:
        run_baseline_calibration(stream, args.n_frames, path)
    finally:
        if stream.is_running:
            stream.stop()


def _cmd_flat_field(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(args.output_dir, args.path, DEFAULT_FLAT_FIELD_FILENAME)
    stream = build_camera_stream(
        args.gain_db, exposure_us=args.exposure_us, auto_exposure=args.auto_exposure
    )
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


def _cmd_spatial(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(
        args.output_dir, args.path, DEFAULT_SCALE_FACTOR_FILENAME
    )
    if args.scale_factor is not None:
        calibration = ScaleFactorPositionCalibration(scale_factor=args.scale_factor)
        save_scale_factor(path, calibration, source="manual")
        print(f"saved scale_factor={calibration.scale_factor} to {path}")
        return

    # No --scale-factor given -- report whatever's currently active (a
    # saved override, or DEFAULT_SCALE_FACTOR if none exists yet), same
    # read-only behavior as `noise-model`. No camera involved either way,
    # matching build_bad_pixel_map()'s reasoning (see _cmd_bad_pixel_map).
    calibration, record = load_scale_factor(path)
    print(f"scale_factor={calibration.scale_factor} (source={record.source})")


def _cmd_conversion_gain(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(
        args.output_dir, args.path, DEFAULT_CONVERSION_GAIN_FILENAME
    )
    # No dedicated flag for conversion-gain's initial exposure -- the sweep
    # (exposure_min_us..exposure_max_us) is caller-driven from the very
    # first level, so seeding the pre-sweep stream at exposure_min_us (the
    # sweep's own first point, per np.linspace) is exact, not a guess, and
    # needs no separate flag. See build_camera_stream()'s docstring for why
    # an explicit value is required at all -- no config-file default.
    stream = build_camera_stream(args.gain_db, exposure_us=args.exposure_min_us)
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


def _cmd_spectral_capture(args: argparse.Namespace) -> None:
    path = resolve_artifact_path(args.output_dir, args.path, DEFAULT_SPECTRAL_FILENAME)
    if args.baseline is None:
        baseline_path = resolve_artifact_path(
            args.output_dir, None, DEFAULT_BASELINE_FILENAME
        )
    else:
        baseline_path = Path(args.baseline)
    if args.flat_field is None:
        flat_field_path = resolve_artifact_path(
            args.output_dir, None, DEFAULT_FLAT_FIELD_FILENAME
        )
    else:
        flat_field_path = Path(args.flat_field)
    if args.bad_pixel_map is None:
        bad_pixel_map_path = resolve_artifact_path(
            args.output_dir, None, DEFAULT_BAD_PIXEL_MAP_FILENAME
        )
    else:
        bad_pixel_map_path = Path(args.bad_pixel_map)

    baseline_result, baseline_record = load_baseline(baseline_path)
    flat_field, flat_field_record = load_flat_field(flat_field_path)
    bad_pixel_mask, _ = load_bad_pixel_map(bad_pixel_map_path)
    sensor_calibration = CalibrationSet(
        baseline=baseline_result.baseline,
        baseline_record=baseline_record,
        flat_field=flat_field,
        flat_field_record=flat_field_record,
        bad_pixel_mask=bad_pixel_mask,
        background_sigma=baseline_result.background_sigma,
    )

    stream = build_camera_stream(
        args.gain_db, exposure_us=args.exposure_us, auto_exposure=args.auto_exposure
    )
    stream.start()
    try:
        run_spectral_calibration(
            stream, args.n_frames, sensor_calibration, path, degree=args.degree,
        )
    finally:
        if stream.is_running:
            stream.stop()


def _cmd_spectral_manual(args: argparse.Namespace) -> None:
    if len(args.coefficients) != len(args.coefficient_sigma):
        args.parser.error(
            "--coefficients and --coefficient-sigma must have the same length "
            f"(got {len(args.coefficients)} and {len(args.coefficient_sigma)})"
        )
    path = resolve_artifact_path(args.output_dir, args.path, DEFAULT_SPECTRAL_FILENAME)
    record = CalibrationRecord(
        exposure_us=MANUAL_SPECTRAL_EXPOSURE_US,
        gain_db=MANUAL_SPECTRAL_GAIN_DB,
        timestamp=time.time(),
        source_frame_count=1,
    )
    result = build_manual_spectral_calibration(
        args.coefficients, args.coefficient_sigma, record
    )
    save_spectral_calibration(path, result)


def build_parser() -> argparse.ArgumentParser:

    '''
    Builds the full argparse parser (every subcommand, every flag) with
    no side effects -- split out from main() so tests can exercise
    argument parsing directly (parser.parse_args([...])) without
    triggering logging setup or a subcommand's actual func (which may
    require a camera).
    '''

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
    _add_exposure_args(baseline_parser)
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
    _add_exposure_args(flat_field_parser)
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

    spatial_parser = subparsers.add_parser(
        "spatial",
        help="Set or report the spatial (pixel-to-position) scale factor -- no camera involved.",
    )
    _add_output_args(spatial_parser)
    spatial_parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        dest="scale_factor",
        help=(
            "Manually-measured relay-optics scale factor to save (source=manual). "
            "Omit to report the currently active value instead (a saved override, "
            "or DEFAULT_SCALE_FACTOR if none exists yet)."
        ),
    )
    spatial_parser.set_defaults(func=_cmd_spatial)

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

    spectral_capture_parser = subparsers.add_parser(
        "spectral-capture",
        help="Acquire lamp frames and fit a pixel->wavelength_nm calibration.",
    )
    _add_gain_db(spectral_capture_parser)
    _add_exposure_args(spectral_capture_parser)
    _add_output_args(spectral_capture_parser)
    spectral_capture_parser.add_argument(
        "--n-frames",
        type=int,
        default=DEFAULT_N_FRAMES,
        dest="n_frames",
        help=f"Number of lamp frames to average (default: {DEFAULT_N_FRAMES}).",
    )
    spectral_capture_parser.add_argument(
        "--degree",
        type=int,
        default=DEFAULT_SPECTRAL_DEGREE,
        help=f"Polynomial degree for the pixel->wavelength_nm fit (default: {DEFAULT_SPECTRAL_DEGREE}).",
    )
    spectral_capture_parser.add_argument(
        "--baseline",
        default=None,
        help=(
            f"Baseline artifact to preprocess lamp frames with (default: "
            f"{DEFAULT_ARTIFACT_DIR}/{DEFAULT_BASELINE_FILENAME})."
        ),
    )
    spectral_capture_parser.add_argument(
        "--flat-field",
        default=None,
        dest="flat_field",
        help=(
            f"Flat-field artifact to preprocess lamp frames with (default: "
            f"{DEFAULT_ARTIFACT_DIR}/{DEFAULT_FLAT_FIELD_FILENAME})."
        ),
    )
    spectral_capture_parser.add_argument(
        "--bad-pixel-map",
        default=None,
        dest="bad_pixel_map",
        help=(
            f"Bad-pixel-map artifact to preprocess lamp frames with (default: "
            f"{DEFAULT_ARTIFACT_DIR}/{DEFAULT_BAD_PIXEL_MAP_FILENAME})."
        ),
    )
    spectral_capture_parser.set_defaults(func=_cmd_spectral_capture)

    spectral_manual_parser = subparsers.add_parser(
        "spectral-manual",
        help="Enter a pixel->wavelength_nm calibration by hand, bypassing lamp capture.",
    )
    spectral_manual_parser.add_argument(
        "--coefficients",
        type=float,
        nargs="+",
        required=True,
        help=(
            "Ascending-order pixel->wavelength_nm polynomial coefficients "
            "(c0 c1 c2 ...; wavelength_nm = c0 + c1*pixel + c2*pixel^2 + ...)."
        ),
    )
    spectral_manual_parser.add_argument(
        "--coefficient-sigma",
        type=float,
        nargs="+",
        required=True,
        dest="coefficient_sigma",
        help="1-sigma uncertainty for each coefficient, same length/order as --coefficients.",
    )
    _add_output_args(spectral_manual_parser)
    spectral_manual_parser.set_defaults(func=_cmd_spectral_manual, parser=spectral_manual_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
