'''
Test suite for cli/calibration.py's argument parsing and small pure
helpers (resolve_artifact_path(), build_camera_stream()'s config-fallback
logic). Deliberately does NOT exercise any subcommand's real func() --
those touch a real camera or real calibration artifacts, and are covered
end-to-end elsewhere (calibration/sensor's own workflow tests,
calibration/spatial's own io tests). This file's job is narrower: does
every subcommand accept the flags it's supposed to, with the defaults and
mutual-exclusion rules it's supposed to have.
'''

# Imports

from pathlib import Path

import pytest

from pipeline.cli.calibration import (
    DEFAULT_ARTIFACT_DIR,
    build_camera_stream,
    build_parser,
    resolve_artifact_path,
)

# Constants

# Classes

# Functions


class TestResolveArtifactPath:

    def test_no_flags_uses_default_dir_and_filename(self):
        path = resolve_artifact_path(None, None, "baseline.npz")
        assert path == DEFAULT_ARTIFACT_DIR / "baseline.npz"

    def test_explicit_output_dir_no_path(self):
        path = resolve_artifact_path("custom_dir", None, "baseline.npz")
        assert path == Path("custom_dir") / "baseline.npz"

    def test_absolute_path_ignores_output_dir(self):
        path = resolve_artifact_path("custom_dir", "/tmp/abs/foo.npz", "baseline.npz")
        assert str(path) == "/tmp/abs/foo.npz"

    def test_relative_path_without_output_dir_used_as_is(self):
        path = resolve_artifact_path(None, "relative/foo.npz", "baseline.npz")
        assert str(path) == "relative/foo.npz"

    def test_relative_path_with_output_dir_joined(self):
        path = resolve_artifact_path("custom_dir", "sub/foo.npz", "baseline.npz")
        assert path == Path("custom_dir") / "sub/foo.npz"


class TestBuildCameraStream:

    def test_defaults_to_config_exposure(self):
        stream = build_camera_stream(gain_db=5.0)
        # configs/default.yaml's camera.exposure_time -- see conftest/module
        # docstring elsewhere for why this file is intentionally not
        # hardcoded here; just check it's a positive, config-sourced value.
        assert stream.exposure_us > 0
        assert stream.gain_db == 5.0

    def test_explicit_exposure_us_overrides_config(self):
        stream = build_camera_stream(gain_db=5.0, exposure_us=12345.0)
        assert stream.exposure_us == 12345.0

    def test_auto_exposure_flag_threaded_to_backend(self):
        stream = build_camera_stream(gain_db=5.0, auto_exposure=True)
        assert stream._backend.auto_exposure is True

    def test_auto_exposure_defaults_to_false(self):
        stream = build_camera_stream(gain_db=5.0)
        assert stream._backend.auto_exposure is False


class TestArgumentParsing:

    def setup_method(self):
        self.parser = build_parser()

    # --- baseline ---

    def test_baseline_requires_gain_db(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["baseline"])

    def test_baseline_defaults(self):
        args = self.parser.parse_args(["baseline", "--gain-db", "1.0"])
        assert args.gain_db == 1.0
        assert args.auto_exposure is False
        assert args.exposure_us is None
        assert args.n_frames == 50

    def test_baseline_manual_exposure(self):
        args = self.parser.parse_args(
            ["baseline", "--gain-db", "1.0", "--exposure-us", "2000"]
        )
        assert args.exposure_us == 2000.0
        assert args.auto_exposure is False

    def test_baseline_auto_exposure(self):
        args = self.parser.parse_args(["baseline", "--gain-db", "1.0", "--auto-exposure"])
        assert args.auto_exposure is True
        assert args.exposure_us is None

    def test_baseline_auto_and_manual_exposure_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(
                ["baseline", "--gain-db", "1.0", "--auto-exposure", "--exposure-us", "2000"]
            )

    # --- flat-field ---

    def test_flat_field_requires_gain_db(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["flat-field"])

    def test_flat_field_exposure_args_present(self):
        args = self.parser.parse_args(
            ["flat-field", "--gain-db", "1.0", "--auto-exposure"]
        )
        assert args.auto_exposure is True

    # --- bad-pixel-map ---

    def test_bad_pixel_map_no_required_args(self):
        args = self.parser.parse_args(["bad-pixel-map"])
        assert args.flat_field is None

    def test_bad_pixel_map_explicit_flat_field(self):
        args = self.parser.parse_args(["bad-pixel-map", "--flat-field", "foo.npz"])
        assert args.flat_field == "foo.npz"

    # --- spatial ---

    def test_spatial_no_required_args(self):
        args = self.parser.parse_args(["spatial"])
        assert args.scale_factor is None

    def test_spatial_set_scale_factor(self):
        args = self.parser.parse_args(["spatial", "--scale-factor", "1.62"])
        assert args.scale_factor == 1.62

    # --- conversion-gain ---

    def test_conversion_gain_requires_all_sweep_args(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["conversion-gain", "--gain-db", "1.0"])

    def test_conversion_gain_full_args(self):
        args = self.parser.parse_args(
            [
                "conversion-gain",
                "--gain-db", "1.0",
                "--exposure-min-us", "100",
                "--exposure-max-us", "5000",
                "--n-levels", "10",
                "--n-frames-per-level", "5",
            ]
        )
        assert args.exposure_min_us == 100.0
        assert args.exposure_max_us == 5000.0
        assert args.n_levels == 10
        assert args.n_frames_per_level == 5

    def test_conversion_gain_has_no_exposure_us_flag(self):
        # Conversion gain sweeps exposure itself -- --exposure-us/
        # --auto-exposure (baseline/flat-field's fixed-exposure flags)
        # don't apply here and must not be accepted.
        with pytest.raises(SystemExit):
            self.parser.parse_args(
                [
                    "conversion-gain",
                    "--gain-db", "1.0",
                    "--exposure-min-us", "100",
                    "--exposure-max-us", "5000",
                    "--n-levels", "10",
                    "--n-frames-per-level", "5",
                    "--exposure-us", "2000",
                ]
            )

    # --- noise-model ---

    def test_noise_model_no_required_args(self):
        args = self.parser.parse_args(["noise-model"])
        assert args.baseline is None
        assert args.conversion_gain is None

    # --- top level ---

    def test_no_command_exits(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args([])

    def test_unknown_command_exits(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["not-a-real-command"])
