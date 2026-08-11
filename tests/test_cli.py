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
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.cli import calibration as cli
from pipeline.calibration.spectral import load_spectral_calibration
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
        assert path == Path("/tmp/abs/foo.npz")

    def test_relative_path_without_output_dir_used_as_is(self):
        path = resolve_artifact_path(None, "relative/foo.npz", "baseline.npz")
        assert path == Path("relative/foo.npz")

    def test_relative_path_with_output_dir_joined(self):
        path = resolve_artifact_path("custom_dir", "sub/foo.npz", "baseline.npz")
        assert path == Path("custom_dir") / "sub/foo.npz"


class TestBuildCameraStream:

    def test_exposure_us_required_without_auto_exposure(self):
        # No implicit config-file default -- the caller (CLI flag or GUI
        # field) must supply an explicit exposure_us, or opt into
        # auto_exposure instead.
        with pytest.raises(ValueError):
            build_camera_stream(gain_db=5.0)

    def test_explicit_exposure_us_used(self):
        stream = build_camera_stream(gain_db=5.0, exposure_us=12345.0)
        assert stream.exposure_us == 12345.0
        assert stream.gain_db == 5.0

    def test_auto_exposure_flag_threaded_to_backend(self):
        stream = build_camera_stream(gain_db=5.0, auto_exposure=True)
        assert stream._backend.auto_exposure is True

    def test_auto_exposure_defaults_to_false(self):
        stream = build_camera_stream(gain_db=5.0, exposure_us=1000.0)
        assert stream._backend.auto_exposure is False


class TestArgumentParsing:

    def setup_method(self):
        self.parser = build_parser()

    # --- baseline ---

    def test_baseline_requires_gain_db(self):
        with pytest.raises(SystemExit):
            self.parser.parse_args(["baseline"])

    def test_baseline_requires_exposure_choice(self):
        # Neither --exposure-us nor --auto-exposure given -- no implicit
        # config-file default, so this must fail to parse.
        with pytest.raises(SystemExit):
            self.parser.parse_args(["baseline", "--gain-db", "1.0"])

    def test_baseline_defaults(self):
        args = self.parser.parse_args(
            ["baseline", "--gain-db", "1.0", "--exposure-us", "2000"]
        )
        assert args.gain_db == 1.0
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


# ---------------------------------------------------------------------------
# spectral-capture / spectral-manual subcommands
# ---------------------------------------------------------------------------
#
# Deliberately does not touch a real camera anywhere -- spectral-capture's
# CameraStream/run_spectral_calibration wiring is exercised only through
# monkeypatched stand-ins, the same scope every other camera-touching
# subcommand in this module (baseline, flat-field, conversion-gain) is
# left untested against real hardware. spectral-manual touches no camera
# at all, so it's exercised for real (write + load an actual artifact).


class TestSpectralManualArgumentParsing:

    def test_parses_coefficients_and_sigma(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(cli, "_cmd_spectral_manual", lambda args: captured.update(vars(args)))

        cli.main([
            "spectral-manual",
            "--coefficients", "400.0", "0.5",
            "--coefficient-sigma", "0.1", "0.01",
            "--path", str(tmp_path / "spectral.npz"),
        ])

        assert captured["coefficients"] == [400.0, 0.5]
        assert captured["coefficient_sigma"] == [0.1, 0.01]
        assert captured["path"] == str(tmp_path / "spectral.npz")

    def test_coefficients_required(self):
        with pytest.raises(SystemExit):
            cli.main(["spectral-manual", "--coefficient-sigma", "0.1", "0.01"])

    def test_coefficient_sigma_required(self):
        with pytest.raises(SystemExit):
            cli.main(["spectral-manual", "--coefficients", "400.0", "0.5"])

    def test_mismatched_lengths_raises_argparse_error(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main([
                "spectral-manual",
                "--coefficients", "400.0", "0.5",
                "--coefficient-sigma", "0.1",
            ])

        assert exc_info.value.code == 2
        assert "same length" in capsys.readouterr().err


class TestSpectralManualEndToEnd:

    '''
    Manual mode touches no camera at all, so -- unlike spectral-capture --
    it can be exercised for real: build args via the CLI's own parser,
    run the resulting func(args), and confirm a real spectral.npz
    artifact is written and loads back correctly.
    '''

    def test_writes_and_loads_artifact(self, tmp_path):
        path = tmp_path / "spectral.npz"

        cli.main([
            "spectral-manual",
            "--coefficients", "400.0", "0.5",
            "--coefficient-sigma", "0.2", "0.01",
            "--path", str(path),
        ])

        assert path.exists()
        loaded = load_spectral_calibration(path)
        assert loaded.fit.degree == 1
        assert np.allclose(loaded.fit.coefficients, [400.0, 0.5])
        assert np.allclose(loaded.fit.coefficient_sigma, [0.2, 0.01])
        assert loaded.record.source_frame_count == 1
        assert np.isclose(loaded.wavelength_nm(np.array([0.0]))[0], 400.0)

    def test_output_dir_relative_path(self, tmp_path):
        cli.main([
            "spectral-manual",
            "--coefficients", "400.0", "0.5",
            "--coefficient-sigma", "0.2", "0.01",
            "--output-dir", str(tmp_path),
        ])

        assert (tmp_path / cli.DEFAULT_SPECTRAL_FILENAME).exists()


class TestSpectralCaptureArgumentParsing:

    def test_parses_arguments(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(cli, "_cmd_spectral_capture", lambda args: captured.update(vars(args)))

        cli.main([
            "spectral-capture",
            "--gain-db", "2.5",
            "--exposure-us", "500",
            "--n-frames", "20",
            "--degree", "2",
            "--baseline", "custom_baseline.npz",
            "--flat-field", "custom_flat.npz",
            "--bad-pixel-map", "custom_mask.npz",
            "--path", str(tmp_path / "spectral.npz"),
        ])

        assert captured["gain_db"] == 2.5
        assert captured["n_frames"] == 20
        assert captured["degree"] == 2
        assert captured["baseline"] == "custom_baseline.npz"
        assert captured["flat_field"] == "custom_flat.npz"
        assert captured["bad_pixel_map"] == "custom_mask.npz"

    def test_defaults(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli, "_cmd_spectral_capture", lambda args: captured.update(vars(args)))

        cli.main(["spectral-capture", "--gain-db", "0.0", "--auto-exposure"])

        assert captured["n_frames"] == cli.DEFAULT_N_FRAMES
        assert captured["degree"] == cli.DEFAULT_SPECTRAL_DEGREE
        assert captured["baseline"] is None
        assert captured["flat_field"] is None
        assert captured["bad_pixel_map"] is None
        assert captured["path"] is None
        assert captured["auto_exposure"] is True
        assert captured["exposure_us"] is None

    def test_gain_db_required(self):
        with pytest.raises(SystemExit):
            cli.main(["spectral-capture", "--auto-exposure"])

    def test_requires_exposure_choice(self):
        with pytest.raises(SystemExit):
            cli.main(["spectral-capture", "--gain-db", "1.0"])

    def test_auto_and_manual_exposure_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            cli.main([
                "spectral-capture", "--gain-db", "1.0",
                "--auto-exposure", "--exposure-us", "2000",
            ])


class TestSpectralCapturePathResolution:

    '''
    Exercises _cmd_spectral_capture()'s own default-path-resolution logic
    for real, with every collaborator that would otherwise touch a camera
    or the filesystem replaced by a recording stand-in.
    '''

    def _patch_collaborators(self, monkeypatch, recorded):

        def fake_load_baseline(path):
            recorded["baseline_path"] = path
            return SimpleNamespace(baseline=np.zeros((2, 2)), background_sigma=1.0), "baseline_record"

        def fake_load_flat_field(path):
            recorded["flat_field_path"] = path
            return np.ones((2, 2)), "flat_field_record"

        def fake_load_bad_pixel_map(path):
            recorded["bad_pixel_map_path"] = path
            return np.zeros((2, 2), dtype=bool), "bad_pixel_record"

        class FakeStream:
            is_running = False

            def start(self):
                recorded["stream_started"] = True

            def stop(self):
                recorded["stream_stopped"] = True

        def fake_build_camera_stream(gain_db, *, exposure_us=None, auto_exposure=False, **kwargs):
            recorded["gain_db"] = gain_db
            recorded["exposure_us"] = exposure_us
            recorded["auto_exposure"] = auto_exposure
            return FakeStream()

        def fake_run_spectral_calibration(stream, n_frames, sensor_calibration, path, degree=1):
            recorded["n_frames"] = n_frames
            recorded["degree"] = degree
            recorded["path"] = path
            recorded["sensor_calibration"] = sensor_calibration

        monkeypatch.setattr(cli, "load_baseline", fake_load_baseline)
        monkeypatch.setattr(cli, "load_flat_field", fake_load_flat_field)
        monkeypatch.setattr(cli, "load_bad_pixel_map", fake_load_bad_pixel_map)
        monkeypatch.setattr(cli, "build_camera_stream", fake_build_camera_stream)
        monkeypatch.setattr(cli, "run_spectral_calibration", fake_run_spectral_calibration)

    def test_defaults_input_artifact_paths_under_output_dir(self, monkeypatch, tmp_path):
        recorded = {}
        self._patch_collaborators(monkeypatch, recorded)

        cli.main([
            "spectral-capture", "--gain-db", "1.0", "--auto-exposure",
            "--output-dir", str(tmp_path),
        ])

        assert recorded["baseline_path"] == tmp_path / cli.DEFAULT_BASELINE_FILENAME
        assert recorded["flat_field_path"] == tmp_path / cli.DEFAULT_FLAT_FIELD_FILENAME
        assert recorded["bad_pixel_map_path"] == tmp_path / cli.DEFAULT_BAD_PIXEL_MAP_FILENAME
        assert recorded["path"] == tmp_path / cli.DEFAULT_SPECTRAL_FILENAME
        assert recorded["n_frames"] == cli.DEFAULT_N_FRAMES
        assert recorded["degree"] == cli.DEFAULT_SPECTRAL_DEGREE
        assert recorded["sensor_calibration"].background_sigma == 1.0

    def test_explicit_artifact_paths_override_defaults(self, monkeypatch, tmp_path):
        recorded = {}
        self._patch_collaborators(monkeypatch, recorded)

        custom_baseline = tmp_path / "my_baseline.npz"
        cli.main([
            "spectral-capture", "--gain-db", "1.0", "--auto-exposure",
            "--baseline", str(custom_baseline),
        ])

        assert recorded["baseline_path"] == custom_baseline
        assert recorded["flat_field_path"] == Path(cli.DEFAULT_ARTIFACT_DIR) / cli.DEFAULT_FLAT_FIELD_FILENAME
        assert recorded["stream_started"] is True
