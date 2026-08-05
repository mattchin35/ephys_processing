"""Tests for reusable Open Ephys AP preprocessing helpers."""

from pathlib import Path
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_NAME = "src.preprocessing.ap_preprocessing_openephys"


def reload_ap_module():
    """Import the AP preprocessing module fresh; returns the module object."""
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_module_import_does_not_load_or_write(monkeypatch) -> None:
    """Importing the module must not load raw data or write derived files."""
    import spikeinterface.extractors as se
    import spikeinterface.core as sc

    def fail_if_called(*args, **kwargs):
        raise AssertionError("AP preprocessing should not run at import time")

    monkeypatch.setattr(se, "read_openephys", fail_if_called)
    monkeypatch.setattr(sc, "write_binary_recording", fail_if_called)

    reload_ap_module()


class FakeApRecording:
    """SpikeInterface-like AP recording double with valid metadata."""

    channel_ids = np.array(["CH0", "CH1", "CH2", "CH3"])

    def get_num_segments(self):
        """Return the number of AP segments."""
        return 1

    def get_sampling_frequency(self):
        """Return AP sampling frequency in Hz."""
        return 30000.0

    def get_probe(self):
        """Return attached probe geometry."""
        return object()

    def get_property_keys(self):
        """Return available channel property names."""
        return ["contact_vector", "inter_sample_shift"]

    def get_num_channels(self):
        """Return AP channel count."""
        return 4

    def get_channel_locations(self):
        """Return channel locations in um with shape (n_channels, 2)."""
        return np.array(
            [
                [0.0, 0.0],
                [32.0, 0.0],
                [0.0, 1000.0],
                [32.0, 1000.0],
            ]
        )

    def get_channel_groups(self):
        """Return fallback shank/group labels with shape (n_channels,)."""
        return np.array([0, 0, 1, 1])

    def get_property(self, key):
        """Return channel property arrays with one row per channel."""
        if key != "contact_vector":
            raise KeyError(key)

        contact_vector = np.zeros(
            4,
            dtype=[
                ("shank_ids", "U8"),
            ],
        )
        contact_vector["shank_ids"] = np.array(["left", "left", "right", "right"])
        return contact_vector

    def get_num_samples(self, segment_index=0):
        """Return AP sample count for one segment."""
        assert segment_index == 0
        return 1000

    def get_dtype(self):
        """Return AP recording dtype."""
        return np.dtype("float32")

    def get_traces(self, start_frame=None, end_frame=None, segment_index=0, return_scaled=False):
        """Return deterministic AP sample chunks with shape (n_samples, n_channels)."""
        assert segment_index == 0
        assert return_scaled is False
        n_samples = end_frame - start_frame
        return np.full((n_samples, self.get_num_channels()), 10.0, dtype=np.float32)


class FakeMultiSegmentRecording(FakeApRecording):
    """AP recording double with multiple segments."""

    def get_num_segments(self):
        """Return more than one segment."""
        return 2


class FakeWrongSamplingRateRecording(FakeApRecording):
    """AP recording double with non-AP sampling frequency."""

    def get_sampling_frequency(self):
        """Return a non-30-kHz sampling frequency in Hz."""
        return 2500.0


class FakeNoProbeRecording(FakeApRecording):
    """AP recording double without probe geometry."""

    def get_probe(self):
        """Return no attached probe geometry."""
        return None


class FakeNoInterSampleShiftRecording(FakeApRecording):
    """AP recording double without Neuropixels shift metadata."""

    def get_property_keys(self):
        """Return channel property names without inter_sample_shift."""
        return ["contact_vector"]


class FakeCrossShankRecording(FakeApRecording):
    """AP recording double with overlapping shank geometry."""

    def get_channel_locations(self):
        """Return cross-shank locations within local-CAR radius."""
        return np.array(
            [
                [0.0, 0.0],
                [32.0, 0.0],
                [0.0, 25.0],
                [32.0, 25.0],
            ]
        )


class FakeOutOfRangeRecording(FakeApRecording):
    """AP recording double with samples outside int16 range."""

    def get_traces(self, start_frame=None, end_frame=None, segment_index=0, return_scaled=False):
        """Return out-of-range sample chunks."""
        n_samples = end_frame - start_frame
        return np.full((n_samples, self.get_num_channels()), 40000.0, dtype=np.float32)


class FakeScaledApRecording(FakeApRecording):
    """AP recording double with Open Ephys voltage-scaling metadata."""

    def get_property_keys(self):
        """Return available channel property names including voltage scaling."""
        return [
            "contact_vector",
            "inter_sample_shift",
            "gain_to_uV",
            "offset_to_uV",
            "physical_unit",
        ]

    def get_property(self, key):
        """Return channel property arrays with one row per channel."""
        if key == "gain_to_uV":
            return np.array([0.195, 0.195, 0.195, 0.195], dtype=float)
        if key == "offset_to_uV":
            return np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        if key == "physical_unit":
            return np.array(["uV", "uV", "uV", "uV"], dtype=object)
        return super().get_property(key)


class FakeNonUniformGainApRecording(FakeScaledApRecording):
    """AP recording double with non-uniform voltage gains."""

    def get_property(self, key):
        """Return non-uniform gains for the gain property."""
        if key == "gain_to_uV":
            return np.array([0.195, 0.195, 0.25, 0.195], dtype=float)
        return super().get_property(key)


def patch_identity_preprocessing(monkeypatch, module) -> None:
    """Patch SpikeInterface preprocessing calls to return the same fake recording."""
    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: recording,
        highpass_filter=lambda recording, **kwargs: recording,
        common_reference=lambda recording, **kwargs: recording,
    )
    monkeypatch.setattr(module, "spre", fake_spre)


def test_validate_ap_recording_rejects_multiple_segments() -> None:
    """AP preprocessing requires a single segment for one exported binary."""
    module = reload_ap_module()

    with pytest.raises(ValueError, match="one segment"):
        module.validate_ap_recording(FakeMultiSegmentRecording())


def test_validate_ap_recording_rejects_non_30khz_sampling() -> None:
    """AP preprocessing requires approximately 30-kHz input data."""
    module = reload_ap_module()

    with pytest.raises(ValueError, match="30 kHz"):
        module.validate_ap_recording(FakeWrongSamplingRateRecording())


def test_validate_ap_recording_requires_probe_geometry() -> None:
    """AP preprocessing requires attached probe geometry."""
    module = reload_ap_module()

    with pytest.raises(ValueError, match="probe geometry"):
        module.validate_ap_recording(FakeNoProbeRecording())


def test_validate_ap_recording_requires_inter_sample_shift() -> None:
    """AP preprocessing requires Neuropixels inter-sample-shift metadata."""
    module = reload_ap_module()

    with pytest.raises(ValueError, match="inter-sample-shift"):
        module.validate_ap_recording(FakeNoInterSampleShiftRecording())


def test_validate_local_car_geometry_uses_contact_vector_shank_ids() -> None:
    """Local-CAR geometry validation uses contact-vector shank IDs when available."""
    module = reload_ap_module()

    module.validate_local_car_geometry(
        recording=FakeApRecording(),
        outer_radius_um=140.0,
    )


def test_validate_local_car_geometry_rejects_cross_shank_neighbors_within_radius() -> None:
    """Local-CAR geometry validation rejects cross-shank neighbors inside radius."""
    module = reload_ap_module()

    with pytest.raises(RuntimeError, match="different shanks"):
        module.validate_local_car_geometry(
            recording=FakeCrossShankRecording(),
            outer_radius_um=140.0,
        )


def test_estimate_output_range_is_deterministic_for_seed() -> None:
    """Range QC samples deterministic chunks when given the same seed."""
    module = reload_ap_module()

    first_range = module.estimate_output_range(
        recording=FakeApRecording(),
        num_random_chunks=3,
        chunk_duration_s=0.01,
        random_seed=7,
    )
    second_range = module.estimate_output_range(
        recording=FakeApRecording(),
        num_random_chunks=3,
        chunk_duration_s=0.01,
        random_seed=7,
    )

    assert first_range == second_range == (10.0, 10.0)


def test_get_recording_scaling_metadata_reads_gain_offset_and_units() -> None:
    """Scaling metadata captures Open Ephys gain, offset, unit, and channel order."""
    module = reload_ap_module()

    scaling_metadata = module.get_recording_scaling_metadata(FakeScaledApRecording())

    assert scaling_metadata == {
        "has_scaleable_traces": True,
        "channel_ids": ["CH0", "CH1", "CH2", "CH3"],
        "gain_to_uV_by_channel": [0.195, 0.195, 0.195, 0.195],
        "offset_to_uV_by_channel": [1.0, 2.0, 3.0, 4.0],
        "physical_unit_by_channel": ["uV", "uV", "uV", "uV"],
    }


def test_get_recording_scaling_metadata_handles_missing_scaling() -> None:
    """Missing raw scaling metadata is represented explicitly without crashing."""
    module = reload_ap_module()

    scaling_metadata = module.get_recording_scaling_metadata(FakeApRecording())

    assert scaling_metadata == {
        "has_scaleable_traces": False,
        "channel_ids": ["CH0", "CH1", "CH2", "CH3"],
        "gain_to_uV_by_channel": None,
        "offset_to_uV_by_channel": None,
        "physical_unit_by_channel": None,
    }


def test_build_ap_binary_scaling_uses_zero_offsets_for_filtered_output() -> None:
    """AP high-pass/CAR output should not reintroduce the raw DC offset."""
    module = reload_ap_module()

    ap_binary_scaling = module.build_ap_binary_scaling_metadata(
        recording=FakeScaledApRecording(),
        export_scale_factor=1.0,
    )

    assert ap_binary_scaling["has_scaleable_traces"]
    assert ap_binary_scaling["gain_to_uV_by_channel"] == [0.195, 0.195, 0.195, 0.195]
    assert ap_binary_scaling["offset_to_uV_by_channel"] == [0.0, 0.0, 0.0, 0.0]
    assert ap_binary_scaling["data_units"] == "unscaled_binary_values"


def test_build_ap_binary_scaling_adjusts_gain_for_export_scale_factor() -> None:
    """If export scaling is added later, the saved gain is adjusted accordingly."""
    module = reload_ap_module()

    ap_binary_scaling = module.build_ap_binary_scaling_metadata(
        recording=FakeScaledApRecording(),
        export_scale_factor=2.0,
    )

    assert ap_binary_scaling["gain_to_uV_by_channel"] == [0.0975, 0.0975, 0.0975, 0.0975]
    assert ap_binary_scaling["export_scale_factor"] == 2.0


def test_build_ap_binary_scaling_warns_if_channel_gains_differ() -> None:
    """Non-uniform gains are allowed but documented with a warning."""
    module = reload_ap_module()

    with pytest.warns(UserWarning, match="Non-uniform gain_to_uV"):
        ap_binary_scaling = module.build_ap_binary_scaling_metadata(
            recording=FakeNonUniformGainApRecording(),
            export_scale_factor=1.0,
        )

    assert ap_binary_scaling["gain_to_uV_by_channel"] == [0.195, 0.195, 0.25, 0.195]


def test_preprocess_ap_for_kilosort_runs_phase_shift_highpass_car_order(
    monkeypatch,
    tmp_path,
) -> None:
    """AP workflow runs phase shift, high-pass, then local CAR before writing."""
    module = reload_ap_module()
    raw_recording = FakeApRecording()
    shifted_recording = object()
    highpass_recording = FakeApRecording()
    car_recording = FakeApRecording()
    calls = []

    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: calls.append(
            ("phase_shift", recording, dtype)
        ) or shifted_recording,
        highpass_filter=lambda recording, **kwargs: calls.append(
            ("highpass_filter", recording, kwargs)
        ) or highpass_recording,
        common_reference=lambda recording, **kwargs: calls.append(
            ("common_reference", recording, kwargs)
        ) or car_recording,
    )

    def fake_write_binary_recording(**kwargs):
        calls.append(("write_binary_recording", kwargs))

    monkeypatch.setattr(module, "spre", fake_spre)
    monkeypatch.setattr(module, "estimate_output_range", lambda **kwargs: (-10.0, 10.0))
    monkeypatch.setattr(module, "write_binary_recording", fake_write_binary_recording)

    module.preprocess_ap_for_kilosort(
        recording=raw_recording,
        output_folder=tmp_path,
        progress_bar=False,
    )

    assert calls == [
        ("phase_shift", raw_recording, "float32"),
        (
            "highpass_filter",
            shifted_recording,
            {
                "freq_min": 300.0,
                "filter_order": 3,
                "ftype": "butter",
                "filter_mode": "sos",
                "direction": "forward-backward",
                "margin_ms": "auto",
                "dtype": "float32",
            },
        ),
        (
            "common_reference",
            highpass_recording,
            {
                "reference": "local",
                "operator": "average",
                "local_radius": (40.0, 140.0),
                "min_local_neighbors": 5,
                "dtype": "float32",
            },
        ),
        (
            "write_binary_recording",
            {
                "recording": car_recording,
                "file_paths": tmp_path / "ap_preprocessed.dat",
                "dtype": "int16",
                "add_file_extension": False,
                "n_jobs": 8,
                "chunk_duration": "1s",
                "progress_bar": False,
                "verbose": True,
            },
        ),
    ]


def test_preprocess_ap_for_kilosort_writes_metadata_json(monkeypatch, tmp_path) -> None:
    """AP workflow writes sidecar metadata beside the output binary."""
    module = reload_ap_module()

    patch_identity_preprocessing(monkeypatch, module)
    monkeypatch.setattr(module, "estimate_output_range", lambda **kwargs: (-10.0, 10.0))
    monkeypatch.setattr(module, "write_binary_recording", lambda **kwargs: None)

    result = module.preprocess_ap_for_kilosort(
        recording=FakeApRecording(),
        output_folder=tmp_path,
        progress_bar=False,
    )

    metadata_path = tmp_path / "ap_preprocessing.json"
    assert metadata_path.is_file()
    assert result["ap_metadata_path"] == metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["output_binary"] == "ap_preprocessed.dat"
    assert metadata["binary_layout"] == "time_major_channel_interleaved"
    assert metadata["random_seed"] == 0
    assert metadata["preprocessing"][0]["name"] == "neuropixels_phase_shift"
    assert metadata["preprocessing"][1]["name"] == "highpass_filter"
    assert metadata["preprocessing"][2]["name"] == "local_common_average_reference"
    assert "raw_recording_scaling" in metadata
    assert "ap_binary_scaling" in metadata


def test_preprocess_ap_for_kilosort_writes_scaling_metadata(monkeypatch, tmp_path) -> None:
    """AP metadata records how to convert exported binary values to uV."""
    module = reload_ap_module()

    patch_identity_preprocessing(monkeypatch, module)
    monkeypatch.setattr(module, "estimate_output_range", lambda **kwargs: (-10.0, 10.0))
    monkeypatch.setattr(module, "write_binary_recording", lambda **kwargs: None)

    module.preprocess_ap_for_kilosort(
        recording=FakeScaledApRecording(),
        output_folder=tmp_path,
        progress_bar=False,
    )

    metadata = json.loads((tmp_path / "ap_preprocessing.json").read_text(encoding="utf-8"))
    assert metadata["raw_recording_scaling"]["gain_to_uV_by_channel"] == [0.195, 0.195, 0.195, 0.195]
    assert metadata["raw_recording_scaling"]["offset_to_uV_by_channel"] == [1.0, 2.0, 3.0, 4.0]
    assert metadata["ap_binary_scaling"]["gain_to_uV_by_channel"] == [0.195, 0.195, 0.195, 0.195]
    assert metadata["ap_binary_scaling"]["offset_to_uV_by_channel"] == [0.0, 0.0, 0.0, 0.0]
    assert metadata["ap_binary_scaling"]["conversion"] == "trace_uV = trace_value * gain_to_uV + offset_to_uV"


def test_preprocess_ap_for_kilosort_returns_output_summary(monkeypatch, tmp_path) -> None:
    """AP workflow returns output paths, dimensions, dtype, and QC range."""
    module = reload_ap_module()

    patch_identity_preprocessing(monkeypatch, module)
    monkeypatch.setattr(module, "estimate_output_range", lambda **kwargs: (-10.0, 10.0))
    monkeypatch.setattr(module, "write_binary_recording", lambda **kwargs: None)

    result = module.preprocess_ap_for_kilosort(
        recording=FakeApRecording(),
        output_folder=tmp_path,
        progress_bar=False,
    )

    assert result["ap_binary_path"] == tmp_path / "ap_preprocessed.dat"
    assert result["ap_metadata_path"] == tmp_path / "ap_preprocessing.json"
    assert result["sampling_frequency_hz"] == 30000.0
    assert result["num_channels"] == 4
    assert result["num_segments"] == 1
    assert result["num_samples_by_segment"] == [1000]
    assert result["dtype"] == "int16"
    assert result["estimated_random_chunk_min"] == -10.0
    assert result["estimated_random_chunk_max"] == 10.0


def test_preprocess_ap_for_kilosort_rejects_int16_range_overflow(
    monkeypatch,
    tmp_path,
) -> None:
    """AP workflow rejects unsafe integer casts based on sampled output range."""
    module = reload_ap_module()

    patch_identity_preprocessing(monkeypatch, module)
    monkeypatch.setattr(module, "estimate_output_range", lambda **kwargs: (-10.0, 40000.0))
    monkeypatch.setattr(
        module,
        "write_binary_recording",
        lambda **kwargs: pytest.fail("Binary should not be written after range failure"),
    )

    with pytest.raises(RuntimeError, match="int16 range"):
        module.preprocess_ap_for_kilosort(
            recording=FakeApRecording(),
            output_folder=tmp_path,
            progress_bar=False,
        )


def test_main_loads_stream_and_runs_ap_workflow(monkeypatch, tmp_path) -> None:
    """IDE-oriented main loads the hardcoded stream and runs AP preprocessing."""
    module = reload_ap_module()
    expected_recording = FakeApRecording()
    expected_output_folder = tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA"
    expected_result = {"ap_binary_path": expected_output_folder / "ap_preprocessed.dat"}
    calls = []

    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda **kwargs: calls.append(("load", kwargs)) or expected_recording,
    )
    monkeypatch.setattr(
        module,
        "build_stream_output_dir",
        lambda **kwargs: calls.append(("build_dir", kwargs)) or expected_output_folder,
    )
    monkeypatch.setattr(
        module,
        "preprocess_ap_for_kilosort",
        lambda **kwargs: calls.append(("preprocess", kwargs)) or expected_result,
    )

    result = module.main()

    assert result is expected_result
    assert calls == [
        (
            "load",
            {
                "raw_root": module.Path(
                    "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
                    "CT026_20260727_alternating_latent/ephys/raw"
                ),
                "experiment_name": "experiment1",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                "load_sync_timestamps": False,
            },
        ),
        (
            "build_dir",
            {
                "output_root": module.Path(
                    "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
                    "CT026_20260727_alternating_latent/ephys/derived"
                ),
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
            },
        ),
        (
            "preprocess",
            {
                "recording": expected_recording,
                "output_folder": expected_output_folder,
                "highpass_hz": 300.0,
                "local_car_inner_um": 40.0,
                "local_car_outer_um": 140.0,
                "min_local_neighbors": 5,
                "working_dtype": "float32",
                "output_dtype": "int16",
                "n_jobs": 8,
                "chunk_duration": "1s",
                "num_random_chunks": 20,
                "random_seed": 0,
                "progress_bar": True,
            },
        ),
    ]
