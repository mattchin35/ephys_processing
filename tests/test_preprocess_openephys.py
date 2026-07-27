"""Tests for Open Ephys raw-data validation helpers.

The tests mock SpikeInterface so validation logic can be checked without
depending on large local binary recordings.
"""

from pathlib import Path
import importlib
import sys

import numpy as np
import pytest


MODULE_NAME = "src.preprocessing.preprocess_openephys"


def reload_preprocess_openephys_module():
    """Import the Open Ephys preprocessing module fresh; returns the module object."""
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_module_import_does_not_query_real_data(monkeypatch) -> None:
    """Importing the module must not inspect raw files or load recordings."""
    import spikeinterface.extractors as se
    from spikeinterface.extractors.extractor_classes import OpenEphysBinaryRecordingExtractor

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Open Ephys discovery should not run at import time")

    monkeypatch.setattr(
        OpenEphysBinaryRecordingExtractor,
        "get_available_experiments",
        staticmethod(fail_if_called),
    )
    monkeypatch.setattr(
        OpenEphysBinaryRecordingExtractor,
        "get_streams",
        staticmethod(fail_if_called),
    )
    monkeypatch.setattr(se, "read_openephys", fail_if_called)

    reload_preprocess_openephys_module()


def test_find_open_ephys_experiments_returns_plain_strings(monkeypatch) -> None:
    """Experiment discovery returns Python str names; names are Open Ephys experiment labels."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module.OpenEphysBinaryRecordingExtractor,
        "get_available_experiments",
        staticmethod(lambda folder_path: [np.str_("experiment1")]),
    )

    experiment_names = module.find_open_ephys_experiments(Path("/data/session"))

    assert experiment_names == ["experiment1"]
    assert type(experiment_names[0]) is str


def test_find_open_ephys_experiments_rejects_empty_result(monkeypatch) -> None:
    """Experiment discovery raises ValueError when no Open Ephys experiments are found."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module.OpenEphysBinaryRecordingExtractor,
        "get_available_experiments",
        staticmethod(lambda folder_path: []),
    )

    with pytest.raises(ValueError, match="No Open Ephys experiments"):
        module.find_open_ephys_experiments(Path("/data/session"))


def test_find_open_ephys_streams_returns_expected_dataframe(monkeypatch) -> None:
    """Stream discovery returns one row per stream with parsed stream metadata."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module.OpenEphysBinaryRecordingExtractor,
        "get_streams",
        staticmethod(
            lambda folder_path, experiment_names: (
                [
                    "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "Record Node 101#Neuropix-PXI-100.ProbeB",
                    "Record Node 109#NI-DAQmx-106.PXI-6133",
                ],
                ["0", "1", "2"],
            )
        ),
    )

    streams = module.find_open_ephys_streams(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
    )

    assert list(streams.columns) == [
        "stream_id",
        "stream_name",
        "record_node",
        "source_name",
        "is_neuropixels",
        "is_nidaq",
    ]
    assert streams.shape == (3, 6)
    assert streams.loc[0, "record_node"] == "Record Node 101"
    assert streams.loc[0, "source_name"] == "Neuropix-PXI-100.ProbeA"
    assert bool(streams.loc[0, "is_neuropixels"])
    assert bool(streams.loc[2, "is_nidaq"])


def test_find_open_ephys_streams_rejects_invalid_experiment_name(monkeypatch) -> None:
    """Invalid experiment names are reported with a clear ValueError."""
    module = reload_preprocess_openephys_module()

    def raise_neo_error(folder_path, experiment_names):
        raise KeyError(0)

    monkeypatch.setattr(
        module.OpenEphysBinaryRecordingExtractor,
        "get_streams",
        staticmethod(raise_neo_error),
    )

    with pytest.raises(ValueError, match="Could not find streams.*experiment1"):
        module.find_open_ephys_streams(
            raw_root=Path("/data/session"),
            experiment_name="experiment1",
        )


def test_select_neuropixels_stream_uses_single_candidate() -> None:
    """A single Neuropixels stream can be selected without an explicit stream name."""
    module = reload_preprocess_openephys_module()
    streams = module.pd.DataFrame(
        [
            {
                "stream_id": "0",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                "record_node": "Record Node 101",
                "source_name": "Neuropix-PXI-100.ProbeA",
                "is_neuropixels": True,
                "is_nidaq": False,
            },
            {
                "stream_id": "1",
                "stream_name": "Record Node 109#NI-DAQmx-106.PXI-6133",
                "record_node": "Record Node 109",
                "source_name": "NI-DAQmx-106.PXI-6133",
                "is_neuropixels": False,
                "is_nidaq": True,
            },
        ]
    )

    stream_name = module.select_neuropixels_stream(streams)

    assert stream_name == "Record Node 101#Neuropix-PXI-100.ProbeA"


def test_select_neuropixels_stream_requires_name_for_multiple_candidates() -> None:
    """Multiple Neuropixels streams require an explicit stream name."""
    module = reload_preprocess_openephys_module()
    streams = module.pd.DataFrame(
        [
            {
                "stream_id": "0",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                "record_node": "Record Node 101",
                "source_name": "Neuropix-PXI-100.ProbeA",
                "is_neuropixels": True,
                "is_nidaq": False,
            },
            {
                "stream_id": "1",
                "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeB",
                "record_node": "Record Node 101",
                "source_name": "Neuropix-PXI-100.ProbeB",
                "is_neuropixels": True,
                "is_nidaq": False,
            },
        ]
    )

    with pytest.raises(ValueError, match="Multiple Neuropixels streams"):
        module.select_neuropixels_stream(streams)


def test_load_open_ephys_stream_uses_validated_arguments(monkeypatch) -> None:
    """Stream loading passes explicit Open Ephys identifiers to SpikeInterface."""
    module = reload_preprocess_openephys_module()
    calls = []
    expected_recording = object()

    def fake_read_openephys(**kwargs):
        calls.append(kwargs)
        return expected_recording

    monkeypatch.setattr(module.se, "read_openephys", fake_read_openephys)

    recording = module.load_open_ephys_stream(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
        stream_name="Record Node 101#Neuropix-PXI-100.ProbeA",
    )

    assert recording is expected_recording
    assert calls == [
        {
            "folder_path": Path("/data/session"),
            "experiment_name": "experiment1",
            "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
            "load_sync_timestamps": False,
        }
    ]


class FakeProbe:
    """Minimal probe object used to verify geometry detection."""


class FakeRecording:
    """Small SpikeInterface-like recording double for summary tests."""

    def get_sampling_frequency(self):
        """Return sampling frequency in Hz."""
        return 30000.0

    def get_num_segments(self):
        """Return the number of recording segments."""
        return 1

    def get_num_channels(self):
        """Return the number of channels."""
        return 384

    def get_num_samples(self, segment_index=0):
        """Return sample count for one segment."""
        assert segment_index == 0
        return 13336068

    def get_dtype(self):
        """Return the underlying sample dtype."""
        return np.dtype("int16")

    def get_channel_ids(self):
        """Return channel identifiers with shape (n_channels,)."""
        return np.array(["CH0", "CH1", "CH2"])

    def get_property_keys(self):
        """Return available channel property names."""
        return ["location", "inter_sample_shift"]

    def get_probe(self):
        """Return attached probe geometry."""
        return FakeProbe()

    def get_property(self, key):
        """Return channel property arrays with one row per channel."""
        if key == "location":
            return np.zeros((384, 2), dtype=float)
        if key == "inter_sample_shift":
            return np.zeros(384, dtype=float)
        raise KeyError(key)


def test_summarize_open_ephys_stream_reports_core_properties() -> None:
    """Recording summary reports dimensions, units, and available probe metadata."""
    module = reload_preprocess_openephys_module()

    summary = module.summarize_open_ephys_stream(FakeRecording())

    assert summary["sampling_frequency_hz"] == 30000.0
    assert summary["num_segments"] == 1
    assert summary["num_channels"] == 384
    assert summary["num_samples_by_segment"] == [13336068]
    assert summary["dtype"] == "int16"
    assert summary["channel_ids_shape"] == (3,)
    assert summary["property_keys"] == ["location", "inter_sample_shift"]
    assert summary["has_probe"]
    assert summary["has_inter_sample_shift"]
    assert summary["location_shape"] == (384, 2)


def test_main_passes_hardcoded_parameters(monkeypatch) -> None:
    """IDE-oriented main passes editable local parameters to validation."""
    module = reload_preprocess_openephys_module()
    calls = []
    expected_result = {
        "experiment_name": "experiment1",
        "streams": module.pd.DataFrame(),
        "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
        "summary": {},
    }

    def fake_validate_open_ephys_probe(**kwargs):
        calls.append(kwargs)
        return expected_result

    monkeypatch.setattr(module, "validate_open_ephys_probe", fake_validate_open_ephys_probe)

    result = module.main()

    assert result is expected_result
    assert calls == [
        {
            "raw_root": module.Path(
                "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
                "CT026_20260727_alternating_latent/ephys/raw"
            ),
            "experiment_name": "experiment1",
            "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
            "load_sync_timestamps": False,
        }
    ]
