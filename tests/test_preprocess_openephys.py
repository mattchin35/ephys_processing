"""Tests for Open Ephys raw-data validation helpers.

The tests mock SpikeInterface so validation logic can be checked without
depending on large local binary recordings.
"""

from pathlib import Path
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio


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


def test_make_safe_path_component_replaces_path_hostile_characters() -> None:
    """Stream names are converted to readable single path components."""
    module = reload_preprocess_openephys_module()

    safe_name = module.make_safe_path_component(
        "Record Node 101#Neuropix-PXI-100.ProbeA"
    )

    assert safe_name == "Record_Node_101_Neuropix-PXI-100.ProbeA"


def test_build_stream_output_dir_creates_stream_directory(tmp_path) -> None:
    """Derived output directories are created per stream under output_root."""
    module = reload_preprocess_openephys_module()

    stream_output_dir = module.build_stream_output_dir(
        output_root=tmp_path,
        stream_name="Record Node 101#Neuropix-PXI-100.ProbeA",
    )

    assert stream_output_dir == tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA"
    assert stream_output_dir.is_dir()


class FakeRecordingWithoutProbe(FakeRecording):
    """SpikeInterface-like recording double with no attached probe geometry."""

    def get_probe(self):
        """Return no probe geometry."""
        return None


class FakeRecordingWithoutInterSampleShift(FakeRecording):
    """SpikeInterface-like recording double without Neuropixels shift metadata."""

    def get_property_keys(self):
        """Return available channel property names."""
        return ["location"]


def test_plot_probe_channel_map_rejects_missing_probe(tmp_path) -> None:
    """Probe layout plotting requires attached probe geometry."""
    module = reload_preprocess_openephys_module()

    with pytest.raises(ValueError, match="No probe geometry"):
        module.plot_probe_channel_map(
            recording=FakeRecordingWithoutProbe(),
            stream_name="Record Node 101#Neuropix-PXI-100.ProbeA",
            output_root=tmp_path,
            show=False,
            save=True,
        )


def test_plot_probe_channel_map_requires_output_root_when_saving() -> None:
    """Saving a probe layout requires an output root path."""
    module = reload_preprocess_openephys_module()

    with pytest.raises(ValueError, match="output_root"):
        module.plot_probe_channel_map(
            recording=FakeRecording(),
            stream_name="Record Node 101#Neuropix-PXI-100.ProbeA",
            output_root=None,
            show=False,
            save=True,
        )


def test_plot_probe_channel_map_saves_expected_png(monkeypatch, tmp_path) -> None:
    """Probe layout plots are saved inside the selected stream output directory."""
    module = reload_preprocess_openephys_module()
    plot_calls = []

    def fake_plot_probe(*args, **kwargs):
        plot_calls.append((args, kwargs))

    monkeypatch.setattr(module, "plot_probe", fake_plot_probe)

    figure_path = module.plot_probe_channel_map(
        recording=FakeRecording(),
        stream_name="Record Node 101#Neuropix-PXI-100.ProbeA",
        output_root=tmp_path,
        show=False,
        save=True,
    )

    assert figure_path == (
        tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA" / "probe_layout.png"
    )
    assert figure_path.is_file()
    assert plot_calls


def test_validate_open_ephys_probe_plots_after_loading_when_requested(monkeypatch, tmp_path) -> None:
    """Validation plots the loaded stream layout only when explicitly requested."""
    module = reload_preprocess_openephys_module()
    expected_recording = FakeRecording()
    expected_plot_path = tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA" / "probe_layout.png"
    plot_calls = []

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: expected_recording,
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})

    def fake_plot_probe_channel_map(**kwargs):
        plot_calls.append(kwargs)
        return expected_plot_path

    monkeypatch.setattr(module, "plot_probe_channel_map", fake_plot_probe_channel_map)

    result = module.validate_open_ephys_probe(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
        output_root=tmp_path,
        plot_probe_layout=True,
        show_probe_layout=False,
        save_probe_layout=True,
    )

    assert result["probe_layout_path"] == expected_plot_path
    assert plot_calls == [
        {
            "recording": expected_recording,
            "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
            "output_root": tmp_path,
            "show": False,
            "save": True,
        }
    ]


def test_validate_open_ephys_probe_does_not_plot_by_default(monkeypatch) -> None:
    """Validation does not create plot artifacts unless plot_probe_layout is true."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: FakeRecording(),
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})
    monkeypatch.setattr(
        module,
        "plot_probe_channel_map",
        lambda **kwargs: pytest.fail("Plotting should not run by default"),
    )

    result = module.validate_open_ephys_probe(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
    )

    assert result["probe_layout_path"] is None


class FakeChanmapRecording:
    """SpikeInterface-like recording double with consistent chanmap metadata."""

    def get_num_channels(self):
        """Return the number of channels."""
        return 4

    def get_channel_locations(self):
        """Return channel locations in um with shape (n_channels, 2)."""
        return np.array(
            [
                [0.0, 0.0],
                [32.0, 0.0],
                [0.0, 15.0],
                [32.0, 15.0],
            ]
        )

    def get_channel_groups(self):
        """Return fallback group labels with shape (n_channels,)."""
        return np.array([0, 0, 1, 1])

    def get_property_keys(self):
        """Return channel property names."""
        return ["contact_vector"]

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

    def get_sampling_frequency(self):
        """Return sampling frequency in Hz."""
        return 30000.0


class FakeBadLocationRecording(FakeChanmapRecording):
    """Recording double with invalid channel-location shape."""

    def get_channel_locations(self):
        """Return invalid channel locations."""
        return np.array([0.0, 32.0, 64.0])


class FakeNoContactVectorRecording(FakeChanmapRecording):
    """Recording double without contact-vector shank IDs."""

    def get_property_keys(self):
        """Return channel property names without contact_vector."""
        return []

    def get_property(self, key):
        """Raise for unavailable properties."""
        raise KeyError(key)


def test_write_kilosort_chanmap_returns_output_path(tmp_path) -> None:
    """Kilosort chanmap writer returns the exact output .mat path."""
    module = reload_preprocess_openephys_module()
    output_file = tmp_path / "chanMap.mat"

    returned_path = module.write_kilosort_chanmap(
        recording=FakeChanmapRecording(),
        output_file=output_file,
    )

    assert returned_path == output_file


def test_write_kilosort_chanmap_writes_required_mat_fields(tmp_path) -> None:
    """Kilosort chanmap .mat file contains the expected field names."""
    module = reload_preprocess_openephys_module()
    output_file = tmp_path / "chanMap.mat"

    module.write_kilosort_chanmap(
        recording=FakeChanmapRecording(),
        output_file=output_file,
    )

    mat_data = sio.loadmat(output_file)
    assert {
        "chanMap",
        "chanMap0ind",
        "connected",
        "xcoords",
        "ycoords",
        "kcoords",
        "fs",
        "name",
    }.issubset(mat_data.keys())


def test_write_kilosort_chanmap_writes_expected_shapes_and_units(tmp_path) -> None:
    """Kilosort chanmap arrays preserve channel count, um locations, and Hz sampling."""
    module = reload_preprocess_openephys_module()
    output_file = tmp_path / "chanMap.mat"

    module.write_kilosort_chanmap(
        recording=FakeChanmapRecording(),
        output_file=output_file,
    )

    mat_data = sio.loadmat(output_file)
    assert mat_data["chanMap"].shape == (4, 1)
    assert mat_data["chanMap0ind"].shape == (4, 1)
    assert mat_data["connected"].shape == (4, 1)
    assert mat_data["xcoords"].shape == (4, 1)
    assert mat_data["ycoords"].shape == (4, 1)
    assert mat_data["kcoords"].shape == (4, 1)
    assert mat_data["fs"].shape == (1, 1)
    assert mat_data["chanMap"][0, 0] == 1
    assert mat_data["chanMap"][-1, 0] == 4
    assert mat_data["chanMap0ind"][0, 0] == 0
    assert mat_data["chanMap0ind"][-1, 0] == 3
    np.testing.assert_allclose(mat_data["xcoords"].ravel(), [0.0, 32.0, 0.0, 32.0])
    np.testing.assert_allclose(mat_data["ycoords"].ravel(), [0.0, 0.0, 15.0, 15.0])
    assert mat_data["fs"][0, 0] == 30000.0


def test_write_kilosort_chanmap_uses_contact_vector_shank_ids(tmp_path) -> None:
    """Kilosort kcoords prefer contact_vector shank IDs over generic groups."""
    module = reload_preprocess_openephys_module()
    output_file = tmp_path / "chanMap.mat"

    module.write_kilosort_chanmap(
        recording=FakeChanmapRecording(),
        output_file=output_file,
    )

    mat_data = sio.loadmat(output_file)
    np.testing.assert_array_equal(mat_data["kcoords"].ravel(), [1, 1, 2, 2])


def test_write_kilosort_chanmap_falls_back_to_channel_groups(tmp_path) -> None:
    """Kilosort kcoords fall back to channel groups when shank IDs are unavailable."""
    module = reload_preprocess_openephys_module()
    output_file = tmp_path / "chanMap.mat"

    module.write_kilosort_chanmap(
        recording=FakeNoContactVectorRecording(),
        output_file=output_file,
    )

    mat_data = sio.loadmat(output_file)
    np.testing.assert_array_equal(mat_data["kcoords"].ravel(), [1, 1, 2, 2])


def test_write_kilosort_chanmap_rejects_bad_location_shape(tmp_path) -> None:
    """Kilosort chanmap writer requires channel locations with shape (n_channels, >=2)."""
    module = reload_preprocess_openephys_module()

    with pytest.raises(ValueError, match="2-D channel locations"):
        module.write_kilosort_chanmap(
            recording=FakeBadLocationRecording(),
            output_file=tmp_path / "chanMap.mat",
        )


def test_validate_open_ephys_probe_writes_chanmap_when_requested(monkeypatch, tmp_path) -> None:
    """Validation writes chanMap.mat into the selected stream directory when requested."""
    module = reload_preprocess_openephys_module()
    expected_recording = FakeRecording()
    write_calls = []

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: expected_recording,
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})

    def fake_write_kilosort_chanmap(**kwargs):
        write_calls.append(kwargs)
        return kwargs["output_file"]

    monkeypatch.setattr(module, "write_kilosort_chanmap", fake_write_kilosort_chanmap)

    result = module.validate_open_ephys_probe(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
        output_root=tmp_path,
        write_kilosort_chanmap_file=True,
    )

    expected_path = tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA" / "chanMap.mat"
    assert result["kilosort_chanmap_path"] == expected_path
    assert write_calls == [
        {
            "recording": expected_recording,
            "output_file": expected_path,
        }
    ]


def test_validate_open_ephys_probe_does_not_write_chanmap_by_default(monkeypatch) -> None:
    """Validation does not write Kilosort chanmap files unless requested."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: FakeRecording(),
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})
    monkeypatch.setattr(
        module,
        "write_kilosort_chanmap",
        lambda **kwargs: pytest.fail("Chanmap writing should not run by default"),
    )

    result = module.validate_open_ephys_probe(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
    )

    assert result["kilosort_chanmap_path"] is None


def test_validate_open_ephys_probe_requires_output_root_for_chanmap(monkeypatch) -> None:
    """Writing a Kilosort chanmap requires a derived output root."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: FakeRecording(),
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})

    with pytest.raises(ValueError, match="write_kilosort_chanmap_file"):
        module.validate_open_ephys_probe(
            raw_root=Path("/data/session"),
            experiment_name="experiment1",
            write_kilosort_chanmap_file=True,
        )


class FakeLfpRecording:
    """SpikeInterface-like LFP recording double for extraction-result tests."""

    channel_ids = np.array(["CH0", "CH1", "CH2", "CH3"])

    def get_sampling_frequency(self):
        """Return LFP sampling frequency in Hz."""
        return 2500.0

    def get_num_segments(self):
        """Return the number of LFP segments."""
        return 1

    def get_num_channels(self):
        """Return LFP channel count."""
        return 4

    def get_num_samples(self, segment_index=0):
        """Return LFP sample count for one segment."""
        assert segment_index == 0
        return 1000

    def get_dtype(self):
        """Return LFP sample dtype."""
        return np.dtype("float32")


def test_extract_lfp_runs_expected_preprocessing_chain(monkeypatch, tmp_path) -> None:
    """LFP extraction phase-shifts, filters, resamples, and writes lfp.dat."""
    module = reload_preprocess_openephys_module()
    raw_recording = FakeRecording()
    shifted_recording = object()
    filtered_recording = object()
    lfp_recording = FakeLfpRecording()
    calls = []

    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: calls.append(
            ("phase_shift", recording, dtype)
        ) or shifted_recording,
        bandpass_filter=lambda recording, **kwargs: calls.append(
            ("bandpass_filter", recording, kwargs)
        ) or filtered_recording,
        resample=lambda recording, **kwargs: calls.append(
            ("resample", recording, kwargs)
        ) or lfp_recording,
    )

    def fake_write_binary_recording(**kwargs):
        calls.append(("write_binary_recording", kwargs))

    monkeypatch.setattr(module, "spre", fake_spre, raising=False)
    monkeypatch.setattr(
        module,
        "write_binary_recording",
        fake_write_binary_recording,
        raising=False,
    )

    module.extract_lfp(
        recording=raw_recording,
        output_folder=tmp_path,
        progress_bar=False,
    )

    assert calls == [
        ("phase_shift", raw_recording, "float32"),
        (
            "bandpass_filter",
            shifted_recording,
            {
                "freq_min": 1.0,
                "freq_max": 500.0,
                "filter_order": 3,
                "filter_mode": "sos",
                "ftype": "butter",
                "direction": "forward-backward",
                "margin_ms": "auto",
                "ignore_low_freq_error": True,
                "dtype": "float32",
            },
        ),
        (
            "resample",
            filtered_recording,
            {
                "resample_rate": 2500,
                "margin_ms": 100.0,
                "dtype": "float32",
            },
        ),
        (
            "write_binary_recording",
            {
                "recording": lfp_recording,
                "file_paths": tmp_path / "lfp.dat",
                "dtype": "float32",
                "add_file_extension": False,
                "n_jobs": 8,
                "chunk_duration": "30s",
                "progress_bar": False,
                "verbose": True,
            },
        ),
    ]


def test_extract_lfp_normalizes_integer_like_resample_rate(monkeypatch, tmp_path) -> None:
    """LFP extraction passes integer-like float rates to SpikeInterface as ints."""
    module = reload_preprocess_openephys_module()
    resample_calls = []

    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: recording,
        bandpass_filter=lambda recording, **kwargs: recording,
        resample=lambda recording, **kwargs: resample_calls.append(kwargs) or FakeLfpRecording(),
    )

    monkeypatch.setattr(module, "spre", fake_spre, raising=False)
    monkeypatch.setattr(
        module,
        "write_binary_recording",
        lambda **kwargs: None,
        raising=False,
    )

    module.extract_lfp(
        recording=FakeRecording(),
        output_folder=tmp_path,
        resample_rate_hz=2500.0,
        progress_bar=False,
    )

    assert resample_calls[0]["resample_rate"] == 2500
    assert isinstance(resample_calls[0]["resample_rate"], int)


def test_extract_lfp_rejects_non_integer_resample_rate(monkeypatch, tmp_path) -> None:
    """LFP extraction rejects fractional resampling rates before preprocessing."""
    module = reload_preprocess_openephys_module()

    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: pytest.fail("Phase shift should not run"),
        bandpass_filter=lambda recording, **kwargs: pytest.fail("Filter should not run"),
        resample=lambda recording, **kwargs: pytest.fail("Resample should not run"),
    )

    monkeypatch.setattr(module, "spre", fake_spre, raising=False)

    with pytest.raises(ValueError, match="integer"):
        module.extract_lfp(
            recording=FakeRecording(),
            output_folder=tmp_path,
            resample_rate_hz=2500.5,
            progress_bar=False,
        )


def test_extract_lfp_rejects_missing_probe(tmp_path) -> None:
    """LFP extraction requires attached probe geometry before phase shifting."""
    module = reload_preprocess_openephys_module()

    with pytest.raises(ValueError, match="No probe geometry"):
        module.extract_lfp(
            recording=FakeRecordingWithoutProbe(),
            output_folder=tmp_path,
            progress_bar=False,
        )


def test_extract_lfp_requires_inter_sample_shift(tmp_path) -> None:
    """LFP extraction requires Neuropixels inter-sample-shift metadata."""
    module = reload_preprocess_openephys_module()

    with pytest.raises(ValueError, match="inter-sample shifts"):
        module.extract_lfp(
            recording=FakeRecordingWithoutInterSampleShift(),
            output_folder=tmp_path,
            progress_bar=False,
        )


def test_extract_lfp_writes_metadata_json(monkeypatch, tmp_path) -> None:
    """LFP extraction writes sidecar metadata beside lfp.dat, not in a subfolder."""
    module = reload_preprocess_openephys_module()

    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: recording,
        bandpass_filter=lambda recording, **kwargs: recording,
        resample=lambda recording, **kwargs: FakeLfpRecording(),
    )

    monkeypatch.setattr(module, "spre", fake_spre, raising=False)
    monkeypatch.setattr(
        module,
        "write_binary_recording",
        lambda **kwargs: None,
        raising=False,
    )

    module.extract_lfp(
        recording=FakeRecording(),
        output_folder=tmp_path,
        progress_bar=False,
    )

    metadata_path = tmp_path / "lfp_preprocessing.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["output_binary"] == "lfp.dat"
    assert metadata["binary_layout"] == "time_major_channel_interleaved"
    assert metadata["preprocessing"][1]["freq_min_hz"] == 1.0
    assert metadata["preprocessing"][1]["freq_max_hz"] == 500.0
    assert metadata["preprocessing"][2]["resample_rate_hz"] == 2500.0


def test_extract_lfp_returns_output_summary(monkeypatch, tmp_path) -> None:
    """LFP extraction returns paths, dimensions, units, and parameters."""
    module = reload_preprocess_openephys_module()

    fake_spre = SimpleNamespace(
        phase_shift=lambda recording, dtype: recording,
        bandpass_filter=lambda recording, **kwargs: recording,
        resample=lambda recording, **kwargs: FakeLfpRecording(),
    )

    monkeypatch.setattr(module, "spre", fake_spre, raising=False)
    monkeypatch.setattr(
        module,
        "write_binary_recording",
        lambda **kwargs: None,
        raising=False,
    )

    result = module.extract_lfp(
        recording=FakeRecording(),
        output_folder=tmp_path,
        progress_bar=False,
    )

    assert result["lfp_binary_path"] == tmp_path / "lfp.dat"
    assert result["lfp_metadata_path"] == tmp_path / "lfp_preprocessing.json"
    assert result["sampling_frequency_hz"] == 2500.0
    assert result["num_channels"] == 4
    assert result["num_segments"] == 1
    assert result["num_samples_by_segment"] == [1000]
    assert result["dtype"] == "float32"


def test_validate_open_ephys_probe_extracts_lfp_when_requested(monkeypatch, tmp_path) -> None:
    """Validation extracts LFP into the selected stream directory when requested."""
    module = reload_preprocess_openephys_module()
    expected_recording = FakeRecording()
    extract_calls = []
    expected_lfp_result = {
        "lfp_binary_path": tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA" / "lfp.dat"
    }

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: expected_recording,
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})

    def fake_extract_lfp(**kwargs):
        extract_calls.append(kwargs)
        return expected_lfp_result

    monkeypatch.setattr(module, "extract_lfp", fake_extract_lfp)

    result = module.validate_open_ephys_probe(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
        output_root=tmp_path,
        extract_lfp_file=True,
        lfp_progress_bar=False,
    )

    expected_output_folder = tmp_path / "Record_Node_101_Neuropix-PXI-100.ProbeA"
    assert result["lfp_result"] is expected_lfp_result
    assert extract_calls == [
        {
            "recording": expected_recording,
            "output_folder": expected_output_folder,
            "freq_min_hz": 1.0,
            "freq_max_hz": 500.0,
            "filter_order": 3,
            "filter_margin_ms": "auto",
            "resample_rate_hz": 2500,
            "resample_margin_ms": 100.0,
            "dtype": "float32",
            "n_jobs": 8,
            "chunk_duration": "30s",
            "progress_bar": False,
        }
    ]


def test_validate_open_ephys_probe_does_not_extract_lfp_by_default(monkeypatch) -> None:
    """Validation does not materialize LFP unless requested."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: FakeRecording(),
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})
    monkeypatch.setattr(
        module,
        "extract_lfp",
        lambda **kwargs: pytest.fail("LFP extraction should not run by default"),
    )

    result = module.validate_open_ephys_probe(
        raw_root=Path("/data/session"),
        experiment_name="experiment1",
    )

    assert result["lfp_result"] is None


def test_validate_open_ephys_probe_requires_output_root_for_lfp(monkeypatch) -> None:
    """LFP extraction requires a derived output root."""
    module = reload_preprocess_openephys_module()

    monkeypatch.setattr(
        module,
        "find_open_ephys_streams",
        lambda raw_root, experiment_name: module.pd.DataFrame(
            [
                {
                    "stream_id": "0",
                    "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
                    "record_node": "Record Node 101",
                    "source_name": "Neuropix-PXI-100.ProbeA",
                    "is_neuropixels": True,
                    "is_nidaq": False,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        module,
        "load_open_ephys_stream",
        lambda raw_root, experiment_name, stream_name, load_sync_timestamps: FakeRecording(),
    )
    monkeypatch.setattr(module, "summarize_open_ephys_stream", lambda recording: {"ok": True})

    with pytest.raises(ValueError, match="extract_lfp_file"):
        module.validate_open_ephys_probe(
            raw_root=Path("/data/session"),
            experiment_name="experiment1",
            extract_lfp_file=True,
        )


def test_main_passes_hardcoded_parameters(monkeypatch) -> None:
    """IDE-oriented main passes editable local parameters to validation."""
    module = reload_preprocess_openephys_module()
    calls = []
    expected_result = {
        "experiment_name": "experiment1",
        "streams": module.pd.DataFrame(),
        "stream_name": "Record Node 101#Neuropix-PXI-100.ProbeA",
        "summary": {},
        "probe_layout_path": None,
        "kilosort_chanmap_path": None,
        "lfp_result": None,
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
            "output_root": module.Path(
                "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
                "CT026_20260727_alternating_latent/ephys/derived"
            ),
            "plot_probe_layout": True,
            "show_probe_layout": True,
            "save_probe_layout": True,
            "write_kilosort_chanmap_file": True,
            "extract_lfp_file": True,
            "lfp_freq_min_hz": 1.0,
            "lfp_freq_max_hz": 500.0,
            "lfp_filter_order": 3,
            "lfp_filter_margin_ms": "auto",
            "lfp_resample_rate_hz": 2500,
            "lfp_resample_margin_ms": 100.0,
            "lfp_dtype": "float32",
            "lfp_n_jobs": 8,
            "lfp_chunk_duration": "30s",
            "lfp_progress_bar": True,
        }
    ]
