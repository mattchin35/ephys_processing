"""Tests for Open Ephys sorter postprocessing with SpikeInterface."""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio


MODULE_NAME = "src.postprocessing.spikeinterface_oe_postprocessing"


def reload_postprocessing_module():
    """Import the Open Ephys postprocessing module fresh."""
    preprocessing_path = Path(__file__).resolve().parents[1] / "src" / "preprocessing"
    if str(preprocessing_path) not in sys.path:
        sys.path.insert(0, str(preprocessing_path))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def write_ap_metadata(
    stream_folder: Path,
    n_channels: int = 4,
    num_segments: int = 1,
    with_scaling: bool = True,
) -> Path:
    """Write minimal AP preprocessing metadata for tests.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Destination stream folder.
    n_channels : int
        Number of AP channels. Units are channels.
    num_segments : int
        Number of AP recording segments.
    with_scaling : bool
        Whether to include AP binary scaling metadata.

    Returns
    -------
    pathlib.Path
        Path to the created ``ap_preprocessing.json`` file.
    """
    stream_folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "output_binary": "ap_preprocessed.dat",
        "sampling_frequency_hz": 30000.0,
        "num_channels": n_channels,
        "num_segments": num_segments,
        "num_samples_by_segment": [1000] * num_segments,
        "dtype": "int16",
        "binary_layout": "time_major_channel_interleaved",
        "channel_ids_in_binary_order": [f"CH{channel_index}" for channel_index in range(n_channels)],
    }
    if with_scaling:
        metadata["ap_binary_scaling"] = {
            "has_scaleable_traces": True,
            "gain_to_uV_by_channel": [0.195] * n_channels,
            "offset_to_uV_by_channel": [0.0] * n_channels,
            "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
        }
    metadata_path = stream_folder / "ap_preprocessing.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    (stream_folder / "ap_preprocessed.dat").write_bytes(b"0" * 8)
    return metadata_path


def write_channel_quality_csv(stream_folder: Path, locations_um: np.ndarray) -> None:
    """Write channel locations in the channel quality CSV format.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Destination stream folder.
    locations_um : numpy.ndarray, shape (n_channels, 2)
        Channel x/y locations in micrometers.

    Returns
    -------
    None
        ``channel_quality.csv`` is written.
    """
    rows = ["channel_id,label,is_good,inside_brain,x_um,y_um"]
    for channel_index, (x_um, y_um) in enumerate(locations_um):
        rows.append(f"CH{channel_index},good,True,True,{x_um},{y_um}")
    (stream_folder / "channel_quality.csv").write_text("\n".join(rows), encoding="utf-8")


class FakeRecording:
    """Minimal SpikeInterface-like recording for analyzer creation tests."""

    def __init__(self):
        self.locations = None

    def set_channel_locations(self, locations, channel_ids=None):
        """Store channel locations with shape (n_channels, 2) in micrometers."""
        self.locations = np.asarray(locations)

    def has_scaleable_traces(self):
        """Return whether traces can be converted to uV."""
        return True


class FakeAnalyzer:
    """Minimal SortingAnalyzer-like object for quality metric tests."""

    def __init__(self):
        self.compute_calls = []
        self.metrics = pd.DataFrame({"firing_rate": [1.0, 2.0]}, index=[10, 11])

    def compute(self, input, **kwargs):
        """Store requested extension computations."""
        self.compute_calls.append((input, kwargs))

    def get_extension(self, extension_name):
        """Return a fake quality metrics extension."""
        assert extension_name == "quality_metrics"
        return self

    def get_data(self):
        """Return quality metrics as a pandas DataFrame."""
        return self.metrics


class SlowFakeAnalyzer(FakeAnalyzer):
    """Fake analyzer that keeps a compute stage active long enough for a heartbeat."""

    def compute(self, input, **kwargs):
        """Store one computation request, then pause briefly without changing data."""
        super().compute(input, **kwargs)
        time.sleep(0.03)


def test_load_ap_metadata_reads_required_fields(tmp_path) -> None:
    """AP metadata loader validates the derived AP binary contract."""
    module = reload_postprocessing_module()
    write_ap_metadata(tmp_path)

    metadata = module.load_ap_metadata(tmp_path)

    assert metadata["output_binary"] == "ap_preprocessed.dat"
    assert metadata["sampling_frequency_hz"] == 30000.0
    assert metadata["num_channels"] == 4
    assert metadata["num_segments"] == 1
    assert metadata["num_samples_by_segment"] == [1000]
    assert metadata["dtype"] == "int16"
    assert metadata["binary_layout"] == "time_major_channel_interleaved"


def test_load_ap_metadata_requires_single_segment(tmp_path) -> None:
    """First-pass postprocessing only accepts one AP segment."""
    module = reload_postprocessing_module()
    write_ap_metadata(tmp_path, num_segments=2)

    with pytest.raises(ValueError, match="single segment"):
        module.load_ap_metadata(tmp_path)


def test_load_ap_recording_uses_binary_metadata_and_uv_scaling(monkeypatch, tmp_path) -> None:
    """AP binary loading passes metadata and saved uV scaling to SpikeInterface."""
    module = reload_postprocessing_module()
    write_ap_metadata(tmp_path)
    calls = []
    expected_recording = FakeRecording()

    def fake_read_binary(**kwargs):
        calls.append(kwargs)
        return expected_recording

    monkeypatch.setattr(module.si, "read_binary", fake_read_binary)

    recording = module.load_ap_recording(tmp_path)

    assert recording is expected_recording
    assert calls == [
        {
            "file_paths": tmp_path / "ap_preprocessed.dat",
            "sampling_frequency": 30000.0,
            "dtype": "int16",
            "num_channels": 4,
            "channel_ids": ["CH0", "CH1", "CH2", "CH3"],
            "time_axis": 0,
            "gain_to_uV": [0.195, 0.195, 0.195, 0.195],
            "offset_to_uV": [0.0, 0.0, 0.0, 0.0],
            "is_filtered": True,
        }
    ]


def test_load_ap_recording_warns_without_uv_scaling(monkeypatch, tmp_path) -> None:
    """Missing AP scaling is allowed by default but clearly reported."""
    module = reload_postprocessing_module()
    write_ap_metadata(tmp_path, with_scaling=False)
    monkeypatch.setattr(module.si, "read_binary", lambda **kwargs: FakeRecording())

    with pytest.warns(UserWarning, match="uV scaling"):
        module.load_ap_recording(tmp_path)


def test_load_ap_recording_requires_uv_when_requested(tmp_path) -> None:
    """require_uV=True raises when AP binary scaling metadata is absent."""
    module = reload_postprocessing_module()
    write_ap_metadata(tmp_path, with_scaling=False)

    with pytest.raises(ValueError, match="uV scaling"):
        module.load_ap_recording(tmp_path, require_uV=True)


def test_load_channel_locations_prefers_channel_quality_csv(tmp_path) -> None:
    """Channel quality CSV is the preferred geometry source."""
    module = reload_postprocessing_module()
    sorter_folder = tmp_path / "kilosort4"
    sorter_folder.mkdir()
    expected_locations = np.array([[0.0, 10.0], [32.0, 20.0], [0.0, 30.0]])
    write_channel_quality_csv(tmp_path, expected_locations)
    np.save(sorter_folder / "channel_positions.npy", np.zeros((3, 2)))

    locations = module.load_channel_locations(tmp_path, sorter_folder, n_channels=3)

    np.testing.assert_array_equal(locations, expected_locations)


def test_load_channel_locations_falls_back_to_sorter_channel_positions(tmp_path) -> None:
    """Sorter channel_positions.npy is used when channel_quality.csv is absent."""
    module = reload_postprocessing_module()
    sorter_folder = tmp_path / "kilosort4"
    sorter_folder.mkdir()
    expected_locations = np.array([[0.0, 10.0], [32.0, 20.0], [0.0, 30.0]])
    np.save(sorter_folder / "channel_positions.npy", expected_locations)

    locations = module.load_channel_locations(tmp_path, sorter_folder, n_channels=3)

    np.testing.assert_array_equal(locations, expected_locations)


def test_load_channel_locations_falls_back_to_chanmap(tmp_path) -> None:
    """Kilosort chanMap.mat is used when CSV and sorter positions are absent."""
    module = reload_postprocessing_module()
    sorter_folder = tmp_path / "kilosort4"
    sorter_folder.mkdir()
    sio.savemat(
        tmp_path / "chanMap.mat",
        {"xcoords": np.array([0.0, 32.0, 0.0]), "ycoords": np.array([10.0, 20.0, 30.0])},
    )

    locations = module.load_channel_locations(tmp_path, sorter_folder, n_channels=3)

    np.testing.assert_array_equal(locations, np.array([[0.0, 10.0], [32.0, 20.0], [0.0, 30.0]]))


def test_create_sorting_analyzer_for_stream_attaches_locations_and_uses_memory_analyzer(
    monkeypatch,
    tmp_path,
) -> None:
    """Analyzer creation attaches geometry and avoids writing an extra analyzer folder."""
    module = reload_postprocessing_module()
    write_ap_metadata(tmp_path)
    sorter_folder = tmp_path / "kilosort4"
    sorter_folder.mkdir()
    recording = FakeRecording()
    sorting = object()
    analyzer = FakeAnalyzer()
    locations = np.array([[0.0, 10.0], [32.0, 20.0], [0.0, 30.0], [32.0, 40.0]])
    calls = []

    monkeypatch.setattr(module, "load_ap_recording", lambda *args, **kwargs: recording)
    monkeypatch.setattr(module, "load_sorting", lambda *args, **kwargs: sorting)
    monkeypatch.setattr(module, "load_channel_locations", lambda *args, **kwargs: locations)

    def fake_create_sorting_analyzer(**kwargs):
        calls.append(kwargs)
        return analyzer

    monkeypatch.setattr(module.si, "create_sorting_analyzer", fake_create_sorting_analyzer)

    result_analyzer, context = module.create_sorting_analyzer_for_stream(
        stream_folder=tmp_path,
        sorter_folder_name="kilosort4",
    )

    assert result_analyzer is analyzer
    np.testing.assert_array_equal(recording.locations, locations)
    assert calls == [
        {
            "sorting": sorting,
            "recording": recording,
            "format": "memory",
            "sparse": True,
            "return_in_uV": True,
        }
    ]
    assert "folder" not in calls[0]
    assert context["sorter_folder"] == sorter_folder


def test_compute_quality_metrics_without_pca_excludes_principal_components() -> None:
    """The default local pass skips PCA and PCA-dependent metrics."""
    module = reload_postprocessing_module()
    analyzer = FakeAnalyzer()

    metrics, context = module.compute_quality_metrics_for_analyzer(
        analyzer,
        compute_principal_components=False,
        job_kwargs={"n_jobs": 1, "chunk_duration": "1s", "progress_bar": False},
    )

    assert len(analyzer.compute_calls) == 2
    extensions = analyzer.compute_calls[0][0]
    assert "principal_components" not in extensions
    assert "quality_metrics" not in extensions
    quality_metric_input, quality_metric_kwargs = analyzer.compute_calls[1]
    assert quality_metric_input == "quality_metrics"
    assert "nearest_neighbor" not in quality_metric_kwargs["metric_names"]
    assert metrics.equals(analyzer.metrics)
    assert context["compute_principal_components"] is False


def test_compute_quality_metrics_with_pca_uses_staged_metric_computation() -> None:
    """The PCA branch computes extensions, non-PCA metrics, then PCA metrics."""
    module = reload_postprocessing_module()
    analyzer = FakeAnalyzer()

    _, context = module.compute_quality_metrics_for_analyzer(
        analyzer,
        compute_principal_components=True,
        job_kwargs={"n_jobs": 1, "chunk_duration": "1s", "progress_bar": False},
    )

    assert len(analyzer.compute_calls) == 3
    extensions = analyzer.compute_calls[0][0]
    assert extensions["principal_components"] == {"n_components": 3, "mode": "by_channel_local"}
    assert "quality_metrics" not in extensions

    non_pca_input, non_pca_kwargs = analyzer.compute_calls[1]
    assert non_pca_input == "quality_metrics"
    assert non_pca_kwargs["metric_names"] == list(module.NON_PCA_METRIC_NAMES)

    pca_input, pca_kwargs = analyzer.compute_calls[2]
    assert pca_input == "quality_metrics"
    assert pca_kwargs["metric_names"] == list(module.PCA_METRIC_NAMES)
    assert pca_kwargs["delete_existing_metrics"] is False
    assert "nearest_neighbor" in pca_kwargs["metric_names"]
    assert "nn_advanced" not in pca_kwargs["metric_names"]
    assert context["metric_names"] == list(module.NON_PCA_METRIC_NAMES + module.PCA_METRIC_NAMES)
    assert context["compute_principal_components"] is True


def test_compute_quality_metrics_reports_stage_progress(capsys) -> None:
    """Enabled progress prints start and completion messages for every stage."""
    module = reload_postprocessing_module()
    analyzer = FakeAnalyzer()

    module.compute_quality_metrics_for_analyzer(
        analyzer,
        compute_principal_components=True,
        job_kwargs={"n_jobs": 1, "chunk_duration": "1s", "progress_bar": True},
    )

    output = capsys.readouterr().out
    assert "[1/3] Computing waveform and PCA extensions..." in output
    assert "[2/3] Computing non-PCA quality metrics..." in output
    assert "[3/3] Computing PCA quality metrics..." in output
    assert output.count("Finished in") == 3


def test_compute_quality_metrics_hides_stage_progress_when_disabled(capsys) -> None:
    """Disabled progress does not add stage messages to standard output."""
    module = reload_postprocessing_module()
    analyzer = FakeAnalyzer()

    module.compute_quality_metrics_for_analyzer(
        analyzer,
        compute_principal_components=False,
        job_kwargs={"n_jobs": 1, "chunk_duration": "1s", "progress_bar": False},
    )

    assert capsys.readouterr().out == ""


def test_compute_stage_reports_heartbeat(monkeypatch, capsys) -> None:
    """A long-running stage periodically reports that computation is active."""
    module = reload_postprocessing_module()
    monkeypatch.setattr(module, "PROGRESS_HEARTBEAT_SECONDS", 0.005)
    analyzer = SlowFakeAnalyzer()

    module._compute_stage_with_progress(
        sorting_analyzer=analyzer,
        extension_input="quality_metrics",
        extension_kwargs={"metric_names": ["firing_rate"]},
        job_kwargs={"progress_bar": True},
        stage_number=1,
        total_stages=1,
        description="Computing test metrics",
    )

    output = capsys.readouterr().out
    assert "Still running" in output
    assert "elapsed" in output


def test_compute_stage_propagates_errors_and_stops_progress(capsys) -> None:
    """Stage failures propagate without printing a misleading completion message."""
    module = reload_postprocessing_module()

    class FailingAnalyzer(FakeAnalyzer):
        """Analyzer whose compute operation always fails."""

        def compute(self, input, **kwargs):
            """Raise the sentinel computation error."""
            raise RuntimeError("sentinel failure")

    with pytest.raises(RuntimeError, match="sentinel failure"):
        module._compute_stage_with_progress(
            sorting_analyzer=FailingAnalyzer(),
            extension_input="quality_metrics",
            extension_kwargs={"metric_names": ["firing_rate"]},
            job_kwargs={"progress_bar": True},
            stage_number=1,
            total_stages=1,
            description="Computing test metrics",
        )

    output = capsys.readouterr().out
    assert "Computing test metrics..." in output
    assert "Finished in" not in output


def test_postprocess_one_recording_saves_metrics_and_summary_in_sorter_folder(
    monkeypatch,
    tmp_path,
) -> None:
    """Postprocessing writes Phy-friendly metrics.csv and a summary JSON."""
    module = reload_postprocessing_module()
    sorter_folder = tmp_path / "kilosort4"
    sorter_folder.mkdir()
    analyzer = FakeAnalyzer()
    create_context = {"sorter_folder": sorter_folder, "return_in_uV": True}

    monkeypatch.setattr(
        module,
        "create_sorting_analyzer_for_stream",
        lambda **kwargs: (analyzer, create_context),
    )
    monkeypatch.setattr(
        module,
        "compute_quality_metrics_for_analyzer",
        lambda sorting_analyzer, **kwargs: (analyzer.metrics, {"metric_names": ["firing_rate"]}),
    )

    summary = module.postprocess_one_recording(
        stream_folder=tmp_path,
        sorter_folder_name="kilosort4",
        compute_principal_components=False,
    )

    metrics_path = sorter_folder / "metrics.csv"
    summary_path = sorter_folder / "spikeinterface_postprocessing.json"
    assert summary["metrics_path"] == str(metrics_path)
    assert summary["summary_path"] == str(summary_path)
    assert metrics_path.is_file()
    assert summary_path.is_file()
    saved_metrics = pd.read_csv(metrics_path)
    assert list(saved_metrics.columns) == ["cluster_id", "firing_rate"]
    assert saved_metrics["cluster_id"].tolist() == [10, 11]


def test_postprocess_recordings_runs_each_job(monkeypatch, tmp_path) -> None:
    """Batch helper runs each stream/sorter job and returns summaries."""
    module = reload_postprocessing_module()
    calls = []

    def fake_postprocess_one_recording(**kwargs):
        calls.append(kwargs)
        return {"sorter_folder_name": kwargs["sorter_folder_name"]}

    monkeypatch.setattr(module, "postprocess_one_recording", fake_postprocess_one_recording)
    jobs = [
        {"stream_folder": tmp_path / "probe_a", "sorter_folder_name": "kilosort4"},
        {"stream_folder": tmp_path / "probe_b", "sorter_folder_name": "kilosort2_5_2"},
    ]

    summaries = module.postprocess_recordings(jobs)

    assert [summary["sorter_folder_name"] for summary in summaries] == ["kilosort4", "kilosort2_5_2"]
    assert calls == jobs
