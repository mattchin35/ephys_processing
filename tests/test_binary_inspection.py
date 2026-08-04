"""Tests for binary inspection helpers."""

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio


MODULE_NAME = "src.preprocessing.binary_inspection"


def reload_binary_inspection_module():
    """Import the binary inspection module with local helper imports available."""
    preprocessing_path = Path(__file__).resolve().parents[1] / "src" / "preprocessing"
    if str(preprocessing_path) not in sys.path:
        sys.path.insert(0, str(preprocessing_path))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def write_derived_binary(folder: Path, binary_name: str, metadata_name: str, data: np.ndarray, sample_rate_hz: float) -> None:
    """Write a tiny time-major binary and matching Open Ephys preprocessing metadata.

    Parameters
    ----------
    folder : pathlib.Path
        Destination directory.
    binary_name : str
        Name of the binary file to create.
    metadata_name : str
        Name of the JSON sidecar to create.
    data : numpy.ndarray, shape (n_samples, n_channels)
        Time-major samples to write. Values are unitless test data.
    sample_rate_hz : float
        Sampling frequency in Hz.

    Returns
    -------
    None
        Files are written to ``folder``.
    """
    folder.mkdir(parents=True, exist_ok=True)
    data.astype(np.float32).tofile(folder / binary_name)
    metadata = {
        "sampling_frequency_hz": sample_rate_hz,
        "num_channels": int(data.shape[1]),
        "num_segments": 1,
        "num_samples_by_segment": [int(data.shape[0])],
        "dtype": "float32",
        "binary_layout": "time_major_channel_interleaved",
    }
    (folder / metadata_name).write_text(json.dumps(metadata), encoding="utf-8")


def write_channel_quality_csv(folder: Path, y_coords: list[float]) -> None:
    """Write channel geometry in the simple channel quality CSV format.

    Parameters
    ----------
    folder : pathlib.Path
        Destination directory.
    y_coords : list of float, shape (n_channels,)
        Probe y positions in micrometers.

    Returns
    -------
    None
        ``channel_quality.csv`` is written to ``folder``.
    """
    with (folder / "channel_quality.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["channel_id", "label", "is_good", "inside_brain", "x_um", "y_um"],
        )
        writer.writeheader()
        for channel_index, y_coord in enumerate(y_coords):
            writer.writerow(
                {
                    "channel_id": str(channel_index),
                    "label": "good",
                    "is_good": "True",
                    "inside_brain": "True",
                    "x_um": "0.0",
                    "y_um": str(y_coord),
                }
            )


def test_load_open_ephys_derived_binary_reads_time_major_binary(tmp_path) -> None:
    """Open Ephys derived binaries are exposed channel-major for inspection functions."""
    module = reload_binary_inspection_module()
    time_major_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_derived_binary(
        tmp_path,
        "ap_preprocessed.dat",
        "ap_preprocessing.json",
        time_major_data,
        sample_rate_hz=30000.0,
    )

    recording, metadata, sample_rate, shape = module.load_open_ephys_derived_binary(
        binary_path=tmp_path / "ap_preprocessed.dat",
        metadata_path=tmp_path / "ap_preprocessing.json",
    )

    assert sample_rate == 30000
    assert shape == (4, 3)
    assert metadata["fileTimeSecs"] == pytest.approx(3 / 30000)
    assert metadata["nSavedChans"] == 4
    np.testing.assert_array_equal(np.asarray(recording), time_major_data.T)


def test_load_open_ephys_derived_binary_rejects_missing_files(tmp_path) -> None:
    """Missing derived files produce loud errors before inspection starts."""
    module = reload_binary_inspection_module()

    with pytest.raises(FileNotFoundError, match="ap_preprocessed.dat"):
        module.load_open_ephys_derived_binary(
            binary_path=tmp_path / "ap_preprocessed.dat",
            metadata_path=tmp_path / "ap_preprocessing.json",
        )


def test_load_open_ephys_derived_binary_rejects_multi_segment_metadata(tmp_path) -> None:
    """The simple binary loader only accepts one continuous segment."""
    module = reload_binary_inspection_module()
    time_major_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    write_derived_binary(
        tmp_path,
        "lfp.dat",
        "lfp_preprocessing.json",
        time_major_data,
        sample_rate_hz=2500.0,
    )
    metadata_path = tmp_path / "lfp_preprocessing.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["num_segments"] = 2
    metadata["num_samples_by_segment"] = [2, 1]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="single-segment"):
        module.load_open_ephys_derived_binary(
            binary_path=tmp_path / "lfp.dat",
            metadata_path=metadata_path,
        )


def test_get_open_ephys_geometric_sort_uses_channel_quality_csv(tmp_path) -> None:
    """The preferred geometry source is the channel quality CSV."""
    module = reload_binary_inspection_module()
    write_channel_quality_csv(tmp_path, y_coords=[40.0, 0.0, 20.0])

    geometric_sort = module.get_open_ephys_geometric_sort(tmp_path, n_channels=3)

    np.testing.assert_array_equal(geometric_sort, np.array([1, 2, 0]))


def test_get_open_ephys_geometric_sort_falls_back_to_chanmap(tmp_path) -> None:
    """If the CSV is absent, Kilosort chanMap y coordinates are used."""
    module = reload_binary_inspection_module()
    sio.savemat(
        tmp_path / "chanMap.mat",
        {
            "xcoords": np.array([0.0, 0.0, 0.0]),
            "ycoords": np.array([40.0, 0.0, 20.0]),
        },
    )

    geometric_sort = module.get_open_ephys_geometric_sort(tmp_path, n_channels=3)

    np.testing.assert_array_equal(geometric_sort, np.array([1, 2, 0]))


def test_get_open_ephys_geometric_sort_raises_without_geometry(tmp_path) -> None:
    """Inspection should fail loudly if no probe geometry file is available."""
    module = reload_binary_inspection_module()

    with pytest.raises(FileNotFoundError, match="channel_quality.csv.*chanMap.mat"):
        module.get_open_ephys_geometric_sort(tmp_path, n_channels=3)


def test_run_oe_inspection_processes_ap_and_lfp(tmp_path, monkeypatch) -> None:
    """The Open Ephys workflow runs AP and LFP inspection from one stream folder."""
    module = reload_binary_inspection_module()
    ap_data = np.arange(12, dtype=np.float32).reshape(3, 4)
    lfp_data = np.arange(8, dtype=np.float32).reshape(2, 4)
    write_derived_binary(tmp_path, "ap_preprocessed.dat", "ap_preprocessing.json", ap_data, sample_rate_hz=30000.0)
    write_derived_binary(tmp_path, "lfp.dat", "lfp_preprocessing.json", lfp_data, sample_rate_hz=2500.0)
    write_channel_quality_csv(tmp_path, y_coords=[40.0, 0.0, 20.0, 60.0])

    process_calls = []
    save_calls = []

    def fake_process_stream_inspection(**kwargs):
        process_calls.append(kwargs)
        return {
            "stream_tag": kwargs["stream_tag"],
            "inspection_file": f"{kwargs['stream_tag']}.pkl",
            "saved_keys": [],
        }

    def fake_save_inspection_data(data, fname, save_path, note=""):
        save_calls.append((data, fname, save_path, note))

    monkeypatch.setattr(module, "process_stream_inspection", fake_process_stream_inspection)
    monkeypatch.setattr(module.pio, "save_inspection_data", fake_save_inspection_data)

    summary = module.run_oe_inspection(
        stream_folder=tmp_path,
        figure_path=tmp_path / "figures",
        processed_path=tmp_path / "processed",
        session_tag="test_session",
    )

    assert [call["stream_tag"] for call in process_calls] == ["AP", "LFP"]
    assert [call["sample_rate"] for call in process_calls] == [30000, 2500]
    assert [call["run_psd"] for call in process_calls] == [False, True]
    np.testing.assert_array_equal(process_calls[0]["geometric_sort"], np.array([1, 2, 0, 3]))
    assert summary["session_tag"] == "test_session"
    assert summary["stream_folder"] == str(tmp_path)
    assert [stream["stream_tag"] for stream in summary["streams"]] == ["AP", "LFP"]
    assert save_calls[0][1] == "test_session_oe_inspection_index"


def test_run_oe_inspection_requires_ap_and_lfp_outputs(tmp_path) -> None:
    """The workflow reports missing derived Open Ephys files before partial processing."""
    module = reload_binary_inspection_module()

    with pytest.raises(FileNotFoundError, match="ap_preprocessed.dat.*lfp.dat"):
        module.run_oe_inspection(
            stream_folder=tmp_path,
            figure_path=tmp_path / "figures",
            processed_path=tmp_path / "processed",
            session_tag="test_session",
        )
