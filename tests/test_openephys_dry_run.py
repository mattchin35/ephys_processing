"""Metadata-only validation tests; synthetic inputs never contain real data."""

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def workflow(monkeypatch, tmp_path):
    """Use pytest fixtures; return (module, metadata-only recording), no units."""
    module = importlib.import_module("run_openephys_preprocessing")
    monkeypatch.setattr(module, "SESSION_ROOT", tmp_path)
    (tmp_path / "ephys/raw").mkdir(parents=True)
    locations = np.column_stack((np.zeros(16), np.arange(16) * 20.0))
    recording = SimpleNamespace(
        get_num_segments=lambda: 1,
        get_sampling_frequency=lambda: 30000.0,
        get_num_samples=lambda segment_index=0: 60000,
        get_num_channels=lambda: 16,
        get_channel_ids=lambda: np.arange(16),
        get_dtype=lambda: np.dtype("int16"),
        get_probe=lambda: object(),
        get_property_keys=lambda: ["location", "inter_sample_shift"],
        get_property=lambda key: locations,
        get_channel_locations=lambda: locations,
        get_channel_groups=lambda: np.zeros(16),
    )
    def forbidden(*args, **kwargs):
        """Accept arbitrary call arguments; raise if samples or processing are used."""
        raise AssertionError("Dry run attempted processing, sample reading, or directory creation")
    recording.get_traces = forbidden
    monkeypatch.setattr(module, "find_open_ephys_experiments", lambda root: ["experiment1"])
    monkeypatch.setattr(module, "find_open_ephys_streams", lambda *args: pd.DataFrame({
        "stream_name": [module.STREAM_NAME], "is_neuropixels": [True],
    }))
    monkeypatch.setattr(module, "load_open_ephys_stream", lambda **kwargs: recording)
    for name in ("run_preprocess", "run_inspection", "build_stream_output_dir"):
        monkeypatch.setattr(module, name, forbidden)
    return module, recording


def test_dry_run_reports_plan_without_writes(workflow, capsys):
    """Use a synthetic workflow fixture and capture; return None after report checks."""
    module, _ = workflow
    assert module.DRY_RUN is True
    before = set(module.SESSION_ROOT.rglob("*"))
    result = module.main([])
    assert result["dry_run"] is True
    assert result["recording"]["duration_s_by_segment"] == [2.0]
    assert "ap_preprocessed.dat" in [path.name for path in result["planned_outputs"]]
    assert set(module.SESSION_ROOT.rglob("*")) == before
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "AP_HIGHPASS_HZ" in output
    assert module.STREAM_NAME in output


def test_disabled_dry_run_uses_existing_workflow(workflow, monkeypatch):
    """Use fixtures; return None after checking real execution dispatch without data."""
    module, _ = workflow
    monkeypatch.setattr(module, "DRY_RUN", False)
    monkeypatch.setattr(module, "run_hpc_workflow", lambda stage: {"stage": stage})
    assert module.main(["--stage", "preprocess"]) == {"stage": "preprocess"}


@pytest.mark.parametrize("problem, message", [
    ("raw", "Raw"), ("experiment", "experiment"), ("stream", "stream"),
    ("rate", "30 kHz"), ("segments", "one segment"),
    ("probe", "probe"), ("shift", "shift"), ("geometry", "locations"),
    ("permission", "writ"),
])
def test_invalid_configuration_fails(workflow, monkeypatch, problem, message):
    """Use fixtures and scalar problem/message strings; assert informative errors."""
    module, recording = workflow
    if problem == "raw":
        (module.SESSION_ROOT / "ephys/raw").rmdir()
    elif problem == "experiment":
        monkeypatch.setattr(module, "EXPERIMENT_NAME", "missing")
    elif problem == "stream":
        monkeypatch.setattr(module, "find_open_ephys_streams", lambda *args: pd.DataFrame({
            "stream_name": ["other"], "is_neuropixels": [True],
        }))
    elif problem == "rate":
        recording.get_sampling_frequency = lambda: 2500.0
    elif problem == "segments":
        recording.get_num_segments = lambda: 2
    elif problem == "probe":
        recording.get_probe = lambda: None
    elif problem == "shift":
        recording.get_property_keys = lambda: ["location"]
    elif problem == "geometry":
        recording.get_channel_locations = lambda: np.zeros((16, 3))
    else:
        monkeypatch.setattr(module.os, "access", lambda *args: False)
    with pytest.raises((ValueError, FileNotFoundError, PermissionError), match=message):
        module.validate_workflow_configuration("preprocess")


def test_existing_outputs_are_preserved(workflow):
    """Use workflow fixture; return None after checking existing output bytes."""
    module, _ = workflow
    folder = module.SESSION_ROOT / "ephys/derived" / module.make_safe_path_component(module.STREAM_NAME)
    folder.mkdir(parents=True)
    path = folder / "ap_preprocessed.dat"
    path.write_bytes(b"unchanged")
    result = module.validate_workflow_configuration("preprocess")
    assert path in result["existing_outputs"]
    assert path.read_bytes() == b"unchanged"


def test_inspect_requires_outputs_but_all_can_plan_them(workflow):
    """Use workflow fixture; return None after checking stage-specific dependencies."""
    module, _ = workflow
    with pytest.raises(FileNotFoundError, match="ap_preprocessed"):
        module.validate_workflow_configuration("inspect")
    assert module.validate_workflow_configuration("all")["stage"] == "all"


def test_all_requires_disabled_products_to_exist(workflow, monkeypatch):
    """Use fixtures; return None after rejecting missing LFP when extraction is off."""
    module, _ = workflow
    monkeypatch.setattr(module, "EXTRACT_LFP_FILE", False)
    with pytest.raises(FileNotFoundError, match="lfp.dat"):
        module.validate_workflow_configuration("all")


def test_inspect_validates_derived_metadata_without_raw(workflow):
    """Use synthetic binary/JSON/CSV files; assert inspection needs no raw input."""
    module, _ = workflow
    (module.SESSION_ROOT / "ephys/raw").rmdir()
    folder = module.SESSION_ROOT / "ephys/derived" / module.make_safe_path_component(module.STREAM_NAME)
    folder.mkdir(parents=True)
    metadata = dict(binary_layout="time_major_channel_interleaved", num_segments=1,
                    num_samples_by_segment=[2], num_channels=2, dtype="int16",
                    sampling_frequency_hz=30000)
    for binary, sidecar in [("ap_preprocessed.dat", "ap_preprocessing.json"),
                            ("lfp.dat", "lfp_preprocessing.json")]:
        (folder / binary).write_bytes(bytes(8))
        (folder / sidecar).write_text(json.dumps(metadata))
    (folder / "channel_quality.csv").write_text("y_um\n0\n20\n")
    assert module.validate_workflow_configuration("inspect")["stage"] == "inspect"
    (folder / "lfp.dat").write_bytes(bytes(4))
    with pytest.raises(ValueError, match="size mismatch"):
        module.validate_workflow_configuration("inspect")
