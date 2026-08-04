"""Tests for the Open Ephys HPC entry point."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "main_oe_hpc"


def reload_main_oe_hpc_module():
    """Import the Open Ephys HPC entry module after clearing cached state."""
    preprocessing_path = PROJECT_ROOT / "src" / "preprocessing"
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(preprocessing_path) not in sys.path:
        sys.path.insert(0, str(preprocessing_path))
    sys.modules.pop(MODULE_NAME, None)
    return importlib.import_module(MODULE_NAME)


def test_preprocess_stage_calls_validate_open_ephys_probe(monkeypatch, tmp_path) -> None:
    """The preprocess stage calls the Open Ephys preprocessing workflow once.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Test helper used to replace runtime paths and heavy preprocessing calls.
    tmp_path : pathlib.Path
        Temporary filesystem root. Units are path components.

    Returns
    -------
    None
        Assertions verify the preprocessing call contract.
    """
    module = reload_main_oe_hpc_module()
    calls = []
    expected_result = {"preprocess": "ok"}

    monkeypatch.setattr(module, "SESSION_ROOT", tmp_path / "session")
    monkeypatch.setattr(module, "EXPERIMENT_NAME", "experiment1")
    monkeypatch.setattr(module, "STREAM_NAME", "Record Node 101#Neuropix-PXI-110.ProbeA")

    def fake_validate_open_ephys_probe(**kwargs):
        calls.append(kwargs)
        return expected_result

    monkeypatch.setattr(module, "validate_open_ephys_probe", fake_validate_open_ephys_probe)

    summary = module.run_hpc_workflow(stage="preprocess")

    assert summary["stage"] == "preprocess"
    assert summary["preprocess_result"] is expected_result
    assert "inspection_result" not in summary
    assert calls[0]["raw_root"] == tmp_path / "session" / "ephys" / "raw"
    assert calls[0]["output_root"] == tmp_path / "session" / "ephys" / "derived"
    assert calls[0]["experiment_name"] == "experiment1"
    assert calls[0]["stream_name"] == "Record Node 101#Neuropix-PXI-110.ProbeA"


def test_inspect_stage_calls_run_oe_inspection(monkeypatch, tmp_path) -> None:
    """The inspect stage points binary inspection at the derived stream folder."""
    module = reload_main_oe_hpc_module()
    calls = []
    expected_result = {"inspection": "ok"}
    expected_stream_folder = tmp_path / "session" / "ephys" / "derived" / "safe_stream"

    monkeypatch.setattr(module, "SESSION_NAME", "test_session")
    monkeypatch.setattr(module, "SESSION_ROOT", tmp_path / "session")
    monkeypatch.setattr(module, "STREAM_NAME", "Record Node 101#Neuropix-PXI-110.ProbeA")

    def fake_build_stream_output_dir(output_root, stream_name):
        calls.append(("build_stream_output_dir", output_root, stream_name))
        return expected_stream_folder

    def fake_run_oe_inspection(**kwargs):
        calls.append(("run_oe_inspection", kwargs))
        return expected_result

    monkeypatch.setattr(module, "build_stream_output_dir", fake_build_stream_output_dir)
    monkeypatch.setattr(module, "run_oe_inspection", fake_run_oe_inspection)

    summary = module.run_hpc_workflow(stage="inspect")

    assert summary["stage"] == "inspect"
    assert summary["inspection_result"] is expected_result
    assert "preprocess_result" not in summary
    assert calls[0] == (
        "build_stream_output_dir",
        tmp_path / "session" / "ephys" / "derived",
        "Record Node 101#Neuropix-PXI-110.ProbeA",
    )
    assert calls[1][1] == {
        "stream_folder": expected_stream_folder,
        "figure_path": tmp_path / "session" / "figures",
        "processed_path": tmp_path / "session" / "processed",
        "session_tag": "test_session",
    }


def test_all_stage_runs_preprocess_then_inspection(monkeypatch) -> None:
    """The all stage preserves the required preprocessing-before-inspection order."""
    module = reload_main_oe_hpc_module()
    calls = []

    def fake_run_preprocess():
        calls.append("preprocess")
        return {"preprocess": "ok"}

    def fake_run_inspection():
        calls.append("inspect")
        return {"inspection": "ok"}

    monkeypatch.setattr(module, "run_preprocess", fake_run_preprocess)
    monkeypatch.setattr(module, "run_inspection", fake_run_inspection)

    summary = module.run_hpc_workflow(stage="all")

    assert calls == ["preprocess", "inspect"]
    assert summary["preprocess_result"] == {"preprocess": "ok"}
    assert summary["inspection_result"] == {"inspection": "ok"}


def test_run_hpc_workflow_rejects_invalid_stage() -> None:
    """Invalid stages raise a clear ValueError before running work."""
    module = reload_main_oe_hpc_module()

    with pytest.raises(ValueError, match="stage"):
        module.run_hpc_workflow(stage="bad")


def test_hpc_openephys_shell_script_has_required_sbatch_options() -> None:
    """The Open Ephys Slurm wrapper keeps the required scheduler directives."""
    script_text = (PROJECT_ROOT / "hpc_openephys.sh").read_text(encoding="utf-8")

    required_lines = [
        "#SBATCH -p normal",
        "#SBATCH --job-name=openephys_preprocess",
        "#SBATCH -n 16",
        "#SBATCH --ntasks=1",
        "#SBATCH --mem=128gb",
        "#SBATCH -t 24:00:00",
        "#SBATCH -o /gs/gsfs0/users/mchin1/logs/openephys_preprocess_%j.log",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=matthew.chin@einsteinmed.edu",
    ]
    for required_line in required_lines:
        assert required_line in script_text


def test_hpc_openephys_shell_script_runs_main_oe_hpc() -> None:
    """The Slurm wrapper activates conda and runs the Open Ephys Python entry point."""
    script_path = PROJECT_ROOT / "hpc_openephys.sh"
    script_text = script_path.read_text(encoding="utf-8")

    assert "conda activate spikeinterface" in script_text
    assert "uv run" not in script_text
    assert 'python main_oe_hpc.py "$@"' in script_text

    syntax_check = subprocess.run(
        ["bash", "-n", str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax_check.returncode == 0, syntax_check.stderr
