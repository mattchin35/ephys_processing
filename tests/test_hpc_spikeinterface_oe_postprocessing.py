"""Tests for the no-argument SpikeInterface postprocessing Slurm wrapper."""

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "src" / "shell_scripts" / "hpc_spikeinterface_oe_postprocessing.sh"


def test_slurm_wrapper_has_requested_resources_and_log_path() -> None:
    """The wrapper requests the agreed small CPU job and persistent log path."""
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    required_lines = [
        "#SBATCH --partition=normal",
        "#SBATCH --job-name=spikeinterface_oe_postprocess",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=16G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --output=/gs/gsfs0/users/mchin1/logs/spikeinterface_oe_postprocess_%j.log",
    ]
    for required_line in required_lines:
        assert required_line in script_text


def test_slurm_wrapper_runs_hardcoded_python_entry_point_with_uv() -> None:
    """The wrapper contains environment concerns but no recording-specific arguments."""
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script_text
    assert 'cd "/gs/gsfs0/users/mchin1/ephys_processing"' in script_text
    assert "uv run --frozen python src/postprocessing/spikeinterface_oe_postprocessing.py" in script_text
    assert "conda" not in script_text
    assert '"$@"' not in script_text
    assert "CT026" not in script_text


def test_slurm_wrapper_has_valid_bash_syntax() -> None:
    """Bash accepts the production wrapper syntax."""
    syntax_check = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert syntax_check.returncode == 0, syntax_check.stderr
