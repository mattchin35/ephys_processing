"""Smoke tests for the uv-managed Open Ephys preprocessing environment.

These tests document the expected environment contract for development:
Python 3.12 and the core scientific/ephys packages needed before writing
Open Ephys preprocessing functions.
"""

import importlib
import importlib.metadata
import sys


REQUIRED_DISTRIBUTIONS = (
    "matplotlib",
    "numpy",
    "open-ephys-python-tools",
    "pandas",
    "pyarrow",
    "scipy",
    "spikeinterface",
)

IMPORT_SMOKE_TESTS = (
    "matplotlib",
    "numpy",
    "open_ephys",
    "pandas",
    "pyarrow",
    "scipy",
    "spikeinterface",
)


def test_environment_uses_python_312() -> None:
    """Python runtime must be CPython 3.12.x; returns no value."""
    assert sys.version_info[:2] == (3, 12)


def test_required_distributions_are_installed() -> None:
    """Required package distributions must be resolvable; returns no value."""
    missing_distributions = []
    for distribution_name in REQUIRED_DISTRIBUTIONS:
        try:
            importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            missing_distributions.append(distribution_name)

    assert missing_distributions == []


def test_core_packages_import() -> None:
    """Core package import names must import successfully; returns no value."""
    for module_name in IMPORT_SMOKE_TESTS:
        importlib.import_module(module_name)
