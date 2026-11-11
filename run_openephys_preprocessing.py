"""Shared Open Ephys entry point for workstation and Slurm execution.

Edit the session configuration below, then run this file with uv. Processing
and inspection use the existing helpers in src/preprocessing.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSING_ROOT = PROJECT_ROOT / "src" / "preprocessing"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_ROOT))

from src.preprocessing.binary_inspection import (
    get_open_ephys_geometric_sort,
    load_open_ephys_derived_binary,
    run_oe_inspection,
)
from src.preprocessing.ap_preprocessing_openephys import (
    validate_ap_recording,
    validate_local_car_geometry,
)
from src.preprocessing.preprocess_openephys import (
    build_stream_output_dir,
    find_open_ephys_experiments,
    find_open_ephys_streams,
    load_open_ephys_stream,
    make_safe_path_component,
    select_neuropixels_stream,
    summarize_open_ephys_stream,
    validate_open_ephys_probe,
)


# Session configuration: edit SESSION_ROOT for the workstation or cluster.
DRY_RUN = True  # Validate and print the plan; set False to process data.
SESSION_NAME = "CT026_20260801_latent_inference"
SUBJECT_ID = "CT026"
SESSION_ROOT = Path.home() / "contextProjectData" / SUBJECT_ID / SESSION_NAME
EXPERIMENT_NAME = "experiment1"
STREAM_NAME = "Record Node 101#Neuropix-PXI-110.ProbeA"


# Open Ephys loading and validation settings.
LOAD_SYNC_TIMESTAMPS = False
PLOT_PROBE_LAYOUT = True
SHOW_PROBE_LAYOUT = False
SAVE_PROBE_LAYOUT = True
WRITE_KILOSORT_CHANMAP_FILE = True


# LFP extraction settings.
EXTRACT_LFP_FILE = True
LFP_FREQ_MIN_HZ = 1.0
LFP_FREQ_MAX_HZ = 500.0
LFP_FILTER_ORDER = 3
LFP_FILTER_MARGIN_MS = "auto"
LFP_RESAMPLE_RATE_HZ = 2500
LFP_RESAMPLE_MARGIN_MS = 100.0
LFP_DTYPE = "float32"
LFP_N_JOBS = 1
LFP_CHUNK_DURATION = "30s"
LFP_POOL_ENGINE = "process"
LFP_MP_CONTEXT = None
LFP_PROGRESS_BAR = True


# AP preprocessing settings.
EXTRACT_AP_FILE = True
AP_HIGHPASS_HZ = 300.0
AP_LOCAL_CAR_INNER_UM = 40.0
AP_LOCAL_CAR_OUTER_UM = 140.0
AP_MIN_LOCAL_NEIGHBORS = 5
AP_WORKING_DTYPE = "float32"
AP_OUTPUT_DTYPE = "int16"
AP_N_JOBS = 8
AP_CHUNK_DURATION = "1s"
AP_NUM_RANDOM_CHUNKS = 20
AP_RANDOM_SEED = 0
AP_PROGRESS_BAR = True


# Channel quality settings.
DETECT_CHANNEL_QUALITY_FILE = True
CHANNEL_QUALITY_SOURCE = "ap"
CHANNEL_QUALITY_METHOD = "coherence+psd"
CHANNEL_QUALITY_OUTSIDE_LOCATION = "top"
CHANNEL_QUALITY_DIRECTION = "y"
CHANNEL_QUALITY_SEED = 0
CHANNEL_QUALITY_NUM_RANDOM_CHUNKS = 20
CHANNEL_QUALITY_CHUNK_DURATION_S = 0.3


VALID_STAGES = ("preprocess", "inspect", "all")


def validate_workflow_configuration(stage: str) -> dict[str, Any]:
    """Check the configured workflow without writing outputs or reading traces.

    Parameters
    ----------
    stage : str
        ``preprocess``, ``inspect``, or ``all``. Session paths and processing
        parameters come from this module's constants, with units in their names.

    Returns
    -------
    dict[str, Any]
        Validation report with filesystem Paths, settings, existing/planned
        outputs, and recording metadata (counts, Hz, and durations in seconds).
        Inspection binaries are mapped read-only for metadata/size validation;
        signal samples are never accessed. Errors raise before processing.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f"Invalid stage: {stage!r}; expected one of {VALID_STAGES}")
    raw_root = SESSION_ROOT / "ephys/raw"
    stream_folder = SESSION_ROOT / "ephys/derived" / make_safe_path_component(STREAM_NAME)
    report = {
        "dry_run": True, "stage": stage, "session_name": SESSION_NAME,
        "session_root": SESSION_ROOT, "raw_root": raw_root,
        "experiment_name": EXPERIMENT_NAME, "stream_name": STREAM_NAME,
        "stream_folder": stream_folder, "planned_outputs": [],
        "settings": {name: value for name, value in globals().items()
                     if name.startswith(("AP_", "LFP_", "CHANNEL_QUALITY_", "EXTRACT_",
                                         "DETECT_", "PLOT_", "SHOW_", "SAVE_", "WRITE_", "LOAD_"))},
    }
    planned = report["planned_outputs"]
    if stage in ("preprocess", "all"):
        report["recording"] = _validate_raw_configuration(raw_root)
        products = [
            (PLOT_PROBE_LAYOUT and SAVE_PROBE_LAYOUT, ["probe_layout.png"]),
            (WRITE_KILOSORT_CHANMAP_FILE, ["chanMap.mat"]),
            (EXTRACT_LFP_FILE, ["lfp.dat", "lfp_preprocessing.json"]),
            (EXTRACT_AP_FILE, ["ap_preprocessed.dat", "ap_preprocessing.json"]),
            (DETECT_CHANNEL_QUALITY_FILE, ["channel_quality.json", "channel_quality.csv"]),
        ]
        for enabled, names in products:
            if enabled:
                planned.extend(stream_folder / name for name in names)
        if planned:
            _check_output_location(stream_folder)

    if stage in ("inspect", "all"):
        n_channels = None
        for binary_name, metadata_name in [
            ("ap_preprocessed.dat", "ap_preprocessing.json"),
            ("lfp.dat", "lfp_preprocessing.json"),
        ]:
            binary_path = stream_folder / binary_name
            if binary_path in planned:
                count = report["recording"]["num_channels"]
            else:
                data, metadata, sample_rate, shape = load_open_ephys_derived_binary(
                    binary_path, stream_folder / metadata_name,
                )
                del data  # Release the read-only mapping without touching samples.
                if sample_rate <= 0 or min(shape) <= 0:
                    raise ValueError(f"Invalid sample rate or shape in {metadata_name}")
                count = shape[0]
            if n_channels is not None and count != n_channels:
                raise ValueError("AP and LFP channel counts do not match")
            n_channels = count
        if not any(stream_folder / name in planned
                   for name in ("chanMap.mat", "channel_quality.csv")):
            get_open_ephys_geometric_sort(stream_folder, n_channels=n_channels)
        for folder in (SESSION_ROOT / "figures", SESSION_ROOT / "processed"):
            _check_output_location(folder)
        report["inspection"] = {
            "figure_path": SESSION_ROOT / "figures",
            "processed_path": SESSION_ROOT / "processed",
            "depends_on_preprocessing": stage == "all",
        }
    report["existing_outputs"] = [path for path in planned if path.exists()]
    report["limitations"] = (
        "No signal samples read: filtering, sampled AP range, signal quality, "
        "available disk space, and actual write success are not verified. "
        "Existing planned outputs may be replaced during real processing."
    )
    return report


def _validate_raw_configuration(raw_root: Path) -> dict[str, Any]:
    """Accept a raw-directory Path; return metadata counts, Hz, and durations in s.

    Reads only recording/probe metadata; trace axes would be (samples, channels).
    Raises for missing inputs or metadata incompatible with enabled operations.
    """
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw Open Ephys directory not found: {raw_root}")
    experiments = find_open_ephys_experiments(raw_root)
    if EXPERIMENT_NAME not in experiments:
        raise ValueError(f"Unknown experiment {EXPERIMENT_NAME!r}; available: {experiments}")
    streams = find_open_ephys_streams(raw_root, EXPERIMENT_NAME)
    select_neuropixels_stream(streams, stream_name=STREAM_NAME)
    recording = load_open_ephys_stream(
        raw_root=raw_root, experiment_name=EXPERIMENT_NAME,
        stream_name=STREAM_NAME, load_sync_timestamps=LOAD_SYNC_TIMESTAMPS,
    )
    if EXTRACT_AP_FILE:
        validate_ap_recording(recording)
        validate_local_car_geometry(recording, outer_radius_um=AP_LOCAL_CAR_OUTER_UM)
    summary = summarize_open_ephys_stream(recording)
    if EXTRACT_LFP_FILE or PLOT_PROBE_LAYOUT:
        if not summary["has_probe"] or not summary["has_inter_sample_shift"]:
            raise ValueError("Probe geometry and inter-sample-shift metadata are required")
    if WRITE_KILOSORT_CHANMAP_FILE or DETECT_CHANNEL_QUALITY_FILE:
        locations = recording.get_channel_locations()
        if locations.shape[0] != summary["num_channels"] or locations.shape[1] < 2:
            raise ValueError("Channel locations must have shape (n_channels, >=2) in um")
    if DETECT_CHANNEL_QUALITY_FILE and CHANNEL_QUALITY_SOURCE != "ap":
        raise ValueError("Channel quality currently supports only source 'ap'")
    rate = summary["sampling_frequency_hz"]
    if rate <= 0 or not summary["num_samples_by_segment"] or any(
        count <= 0 for count in summary["num_samples_by_segment"]
    ):
        raise ValueError("Recording must contain samples and have a positive sample rate")
    summary["duration_s_by_segment"] = [count / rate for count in summary["num_samples_by_segment"]]
    return summary


def _check_output_location(path: Path) -> None:
    """Accept an output-directory Path; return None or raise on permission/path errors.

    Checks the nearest existing ancestor for write/search permission without
    creating directories. Permission checks do not guarantee a future write.
    """
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError(f"Output directory is blocked by a file: {ancestor}")
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError(f"Output ancestor is not writable/searchable: {ancestor}")


def run_preprocess() -> dict[str, Any]:
    """Run Open Ephys preprocessing for the configured session.

    Parameters
    ----------
    None
        Runtime configuration is read from module-level constants. Paths are
        filesystem paths and numeric filter settings use Hz, milliseconds,
        seconds, or micrometers as indicated by each constant name.

    Returns
    -------
    dict[str, Any]
        Summary returned by ``validate_open_ephys_probe``. Array-like recording
        data are not returned by this entry point.
    """
    raw_root = SESSION_ROOT / "ephys" / "raw"
    output_root = SESSION_ROOT / "ephys" / "derived"

    return validate_open_ephys_probe(
        raw_root=raw_root,
        experiment_name=EXPERIMENT_NAME,
        stream_name=STREAM_NAME,
        load_sync_timestamps=LOAD_SYNC_TIMESTAMPS,
        output_root=output_root,
        plot_probe_layout=PLOT_PROBE_LAYOUT,
        show_probe_layout=SHOW_PROBE_LAYOUT,
        save_probe_layout=SAVE_PROBE_LAYOUT,
        write_kilosort_chanmap_file=WRITE_KILOSORT_CHANMAP_FILE,
        extract_lfp_file=EXTRACT_LFP_FILE,
        extract_ap_file=EXTRACT_AP_FILE,
        lfp_freq_min_hz=LFP_FREQ_MIN_HZ,
        lfp_freq_max_hz=LFP_FREQ_MAX_HZ,
        lfp_filter_order=LFP_FILTER_ORDER,
        lfp_filter_margin_ms=LFP_FILTER_MARGIN_MS,
        lfp_resample_rate_hz=LFP_RESAMPLE_RATE_HZ,
        lfp_resample_margin_ms=LFP_RESAMPLE_MARGIN_MS,
        lfp_dtype=LFP_DTYPE,
        lfp_n_jobs=LFP_N_JOBS,
        lfp_chunk_duration=LFP_CHUNK_DURATION,
        lfp_pool_engine=LFP_POOL_ENGINE,
        lfp_mp_context=LFP_MP_CONTEXT,
        lfp_progress_bar=LFP_PROGRESS_BAR,
        ap_highpass_hz=AP_HIGHPASS_HZ,
        ap_local_car_inner_um=AP_LOCAL_CAR_INNER_UM,
        ap_local_car_outer_um=AP_LOCAL_CAR_OUTER_UM,
        ap_min_local_neighbors=AP_MIN_LOCAL_NEIGHBORS,
        ap_working_dtype=AP_WORKING_DTYPE,
        ap_output_dtype=AP_OUTPUT_DTYPE,
        ap_n_jobs=AP_N_JOBS,
        ap_chunk_duration=AP_CHUNK_DURATION,
        ap_num_random_chunks=AP_NUM_RANDOM_CHUNKS,
        ap_random_seed=AP_RANDOM_SEED,
        ap_progress_bar=AP_PROGRESS_BAR,
        detect_channel_quality_file=DETECT_CHANNEL_QUALITY_FILE,
        channel_quality_source=CHANNEL_QUALITY_SOURCE,
        channel_quality_method=CHANNEL_QUALITY_METHOD,
        channel_quality_outside_location=CHANNEL_QUALITY_OUTSIDE_LOCATION,
        channel_quality_direction=CHANNEL_QUALITY_DIRECTION,
        channel_quality_seed=CHANNEL_QUALITY_SEED,
        channel_quality_num_random_chunks=CHANNEL_QUALITY_NUM_RANDOM_CHUNKS,
        channel_quality_chunk_duration_s=CHANNEL_QUALITY_CHUNK_DURATION_S,
    )


def run_inspection() -> dict[str, Any]:
    """Run binary inspection for the configured derived Open Ephys stream.

    Parameters
    ----------
    None
        Runtime configuration is read from module-level constants. The stream
        folder is ``SESSION_ROOT / "ephys/derived" / safe_stream_name``.

    Returns
    -------
    dict[str, Any]
        Summary returned by ``run_oe_inspection``. AP and LFP binaries are read
        as channel-major ``(n_channels, n_samples)`` memmap views downstream.
    """
    output_root = SESSION_ROOT / "ephys" / "derived"
    stream_folder = build_stream_output_dir(output_root=output_root, stream_name=STREAM_NAME)
    figure_path = SESSION_ROOT / "figures"
    processed_path = SESSION_ROOT / "processed"

    return run_oe_inspection(
        stream_folder=stream_folder,
        figure_path=figure_path,
        processed_path=processed_path,
        session_tag=SESSION_NAME,
    )


def run_hpc_workflow(stage: str) -> dict[str, Any]:
    """Run one Open Ephys workflow stage on a workstation or cluster.

    Parameters
    ----------
    stage : str
        Workflow stage. Must be ``"preprocess"``, ``"inspect"``, or ``"all"``.
        The ``"all"`` stage runs preprocessing before binary inspection.

    Returns
    -------
    dict[str, Any]
        Summary containing the requested stage and the result dictionaries from
        the stages that ran.
    """
    if stage not in VALID_STAGES:
        raise ValueError("stage must be one of {}; got {!r}".format(VALID_STAGES, stage))

    summary = {"stage": stage, "session_name": SESSION_NAME, "session_root": str(SESSION_ROOT)}
    if stage in ("preprocess", "all"):
        print("Running Open Ephys preprocessing for {}".format(SESSION_NAME))
        summary["preprocess_result"] = run_preprocess()
    if stage in ("inspect", "all"):
        print("Running Open Ephys binary inspection for {}".format(SESSION_NAME))
        summary["inspection_result"] = run_inspection()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the shared Open Ephys entry point.

    Parameters
    ----------
    argv : list[str] or None
        Command-line tokens excluding the executable name. Shape is
        ``(n_arguments,)``. If ``None``, ``argparse`` reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``stage`` as a string.
    """
    parser = argparse.ArgumentParser(description="Run Open Ephys preprocessing and inspection on a workstation or cluster.")
    parser.add_argument(
        "--stage",
        choices=VALID_STAGES,
        default="all",
        help="Workflow stage to run. Use separate preprocess and inspect jobs for long recordings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the shared Open Ephys entry point from the command line.

    Parameters
    ----------
    argv : list[str] or None
        Command-line tokens excluding the executable name. Shape is
        ``(n_arguments,)``.

    Returns
    -------
    dict[str, Any]
        Validation report when DRY_RUN is True, otherwise the workflow summary.
    """
    args = parse_args(argv)
    if DRY_RUN:
        summary = validate_workflow_configuration(stage=args.stage)
        print("DRY RUN: validation passed; no processing outputs written.")
        print(json.dumps(summary, indent=2, default=str))
        return summary
    summary = run_hpc_workflow(stage=args.stage)
    print("Finished Open Ephys HPC stage {}".format(args.stage))
    return summary


if __name__ == "__main__":
    main()

