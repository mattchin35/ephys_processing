"""First-pass SpikeInterface postprocessing for Open Ephys sorter outputs.

This module rebuilds a SpikeInterface AP recording from a derived stream folder,
loads a Phy/Kilosort-style sorter output folder, computes quality metrics with a
memory-backed ``SortingAnalyzer``, and writes simple outputs into the sorter
folder.
"""

from __future__ import annotations

import csv
import json
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import spikeinterface as si
import spikeinterface.extractors as se


POSTPROCESSING_VERSION = "0.1.0"
REQUIRED_AP_METADATA_FIELDS = (
    "output_binary",
    "sampling_frequency_hz",
    "num_channels",
    "num_segments",
    "num_samples_by_segment",
    "dtype",
    "binary_layout",
)
EXPECTED_BINARY_LAYOUT = "time_major_channel_interleaved"

NON_PCA_EXTENSIONS = (
    "random_spikes",
    "waveforms",
    "templates",
    "noise_levels",
    "spike_amplitudes",
    "spike_locations",
    "unit_locations",
    "template_similarity",
    "correlograms",
)
NON_PCA_METRIC_NAMES = (
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "synchrony",
    "firing_range",
    "amplitude_cv",
    "amplitude_cutoff",
    "noise_cutoff",
    "amplitude_median",
    "drift",
    "sd_ratio",
)
PCA_METRIC_NAMES = (
    "mahalanobis",
    "d_prime",
    "nearest_neighbor",
    "silhouette",
)
PROGRESS_HEARTBEAT_SECONDS = 30.0
DEFAULT_JOB_KWARGS = {
    "n_jobs": 1,
    "chunk_duration": "1s",
    "progress_bar": True,
}


# Hardcoded no-argument entry-point configuration. Keep recording-specific
# settings here rather than in the reusable Slurm wrapper.
WORKSTATION_DATA_ROOT = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData")
CLUSTER_DATA_ROOT = Path("/gs/gsfs0/users/mchin1/contextProjectData")
SUBJECT_ID = "CT026"
SESSION_NAME = "CT026_20260810_latent_inference"
STREAM_FOLDER_NAME = "Record_Node_101_Neuropix-PXI-103.ProbeB"
SORTER_FOLDER_NAME = "Kilosort4.1.3_2026-09-16_132246"
KEEP_GOOD_ONLY = False
REQUIRE_UV = False
COMPUTE_PRINCIPAL_COMPONENTS = True
POSTPROCESSING_JOB_KWARGS = {
    "n_jobs": 1,
    "chunk_duration": "1s",
    "progress_bar": True,
}


def load_ap_metadata(stream_folder: Path) -> dict[str, Any]:
    """Load and validate derived AP preprocessing metadata.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Derived Open Ephys stream folder containing ``ap_preprocessing.json``.
        Paths are filesystem path components.

    Returns
    -------
    dict[str, Any]
        AP metadata. Sampling frequency is in Hz, channel count is in channels,
        and ``num_samples_by_segment`` has shape ``(1,)`` for this first-pass
        single-segment workflow.
    """
    metadata_path = Path(stream_folder) / "ap_preprocessing.json"
    if not metadata_path.exists():
        raise FileNotFoundError("AP preprocessing metadata not found: {}".format(metadata_path))

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing_fields = [field for field in REQUIRED_AP_METADATA_FIELDS if field not in metadata]
    if missing_fields:
        raise ValueError("AP metadata missing required fields: {}".format(", ".join(missing_fields)))

    if metadata["binary_layout"] != EXPECTED_BINARY_LAYOUT:
        raise ValueError(
            "AP binary layout must be '{}'; got '{}'".format(
                EXPECTED_BINARY_LAYOUT,
                metadata["binary_layout"],
            )
        )
    if int(metadata["num_segments"]) != 1 or len(metadata["num_samples_by_segment"]) != 1:
        raise ValueError("First-pass Open Ephys postprocessing requires a single segment AP binary.")

    return metadata


def load_ap_recording(
    stream_folder: Path,
    require_uV: bool = False,
):
    """Rebuild the derived AP binary as a SpikeInterface recording.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Derived Open Ephys stream folder containing ``ap_preprocessed.dat`` and
        ``ap_preprocessing.json``.
    require_uV : bool
        If ``True``, raise an error when ``ap_binary_scaling`` metadata are not
        present. If ``False``, load the recording without uV scaling and warn.

    Returns
    -------
    recording
        SpikeInterface recording extractor. Traces use SpikeInterface's
        ``(n_samples, n_channels)`` convention. The generic binary reader is
        configured with ``time_axis=0`` and, when metadata are available,
        per-channel ``gain_to_uV`` and ``offset_to_uV`` arrays with shape
        ``(n_channels,)``.
    """
    stream_folder = Path(stream_folder)
    metadata = load_ap_metadata(stream_folder)
    ap_binary_path = stream_folder / metadata["output_binary"]
    if not ap_binary_path.exists():
        raise FileNotFoundError("AP binary not found: {}".format(ap_binary_path))

    scaling = metadata.get("ap_binary_scaling")
    has_uV_scaling = bool(scaling and scaling.get("has_scaleable_traces"))
    if has_uV_scaling:
        gain_to_uV = scaling["gain_to_uV_by_channel"]
        offset_to_uV = scaling["offset_to_uV_by_channel"]
    else:
        if require_uV:
            raise ValueError("AP binary uV scaling metadata are required but were not found.")
        warnings.warn(
            "AP binary uV scaling metadata are missing; amplitude-related metrics will use binary units.",
            UserWarning,
            stacklevel=2,
        )
        gain_to_uV = None
        offset_to_uV = None

    return si.read_binary(
        file_paths=ap_binary_path,
        sampling_frequency=float(metadata["sampling_frequency_hz"]),
        dtype=str(metadata["dtype"]),
        num_channels=int(metadata["num_channels"]),
        channel_ids=metadata.get("channel_ids_in_binary_order"),
        time_axis=0,
        gain_to_uV=gain_to_uV,
        offset_to_uV=offset_to_uV,
        is_filtered=True,
    )


def load_sorting(
    sorter_folder: Path,
    keep_good_only: bool = False,
):
    """Load a Phy/Kilosort-style sorter output folder.

    Parameters
    ----------
    sorter_folder : pathlib.Path
        Sorter output folder, for example ``stream_folder / "kilosort4"`` or a
        Kilosort 2.5.2 output folder.
    keep_good_only : bool
        Whether SpikeInterface should keep only units marked as good by the
        sorter/Phy labels.

    Returns
    -------
    sorting
        SpikeInterface sorting extractor. Unit IDs are determined by the sorter
        output files and spike times are in samples at the sorter sampling rate.
    """
    sorter_folder = Path(sorter_folder)
    if not sorter_folder.exists():
        raise FileNotFoundError("Sorter folder not found: {}".format(sorter_folder))
    return se.read_kilosort(
        folder_path=sorter_folder,
        keep_good_only=keep_good_only,
        remove_empty_units=True,
    )


def load_channel_locations(
    stream_folder: Path,
    sorter_folder: Path,
    n_channels: int,
) -> np.ndarray:
    """Load AP channel locations for attaching geometry to a recording.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Derived Open Ephys stream folder. Preferred geometry source is
        ``channel_quality.csv`` with ``x_um`` and ``y_um`` columns; fallback is
        ``chanMap.mat``.
    sorter_folder : pathlib.Path
        Sorter output folder. Fallback geometry source is
        ``channel_positions.npy``.
    n_channels : int
        Expected number of channels. Units are channels.

    Returns
    -------
    numpy.ndarray
        Channel locations with shape ``(n_channels, 2)`` in micrometers.
    """
    stream_folder = Path(stream_folder)
    sorter_folder = Path(sorter_folder)
    channel_quality_path = stream_folder / "channel_quality.csv"
    sorter_positions_path = sorter_folder / "channel_positions.npy"
    chanmap_path = stream_folder / "chanMap.mat"

    if channel_quality_path.exists():
        with channel_quality_path.open("r", newline="", encoding="utf-8") as input_file:
            rows = list(csv.DictReader(input_file))
        if not rows or "x_um" not in rows[0] or "y_um" not in rows[0]:
            raise ValueError("{} must contain x_um and y_um columns".format(channel_quality_path))
        locations = np.asarray([[float(row["x_um"]), float(row["y_um"])] for row in rows], dtype=float)
        return _validate_channel_locations(locations, n_channels=n_channels, source_path=channel_quality_path)

    if sorter_positions_path.exists():
        locations = np.asarray(np.load(sorter_positions_path), dtype=float)
        if locations.ndim == 2 and locations.shape[1] > 2:
            locations = locations[:, :2]
        return _validate_channel_locations(locations, n_channels=n_channels, source_path=sorter_positions_path)

    if chanmap_path.exists():
        chanmap = sio.loadmat(chanmap_path)
        if "xcoords" not in chanmap or "ycoords" not in chanmap:
            raise ValueError("{} must contain xcoords and ycoords".format(chanmap_path))
        locations = np.column_stack(
            [
                np.asarray(chanmap["xcoords"], dtype=float).reshape(-1),
                np.asarray(chanmap["ycoords"], dtype=float).reshape(-1),
            ]
        )
        return _validate_channel_locations(locations, n_channels=n_channels, source_path=chanmap_path)

    raise FileNotFoundError(
        "No channel geometry found. Expected channel_quality.csv, channel_positions.npy, or chanMap.mat."
    )


def create_sorting_analyzer_for_stream(
    stream_folder: Path,
    sorter_folder_name: str,
    keep_good_only: bool = False,
    require_uV: bool = False,
):
    """Create an in-memory SortingAnalyzer for one derived stream and sorter.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Derived Open Ephys stream folder containing the AP binary and metadata.
    sorter_folder_name : str
        Name of the sorter output folder inside ``stream_folder``.
    keep_good_only : bool
        Whether to load only units labelled as good.
    require_uV : bool
        Whether AP binary uV scaling metadata are required.

    Returns
    -------
    tuple
        ``(sorting_analyzer, context)``. The analyzer is memory-backed and
        contains AP traces with shape ``(n_samples, n_channels)`` downstream.
        ``context`` is a JSON-serializable dictionary describing paths and
        whether traces can be returned in uV.
    """
    stream_folder = Path(stream_folder)
    sorter_folder = stream_folder / sorter_folder_name
    metadata = load_ap_metadata(stream_folder)
    recording = load_ap_recording(stream_folder=stream_folder, require_uV=require_uV)
    sorting = load_sorting(sorter_folder=sorter_folder, keep_good_only=keep_good_only)
    locations_um = load_channel_locations(
        stream_folder=stream_folder,
        sorter_folder=sorter_folder,
        n_channels=int(metadata["num_channels"]),
    )
    recording.set_channel_locations(locations_um)
    return_in_uV = _recording_has_uV_scaling(recording)

    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
        sparse=True,
        return_in_uV=return_in_uV,
    )
    context = {
        "stream_folder": str(stream_folder),
        "sorter_folder": sorter_folder,
        "sorter_folder_name": sorter_folder_name,
        "keep_good_only": bool(keep_good_only),
        "require_uV": bool(require_uV),
        "return_in_uV": bool(return_in_uV),
    }
    return sorting_analyzer, context


def compute_quality_metrics_for_analyzer(
    sorting_analyzer,
    compute_principal_components: bool = False,
    job_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute first-pass SpikeInterface quality metrics.

    Parameters
    ----------
    sorting_analyzer
        SpikeInterface ``SortingAnalyzer`` for one sorter output and AP
        recording. Waveform-like extensions use samples with shape
        ``(n_samples, n_channels)`` internally.
    compute_principal_components : bool
        Whether to compute principal components and include PCA-dependent
        quality metrics.
    job_kwargs : dict[str, Any] or None
        SpikeInterface job settings such as ``n_jobs``, ``chunk_duration``, and
        ``progress_bar``. Units follow SpikeInterface conventions; chunk
        duration strings are in seconds.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, Any]]
        Quality metrics table indexed by unit ID, plus JSON-serializable
        context describing computed extensions and metric names.
    """
    if job_kwargs is None:
        job_kwargs = dict(DEFAULT_JOB_KWARGS)
    else:
        job_kwargs = dict(job_kwargs)

    extensions = {extension_name: {} for extension_name in NON_PCA_EXTENSIONS}
    metric_names = list(NON_PCA_METRIC_NAMES)
    if compute_principal_components:
        extensions["principal_components"] = {
            "n_components": 3,
            "mode": "by_channel_local",
        }
        metric_names.extend(PCA_METRIC_NAMES)

    total_stages = 3 if compute_principal_components else 2
    extension_description = (
        "Computing waveform and PCA extensions"
        if compute_principal_components
        else "Computing waveform extensions"
    )
    _compute_stage_with_progress(
        sorting_analyzer=sorting_analyzer,
        extension_input=extensions,
        extension_kwargs={},
        job_kwargs=job_kwargs,
        stage_number=1,
        total_stages=total_stages,
        description=extension_description,
    )
    _compute_stage_with_progress(
        sorting_analyzer=sorting_analyzer,
        extension_input="quality_metrics",
        extension_kwargs={"metric_names": list(NON_PCA_METRIC_NAMES)},
        job_kwargs=job_kwargs,
        stage_number=2,
        total_stages=total_stages,
        description="Computing non-PCA quality metrics",
    )
    if compute_principal_components:
        _compute_stage_with_progress(
            sorting_analyzer=sorting_analyzer,
            extension_input="quality_metrics",
            extension_kwargs={
                "metric_names": list(PCA_METRIC_NAMES),
                "delete_existing_metrics": False,
            },
            job_kwargs=job_kwargs,
            stage_number=3,
            total_stages=total_stages,
            description="Computing PCA quality metrics",
        )

    quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    context = {
        "compute_principal_components": bool(compute_principal_components),
        "extensions": extensions,
        "metric_names": metric_names,
        "job_kwargs": job_kwargs,
    }
    return quality_metrics, context


def _compute_stage_with_progress(
    sorting_analyzer,
    extension_input,
    extension_kwargs: dict[str, Any],
    job_kwargs: dict[str, Any],
    stage_number: int,
    total_stages: int,
    description: str,
) -> None:
    """Compute one analyzer stage with optional elapsed-time updates.

    Parameters
    ----------
    sorting_analyzer
        SpikeInterface ``SortingAnalyzer`` or compatible object exposing
        ``compute(extension_input, **kwargs)``.
    extension_input
        Extension name or extension-parameter dictionary accepted by
        ``SortingAnalyzer.compute``.
    extension_kwargs : dict[str, Any]
        Extension-specific keyword arguments. Values follow SpikeInterface's
        extension conventions.
    job_kwargs : dict[str, Any]
        SpikeInterface job settings. ``progress_bar`` controls both native
        progress bars and these stage messages; time strings are in seconds.
    stage_number : int
        One-based position of this stage.
    total_stages : int
        Total number of computation stages.
    description : str
        Human-readable stage description without trailing punctuation.

    Returns
    -------
    None
        Results are stored on ``sorting_analyzer`` by SpikeInterface.
    """
    compute_kwargs = dict(extension_kwargs)
    compute_kwargs.update(job_kwargs)
    if not bool(job_kwargs.get("progress_bar", False)):
        sorting_analyzer.compute(extension_input, **compute_kwargs)
        return

    stage_prefix = "[{}/{}]".format(stage_number, total_stages)
    print("{} {}...".format(stage_prefix, description), flush=True)
    start_time = time.monotonic()
    stop_heartbeat = threading.Event()

    def report_heartbeat() -> None:
        """Report elapsed wall time until the computation stage stops."""
        while not stop_heartbeat.wait(PROGRESS_HEARTBEAT_SECONDS):
            elapsed = _format_elapsed_time(time.monotonic() - start_time)
            print("{} Still running ({} elapsed)".format(stage_prefix, elapsed), flush=True)

    heartbeat_thread = threading.Thread(
        target=report_heartbeat,
        name="spikeinterface-progress-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        sorting_analyzer.compute(extension_input, **compute_kwargs)
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join()

    elapsed = _format_elapsed_time(time.monotonic() - start_time)
    print("{} Finished in {}".format(stage_prefix, elapsed), flush=True)


def _format_elapsed_time(elapsed_seconds: float) -> str:
    """Format a nonnegative elapsed duration for progress messages.

    Parameters
    ----------
    elapsed_seconds : float
        Elapsed wall-clock duration in seconds.

    Returns
    -------
    str
        Duration formatted as ``HH:MM:SS`` with whole-second precision.
    """
    total_seconds = max(0, int(elapsed_seconds))
    hours, remaining_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remaining_seconds, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def postprocess_one_recording(
    stream_folder: Path,
    sorter_folder_name: str,
    keep_good_only: bool = False,
    require_uV: bool = False,
    compute_principal_components: bool = False,
    job_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Postprocess one derived Open Ephys stream and sorter output.

    Parameters
    ----------
    stream_folder : pathlib.Path
        Derived Open Ephys stream folder containing AP binary metadata and a
        sorter output folder.
    sorter_folder_name : str
        Name of the sorter output folder inside ``stream_folder``.
    keep_good_only : bool
        Whether to load only units labelled as good.
    require_uV : bool
        Whether AP binary uV scaling metadata are required.
    compute_principal_components : bool
        Whether to compute PCA and PCA-dependent quality metrics.
    job_kwargs : dict[str, Any] or None
        SpikeInterface job settings. Chunk duration strings are in seconds.

    Returns
    -------
    dict[str, Any]
        JSON-serializable summary. Output paths are strings. Metrics are saved
        as ``sorter_folder / "metrics.csv"``.
    """
    stream_folder = Path(stream_folder)
    sorting_analyzer, analyzer_context = create_sorting_analyzer_for_stream(
        stream_folder=stream_folder,
        sorter_folder_name=sorter_folder_name,
        keep_good_only=keep_good_only,
        require_uV=require_uV,
    )
    quality_metrics, metrics_context = compute_quality_metrics_for_analyzer(
        sorting_analyzer=sorting_analyzer,
        compute_principal_components=compute_principal_components,
        job_kwargs=job_kwargs,
    )

    sorter_folder = Path(analyzer_context["sorter_folder"])
    metrics_path = sorter_folder / "metrics.csv"
    summary_path = sorter_folder / "spikeinterface_postprocessing.json"
    quality_metrics.index.name = "cluster_id"
    quality_metrics.to_csv(metrics_path)

    summary = {
        "analysis_version": POSTPROCESSING_VERSION,
        "stream_folder": str(stream_folder),
        "sorter_folder": str(sorter_folder),
        "sorter_folder_name": sorter_folder_name,
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "keep_good_only": bool(keep_good_only),
        "require_uV": bool(require_uV),
        **_json_safe_context(analyzer_context),
        **_json_safe_context(metrics_context),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def postprocess_recordings(recording_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Postprocess multiple Open Ephys sorter jobs.

    Parameters
    ----------
    recording_jobs : list[dict[str, Any]]
        One dictionary per recording. Each dictionary is passed to
        :func:`postprocess_one_recording` as keyword arguments. Paths are
        filesystem paths; sorter folder names are strings.

    Returns
    -------
    list[dict[str, Any]]
        One summary dictionary per job, in input order.
    """
    summaries = []
    for recording_job in recording_jobs:
        summaries.append(postprocess_one_recording(**recording_job))
    return summaries


def select_data_root() -> Path:
    """Select the available hardcoded data root for this machine.

    Parameters
    ----------
    None
        Candidate locations are read from ``CLUSTER_DATA_ROOT`` and
        ``WORKSTATION_DATA_ROOT``. Both values are filesystem paths.

    Returns
    -------
    pathlib.Path
        Existing data root. The cluster root takes precedence when both roots
        exist.

    Raises
    ------
    FileNotFoundError
        If neither configured data root exists.
    """
    if CLUSTER_DATA_ROOT.is_dir():
        return CLUSTER_DATA_ROOT
    if WORKSTATION_DATA_ROOT.is_dir():
        return WORKSTATION_DATA_ROOT
    raise FileNotFoundError(
        "Neither configured data root exists: {} or {}".format(
            CLUSTER_DATA_ROOT,
            WORKSTATION_DATA_ROOT,
        )
    )


def main() -> dict[str, Any]:
    """Run the first-pass postprocessing example with IDE-editable settings.

    Parameters
    ----------
    None
        Edit local variables inside this function before running from an IDE.

    Returns
    -------
    dict[str, Any]
        Summary returned by :func:`postprocess_one_recording`. Metrics are saved
        in the selected sorter folder as ``metrics.csv``.
    """
    stream_folder = (
        select_data_root()
        / SUBJECT_ID
        / SESSION_NAME
        / "ephys"
        / "derived"
        / STREAM_FOLDER_NAME
    )

    summary = postprocess_one_recording(
        stream_folder=stream_folder,
        sorter_folder_name=SORTER_FOLDER_NAME,
        keep_good_only=KEEP_GOOD_ONLY,
        require_uV=REQUIRE_UV,
        compute_principal_components=COMPUTE_PRINCIPAL_COMPONENTS,
        job_kwargs=dict(POSTPROCESSING_JOB_KWARGS),
    )
    print("Wrote metrics: {}".format(summary["metrics_path"]))
    print("Wrote summary: {}".format(summary["summary_path"]))
    return summary


def _validate_channel_locations(
    locations: np.ndarray,
    n_channels: int,
    source_path: Path,
) -> np.ndarray:
    """Validate channel locations from one geometry source.

    Parameters
    ----------
    locations : numpy.ndarray
        Candidate locations with shape ``(n_channels, 2)`` in micrometers.
    n_channels : int
        Expected number of rows. Units are channels.
    source_path : pathlib.Path
        Source file path used for error messages.

    Returns
    -------
    numpy.ndarray
        Float locations with shape ``(n_channels, 2)`` in micrometers.
    """
    if locations.shape != (n_channels, 2):
        raise ValueError(
            "Expected channel locations with shape ({}, 2) from {}; found {}".format(
                n_channels,
                source_path,
                locations.shape,
            )
        )
    return locations.astype(float, copy=False)


def _recording_has_uV_scaling(recording) -> bool:
    """Return whether a recording can provide scaled traces in uV.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor.

    Returns
    -------
    bool
        ``True`` if the recording reports scalable traces; otherwise ``False``.
    """
    if hasattr(recording, "has_scaleable_traces"):
        return bool(recording.has_scaleable_traces())
    return False


def _json_safe_context(context: dict[str, Any]) -> dict[str, Any]:
    """Convert context values to JSON-serializable objects.

    Parameters
    ----------
    context : dict[str, Any]
        Context dictionary. Values can include paths, tuples, lists, dicts, and
        NumPy scalar values.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with path values converted to strings.
    """
    return {key: _json_safe_value(value) for key, value in context.items()}


def _json_safe_value(value):
    """Convert one value to a JSON-safe representation."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
