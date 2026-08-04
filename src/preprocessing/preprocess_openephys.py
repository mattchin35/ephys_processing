"""Validation helpers for raw Open Ephys binary recordings.

This module only discovers and summarizes raw Open Ephys streams. It does not
extract LFP data, compute derived products, or read signal samples into memory.
"""

import json
from pathlib import Path
import re
from typing import Any
import csv

import matplotlib.pyplot as plt
import pandas as pd
import spikeinterface.preprocessing as spre
import spikeinterface.extractors as se
from spikeinterface.core import write_binary_recording
from spikeinterface.extractors.extractor_classes import OpenEphysBinaryRecordingExtractor
from probeinterface.plotting import plot_probe
from scipy.io import savemat
import numpy as np


STREAM_TABLE_COLUMNS = [
    "stream_id",
    "stream_name",
    "record_node",
    "source_name",
    "is_neuropixels",
    "is_nidaq",
]


def find_open_ephys_experiments(raw_root: Path) -> list[str]:
    """Find Open Ephys experiment labels under a raw-data root.

    Parameters
    ----------
    raw_root
        Path to an Open Ephys binary-format root directory. The directory may
        contain Open Ephys record nodes directly or session folders containing
        record nodes. Units are filesystem path components.

    Returns
    -------
    list[str]
        Open Ephys experiment labels, for example ``["experiment1"]``. The
        values are plain Python strings, even if SpikeInterface returns NumPy
        string scalars.

    Raises
    ------
    ValueError
        If no experiment labels can be discovered.
    """
    experiment_names = OpenEphysBinaryRecordingExtractor.get_available_experiments(Path(raw_root))
    experiment_names = [str(experiment_name) for experiment_name in experiment_names]

    if not experiment_names:
        raise ValueError(f"No Open Ephys experiments found under {raw_root}")

    return experiment_names


def find_open_ephys_streams(raw_root: Path, experiment_name: str) -> pd.DataFrame:
    """Find continuous streams for one Open Ephys experiment.

    Parameters
    ----------
    raw_root
        Path to an Open Ephys binary-format root directory. Units are filesystem
        path components.
    experiment_name
        Open Ephys experiment label, for example ``"experiment1"``. This is not
        the timestamped session folder name.

    Returns
    -------
    pandas.DataFrame
        One row per continuous stream. Columns are ``stream_id``, ``stream_name``,
        ``record_node``, ``source_name``, ``is_neuropixels``, and ``is_nidaq``.
        Rows describe metadata only; no continuous signal samples are loaded.

    Raises
    ------
    ValueError
        If SpikeInterface cannot parse streams for the requested experiment, or
        if no streams are returned.
    """
    try:
        stream_names, stream_ids = OpenEphysBinaryRecordingExtractor.get_streams(
            Path(raw_root),
            experiment_names=[str(experiment_name)],
        )
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(
            f"Could not find streams for Open Ephys experiment {experiment_name!r} "
            f"under {raw_root}"
        ) from error

    if not stream_names:
        raise ValueError(
            f"No Open Ephys streams found for experiment {experiment_name!r} "
            f"under {raw_root}"
        )

    rows = []
    for stream_id, stream_name in zip(stream_ids, stream_names):
        record_node, source_name = _split_stream_name(str(stream_name))
        rows.append(
            {
                "stream_id": str(stream_id),
                "stream_name": str(stream_name),
                "record_node": record_node,
                "source_name": source_name,
                "is_neuropixels": _is_neuropixels_stream(source_name),
                "is_nidaq": _is_nidaq_stream(source_name),
            }
        )

    return pd.DataFrame(rows, columns=STREAM_TABLE_COLUMNS)


def select_neuropixels_stream(streams: pd.DataFrame, stream_name: str | None = None) -> str:
    """Select a Neuropixels stream from a stream table.

    Parameters
    ----------
    streams
        Stream table returned by :func:`find_open_ephys_streams`. Shape is
        ``(n_streams, 6)`` with one row per stream and columns defined by
        ``STREAM_TABLE_COLUMNS``. Units are stream identifiers and boolean
        labels.
    stream_name
        Optional exact stream name to select, for example
        ``"Record Node 101#Neuropix-PXI-100.ProbeA"``. If omitted, selection is
        automatic only when exactly one Neuropixels stream is present.

    Returns
    -------
    str
        Selected stream name.

    Raises
    ------
    ValueError
        If the requested stream is unavailable, if no Neuropixels stream exists,
        or if multiple Neuropixels streams require an explicit selection.
    """
    if stream_name is not None:
        matching_streams = streams.loc[streams["stream_name"] == stream_name]
        if matching_streams.empty:
            available_streams = streams["stream_name"].tolist()
            raise ValueError(
                f"Requested stream {stream_name!r} was not found. "
                f"Available streams: {available_streams}"
            )
        return str(stream_name)

    neuropixels_streams = streams.loc[streams["is_neuropixels"]]
    if neuropixels_streams.empty:
        raise ValueError("No Neuropixels streams were found")

    if len(neuropixels_streams) > 1:
        available_streams = neuropixels_streams["stream_name"].tolist()
        raise ValueError(
            "Multiple Neuropixels streams were found; pass stream_name explicitly. "
            f"Available Neuropixels streams: {available_streams}"
        )

    return str(neuropixels_streams.iloc[0]["stream_name"])


def load_open_ephys_stream(
    raw_root: Path,
    experiment_name: str,
    stream_name: str,
    load_sync_timestamps: bool = False,
):
    """Load one raw Open Ephys continuous stream with SpikeInterface.

    Parameters
    ----------
    raw_root
        Path to an Open Ephys binary-format root directory. Units are filesystem
        path components.
    experiment_name
        Open Ephys experiment label, for example ``"experiment1"``.
    stream_name
        Exact stream name from :func:`find_open_ephys_streams`, for example
        ``"Record Node 101#Neuropix-PXI-100.ProbeA"``.
    load_sync_timestamps
        Whether SpikeInterface should attempt to load synchronized timestamps.
        The default is ``False`` because some raw sessions do not provide them
        in a form SpikeInterface can attach.

    Returns
    -------
    spikeinterface.core.BaseRecording
        Lazy SpikeInterface recording extractor. Shape is
        ``(n_samples, n_channels)`` when traces are requested from
        SpikeInterface. Samples remain in their source dtype and physical units
        are described by channel properties such as ``gain_to_uV``.
    """
    return se.read_openephys(
        folder_path=Path(raw_root),
        experiment_name=str(experiment_name),
        stream_name=str(stream_name),
        load_sync_timestamps=load_sync_timestamps,
    )


def summarize_open_ephys_stream(recording) -> dict[str, Any]:
    """Summarize a SpikeInterface recording without reading signal traces.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Continuous trace convention is
        ``(n_samples, n_channels)`` when samples are later requested. Sampling
        frequency is in Hz. Sample values retain the extractor dtype until
        explicit gain conversion is requested elsewhere.

    Returns
    -------
    dict[str, Any]
        Metadata summary containing sampling frequency in Hz, segment count,
        channel count, sample counts per segment, dtype, channel-id shape,
        property keys, probe availability, inter-sample-shift availability, and
        channel-location shape when locations are present.
    """
    num_segments = int(recording.get_num_segments())
    property_keys = list(recording.get_property_keys())
    channel_ids = recording.get_channel_ids()

    summary = {
        "sampling_frequency_hz": float(recording.get_sampling_frequency()),
        "num_segments": num_segments,
        "num_channels": int(recording.get_num_channels()),
        "num_samples_by_segment": [
            int(recording.get_num_samples(segment_index=segment_index))
            for segment_index in range(num_segments)
        ],
        "dtype": str(recording.get_dtype()),
        "channel_ids_shape": tuple(channel_ids.shape),
        "property_keys": property_keys,
        "has_probe": recording.get_probe() is not None,
        "has_inter_sample_shift": "inter_sample_shift" in property_keys,
        "location_shape": None,
    }

    if "location" in property_keys:
        summary["location_shape"] = tuple(recording.get_property("location").shape)

    return summary


def make_safe_path_component(name: str) -> str:
    """Convert a stream label into one readable filesystem path component.

    Parameters
    ----------
    name
        Stream label or other identifier. Units are text characters. The input
        may contain path separators, spaces, or punctuation from SpikeInterface
        stream names.

    Returns
    -------
    str
        Filesystem-safe path component. Shape is scalar text. Runs of unsafe
        characters are replaced with one underscore while letters, numbers,
        periods, hyphens, and underscores are preserved.
    """
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))
    safe_name = safe_name.strip("_")
    if not safe_name:
        raise ValueError("Path component cannot be empty after sanitizing")
    return safe_name


def build_stream_output_dir(output_root: Path, stream_name: str) -> Path:
    """Create and return the derived-output directory for one stream.

    Parameters
    ----------
    output_root
        Root directory for derived Open Ephys outputs. Units are filesystem path
        components. Missing directories are created.
    stream_name
        Exact SpikeInterface stream name, for example
        ``"Record Node 101#Neuropix-PXI-100.ProbeA"``.

    Returns
    -------
    pathlib.Path
        Directory path ``output_root / safe_stream_name``. The directory exists
        when the function returns.
    """
    stream_output_dir = Path(output_root) / make_safe_path_component(stream_name)
    stream_output_dir.mkdir(parents=True, exist_ok=True)
    return stream_output_dir


def plot_probe_channel_map(
    recording,
    stream_name: str,
    output_root: Path | None,
    show: bool = True,
    save: bool = True,
) -> Path | None:
    """Plot and optionally save the probe channel map for one loaded stream.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor for one stream. Continuous
        traces follow SpikeInterface's ``(n_samples, n_channels)`` convention
        when requested, but this function only reads probe/channel metadata.
    stream_name
        Exact stream name used to load the recording. Units are text; this is
        used to create the stream-specific derived-output directory.
    output_root
        Root directory for derived outputs. Required when ``save`` is true.
        Units are filesystem path components.
    show
        Whether to display the figure using matplotlib.
    save
        Whether to save ``probe_layout.png`` into the stream-specific output
        directory.

    Returns
    -------
    pathlib.Path | None
        Saved PNG path when ``save`` is true, otherwise ``None``.

    Raises
    ------
    ValueError
        If no probe geometry is attached, if Neuropixels inter-sample shifts are
        missing, or if ``output_root`` is omitted while saving.
    """
    probe = recording.get_probe()
    if probe is None:
        raise ValueError("No probe geometry was loaded for this stream")

    property_keys = list(recording.get_property_keys())
    if "inter_sample_shift" not in property_keys:
        raise ValueError("No Neuropixels inter-sample shifts were loaded")

    if save and output_root is None:
        raise ValueError("output_root must be provided when save=True")

    fig, ax = plt.subplots(figsize=(6, 14))
    plot_probe(
        probe,
        ax=ax,
        with_contact_id=False,
        with_device_index=False,
        title=False,
    )
    ax.set_title(f"{stream_name} recorded contacts")
    ax.set_xlabel("x position (um)")
    ax.set_ylabel("y position (um)")
    fig.tight_layout()

    figure_path = None
    if save:
        stream_output_dir = build_stream_output_dir(
            output_root=Path(output_root),
            stream_name=stream_name,
        )
        figure_path = stream_output_dir / "probe_layout.png"
        fig.savefig(figure_path, dpi=200)

    if show:
        plt.show(block=False)
    else:
        plt.close(fig)

    return figure_path


def validate_open_ephys_probe(
    raw_root: Path,
    experiment_name: str | None = None,
    stream_name: str | None = None,
    load_sync_timestamps: bool = False,
    output_root: Path | None = None,
    plot_probe_layout: bool = False,
    show_probe_layout: bool = True,
    save_probe_layout: bool = True,
    write_kilosort_chanmap_file: bool = False,
    extract_lfp_file: bool = False,
    extract_ap_file: bool = False,
    detect_channel_quality_file: bool = False,
    lfp_freq_min_hz: float = 1.0,
    lfp_freq_max_hz: float = 500.0,
    lfp_filter_order: int = 3,
    lfp_filter_margin_ms: float | str = "auto",
    lfp_resample_rate_hz: int | float = 2500,
    lfp_resample_margin_ms: float = 100.0,
    lfp_dtype: str = "float32",
    lfp_n_jobs: int = 8,
    lfp_chunk_duration: str = "30s",
    lfp_pool_engine: str = "process",
    lfp_mp_context: str | None = None,
    lfp_progress_bar: bool = True,
    ap_highpass_hz: float = 300.0,
    ap_local_car_inner_um: float = 40.0,
    ap_local_car_outer_um: float = 140.0,
    ap_min_local_neighbors: int = 5,
    ap_working_dtype: str = "float32",
    ap_output_dtype: str = "int16",
    ap_n_jobs: int = 8,
    ap_chunk_duration: str = "1s",
    ap_num_random_chunks: int = 20,
    ap_random_seed: int = 0,
    ap_progress_bar: bool = True,
    channel_quality_source: str = "ap",
    channel_quality_method: str = "coherence+psd",
    channel_quality_outside_location: str = "top",
    channel_quality_direction: str = "y",
    channel_quality_seed: int = 0,
    channel_quality_num_random_chunks: int = 20,
    channel_quality_chunk_duration_s: float = 0.3,
) -> dict[str, Any]:
    """Discover, load, and summarize one raw Open Ephys Neuropixels stream.

    Parameters
    ----------
    raw_root
        Path to an Open Ephys binary-format root directory. Units are filesystem
        path components.
    experiment_name
        Optional Open Ephys experiment label. If omitted, it is selected
        automatically only when exactly one experiment is present.
    stream_name
        Optional exact Neuropixels stream name. If omitted, it is selected
        automatically only when exactly one Neuropixels stream is present.
    load_sync_timestamps
        Whether SpikeInterface should attempt synchronized timestamp loading.
    output_root
        Optional root directory for derived stream outputs. Required when
        ``plot_probe_layout`` and ``save_probe_layout`` are both true.
    plot_probe_layout
        Whether to plot the loaded stream's probe layout.
    show_probe_layout
        Whether to display the probe layout figure.
    save_probe_layout
        Whether to save the probe layout figure in the stream output directory.
    write_kilosort_chanmap_file
        Whether to write ``chanMap.mat`` for Kilosort in the stream output
        directory.
    extract_lfp_file
        Whether to materialize an LFP binary derived from the loaded full-rate
        raw stream.
    extract_ap_file
        Whether to materialize an AP binary derived from the loaded full-rate
        raw stream for Kilosort.
    detect_channel_quality_file
        Whether to detect and save per-channel quality labels.
    lfp_freq_min_hz
        LFP bandpass lower cutoff in Hz.
    lfp_freq_max_hz
        LFP bandpass upper cutoff in Hz.
    lfp_filter_order
        Butterworth filter order for LFP bandpass filtering.
    lfp_filter_margin_ms
        SpikeInterface filter margin in milliseconds, or ``"auto"``.
    lfp_resample_rate_hz
        Output LFP sampling frequency in Hz. Must be an integer or
        integer-like float because SpikeInterface requires integer resampling
        rates.
    lfp_resample_margin_ms
        SpikeInterface resampling margin in milliseconds.
    lfp_dtype
        Output LFP binary dtype.
    lfp_n_jobs
        Number of worker jobs used while writing the LFP binary.
    lfp_chunk_duration
        Chunk duration passed to SpikeInterface while writing the LFP binary.
    lfp_pool_engine
        SpikeInterface worker engine used when ``lfp_n_jobs > 1``. Valid values
        are defined by SpikeInterface and include ``"process"`` and
        ``"thread"``.
    lfp_mp_context
        Multiprocessing start context passed to SpikeInterface when process
        workers are used, for example ``"fork"`` or ``"spawn"``. ``None`` lets
        SpikeInterface choose.
    lfp_progress_bar
        Whether SpikeInterface should display write progress.
    ap_highpass_hz
        AP high-pass cutoff frequency in Hz.
    ap_local_car_inner_um
        Inner exclusion radius for AP local common average reference, in
        micrometers.
    ap_local_car_outer_um
        Outer inclusion radius for AP local common average reference, in
        micrometers.
    ap_min_local_neighbors
        Minimum number of local AP reference channels required by
        SpikeInterface.
    ap_working_dtype
        Floating dtype used inside the AP preprocessing chain.
    ap_output_dtype
        AP binary export dtype.
    ap_n_jobs
        Number of worker jobs used while writing the AP binary.
    ap_chunk_duration
        Chunk duration passed to SpikeInterface while writing the AP binary.
    ap_num_random_chunks
        Number of chunks sampled for AP range QC before integer export.
    ap_random_seed
        Seed controlling AP range-QC chunk selection.
    ap_progress_bar
        Whether SpikeInterface should display AP binary-writing progress.
    channel_quality_source
        Recording source used for channel quality detection. ``"ap"`` uses the
        loaded full-rate stream. ``"lfp"`` is reserved for a future LFP
        recording-construction refactor.
    channel_quality_method
        SpikeInterface bad-channel detection method.
    channel_quality_outside_location
        Probe side where outside-brain channels are expected. SpikeInterface
        accepts values such as ``"top"``, ``"bottom"``, and ``"both"``.
    channel_quality_direction
        Probe location axis used as depth by SpikeInterface.
    channel_quality_seed
        Random seed for chunk sampling during channel quality detection.
    channel_quality_num_random_chunks
        Number of chunks sampled for channel quality detection.
    channel_quality_chunk_duration_s
        Duration of each sampled channel-quality chunk in seconds.

    Returns
    -------
    dict[str, Any]
        Validation result with selected experiment name, stream table, selected
        stream name, stream summary, optional probe-layout figure path, and
        optional Kilosort channel-map and LFP-output paths.

    Raises
    ------
    ValueError
        If automatic experiment or stream selection is ambiguous.
    """
    selected_experiment_name = experiment_name
    if selected_experiment_name is None:
        experiment_names = find_open_ephys_experiments(raw_root)
        if len(experiment_names) > 1:
            raise ValueError(
                "Multiple Open Ephys experiments were found; pass "
                f"experiment_name explicitly. Available experiments: {experiment_names}"
            )
        selected_experiment_name = experiment_names[0]

    streams = find_open_ephys_streams(raw_root, selected_experiment_name)
    selected_stream_name = select_neuropixels_stream(streams, stream_name=stream_name)
    recording = load_open_ephys_stream(
        raw_root=raw_root,
        experiment_name=selected_experiment_name,
        stream_name=selected_stream_name,
        load_sync_timestamps=load_sync_timestamps,
    )
    summary = summarize_open_ephys_stream(recording)
    probe_layout_path = None
    kilosort_chanmap_path = None
    lfp_result = None
    ap_result = None
    channel_quality_result = None

    if plot_probe_layout:
        probe_layout_path = plot_probe_channel_map(
            recording=recording,
            stream_name=selected_stream_name,
            output_root=output_root,
            show=show_probe_layout,
            save=save_probe_layout,
        )

    if write_kilosort_chanmap_file:
        if output_root is None:
            raise ValueError(
                "output_root must be provided when "
                "write_kilosort_chanmap_file=True"
            )
        stream_output_dir = build_stream_output_dir(
            output_root=output_root,
            stream_name=selected_stream_name,
        )
        kilosort_chanmap_path = write_kilosort_chanmap(
            recording=recording,
            output_file=stream_output_dir / "chanMap.mat",
        )

    if extract_lfp_file:
        if output_root is None:
            raise ValueError("output_root must be provided when extract_lfp_file=True")
        stream_output_dir = build_stream_output_dir(
            output_root=output_root,
            stream_name=selected_stream_name,
        )
        lfp_result = extract_lfp(
            recording=recording,
            output_folder=stream_output_dir,
            freq_min_hz=lfp_freq_min_hz,
            freq_max_hz=lfp_freq_max_hz,
            filter_order=lfp_filter_order,
            filter_margin_ms=lfp_filter_margin_ms,
            resample_rate_hz=lfp_resample_rate_hz,
            resample_margin_ms=lfp_resample_margin_ms,
            dtype=lfp_dtype,
            n_jobs=lfp_n_jobs,
            chunk_duration=lfp_chunk_duration,
            pool_engine=lfp_pool_engine,
            mp_context=lfp_mp_context,
            progress_bar=lfp_progress_bar,
        )

    if extract_ap_file:
        if output_root is None:
            raise ValueError("output_root must be provided when extract_ap_file=True")
        from src.preprocessing.ap_preprocessing_openephys import preprocess_ap_for_kilosort

        stream_output_dir = build_stream_output_dir(
            output_root=output_root,
            stream_name=selected_stream_name,
        )
        ap_result = preprocess_ap_for_kilosort(
            recording=recording,
            output_folder=stream_output_dir,
            highpass_hz=ap_highpass_hz,
            local_car_inner_um=ap_local_car_inner_um,
            local_car_outer_um=ap_local_car_outer_um,
            min_local_neighbors=ap_min_local_neighbors,
            working_dtype=ap_working_dtype,
            output_dtype=ap_output_dtype,
            n_jobs=ap_n_jobs,
            chunk_duration=ap_chunk_duration,
            num_random_chunks=ap_num_random_chunks,
            random_seed=ap_random_seed,
            progress_bar=ap_progress_bar,
        )

    if detect_channel_quality_file:
        if output_root is None:
            raise ValueError(
                "output_root must be provided when detect_channel_quality_file=True"
            )
        if channel_quality_source == "ap":
            quality_recording = recording
        elif channel_quality_source == "lfp":
            raise NotImplementedError(
                "channel_quality_source='lfp' will be supported after LFP "
                "recording construction is factored out"
            )
        else:
            raise ValueError(
                "channel_quality_source must be 'ap' or 'lfp'; "
                f"got {channel_quality_source!r}"
            )

        stream_output_dir = build_stream_output_dir(
            output_root=output_root,
            stream_name=selected_stream_name,
        )
        channel_quality_result = detect_channel_quality(
            recording=quality_recording,
            output_folder=stream_output_dir,
            method=channel_quality_method,
            outside_channels_location=channel_quality_outside_location,
            direction=channel_quality_direction,
            seed=channel_quality_seed,
            num_random_chunks=channel_quality_num_random_chunks,
            chunk_duration_s=channel_quality_chunk_duration_s,
        )

    return {
        "experiment_name": selected_experiment_name,
        "streams": streams,
        "stream_name": selected_stream_name,
        "summary": summary,
        "probe_layout_path": probe_layout_path,
        "kilosort_chanmap_path": kilosort_chanmap_path,
        "lfp_result": lfp_result,
        "ap_result": ap_result,
        "channel_quality_result": channel_quality_result,
    }


def _split_stream_name(stream_name: str) -> tuple[str | None, str]:
    """Split a SpikeInterface stream name into record-node and source labels."""
    if "#" not in stream_name:
        return None, stream_name

    record_node, source_name = stream_name.split("#", maxsplit=1)
    return record_node, source_name


def _is_neuropixels_stream(source_name: str) -> bool:
    """Return whether a source label appears to describe a Neuropixels probe."""
    return "Neuropix" in source_name or "Neuropixels" in source_name


def _is_nidaq_stream(source_name: str) -> bool:
    """Return whether a source label appears to describe an NI-DAQ stream."""
    return "NI-DAQ" in source_name or "NIDAQ" in source_name


def write_kilosort_chanmap(recording, output_file: Path) -> Path:
    """Write a MATLAB Kilosort channel map for one loaded recording stream.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. The Kilosort map assumes the
        binary channel columns are in ``recording.channel_ids`` order. Channel
        locations are read in micrometers with shape ``(n_channels, >=2)``.
        Sampling frequency is read in Hz.
    output_file
        Destination ``.mat`` file path. The parent directory is created if it is
        missing. The expected filename in this workflow is ``chanMap.mat``.

    Returns
    -------
    pathlib.Path
        Path to the written MATLAB file.

    Raises
    ------
    ValueError
        If channel locations do not have shape ``(n_channels, >=2)``, if shank
        labels do not match the channel count, or if sampling frequency is not
        finite and positive.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    locations = np.asarray(recording.get_channel_locations())
    if locations.ndim != 2 or locations.shape[1] < 2:
        raise ValueError("Recording must have 2-D channel locations")

    n_channels = recording.get_num_channels()
    if locations.shape[0] != n_channels:
        raise ValueError("Location count does not match channel count")

    sampling_frequency = float(recording.get_sampling_frequency())
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0:
        raise ValueError("Sampling frequency must be finite and positive")

    shank_labels = _get_kilosort_shank_labels(recording=recording, n_channels=n_channels)
    kcoords = _map_labels_to_one_based_indices(shank_labels)

    mat = {
        # chanMap is one-based in a MATLAB .mat channel map.
        "chanMap": np.arange(
            1, n_channels + 1, dtype=np.int32
        )[:, None],

        # Useful to retain for software expecting an explicit zero-based map.
        "chanMap0ind": np.arange(
            n_channels, dtype=np.int32
        )[:, None],

        "connected": np.ones(
            (n_channels, 1), dtype=np.bool_
        ),
        "xcoords": locations[:, 0].astype(np.float64)[:, None],
        "ycoords": locations[:, 1].astype(np.float64)[:, None],
        "kcoords": kcoords[:, None],
        "fs": np.array(
            [[sampling_frequency]],
            dtype=np.float64,
        ),
        "name": output_file.stem,
    }

    savemat(output_file, mat, do_compression=True)
    return output_file


def _get_kilosort_shank_labels(recording, n_channels: int) -> np.ndarray:
    """Get per-channel shank/group labels for Kilosort ``kcoords``.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. If available,
        ``recording.get_property("contact_vector")["shank_ids"]`` is preferred.
        Otherwise ``recording.get_channel_groups()`` is used.
    n_channels
        Number of channels in the recording. Units are channel count.

    Returns
    -------
    numpy.ndarray
        One-dimensional label array with shape ``(n_channels,)``. Labels may be
        strings or numbers and are mapped later to one-based Kilosort indices.

    Raises
    ------
    ValueError
        If the selected label source does not have one entry per channel.
    """
    property_keys = list(recording.get_property_keys())
    if "contact_vector" in property_keys:
        contact_vector = recording.get_property("contact_vector")
        if getattr(contact_vector.dtype, "names", None) and "shank_ids" in contact_vector.dtype.names:
            shank_labels = np.asarray(contact_vector["shank_ids"])
            if shank_labels.shape[0] != n_channels:
                raise ValueError("Shank label count does not match channel count")
            return shank_labels

    shank_labels = np.asarray(recording.get_channel_groups())
    if shank_labels.shape[0] != n_channels:
        raise ValueError("Group label count does not match channel count")
    return shank_labels


def _map_labels_to_one_based_indices(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary labels to MATLAB-friendly one-based integer indices.

    Parameters
    ----------
    labels
        One-dimensional array-like labels with shape ``(n_channels,)``. Units
        are categorical shank/group labels.

    Returns
    -------
    numpy.ndarray
        Integer Kilosort ``kcoords`` values with shape ``(n_channels,)`` and
        one-based indexing.
    """
    unique_labels = list(dict.fromkeys(np.asarray(labels).tolist()))
    label_lookup = {
        label: index + 1
        for index, label in enumerate(unique_labels)
    }
    return np.array(
        [label_lookup[label] for label in np.asarray(labels).tolist()],
        dtype=np.int32,
    )


def detect_channel_quality(
    recording,
    output_folder: Path,
    method: str = "coherence+psd",
    outside_channels_location: str = "top",
    direction: str = "y",
    seed: int = 0,
    num_random_chunks: int = 20,
    chunk_duration_s: float = 0.3,
) -> dict[str, Any]:
    """Detect and save per-channel quality labels for one recording.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Traces follow SpikeInterface's
        ``(n_samples, n_channels)`` convention when sampled. Channel locations
        must be available with shape ``(n_channels, >=2)`` in micrometers.
    output_folder
        Stream-specific derived-output directory. The function writes
        ``channel_quality.json`` directly inside this directory.
    method
        SpikeInterface bad-channel detection method. ``"coherence+psd"``
        returns labels including ``"good"``, ``"dead"``, ``"noise"``, and
        ``"out"``.
    outside_channels_location
        Probe side where outside-brain channels are expected, passed through to
        SpikeInterface. Units are categorical text such as ``"top"``.
    direction
        Channel-location axis used as depth by SpikeInterface, for example
        ``"y"``.
    seed
        Random seed for SpikeInterface's chunk sampling.
    num_random_chunks
        Number of random chunks sampled from the recording.
    chunk_duration_s
        Duration of each sampled chunk in seconds.

    Returns
    -------
    dict[str, Any]
        Channel-quality summary. ``channel_quality_path`` is a ``pathlib.Path``.
        ``channels`` is a list with one record per channel containing channel ID,
        SpikeInterface label, inside-brain boolean, and x/y coordinates in
        micrometers. ``inside_brain`` is ``False`` only for label ``"out"``.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    channel_quality_path = output_folder / "channel_quality.json"
    channel_quality_csv_path = output_folder / "channel_quality.csv"

    bad_channel_ids, channel_labels = spre.detect_bad_channels(
        recording=recording,
        method=method,
        outside_channels_location=outside_channels_location,
        direction=direction,
        seed=seed,
        num_random_chunks=num_random_chunks,
        chunk_duration_s=chunk_duration_s,
    )

    channel_ids = [str(channel_id) for channel_id in _get_recording_channel_ids(recording)]
    channel_labels = np.asarray(channel_labels, dtype=str)
    locations_um = _get_recording_channel_locations(recording)

    if channel_labels.shape != (len(channel_ids),):
        raise ValueError(
            "Channel quality labels must have shape (n_channels,), "
            f"found {channel_labels.shape} for {len(channel_ids)} channel IDs."
        )

    if locations_um.shape[0] != len(channel_ids):
        raise ValueError(
            "Channel location count does not match channel ID count: "
            f"{locations_um.shape[0]} locations for {len(channel_ids)} IDs."
        )

    unique_labels, label_counts = np.unique(channel_labels, return_counts=True)
    counts = {
        str(label): int(count)
        for label, count in zip(unique_labels, label_counts)
    }

    channel_records = []
    for channel_id, label, location_um in zip(channel_ids, channel_labels, locations_um):
        label = str(label)
        channel_records.append(
            {
                "channel_id": channel_id,
                "label": label,
                "inside_brain": label != "out",
                "x_um": float(location_um[0]),
                "y_um": float(location_um[1]),
            }
        )

    metadata = {
        "output_file": channel_quality_path.name,
        "method": method,
        "outside_channels_location": outside_channels_location,
        "direction": direction,
        "seed": int(seed),
        "num_random_chunks": int(num_random_chunks),
        "chunk_duration_s": float(chunk_duration_s),
        "bad_channel_ids": [str(channel_id) for channel_id in bad_channel_ids],
        "counts": counts,
        "inside_brain_definition": "inside_brain is false only when label == 'out'",
        "channels": channel_records,
    }

    channel_quality_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    csv_columns = ["channel_id", "label", "is_good", "inside_brain", "x_um", "y_um"]
    with channel_quality_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(
            {
                **channel_record,
                "is_good": channel_record["label"] == "good",
            }
            for channel_record in channel_records
        )

    result = dict(metadata)
    result["channel_quality_path"] = channel_quality_path
    result["channel_quality_csv_path"] = channel_quality_csv_path
    return result


def extract_lfp(
    recording,
    output_folder: Path,
    freq_min_hz: float = 1.0,
    freq_max_hz: float = 500.0,
    filter_order: int = 3,
    filter_margin_ms: float | str = "auto",
    resample_rate_hz: int | float = 2500,
    resample_margin_ms: float = 100.0,
    dtype: str = "float32",
    n_jobs: int = 8,
    chunk_duration: str = "30s",
    pool_engine: str = "process",
    mp_context: str | None = None,
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Extract and save LFP from one full-rate Neuropixels recording stream.

    Parameters
    ----------
    recording
        SpikeInterface-like full-rate raw recording extractor. Input traces use
        SpikeInterface's ``(n_samples, n_channels)`` convention when materialized.
        Samples keep their source physical-unit scaling metadata; this function
        writes the requested ``dtype`` after filtering and resampling.
    output_folder
        Stream-specific derived-output directory. The LFP files are written
        directly into this folder, not into an LFP subdirectory.
    freq_min_hz
        Bandpass lower cutoff in Hz.
    freq_max_hz
        Bandpass upper cutoff in Hz.
    filter_order
        Butterworth bandpass filter order.
    filter_margin_ms
        SpikeInterface filter margin in milliseconds, or ``"auto"``.
    resample_rate_hz
        Output LFP sampling frequency in Hz. Must be an integer or
        integer-like float because SpikeInterface requires integer resampling
        rates.
    resample_margin_ms
        SpikeInterface resampling margin in milliseconds.
    dtype
        Output binary dtype.
    n_jobs
        Number of worker jobs used while writing the binary.
    chunk_duration
        Chunk duration passed to SpikeInterface during binary writing.
    pool_engine
        SpikeInterface worker engine used when ``n_jobs > 1``. Valid values are
        defined by SpikeInterface and include ``"process"`` and ``"thread"``.
    mp_context
        Multiprocessing start context passed to SpikeInterface when process
        workers are used, for example ``"fork"`` or ``"spawn"``. ``None`` lets
        SpikeInterface choose.
    progress_bar
        Whether SpikeInterface should display write progress.

    Returns
    -------
    dict[str, Any]
        LFP output summary. Includes ``lfp_binary_path``,
        ``lfp_metadata_path``, sampling frequency in Hz, channel count, segment
        count, per-segment sample counts, dtype, and preprocessing parameters.

    Raises
    ------
    ValueError
        If no probe geometry is attached or Neuropixels inter-sample-shift
        metadata are unavailable.
    """
    normalized_resample_rate_hz = _normalize_integer_sample_rate_hz(
        sample_rate_hz=resample_rate_hz,
        parameter_name="resample_rate_hz",
    )

    if recording.get_probe() is None:
        raise ValueError("No probe geometry was loaded for this stream")

    if "inter_sample_shift" not in list(recording.get_property_keys()):
        raise ValueError("No Neuropixels inter-sample shifts were loaded")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    lfp_binary_path = output_folder / "lfp.dat"
    lfp_metadata_path = output_folder / "lfp_preprocessing.json"

    shifted = spre.phase_shift(
        recording,
        dtype=dtype,
    )
    lfp_filtered = spre.bandpass_filter(
        shifted,
        freq_min=freq_min_hz,
        freq_max=freq_max_hz,
        filter_order=filter_order,
        filter_mode="sos",
        ftype="butter",
        direction="forward-backward",
        margin_ms=filter_margin_ms,
        ignore_low_freq_error=True,
        dtype=dtype,
    )
    lfp_downsampled = spre.resample(
        lfp_filtered,
        resample_rate=normalized_resample_rate_hz,
        margin_ms=resample_margin_ms,
        dtype=dtype,
    )

    write_binary_recording(
        recording=lfp_downsampled,
        file_paths=lfp_binary_path,
        dtype=dtype,
        add_file_extension=False,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        pool_engine=pool_engine,
        mp_context=mp_context,
        progress_bar=progress_bar,
        verbose=True,
    )

    num_segments = int(lfp_downsampled.get_num_segments())
    metadata = {
        "output_binary": lfp_binary_path.name,
        "sampling_frequency_hz": float(lfp_downsampled.get_sampling_frequency()),
        "num_channels": int(lfp_downsampled.get_num_channels()),
        "num_segments": num_segments,
        "num_samples_by_segment": [
            int(lfp_downsampled.get_num_samples(segment_index=segment_index))
            for segment_index in range(num_segments)
        ],
        "dtype": str(np.dtype(dtype)),
        "binary_layout": "time_major_channel_interleaved",
        "channel_ids_in_binary_order": [
            str(channel_id)
            for channel_id in _get_recording_channel_ids(lfp_downsampled)
        ],
        "preprocessing": [
            {
                "name": "neuropixels_phase_shift",
                "source_property": "inter_sample_shift",
                "dtype": dtype,
            },
            {
                "name": "bandpass_filter",
                "freq_min_hz": float(freq_min_hz),
                "freq_max_hz": float(freq_max_hz),
                "filter_type": "butter",
                "filter_order": int(filter_order),
                "filter_mode": "sos",
                "direction": "forward-backward",
                "margin_ms": filter_margin_ms,
                "ignore_low_freq_error": True,
                "dtype": dtype,
            },
            {
                "name": "resample",
                "resample_rate_hz": float(normalized_resample_rate_hz),
                "margin_ms": float(resample_margin_ms),
                "dtype": dtype,
            },
        ],
        "write_binary_recording": {
            "n_jobs": int(n_jobs),
            "chunk_duration": chunk_duration,
            "pool_engine": pool_engine,
            "mp_context": mp_context,
            "progress_bar": bool(progress_bar),
        },
    }
    lfp_metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    result = dict(metadata)
    result["lfp_binary_path"] = lfp_binary_path
    result["lfp_metadata_path"] = lfp_metadata_path
    return result


def _normalize_integer_sample_rate_hz(
    sample_rate_hz: int | float,
    parameter_name: str,
) -> int:
    """Normalize a sample rate to the integer required by SpikeInterface.

    Parameters
    ----------
    sample_rate_hz
        Sampling frequency in Hz. Integer-like values such as ``2500`` or
        ``2500.0`` are accepted. Fractional, non-finite, or non-positive values
        are rejected.
    parameter_name
        Name of the caller-facing parameter for error messages. Units are text.

    Returns
    -------
    int
        Sampling frequency in Hz as a Python integer, suitable for
        ``spikeinterface.preprocessing.resample(resample_rate=...)``.
    """
    sample_rate_float = float(sample_rate_hz)
    if (
        not np.isfinite(sample_rate_float)
        or sample_rate_float <= 0
        or not sample_rate_float.is_integer()
    ):
        raise ValueError(
            f"{parameter_name} must be a positive integer sampling rate in Hz; "
            f"got {sample_rate_hz!r}."
        )

    return int(sample_rate_float)


def _get_recording_channel_ids(recording) -> list[Any]:
    """Return channel IDs from a SpikeInterface-like recording.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Channel IDs have shape
        ``(n_channels,)`` and identify binary channel order.

    Returns
    -------
    list[Any]
        Channel IDs in binary channel order.
    """
    if hasattr(recording, "channel_ids"):
        return list(recording.channel_ids)
    return list(recording.get_channel_ids())


def _get_recording_channel_locations(recording) -> np.ndarray:
    """Return channel locations from a SpikeInterface-like recording.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Channel locations are expected
        to have shape ``(n_channels, >=2)`` in micrometers and are read from
        ``get_channel_locations()`` when available, otherwise from the
        ``"location"`` channel property.

    Returns
    -------
    numpy.ndarray
        Floating-point location array with shape ``(n_channels, >=2)`` in
        micrometers.
    """
    if hasattr(recording, "get_channel_locations"):
        locations_um = np.asarray(recording.get_channel_locations(), dtype=float)
    elif "location" in list(recording.get_property_keys()):
        locations_um = np.asarray(recording.get_property("location"), dtype=float)
    else:
        raise ValueError("Recording must provide channel locations")

    if locations_um.ndim != 2 or locations_um.shape[1] < 2:
        raise ValueError(
            "Recording channel locations must have shape (n_channels, >=2) in um"
        )

    return locations_um


def main() -> dict[str, Any]:
    """Run a local validation demo for one hardcoded raw Open Ephys stream."""
    session_path = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260801_latent_inference")
    raw_root = session_path / "ephys/raw"
    output_root = session_path / "ephys/derived"
    experiment_name = "experiment1"
    stream_name = "Record Node 101#Neuropix-PXI-110.ProbeA"

    load_sync_timestamps = False
    plot_probe_layout = True
    show_probe_layout = True
    save_probe_layout = True
    write_kilosort_chanmap_file = True
    extract_lfp_file = True
    extract_ap_file = True
    lfp_freq_min_hz = 1.0
    lfp_freq_max_hz = 500.0
    lfp_filter_order = 3
    lfp_filter_margin_ms = "auto"
    lfp_resample_rate_hz = 2500
    lfp_resample_margin_ms = 100.0
    lfp_dtype = "float32"
    lfp_n_jobs = 1 # use 1 with process if things fail
    lfp_chunk_duration = "30s"
    lfp_pool_engine = "process"  # use process with 1 job if things fail
    lfp_mp_context = None
    lfp_progress_bar = True
    ap_highpass_hz = 300.0
    ap_local_car_inner_um = 40.0
    ap_local_car_outer_um = 140.0
    ap_min_local_neighbors = 5
    ap_working_dtype = "float32"
    ap_output_dtype = "int16"
    ap_n_jobs = 8
    ap_chunk_duration = "1s"
    ap_num_random_chunks = 20
    ap_random_seed = 0
    ap_progress_bar = True
    detect_channel_quality_file = True
    channel_quality_source = "ap"
    channel_quality_method = "coherence+psd"
    channel_quality_outside_location = "top"
    channel_quality_direction = "y"
    channel_quality_seed = 0
    channel_quality_num_random_chunks = 20
    channel_quality_chunk_duration_s = 0.3

    result = validate_open_ephys_probe(
        raw_root=raw_root,
        experiment_name=experiment_name,
        stream_name=stream_name,
        load_sync_timestamps=load_sync_timestamps,
        output_root=output_root,
        plot_probe_layout=plot_probe_layout,
        show_probe_layout=show_probe_layout,
        save_probe_layout=save_probe_layout,
        write_kilosort_chanmap_file=write_kilosort_chanmap_file,
        extract_lfp_file=extract_lfp_file,
        extract_ap_file=extract_ap_file,
        lfp_freq_min_hz=lfp_freq_min_hz,
        lfp_freq_max_hz=lfp_freq_max_hz,
        lfp_filter_order=lfp_filter_order,
        lfp_filter_margin_ms=lfp_filter_margin_ms,
        lfp_resample_rate_hz=lfp_resample_rate_hz,
        lfp_resample_margin_ms=lfp_resample_margin_ms,
        lfp_dtype=lfp_dtype,
        lfp_n_jobs=lfp_n_jobs,
        lfp_chunk_duration=lfp_chunk_duration,
        lfp_pool_engine=lfp_pool_engine,
        lfp_mp_context=lfp_mp_context,
        lfp_progress_bar=lfp_progress_bar,
        ap_highpass_hz=ap_highpass_hz,
        ap_local_car_inner_um=ap_local_car_inner_um,
        ap_local_car_outer_um=ap_local_car_outer_um,
        ap_min_local_neighbors=ap_min_local_neighbors,
        ap_working_dtype=ap_working_dtype,
        ap_output_dtype=ap_output_dtype,
        ap_n_jobs=ap_n_jobs,
        ap_chunk_duration=ap_chunk_duration,
        ap_num_random_chunks=ap_num_random_chunks,
        ap_random_seed=ap_random_seed,
        ap_progress_bar=ap_progress_bar,
        detect_channel_quality_file=detect_channel_quality_file,
        channel_quality_source=channel_quality_source,
        channel_quality_method=channel_quality_method,
        channel_quality_outside_location=channel_quality_outside_location,
        channel_quality_direction=channel_quality_direction,
        channel_quality_seed=channel_quality_seed,
        channel_quality_num_random_chunks=channel_quality_num_random_chunks,
        channel_quality_chunk_duration_s=channel_quality_chunk_duration_s,
    )
    print("Selected experiment:", result["experiment_name"])
    print(result["streams"])
    print("Selected stream:", result["stream_name"])
    print(result["summary"])
    print("Probe layout path:", result["probe_layout_path"])
    print("Kilosort chanmap path:", result["kilosort_chanmap_path"])
    print("LFP result:", result["lfp_result"])
    print("AP result:", result["ap_result"])
    print("Channel quality result:", result["channel_quality_result"])
    return result


if __name__ == "__main__":
    main()
