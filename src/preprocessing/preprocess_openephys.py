"""Validation helpers for raw Open Ephys binary recordings.

This module only discovers and summarizes raw Open Ephys streams. It does not
extract LFP data, compute derived products, or read signal samples into memory.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import spikeinterface.extractors as se
from spikeinterface.extractors.extractor_classes import OpenEphysBinaryRecordingExtractor


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


def validate_open_ephys_probe(
    raw_root: Path,
    experiment_name: str | None = None,
    stream_name: str | None = None,
    load_sync_timestamps: bool = False,
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

    Returns
    -------
    dict[str, Any]
        Validation result with selected experiment name, stream table, selected
        stream name, and stream summary. Signal samples are not read.

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

    return {
        "experiment_name": selected_experiment_name,
        "streams": streams,
        "stream_name": selected_stream_name,
        "summary": summarize_open_ephys_stream(recording),
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


def main() -> dict[str, Any]:
    """Run a local validation demo for one hardcoded raw Open Ephys stream."""
    raw_root = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
        "CT026_20260727_alternating_latent/ephys/raw"
    )
    experiment_name = "experiment1"
    stream_name = "Record Node 101#Neuropix-PXI-100.ProbeA"
    load_sync_timestamps = False

    result = validate_open_ephys_probe(
        raw_root=raw_root,
        experiment_name=experiment_name,
        stream_name=stream_name,
        load_sync_timestamps=load_sync_timestamps,
    )
    print("Selected experiment:", result["experiment_name"])
    print(result["streams"])
    print("Selected stream:", result["stream_name"])
    print(result["summary"])
    return result


if __name__ == "__main__":
    main()
