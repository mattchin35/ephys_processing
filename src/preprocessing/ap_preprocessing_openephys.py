"""Reusable AP preprocessing helpers for Open Ephys Neuropixels streams.

The main preprocessing order is:

1. Neuropixels phase shift correction.
2. High-pass filtering.
3. Local common average reference.

The module is safe to import. Running the hardcoded example requires calling
``main()`` or executing this file directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import spikeinterface.preprocessing as spre
from spikeinterface.core import write_binary_recording

from src.preprocessing.preprocess_openephys import (
    build_stream_output_dir,
    load_open_ephys_stream,
)


AP_PREPROCESSING_VERSION = "0.1.0"


# ---------------------------------------------------------------------
# User-facing validation and preprocessing functions
# ---------------------------------------------------------------------

def validate_ap_recording(
    recording,
    expected_sample_rate_hz: float = 30_000.0,
    sample_rate_tolerance_hz: float = 1.0,
) -> None:
    """Validate that a recording is suitable for AP preprocessing.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Traces are expected to use
        SpikeInterface convention ``(n_samples, n_channels)`` when requested.
        Samples are AP-band raw values in the extractor's native units.
    expected_sample_rate_hz
        Expected AP sampling frequency in Hz. Default is 30,000 Hz.
    sample_rate_tolerance_hz
        Absolute allowed difference from ``expected_sample_rate_hz``, in Hz.

    Returns
    -------
    None
        Raises an exception if validation fails. The recording is not modified.
    """
    num_segments = int(recording.get_num_segments())
    if num_segments != 1:
        raise ValueError(
            f"AP preprocessing expects one segment, found {num_segments}."
        )

    sample_rate_hz = float(recording.get_sampling_frequency())
    if not np.isclose(
        sample_rate_hz,
        expected_sample_rate_hz,
        atol=sample_rate_tolerance_hz,
        rtol=0.0,
    ):
        raise ValueError(
            "Expected approximately 30 kHz AP data, "
            f"found {sample_rate_hz:g} Hz."
        )

    if recording.get_probe() is None:
        raise ValueError("AP preprocessing requires attached probe geometry.")

    property_keys = set(recording.get_property_keys())
    if "inter_sample_shift" not in property_keys:
        raise ValueError(
            "AP preprocessing requires Neuropixels inter-sample-shift metadata."
        )


def validate_local_car_geometry(
    recording,
    outer_radius_um: float,
) -> None:
    """Validate geometry before applying local common average reference.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Channel locations must be
        available as an array with shape ``(n_channels, 2)`` in micrometers.
        Contact-vector ``shank_ids`` or channel groups must provide one shank
        label per channel with shape ``(n_channels,)``.
    outer_radius_um
        Local-CAR outer radius in micrometers. Channels from different shanks
        must not occur within this radius when coordinates are interpreted in a
        common two-dimensional coordinate system.

    Returns
    -------
    None
        Raises an exception if geometry is missing or if cross-shank neighbors
        would fall inside the local-CAR neighborhood. The recording is not
        modified.
    """
    n_channels = int(recording.get_num_channels())
    locations_um = np.asarray(recording.get_channel_locations(), dtype=float)
    shank_labels = _get_shank_labels(recording=recording, n_channels=n_channels)

    if locations_um.shape != (n_channels, 2):
        raise ValueError(
            "Expected channel locations with shape (n_channels, 2) in um, "
            f"found {locations_um.shape}."
        )

    if shank_labels.shape != (n_channels,):
        raise ValueError(
            "Expected one shank label per channel, "
            f"found shape {shank_labels.shape}."
        )

    for channel_index in range(n_channels):
        channels_on_different_shanks = shank_labels != shank_labels[channel_index]
        if not np.any(channels_on_different_shanks):
            continue

        distances_um = np.linalg.norm(
            locations_um[channels_on_different_shanks]
            - locations_um[channel_index],
            axis=1,
        )

        if np.any(distances_um <= outer_radius_um):
            raise RuntimeError(
                "Channels from different shanks occur within the "
                f"{outer_radius_um:g}-um local-CAR radius. Correct the probe "
                "geometry or preprocess shanks separately before local CAR."
            )


def estimate_output_range(
    recording,
    num_random_chunks: int = 20,
    chunk_duration_s: float = 1.0,
    random_seed: int = 0,
) -> tuple[float, float]:
    """Estimate trace range from deterministic random chunks.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Traces are read with shape
        ``(n_samples, n_channels)`` using ``return_scaled=False``. Sample units
        are the current units of the preprocessing chain.
    num_random_chunks
        Number of chunks to sample across all segments.
    chunk_duration_s
        Duration of each sampled chunk in seconds.
    random_seed
        Seed for selecting segment indices and start frames.

    Returns
    -------
    tuple[float, float]
        Estimated minimum and maximum sample values in the current recording
        units, computed from the sampled chunks.
    """
    if num_random_chunks <= 0:
        raise ValueError("num_random_chunks must be positive.")
    if chunk_duration_s <= 0:
        raise ValueError("chunk_duration_s must be positive.")

    sample_rate_hz = float(recording.get_sampling_frequency())
    chunk_size_samples = max(1, int(round(chunk_duration_s * sample_rate_hz)))
    num_segments = int(recording.get_num_segments())
    rng = np.random.default_rng(random_seed)

    chunk_min = np.inf
    chunk_max = -np.inf

    for _ in range(num_random_chunks):
        segment_index = int(rng.integers(0, num_segments))
        num_samples = int(recording.get_num_samples(segment_index=segment_index))
        if num_samples <= 0:
            continue

        max_start = max(0, num_samples - chunk_size_samples)
        start_frame = int(rng.integers(0, max_start + 1))
        end_frame = min(start_frame + chunk_size_samples, num_samples)

        traces = recording.get_traces(
            start_frame=start_frame,
            end_frame=end_frame,
            segment_index=segment_index,
            return_scaled=False,
        )
        traces = np.asarray(traces)
        if traces.size == 0:
            continue

        chunk_min = min(chunk_min, float(np.min(traces)))
        chunk_max = max(chunk_max, float(np.max(traces)))

    if not np.isfinite(chunk_min) or not np.isfinite(chunk_max):
        raise ValueError("Could not sample any AP data for range estimation.")

    return chunk_min, chunk_max


def preprocess_ap_for_kilosort(
    recording,
    output_folder: Path,
    highpass_hz: float = 300.0,
    local_car_inner_um: float = 40.0,
    local_car_outer_um: float = 140.0,
    min_local_neighbors: int = 5,
    working_dtype: str = "float32",
    output_dtype: str = "int16",
    n_jobs: int = 8,
    chunk_duration: str = "1s",
    num_random_chunks: int = 20,
    random_seed: int = 0,
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Run AP preprocessing and write a Kilosort-ready binary plus metadata.

    Parameters
    ----------
    recording
        SpikeInterface-like AP recording extractor. Input traces use
        SpikeInterface convention ``(n_samples, n_channels)``. Samples are raw
        AP values in extractor-native units, and the recording must contain one
        segment, attached probe geometry, and Neuropixels inter-sample-shift
        metadata.
    output_folder
        Stream-specific derived-output directory. The function writes
        ``ap_preprocessed.dat`` and ``ap_preprocessing.json`` directly inside
        this directory.
    highpass_hz
        High-pass cutoff frequency in Hz.
    local_car_inner_um
        Inner exclusion radius for local CAR in micrometers.
    local_car_outer_um
        Outer inclusion radius for local CAR in micrometers.
    min_local_neighbors
        Minimum number of local reference channels required by
        SpikeInterface's local common reference.
    working_dtype
        Floating dtype used inside the preprocessing chain.
    output_dtype
        Binary export dtype. Default is ``"int16"`` for Kilosort.
    n_jobs
        Number of parallel workers used by SpikeInterface during binary write.
    chunk_duration
        SpikeInterface job chunk duration string for binary writing, for
        example ``"1s"``.
    num_random_chunks
        Number of chunks sampled for range QC before integer export.
    random_seed
        Seed controlling range-QC chunk selection.
    progress_bar
        Whether SpikeInterface displays binary-writing progress.

    Returns
    -------
    dict[str, Any]
        Output summary and metadata. Paths are returned as ``pathlib.Path``
        objects under ``ap_binary_path`` and ``ap_metadata_path``. Shape and
        unit metadata include sampling frequency in Hz, one sample count per
        segment, channel count, output dtype, and estimated sampled range in
        current recording units.
    """
    validate_ap_recording(recording)

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    ap_binary_path = output_folder / "ap_preprocessed.dat"
    ap_metadata_path = output_folder / "ap_preprocessing.json"

    ap_shifted = spre.phase_shift(
        recording,
        dtype=working_dtype,
    )

    ap_highpass = spre.highpass_filter(
        ap_shifted,
        freq_min=highpass_hz,
        filter_order=3,
        ftype="butter",
        filter_mode="sos",
        direction="forward-backward",
        margin_ms="auto",
        dtype=working_dtype,
    )

    validate_local_car_geometry(
        recording=ap_highpass,
        outer_radius_um=local_car_outer_um,
    )

    ap_preprocessed = spre.common_reference(
        ap_highpass,
        reference="local",
        operator="average",
        local_radius=(local_car_inner_um, local_car_outer_um),
        min_local_neighbors=min_local_neighbors,
        dtype=working_dtype,
    )

    estimated_min, estimated_max = estimate_output_range(
        recording=ap_preprocessed,
        num_random_chunks=num_random_chunks,
        chunk_duration_s=1.0,
        random_seed=random_seed,
    )
    _validate_integer_output_range(
        minimum=estimated_min,
        maximum=estimated_max,
        output_dtype=output_dtype,
    )

    write_binary_recording(
        recording=ap_preprocessed,
        file_paths=ap_binary_path,
        dtype=output_dtype,
        add_file_extension=False,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        progress_bar=progress_bar,
        verbose=True,
    )

    num_segments = int(ap_preprocessed.get_num_segments())
    metadata = {
        "analysis_version": AP_PREPROCESSING_VERSION,
        "output_binary": ap_binary_path.name,
        "sampling_frequency_hz": float(ap_preprocessed.get_sampling_frequency()),
        "num_channels": int(ap_preprocessed.get_num_channels()),
        "num_segments": num_segments,
        "num_samples_by_segment": [
            int(ap_preprocessed.get_num_samples(segment_index=segment_index))
            for segment_index in range(num_segments)
        ],
        "dtype": str(output_dtype),
        "working_dtype": str(working_dtype),
        "binary_layout": "time_major_channel_interleaved",
        "channel_ids_in_binary_order": _get_channel_ids(ap_preprocessed),
        "random_seed": int(random_seed),
        "preprocessing": [
            {
                "name": "neuropixels_phase_shift",
                "source_property": "inter_sample_shift",
                "dtype": str(working_dtype),
            },
            {
                "name": "highpass_filter",
                "cutoff_hz": float(highpass_hz),
                "filter_type": "butter",
                "filter_order": 3,
                "filter_mode": "sos",
                "direction": "forward-backward",
                "margin_ms": "auto",
                "dtype": str(working_dtype),
            },
            {
                "name": "local_common_average_reference",
                "operator": "average",
                "inner_radius_um": float(local_car_inner_um),
                "outer_radius_um": float(local_car_outer_um),
                "min_local_neighbors": int(min_local_neighbors),
                "dtype": str(working_dtype),
            },
        ],
        "estimated_random_chunk_min": float(estimated_min),
        "estimated_random_chunk_max": float(estimated_max),
    }

    ap_metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        **metadata,
        "ap_binary_path": ap_binary_path,
        "ap_metadata_path": ap_metadata_path,
    }


def main() -> dict[str, Any]:
    """Run the AP preprocessing example with IDE-editable parameters.

    Parameters
    ----------
    None
        Edit the local variables inside this function before running from an
        IDE. The raw and derived roots are filesystem paths.

    Returns
    -------
    dict[str, Any]
        Output summary returned by :func:`preprocess_ap_for_kilosort`. Paths are
        ``pathlib.Path`` objects. Sampling frequency is in Hz; channel locations
        remain in micrometers inside the source recording metadata.
    """
    raw_root = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
        "CT026_20260727_alternating_latent/ephys/raw"
    )
    output_root = Path(
        "/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/"
        "CT026_20260727_alternating_latent/ephys/derived"
    )
    experiment_name = "experiment1"
    stream_name = "Record Node 101#Neuropix-PXI-100.ProbeA"
    load_sync_timestamps = False

    highpass_hz = 300.0
    local_car_inner_um = 40.0
    local_car_outer_um = 140.0
    min_local_neighbors = 5
    working_dtype = "float32"
    output_dtype = "int16"
    n_jobs = 8
    chunk_duration = "1s"
    num_random_chunks = 20
    random_seed = 0
    progress_bar = True

    recording = load_open_ephys_stream(
        raw_root=raw_root,
        experiment_name=experiment_name,
        stream_name=stream_name,
        load_sync_timestamps=load_sync_timestamps,
    )
    stream_output_dir = build_stream_output_dir(
        output_root=output_root,
        stream_name=stream_name,
    )

    result = preprocess_ap_for_kilosort(
        recording=recording,
        output_folder=stream_output_dir,
        highpass_hz=highpass_hz,
        local_car_inner_um=local_car_inner_um,
        local_car_outer_um=local_car_outer_um,
        min_local_neighbors=min_local_neighbors,
        working_dtype=working_dtype,
        output_dtype=output_dtype,
        n_jobs=n_jobs,
        chunk_duration=chunk_duration,
        num_random_chunks=num_random_chunks,
        random_seed=random_seed,
        progress_bar=progress_bar,
    )

    if "ap_binary_path" in result:
        print(f"Wrote AP binary: {result['ap_binary_path']}")
    if "ap_metadata_path" in result:
        print(f"Wrote AP metadata: {result['ap_metadata_path']}")
    return result


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _get_shank_labels(recording, n_channels: int) -> np.ndarray:
    """Return one shank label per channel.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. A structured ``contact_vector``
        property with ``shank_ids`` is preferred. If unavailable, channel groups
        are used.
    n_channels
        Number of channels expected in the returned label array.

    Returns
    -------
    numpy.ndarray
        One-dimensional string array with shape ``(n_channels,)``. Values are
        shank identifiers copied from recording metadata.
    """
    property_keys = set(recording.get_property_keys())

    if "contact_vector" in property_keys:
        contact_vector = recording.get_property("contact_vector")
        if getattr(contact_vector, "dtype", None) is not None:
            dtype_names = contact_vector.dtype.names or ()
            if "shank_ids" in dtype_names:
                return np.asarray(contact_vector["shank_ids"], dtype=str)

    return np.asarray(recording.get_channel_groups(), dtype=str)


def _get_channel_ids(recording) -> list[str]:
    """Return channel IDs in binary column order.

    Parameters
    ----------
    recording
        SpikeInterface-like recording extractor. Channel IDs are expected in
        recording order and have shape ``(n_channels,)``.

    Returns
    -------
    list[str]
        One string per channel, in the same order as binary columns.
    """
    if hasattr(recording, "get_channel_ids"):
        channel_ids = recording.get_channel_ids()
    else:
        channel_ids = recording.channel_ids

    return [str(channel_id) for channel_id in channel_ids]


def _validate_integer_output_range(
    minimum: float,
    maximum: float,
    output_dtype: str,
) -> None:
    """Validate that sampled values fit in an integer output dtype.

    Parameters
    ----------
    minimum
        Estimated minimum sample value in the current recording units.
    maximum
        Estimated maximum sample value in the current recording units.
    output_dtype
        NumPy dtype string for binary export.

    Returns
    -------
    None
        Raises an exception if ``output_dtype`` is an integer dtype and the
        sampled values exceed the representable range.
    """
    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.integer):
        return

    limits = np.iinfo(dtype)
    if minimum < limits.min or maximum > limits.max:
        raise RuntimeError(
            f"The sampled preprocessed data exceed the {dtype.name} range "
            f"({limits.min} to {limits.max}). Do not silently cast or clip."
        )


if __name__ == "__main__":
    main()
