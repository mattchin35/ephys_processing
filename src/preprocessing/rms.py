from __future__ import annotations

from importlib.metadata import metadata

import numpy as np
from typing import Tuple, Dict, Any, Callable
from functools import partial
import spikeinterface.full as si
import subset_helpers as subset
import preprocess_io as io

eps = np.finfo(float).eps


def rms(x: np.ndarray, axis: int = None) -> np.ndarray:
    x = np.asarray(x)
    calc_dtype = np.result_type(x.dtype, np.float32)
    x = x.astype(calc_dtype, copy=False)
    return np.sqrt(np.mean(np.square(x), axis=axis))  # flatten along the specified axis; the numpy fxn already handles axis options


def get_rms_step_samples(sample_rate: int, skip_window: float) -> int:
    step_samples = int(sample_rate * skip_window)
    if step_samples <= 0:
        raise ValueError("skip_window must produce a positive step size")
    return step_samples


def count_rms_windows(nsamples: int, sample_rate: int, skip_window: float) -> int:
    if nsamples <= 0:
        return 0
    step_samples = get_rms_step_samples(sample_rate, skip_window)
    return ((nsamples - 1) // step_samples) + 1


def calculate_session_rms_preallocated(
    get_window: Callable,
    nsamples: int,
    n_channels: int,
    sample_rate: int,
    skip_window: float = 300,
    dtype: np.dtype = np.float32,
    channel_scale_factors: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform RMS inspection calculations using preallocated output arrays.
    """
    n_windows = count_rms_windows(nsamples, sample_rate, skip_window)
    if n_windows == 0:
        return (
            np.empty((n_channels, 0), dtype=dtype),
            np.empty((0,), dtype=dtype),
        )

    step_samples = get_rms_step_samples(sample_rate, skip_window)
    window_rms = np.empty((n_channels, n_windows), dtype=dtype)
    times = np.empty(n_windows, dtype=dtype)

    ix = 0
    window_index = 0
    while ix < nsamples:
        times[window_index] = ix / sample_rate
        window = get_window(ix=ix)
        window_rms_vector = rms(window, axis=1)
        if channel_scale_factors is not None:
            window_rms_vector = window_rms_vector * channel_scale_factors
        window_rms[:, window_index] = window_rms_vector.astype(dtype, copy=False)
        ix += step_samples
        window_index += 1

    return window_rms, times


def calculate_session_rms(get_window: Callable, nsamples: int, sample_rate: int, skip_window: float = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform RMS inspection calculations on a spikeinterface recording extractor.
    :param skip_window: time to skip between windows in seconds
    :param get_window: a function which takes in an ix and returns a window of data.
    :return:
    """
    first_window = get_window(ix=0)
    return calculate_session_rms_preallocated(
        get_window=get_window,
        nsamples=nsamples,
        n_channels=first_window.shape[0],
        sample_rate=sample_rate,
        skip_window=skip_window,
    )

# def rms_helper(get_window: Callable, sample_rate: int, nsamples: int, skip_window: int, tag: str, data_dict=None) -> Dict[str, Any]:
#     rms, window_times = calculate_session_rms(get_window, sample_rate=sample_rate, nsamples=nsamples, skip_window=skip_window)
#     median, q1, q9 = get_quantiles(rms)
#     # convert rms to dB
#     # https://rexburghams.org/assets/decibeltutorial.pdf
#     db = 20 * np.log10(rms / (median+eps))  # need a case to handle nans
#     if data_dict is None:
#         data_dict = {}
#
#     data_dict['{}_rms'.format(tag)] = rms
#     data_dict['{}_db'.format(tag)] = db
#     data_dict['{}_median'.format(tag)] = median
#     data_dict['{}_q1'.format(tag)] = q1
#     data_dict['{}_q9'.format(tag)] = q9
#     data_dict['{}_rms_times'.format(tag)] = window_times
#     return data_dict

def collect_rms_stats(rms: np.ndarray, window_times: np.ndarray, tag: str, data_dict=None):
    if data_dict is None:
        data_dict = {}

    median, q1, q9 = get_quantiles(rms, axis=1)
    # convert rms to dB
    # https://rexburghams.org/assets/decibeltutorial.pdf
    # Clamp zero-valued RMS inputs before log10; very large negative dB values can indicate a dead,
    # disconnected, or all-zero channel/window.
    safe_rms = np.maximum(rms, eps)
    safe_median = np.maximum(median[:, np.newaxis], eps)
    db = 20 * np.log10(safe_rms / safe_median)  # normalize each channel by its own median

    data_dict['{}_rms'.format(tag)] = rms
    data_dict['{}_db'.format(tag)] = db
    data_dict['{}_median'.format(tag)] = median
    data_dict['{}_q1'.format(tag)] = q1
    data_dict['{}_q9'.format(tag)] = q9
    data_dict['{}_rms_times'.format(tag)] = window_times
    return data_dict


def np_windowed_rms(recording: np.ndarray, sample_rate: int, tag: str, window_size: float, skip_window: float,
                        data_dict: Dict[str, Any] = None, metadata=None, chanlist=[]) -> dict:
    # get_window = partial(subset.get_np_window, data=recording, sample_rate=sample_rate, t_seconds=rms_window)
    # rms, window_times = calculate_session_rms(get_window, sample_rate=sample_rate, nsamples=sample_rate, skip_window=skip_window)
    # data_dict = rms_helper(get_window, sample_rate, nsamples=recording.shape[1], skip_window=skip_window, tag=tag,
    #                        data_dict=data_dict)
    get_window = partial(
        subset.get_np_window,
        data=recording,
        sample_rate=sample_rate,
        window_size=window_size,
        correct_gain=False,
        metadata=metadata,
        chanlist=chanlist,
    )
    channel_scale_factors = None
    if metadata is not None and "typeThis" in metadata:
        if len(chanlist) == 0:
            chanlist = list(range(recording.shape[0]))
        channel_scale_factors = io.get_binary_gain_factors(metadata, chanlist)
    rms, window_times = calculate_session_rms_preallocated(
        get_window=get_window,
        nsamples=recording.shape[1],
        n_channels=recording.shape[0],
        sample_rate=sample_rate,
        skip_window=skip_window,
        channel_scale_factors=channel_scale_factors,
    )
    data_dict = collect_rms_stats(rms, window_times, tag, data_dict)
    return data_dict


def spikeinterface_windowed_rms(recording: si.SpikeGLXRecordingExtractor, sample_rate: int, tag: str, rms_window: int, skip_window: int,
                         data_dict: Dict[str, Any] = None) -> dict:

    get_window = partial(subset.get_spikeinterface_window, recording=recording, sample_rate=sample_rate, window=rms_window)
    rms, window_times = calculate_session_rms_preallocated(
        get_window=get_window,
        nsamples=recording.get_num_samples(),
        n_channels=recording.get_num_channels(),
        sample_rate=sample_rate,
        skip_window=skip_window,
    )
    # data_dict = collect_rms_stats(window_times, tag, data_dict)
    # data_dict = rms_helper(get_window, sample_rate, nsamples=recording.get_num_samples(), skip_window=skip_window,
    #                        tag=tag, data_dict=data_dict)
    data_dict = collect_rms_stats(rms, window_times, tag, data_dict)
    return data_dict


def get_quantiles(rms_data: np.ndarray, axis: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the median, 10th percentile, and 90th percentile of the RMS data for each channel.
    """
    median = np.median(rms_data, axis=axis)
    q1 = np.quantile(rms_data, 0.1, axis=axis)
    q9 = np.quantile(rms_data, 0.9, axis=axis)
    return median, q1, q9
