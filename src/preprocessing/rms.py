from importlib.metadata import metadata

import numpy as np
from typing import Tuple, Dict, Any, Callable
from functools import partial
import spikeinterface.full as si
import subset_helpers as subset
import preprocess_io as io

eps = np.finfo(float).eps


def rms(x: np.ndarray, axis: int = None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x), axis=axis))  # flatten along the specified axis; the numpy fxn already handles axis options


def calculate_session_rms(get_window: Callable, nsamples: int, sample_rate: int, skip_window: float = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform RMS inspection calculations on a spikeinterface recording extractor.
    :param skip_window: time to skip between windows in seconds
    :param get_window: a function which takes in an ix and returns a window of data.
    :return:
    """
    # assert data_type in ['spikeinterface', 'numpy'], "data_type must be either 'spikeinterface' or 'numpy'."
    # if data_type == 'spikeinterface':
    #     get_window = partial(subset.get_spikeinterface_window, recording=data, sample_rate=sample_rate, window=window_size)
    # elif data_type == 'numpy':
    #     get_window = partial(subset.get_np_window, data=data, sample_rate=sample_rate, window_size=window_size)

    ix = 0
    window_rms = []
    times = []
    while ix < nsamples:
        times.append(ix / sample_rate)
        window = get_window(ix=ix)
        tmp_rms = rms(window, axis=1)
        window_rms.append(tmp_rms)
        ix += int(sample_rate * skip_window)

    return np.stack(window_rms, axis=1), np.array(times)

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

    median, q1, q9 = get_quantiles(rms)
    # convert rms to dB
    # https://rexburghams.org/assets/decibeltutorial.pdf
    db = 20 * np.log10(rms / (median + eps))  # need a case to handle nans

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
    get_window = partial(subset.get_np_window, data=recording, sample_rate=sample_rate, window_size=window_size,
                         correct_gain=True, metadata=metadata, chanlist=chanlist)
    rms, window_times = calculate_session_rms(get_window, nsamples=recording.shape[1], sample_rate=sample_rate,
                                              skip_window=skip_window)
    data_dict = collect_rms_stats(rms, window_times, tag, data_dict)
    return data_dict


def spikeinterface_windowed_rms(recording: si.SpikeGLXRecordingExtractor, sample_rate: int, tag: str, rms_window: int, skip_window: int,
                         data_dict: Dict[str, Any] = None) -> dict:

    get_window = partial(subset.get_spikeinterface_window, recording=recording, sample_rate=sample_rate, window=rms_window)
    rms, window_times = calculate_session_rms(get_window, sample_rate=sample_rate, nsamples=sample_rate,
                                              skip_window=skip_window)
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
