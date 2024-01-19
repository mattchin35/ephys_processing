import numpy as np
from typing import Tuple, Dict, Any, Callable, Iterable
from functools import partial
import spikeinterface.full as si
import rms
import subset_helpers as sh


def threshold_detection(data: np.ndarray, t_session: float, threshold: float = 0.5, axis: int = None) -> np.ndarray:
    """
    Detect threshold crossings in the data.
    :param data: data array
    :param threshold: threshold value
    :param axis: axis to calculate threshold crossings along
    :return: threshold crossing array
    """
    if axis is None:
        axis = 0
    threshold_crossings = np.diff(data > threshold, axis=axis, prepend=False)
    nspikes = np.sum(threshold_crossings, axis=axis) / 2  # divide by 2 because we're counting both the positive and negative threshold crossings
    avg_spikes = nspikes / t_session
    return avg_spikes


def get_channelwise_threshold_crossings(get_channel: Callable, channel_ids: Iterable[Any], threshold: float, t_session: float) \
        -> np.ndarray:
    """
    Calculate PSD for a single channel of data.
    """
    avg_crossings = []
    nchannels = len(channel_ids)
    for i, ch in enumerate(channel_ids):
        data = get_channel(channel_id=ch)
        tmp_crossings = threshold_detection(data, t_session=t_session, threshold=threshold)
        print('Calculated {} spikes/sec for channel {} of {}'.format(tmp_crossings, i, nchannels))
        avg_crossings.append(tmp_crossings)

    return np.array(avg_crossings)


def spikeinterface_channelwise_threshold_detection(recording: si.SpikeGLXRecordingExtractor, threshold: float, tag: str, t_session: float,
                       data_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate PSD one channel at a time to prevent memory overruns.
    """
    channel_ids = recording.get_channel_ids()
    get_channel = partial(sh.get_spikeinterface_channel, recording=recording)
    avg_spikes = get_channelwise_threshold_crossings(get_channel=get_channel, channel_ids=channel_ids,
                                                     t_session=t_session,
                                                     threshold=threshold)

    if data_dict is None:
        data_dict = {}
    data_dict['{}_avg_spikes'.format(tag)] = avg_spikes
    return data_dict


def np_channelwise_threshold_detection(recording: np.ndarray, threshold: float, tag: str, t_session: float,
                       data_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate PSD one channel at a time to prevent memory overruns.
    """
    channel_ids = np.arange(recording.shape[0])
    get_channel = partial(sh.get_np_channel, recording=recording)
    avg_spikes = get_channelwise_threshold_crossings(get_channel=get_channel, channel_ids=channel_ids, t_session=t_session,
                                     threshold=threshold)

    if data_dict is None:
        data_dict = {}
    data_dict['{}_avg_spikes'.format(tag)] = avg_spikes
    return data_dict


def fullsize_threshold_detection(recording: np.ndarray, threshold: float, tag: str, t_session: float,
                       data_dict: Dict[str, Any] = None) -> Dict[str, Any]:

    avg_spikes = threshold_detection(recording, t_session=t_session, threshold=threshold, axis=1)
    if data_dict is None:
        data_dict = {}
    data_dict['{}_avg_spikes'.format(tag)] = avg_spikes
    return data_dict
