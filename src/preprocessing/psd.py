import numpy as np
import scipy as sp
from typing import Tuple, Dict, Any, Iterable, Callable
from functools import partial
import spikeinterface.full as si
import subset_helpers as sh


def get_channelwise_PSD(get_channel: Callable, channel_ids: Iterable[Any], sample_rate: int, nperseg: int = 1024) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate PSD for a single channel of data.
    """
    psd = []
    nchannels = len(channel_ids)
    for i, ch in enumerate(channel_ids):
        print('Calculating PSD for channel {} of {}'.format(i, nchannels))
        data = get_channel(channel_id=ch)
        freqs, tmp_psd = sp.signal.welch(data, sample_rate, nperseg=nperseg)
        psd.append(tmp_psd)

    return freqs, np.stack(psd, axis=0)


def np_channelwise_PSD(recording: np.ndarray, sample_rate: int, tag: str, nperseg: int = 1024,
                       data_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate PSD one channel at a time to prevent memory overruns.
    """
    channel_ids = np.arange(recording.shape[0])
    get_channel = partial(sh.get_np_channel, recording=recording)
    freqs, psd = get_channelwise_PSD(get_channel=get_channel, channel_ids=channel_ids, sample_rate=sample_rate, nperseg=nperseg)

    if data_dict is None:
        data_dict = {}

    data_dict['{}_freq'.format(tag)] = freqs
    data_dict['{}_psd'.format(tag)] = psd
    return data_dict


def spikeinterface_channelwise_PSD(recording: si.SpikeGLXRecordingExtractor, sample_rate: int, tag: str, nperseg: int = 1024,
                       data_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate PSD one channel at a time to prevent memory overruns.
    """
    channel_ids = recording.get_channel_ids()
    get_channel = partial(sh.get_spikeinterface_channel, recording=recording)
    freqs, psd = get_channelwise_PSD(get_channel=get_channel, channel_ids=channel_ids,
                                                sample_rate=sample_rate, nperseg=nperseg)

    if data_dict is None:
        data_dict = {}

    data_dict['{}_freq'.format(tag)] = freqs
    data_dict['{}_psd'.format(tag)] = psd
    return data_dict


def process_fullsize_PSD(data: np.ndarray, sample_rate: int, tag: str, nperseg=1024, data_dict: Dict[str, Any]=None) -> Dict[str, Any]:
    freq, psd = sp.signal.welch(data, sample_rate, nperseg=nperseg, axis=1)  # choose npersg to be a power of 2; 1024 or 2048 are good choices just because
    if data_dict is None:
        data_dict = {}
    data_dict['{}_freq'.format(tag)] = freq
    data_dict['{}_psd'.format(tag)] = psd
    # plot_PSD(freq, psd, title='{} PSD'.format(tag))
    return data_dict