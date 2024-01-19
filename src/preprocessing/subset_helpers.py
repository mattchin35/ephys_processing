"""
Retrieve a subset of the data.
"""
import numpy as np
import spikeinterface.full as si


class SpikeInterfaceWindowIterator:
    """I could convert the functool partials to classes like this; the functools are more concise though."""
    def __init__(self, recording: si.SpikeGLXRecordingExtractor, sample_rate: int, window: int = 1):
        self.recording = recording
        self.nsamples = recording.get_num_samples()
        self.sample_rate = sample_rate
        self.window = window
        self.ix = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ix < self.nsamples:
            end_frame = min(self.ix + self.sample_rate * self.window, self.nsamples)
            traces = self.recording.get_traces(start_frame=self.ix, end_frame=end_frame)
            self.ix += self.sample_rate * self.window
            return traces.T
        else:
            raise StopIteration


def get_np_channel(recording: np.ndarray, channel_id: int) -> np.ndarray:
    """
    Get a single channel of data from a numpy array. Array should be n_channels x n_samples.
    """
    data = recording[channel_id]
    return data


def get_spikeinterface_channel(recording: si.SpikeGLXRecordingExtractor, channel_id: str) -> np.ndarray:
    data = recording.get_traces(channel_ids=[channel_id])
    return data


def get_np_window(data: np.ndarray, ix: int, sample_rate: int, rms_window: int = 1) -> np.ndarray:
    """
    Get a window of data from a np.ndarray. Windows must have shape n_channels x n_samples.
    """
    window = data[:, ix:min(ix + sample_rate * rms_window, data.shape[1])]
    return window


def get_spikeinterface_window(recording: si.SpikeGLXRecordingExtractor, ix: int, sample_rate: int, window: int = 1)\
        -> np.ndarray:
    """
    Get a window of data from a spikeinterface recording extractor. Windows must have shape n_channels x n_samples.
    """
    window = recording.get_traces(start_frame=ix, end_frame=min(ix + sample_rate * window, recording.get_num_samples()))
    return window.T

