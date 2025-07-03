import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression
from icecream import ic
from pathlib import Path
import time
import get_events as ge
import preprocess_io as pio
import matplotlib.pyplot as plt
from src.visualization import viz_preprocessing as viz


def interpolate_artifacts(data: np.ndarray, ix_artifacts: np.ndarray, window_before:int, window_after:int, method: str='cubic') -> np.ndarray:
    assert method in ['zeros', 'linear', 'cubic']
    for n, ix in enumerate(ix_artifacts):
        if n % 100 == 0:
            print('interpolating artifact {} out of {}'.format(n, ix_artifacts.shape[0]))

        tstart = max(0, int(ix - window_before))
        tend = min(data.shape[1], int(ix + window_after))
        if method == 'zeros':
            data[:, tstart:tend] = 0
        elif method == 'linear':
            for channel in range(data.shape[0]):
                data[channel, tstart:tend] = np.interp(x=np.arange(tstart, tend), xp=[tstart, tend],
                                                       fp=[data[channel, tstart], data[channel, tend]])
        elif method == 'cubic':
            for channel in range(data.shape[0]):
                fn = sp.interpolate.PchipInterpolator(x=[tstart, tend],
                                                      y=[data[channel, tstart], data[channel, tend]])
                data[channel, tstart:tend] = fn(np.arange(tstart, tend))

    return data


def stats_regression_reference(data: np.ndarray) -> np.ndarray:
    means = np.mean(data, axis=0)
    medians = np.median(data, axis=0)
    stds = np.std(data, axis=0)
    maxs = np.max(np.abs(data), axis=0)
    features = np.array([means, medians, stds, maxs]).T
    # features = np.array([medians, stds, maxs]).T

    data_referenced = np.zeros_like(data)
    for i, d in enumerate(data):
        model = LinearRegression().fit(features, d)
        data_referenced[i] = d - model.predict(features)
    return data_referenced


def main():
    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    session_name = 'CT009_current_20250302'
    opts = 'filter-gfix-tartifacts'
    # recording_path = processed_data_root.joinpath('{}/run1_g0'.format(session_name))  # for raw data
    # imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')  # for raw data
    recording_path = processed_data_root.joinpath('{}_{}/catgt_run1_g0'.format(session_name, opts))  # for filtered data
    imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_tcat.imec0.ap.bin')  # for filtered data
    ni_file = raw_data_root.joinpath('{}/run1_g0/run1_g0_t0.nidq.bin'.format(session_name))
    if opts:
        tag = session_name + '_' + opts
    else:
        tag = session_name

    # read in data, assuming it's neuropixel data
    ap_data, ap_meta, ap_srate, ap_shape = pio.read_binary(imec_file_ap, allow_write=True)
    ap_data = ap_data[:384, :]  # only use first 384 channels
    all_events, offsets, onsets = ge.sync_for_demonstration(imec_file_ap, ni_file, debounce=0.0002)

    # inter_event_intervals = np.sort((all_events[1:] - all_events[:-1]) / ap_srate)[:10] * 1e3
    # n_events = all_events.shape[0]
    # plt.hist(np.sort((all_events[1:] - all_events[:-1]) / ap_srate)[:int(n_events/4)] * 1e3, bins=40)
    # plt.title('Inter-event intervals, bottom 1/4 (ms)')
    # plt.show()
    tst = time.perf_counter()

    date = '2025-03-17'
    # windowsize would be window time (say 2 ms) * sampling rate

    t1 = 5 * 60
    t2 = t1 + 5
    t1_ix = int(t1 * ap_srate)
    t2_ix = int(t2 * ap_srate)
    onsets = onsets[(onsets > t1_ix) & (onsets < t2_ix)]

    window_ms = 5
    int(window_ms * 1e-3 * ap_srate)
    # interpolated_data = interpolate_artifacts(ap_data, ix_artifacts=onsets, windowsize=int(window_ms * 1e-3 * ap_srate), method='zeros')
    interpolate_artifacts(ap_data, ix_artifacts=all_events, window_before=int(window_ms * 1e-3 * ap_srate),
                          window_after=int(window_ms * 1e-3 * ap_srate), method='cubic')
    ic('interpolation time', time.perf_counter() - tst)

    onsets -= t1_ix
    # viz.plot_sample_data(ap_data, t1_ix, t2_ix, ap_srate, tag, processing_step='preprocessed', event_times=onsets)
    ap_data.flush()  # write changes to disk for numpy memory mapped file
    
    
if __name__ == '__main__':
    main()
