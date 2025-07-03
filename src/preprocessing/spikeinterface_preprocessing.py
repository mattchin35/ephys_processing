import time
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import spikeinterface.full as si
from pathlib import Path
import pickle as pkl
import preprocess_io as pio
from time import perf_counter
# import src.DemoReadSGLXData.readSGLX as sglx # will use readMeta, SampRate, makeMemMapRaw, ExtractDigital, GainCorrectIM, GainCorrectNI
import get_events as ge
import sync_lines as sync
from sklearn.linear_model import LinearRegression


# hard-coded parameters for neuropixel recordings
lfp_sample_rate = 2500
ap_sample_rate = 30000


def load_sglx_data(spikeglx_folder: Path) -> List[si.SpikeGLXRecordingExtractor]:
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    ic(stream_names)
    recordings = [si.read_spikeglx(spikeglx_folder, stream_name=name, load_sync_channel=False) for name in stream_names]
    return recordings


def load_channel_quality_ids(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        channel_quality_ids = pkl.load(f)
    return channel_quality_ids


def save_cleaned_data(recording: si.SpikeGLXRecordingExtractor, channel_quality_ids: Tuple[np.ndarray, np.ndarray],
                      save_path: Path, **kwargs) -> None:

    recording.save(folder=save_path, **kwargs)
    with open(save_path, 'wb') as f:
        pkl.dump(channel_quality_ids, f)


def savefig(figure: plt.Figure, name: str, dpi=300) -> None:
    save_dir = Path('../../reports/figures/preprocessed/destriping/')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (name + '.png')
    figure.savefig(p, dpi=dpi)


def destripe_IBL(rec: si.SpikeGLXRecordingExtractor, catgt_preprocessed=False) -> Tuple[si.SpikeGLXRecordingExtractor, Tuple[np.ndarray, np.ndarray]]:
    ic('[**] IBL destriping [**] ')
    if not catgt_preprocessed:
        ic('highpass filtering')
        rec = si.highpass_filter(recording=rec, freq_min=300)  # 300 is the default

        ic('phase shifting')
        rec = si.phase_shift(recording=rec)
    else:
        ic('skipping highpass filtering and phase shifting')

    ic('detecting and interpolating bad channels')
    bad_channel_ids, channel_labels = si.detect_bad_channels(recording=rec)
    rec = si.interpolate_bad_channels(recording=rec, bad_channel_ids=bad_channel_ids)

    ic('highpass spatial filtering')
    rec = si.highpass_spatial_filter(recording=rec, n_channel_pad=60)  # 60 is the default
    return rec, (bad_channel_ids, channel_labels)


def destripe_CatGT(rec: si.SpikeGLXRecordingExtractor) -> Tuple[si.SpikeGLXRecordingExtractor, Tuple[np.ndarray, np.ndarray]]:
    ic('[**] CatGT destriping [**] ')

    ic('highpass filtering')
    rec = si.highpass_filter(rec, freq_min=300)

    ic('phase shifting')
    rec = si.phase_shift(recording=rec)

    ic('detecting bad channels (bad channels will not be removed or modified)')
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec)

    ic('common referencing')
    rec = si.common_reference(recording=rec, operator="median", reference="global")
    return rec, (bad_channel_ids, channel_labels)


def destripe_hybrid(rec: si.SpikeGLXRecordingExtractor, catgt_preprocessed=False) -> Tuple[si.SpikeGLXRecordingExtractor, Tuple[np.ndarray, np.ndarray]]:
    ic('[**] Hybrid destriping [**] ')
    if not catgt_preprocessed:
        ic('highpass filtering')
        rec = si.highpass_filter(rec, freq_min=300)

        ic('phase shifting')
        rec = si.phase_shift(rec)

    ic('detecting and removing bad channels')
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec)
    rec = rec.remove_channels(bad_channel_ids)

    ic('highpass spatial filtering')
    rec = si.highpass_spatial_filter(recording=rec)

    if not catgt_preprocessed:
        ic('common referencing')
        rec = si.common_reference(rec, operator="median", reference="global")

    return rec, (bad_channel_ids, channel_labels)


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


def destripe_viz(rec: si.SpikeGLXRecordingExtractor):
    """C"""

    ic('highpass filtering')
    rec1 = si.highpass_filter(rec, freq_min=300)

    ic('phase shifting')
    rec2 = si.phase_shift(rec1)

    ic('detecting and removing bad channels')
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec2)
    rec3 = rec1.remove_channels(bad_channel_ids)

    ic('common referencing')
    rec4 = si.common_reference(rec3, operator="median", reference="global")

    # ic('highpass spatial filtering')
    rec5 = si.highpass_spatial_filter(recording=rec3)

    compare_processed(recordings=[rec, rec1, rec2, rec3, rec4, rec5],
                      labels=['base', 'hipass', 'phase shift', 'removed channels', 'common ref', 'hipass spatial'])


def plot_probe_layout(rec: si.SpikeGLXRecordingExtractor):
    fig, ax = plt.subplots(figsize=(15, 10))
    si.plot_probe_map(rec, ax=ax, with_channel_ids=True)
    ax.set_ylim(-100, 100)
    plt.show()


def compare_processed(recordings: List[si.SpikeGLXRecordingExtractor], labels: List[str]):
    nplots = len(recordings)
    f, ax = plt.subplots(ncols=nplots, figsize=(25, 10))

    for i, rec in enumerate(recordings):
        si.plot_traces(rec, backend='matplotlib', clim=(-50, 50), ax=ax[i])
        ax[i].set_title(labels[i])
    savefig(f, 'compare_processed_traces')

    # plot some channels
    some_chans = recordings[0].channel_ids[[0, 20, 40]]
    f1, ax1 = plt.subplots(figsize=(20, 10))
    si.plot_0traces({'{}'.format(labels[i]): recordings[i] for i in range(nplots)}, backend='matplotlib', mode='line',
                   ax=ax1, channel_ids=some_chans)
    savefig(f1, 'compare_processed_channels')


def run_preprocessing_ap_lfp(ap_recording: si.SpikeGLXRecordingExtractor, lfp_recording: si.SpikeGLXRecordingExtractor,
                      save_path: Path) -> None:
    cleanap_folder = save_path / 'clean_ap'
    cleanlfp_folder = save_path / 'clean_lfp'

    # destripe the AP data
    ap_catgt, _ = destripe_CatGT(recordings[0])
    # ap_ibl, ibl_badchannels = destripe_IBL(recordings[0])
    ap_processed, (bad_channels, channel_quality) = destripe_hybrid(ap_recording)
    channel_quality_dict = dict(bad_channels=bad_channels, channel_quality=channel_quality)
    bad_channel_ix = channel_quality != 'good'

    # save the cleaned AP data
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True,
                      format='zarr')  # format='zarr' (compressed) or 'binary'

    # phase-shift the lfp data
    lfp_rec = si.phase_shift(lfp_recording)
    lfp_bad_channels = lfp_rec.get_channel_ids()[bad_channel_ix]
    lfp_rec = lfp_rec.remove_channels(lfp_bad_channels)

    ap_processed.save(folder=cleanap_folder, **job_kwargs)
    lfp_rec.save(folder=cleanlfp_folder, **job_kwargs)
    channel_quality_fname = save_path / 'channel_quality_ids.pkl'
    with open(channel_quality_fname, 'wb') as f:
        pkl.dump(channel_quality_dict, f)

def run_preprocessing_ap(recording: si.SpikeGLXRecordingExtractor, save_path: Path, catgt_preprocessed: bool, method=None,
                         ix_artifacts=None, plot=False) -> None:
    cleanap_folder = save_path / 'clean_ap'

    if method is None:
        ic('no referencing method applied')
        bad_channel_ids, channel_quality = si.detect_bad_channels(recording)
    elif method == 'catgt':
        ap_processed, (bad_channels, channel_quality) = destripe_CatGT(recording)
    elif method == 'destripe':
        ap_processed, (bad_channels, channel_quality) = destripe_IBL(recording, catgt_preprocessed)
    elif method == 'hybrid':
        ap_processed, (bad_channels, channel_quality) = destripe_hybrid(recording, catgt_preprocessed)
    else:
        raise ValueError('Invalid method')

    if ix_artifacts is not None:
        ap_processed = remove_timed_artifacts(recording, ix_artifacts, ms_before=2, ms_after=2)

    if plot:
        channel_ids = ["imec0.ap#AP{}".format(i) for i in range(5)]
        w = si.plot_traces(ap_processed, channel_ids=channel_ids, time_range=(360, 365))
        plt.show()

    # save the cleaned AP data
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True,
                      format='zarr')  # format='zarr' (compressed) or 'binary'

    ap_processed.save(folder=cleanap_folder, **job_kwargs)
    # recording.save(folder=cleanap_folder, **job_kwargs)
    # si.write_binary_recording(recording=ap_processed, save_path=cleanap_folder, dtype='int16')

    if method is not None:
        channel_quality_dict = dict(bad_channels=bad_channels, channel_quality=channel_quality)
        bad_channel_ix = channel_quality != 'good'
        channel_quality_fname = save_path / 'channel_quality_ids.pkl'
        with open(channel_quality_fname, 'wb') as f:
            pkl.dump(channel_quality_dict, f)


def remove_timed_artifacts(recording: si.SpikeGLXRecordingExtractor, ix_artifacts: np.ndarray, ms_before: float = 1, ms_after:float = 1) \
        -> si.SpikeGLXRecordingExtractor:
    ic('removing artifacts')
    removed_recording = si.remove_artifacts(recording, ix_artifacts, ms_before=ms_before, ms_after=ms_after, mode='linear')
    return removed_recording


def get_lfp_from_wideband(wideband: si.SpikeGLXRecordingExtractor, lfp_sample_rate=2500) -> si.SpikeGLXRecordingExtractor:
    """
    For use with Neuropixel 2.0 recordings. 1.0 probes provide separate LFP and AP bands - do not downsample AP band
    data.
    """
    lfp = si.resample(wideband, lfp_sample_rate)
    return lfp


def main_unprocessed():
    # set data paths
    # data_root = Path.home().joinpath('Documents', 'testdata')
    # ibl_folder = data_root.joinpath('IBL_testdata')
    # spikeglx_folder = data_root.joinpath('HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0/')

    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    session_name = 'CT009_current_20250302'
    # recording_path = raw_data_root.joinpath('{}/run1_g0'.format(session_name))  # for raw data
    # imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')  # for raw data
    recording_path = processed_data_root.joinpath('{}_filter-gfix/catgt_run1_g0'.format(session_name))  # for catgt filtered data
    imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_tcat.imec0.ap.bin')  # for catgt filtered data
    ni_file = raw_data_root.joinpath('{}/run1_g0/run1_g0_t0.nidq.bin'.format(session_name))
    tag = session_name

    save_path = processed_data_root.joinpath('CT009_current_20250302_Tartifacts')
    recordings = load_sglx_data(recording_path)  # spikeinterface loading, not binary
    ic(recordings)

    # get lick events for artifact removal
    all_events, offsets, onsets = ge.sync_for_demonstration(imec_file_ap, ni_file, debounce=0.0002)
    date = '2025-03-17'
    sess_stats_dict = pio.load_inspection_data(date, 'session_stats')
    std_threshold = 50
    ix_std = np.nonzero(sess_stats_dict['std'] > std_threshold)[0]
    ix_artifacts = np.concatenate([ix_std, onsets])

    # plot_probe_layout(recordings[0])
    # run_preprocessing_NP1(ap_recording=recordings[0], lfp_recording=recordings[1], save_path=save_path)
    # run_preprocessing_ap(recording=recordings[0], save_path=save_path, catgt_preprocessed=True, method='destripe', plot=True)
    run_preprocessing_ap(recording=recordings[0], save_path=save_path, catgt_preprocessed=True, method=None, ix_artifacts=ix_artifacts)
    # compare_processed(recordings=[recordings[0], ap_ibl, ap_catgt, ap_custom], labels=['base', 'ibl', 'catgt', 'custom'])



if __name__ == '__main__':
    ic('running')
    main_unprocessed()
