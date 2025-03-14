"""
A SpikeInterface based preprocessing pipeline for Neuropixel 1.0 spikeGLX recordings.
Will need to modify for Neuropixel 2.0 recordings.
Data which was not collected via spikeGLX

# https://spikeinterface.readthedocs.io/en/latest/how_to/get_started.html
# https://spikeinterface.readthedocs.io/en/latest/how_to/analyse_neuropixels.html
# https://spikeinterface.readthedocs.io/en/latest/modules/preprocessing.html
# also see the spikeGLX CatGT documentation
# https://billkarsh.github.io/SpikeGLX/help/catgt_tshift/catgt_tshift/
# future: look into EMG denoising? AC current filtering?
"""

from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import spikeinterface.full as si
from pathlib import Path
import pickle as pkl
from time import perf_counter
# import src.DemoReadSGLXData.readSGLX as sglx # will use readMeta, SampRate, makeMemMapRaw, ExtractDigital, GainCorrectIM, GainCorrectNI


# hard-coded parameters for neuropixel recordings
lfp_sample_rate = 1250
ap_sample_rate = 30000


def load_sglx_data(spikeglx_folder: Path) -> List[si.SpikeGLXRecordingExtractor]:
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    ic(stream_names)
    recordings = [si.read_spikeglx(spikeglx_folder, stream_name=name, load_sync_channel=False) for name in stream_names]
    return recordings


def load_cleaned_data(path: Path) -> Tuple[si.ZarrRecordingExtractor]:
    rec = si.load_extractor(path)
    return rec


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
    save_dir = Path('../../reports/figures/preprocessing/destriping/')
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (name + '.png')
    figure.savefig(p, dpi=dpi)


def get


def destripe_IBL(rec: si.SpikeGLXRecordingExtractor, catgt_preprocessed=False) -> Tuple[si.SpikeGLXRecordingExtractor, Tuple[np.ndarray, np.ndarray]]:
    ic('[**] IBL destriping [**] ')

    if not catgt_preprocessed:
        ic('highpass filtering')
        rec = si.highpass_filter(recording=rec, freq_min=300)  # 300 is the default

        ic('phase shifting')
        rec = si.phase_shift(recording=rec)

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

    if not catgt_preprocessed:
        ic('common referencing')
        rec = si.common_reference(rec, operator="median", reference="global")

    ic('highpass spatial filtering')
    rec = si.highpass_spatial_filter(recording=rec)

    return rec, (bad_channel_ids, channel_labels)


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


def run_preprocessing_ap(recording: si.SpikeGLXRecordingExtractor, save_path: Path, catgt_preprocessed: bool) -> None:
    cleanap_folder = save_path / 'clean_ap'

    # destripe the AP data
    # ap_processed, (bad_channels, channel_quality) = destripe_CatGT(recording)
    ap_processed, (bad_channels, channel_quality) = destripe_IBL(recording, catgt_preprocessed)
    # ap_processed, (bad_channels, channel_quality) = destripe_hybrid(recording, catgt_preprocessed)
    channel_quality_dict = dict(bad_channels=bad_channels, channel_quality=channel_quality)
    bad_channel_ix = channel_quality != 'good'

    # save the cleaned AP data
    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True,
                      format='binary')  # format='zarr' (compressed) or 'binary'

    ap_processed.save(folder=cleanap_folder, **job_kwargs)
    channel_quality_fname = save_path / 'channel_quality_ids.pkl'
    with open(channel_quality_fname, 'wb') as f:
        pkl.dump(channel_quality_dict, f)


def get_lfp_from_wideband(wideband: si.SpikeGLXRecordingExtractor) -> si.SpikeGLXRecordingExtractor:
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

    recording_path = raw_data_root.joinpath('CT009_current_20250302/run1_g0/run1_g0_imec0')
    # recording_path = processed_data_root.joinpath('CT009_current_20250302_filter-gfix/catgt_run1_g0/run1_g0_imec0')

    save_path = processed_data_root.joinpath('CT009_current_20250302_hipassfilter-spatialfilter')
    recordings = load_sglx_data(recording_path)
    ic(recordings)

    # plot_probe_layout(recordings[0])
    # run_preprocessing_NP1(ap_recording=recordings[0], lfp_recording=recordings[1], save_path=save_path)
    run_preprocessing_ap(recording=recordings[0], save_path=save_path, catgt_preprocessed=False)
    # compare_processed(recordings=[recordings[0], ap_ibl, ap_catgt, ap_custom], labels=['base', 'ibl', 'catgt', 'custom'])


if __name__ == '__main__':
    ic('running')
    main_unprocessed()
