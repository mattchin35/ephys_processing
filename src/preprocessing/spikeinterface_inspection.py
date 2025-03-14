"""
Goal for this file is to take preprocessed data to get an idea of what its activity looks like.
Should be able to overlay primitive spikes ("peaks"/threshold crossings) from spikeinterface on top of the RMS data.
Should also be able to look at primitive spikes/second on each channel.
"""
from pathlib import Path
import preprocess_io as pio
from icecream import ic
from time import perf_counter
import rms
import psd
import threshold_detection as td
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import numpy as np
import matplotlib.pyplot as plt
from src.visualization import viz_preprocessing as viz
from src.visualization import save as save
from dataclasses import dataclass


@dataclass()
class InspectionParams:
    window: int = 1
    skip: int = 300
    nperseg: int = 1024
    ap_srate: int = 30000
    lfp_srate: int = 2500
    run_reduced_rms: bool = True
    run_full_rms: bool = True
    run_PSD: bool = False

    threshold: int = 10
    run_threshold_detection: bool = False


def si_checks(path: Path):
    ap_folder = path.joinpath('clean_ap.zarr')
    rec = pio.load_preprocessed_data(ap_folder)

    # we can estimate the noise on the scaled traces (microV) or on the raw one (which is in our case int16).
    noise_levels_microV = si.get_noise_levels(rec, return_scaled=True)
    noise_levels_int16 = si.get_noise_levels(rec, return_scaled=False)

    fig_noise, ax_noise = plt.subplots()
    _ = ax_noise.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
    ax_noise.set_xlabel('noise  [microV]')
    plt.title('Noise levels (microV)')
    plt.tight_layout()
    save.savefig(fig_noise, 'preprocessing', 'noise_levels')

    job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=True)
    peaks = detect_peaks(rec, method='locally_exclusive', noise_levels=noise_levels_int16,
                         detect_threshold=10, radius_um=50., **job_kwargs)

    peak_locations = localize_peaks(rec, peaks, method='center_of_mass', radius_um=50., **job_kwargs)

    # check for drifts
    fs = rec.sampling_frequency
    f_drift, ax_drift = plt.subplots(figsize=(10, 8))
    ax_drift.scatter(peaks['sample_index'] / fs, peak_locations['y'], color='k', marker='.', alpha=0.002)
    plt.title('Peaks (for drift detection)')
    plt.tight_layout()
    save.savefig(fig_noise, 'preprocessing', 'peaks')

    # we can also use the peak location estimates to have an insight of cluster separation before sorting
    f_clusters, ax_clusters = plt.subplots(figsize=(15, 10))
    si.plot_probe_map(rec, ax=ax_clusters, with_channel_ids=True)
    ax_clusters.set_ylim(-100, 150)

    ax_clusters.scatter(peak_locations['x'], peak_locations['y'], color='purple', alpha=0.002)
    plt.title('Cluster separation before sorting')
    plt.tight_layout()
    save.savefig(fig_noise, 'preprocessing', 'clusters')
    plt.show()


def IBL_metrics(path: Path, params: InspectionParams, tag: str, data_dict=None) -> None:
    if data_dict is None:
        data_dict = {}

    ap_folder = path.joinpath('clean_ap.zarr')
    lfp_folder = path.joinpath('clean_lfp.zarr')

    rec_ap = pio.load_preprocessed_data(ap_folder)
    rec_lfp = pio.load_preprocessed_data(lfp_folder)

    channel_quality_dict = pio.load_channel_quality_ids(path / 'channel_quality_ids.pkl')

    data_dict['channel_quality'] = channel_quality_dict['channel_quality']
    if params.run_reduced_rms:
        ic('processing reduced RMS')
        st = perf_counter()
        data_dict = rms.spikeinterface_windowed_rms(recording=rec_ap, sample_rate=params.ap_srate, tag='AP_reduced',
                                rms_window=params.window, skip_window=params.skip, data_dict=data_dict)
        data_dict = rms.spikeinterface_windowed_rms(recording=rec_lfp, sample_rate=params.lfp_srate, tag='LFP_reduced',
                                rms_window=params.window, skip_window=params.skip, data_dict=data_dict)
        ic('done processing reduced RMS in {} seconds'.format(perf_counter() - st))

    if params.run_full_rms:
        ic('processing full RMS')
        st = perf_counter()
        data_dict = rms.spikeinterface_windowed_rms(recording=rec_ap, sample_rate=params.ap_srate, tag='AP_full',
                                rms_window=params.window, skip_window=params.window, data_dict=data_dict)
        data_dict = rms.spikeinterface_windowed_rms(recording=rec_lfp, sample_rate=params.lfp_srate, tag='LFP_full',
                                rms_window=params.window, skip_window=params.window, data_dict=data_dict)
        ic('done processing full RMS in {} seconds'.format(perf_counter() - st))

    if params.run_PSD:
        ic("Calculating PSDs...")
        data_dict = psd.spikeinterface_channelwise_PSD(recording=rec_ap, sample_rate=params.ap_srate, tag='AP',
                                                   nperseg=params.nperseg, data_dict=data_dict)
        data_dict = psd.spikeinterface_channelwise_PSD(recording=rec_lfp, sample_rate=params.lfp_srate, tag='LFP',
                                                   nperseg=params.nperseg, data_dict=data_dict)

    # get primitive spikes
    if params.run_threshold_detection:
        data_dict = td.spikeinterface_channelwise_threshold_detection(recording=rec_ap, threshold=params.threshold,
                                                                      tag='{:.1f}'.format(params.threshold),
                                                                      t_session=rec_ap.get_duration(), data_dict=data_dict)

    return data_dict


def main():
    data_root = Path.home().joinpath('Documents', 'testdata')
    preprocessed_path = data_root.joinpath(
        'HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0/preprocessed/')
    tag = 'HD015_11.30.2023'

    ### set parameters ###
    params = InspectionParams()
    params.window = 1  # seconds
    params.skip = 30  # this should be 300, using 30 for testing
    params.nperseg = 1024  # use 512, 1024, or 2048
    params.ap_srate = 30000
    params.lfp_srate = 2500

    params.run_reduced_rms = True
    params.run_full_rms = True
    params.run_PSD = False
    params.run_threshold_detection = False

    # date = '2024-01-05'
    # fname = '{}_preprocessed_inspection'.format(tag)
    # data_dict = pio.load_inspection_data(date, fname)

    data_dict = IBL_metrics(preprocessed_path, params, tag)
    # si_checks(preprocessed_path)

    # plot_IBL_metrics(fname, date, tag)

    # save all the processing
    fname = '{}_preprocessed_inspection'.format(tag)
    pio.save_inspection_data(data_dict, fname)

    viz.plot_IBL_metrics(data_dict=data_dict, tag=tag)
    # viz.plot_sample_data()


if __name__ == '__main__':
    main()