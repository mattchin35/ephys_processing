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
import get_events as ge


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


def get_session_stats_chunks(si_extractor: si.BaseRecording, apply_gain=False, save=True, tag='') -> dict:
    ix_end = si_extractor.get_num_frames()
    sample_rate = si_extractor.get_time_info()['sampling_frequency']

    data_dict = {}
    ix2 = 0
    means, medians, stds, maxs = [], [], [], []
    ic('Calculating session stats...')
    while ix2 < ix_end:
        ix1 = ix2
        ix2 = np.min([ix1 + sample_rate, ix_end])
        if ix2 % (10 * 60 * sample_rate) == 0:
            ic('now processing until timestep {}'.format(ix2 / sample_rate))

        chunk = si_extractor.get_traces(start_frame=ix1, end_frame=ix2, return_scaled=apply_gain).T
        _means = np.mean(chunk, axis=0)
        _medians = np.median(chunk, axis=0)
        _stds = np.std(chunk, axis=0)
        _maxs = np.max(np.abs(chunk), axis=0)

        if ix2-ix1 == 1:
            _means = np.array([_means])
            _medians = np.array([_medians])
            _stds = np.array([_stds])
            _maxs = np.array([_maxs])

        means.append(_means)
        medians.append(_medians)
        stds.append(_stds)
        maxs.append(_maxs)


    ic('Done calculating session stats')
    means = np.concatenate(means)
    medians = np.concatenate(medians)
    stds = np.concatenate(stds)
    maxs = np.concatenate(maxs)
    features = np.stack([means, medians, stds, maxs], axis=0)

    data_dict['mean'] = means
    data_dict['median'] = medians
    data_dict['std'] = stds
    data_dict['max'] = maxs
    data_dict['all'] = features
    if save:
        fname = 'session_stats'
        pio.save_inspection_data(data_dict, fname)

    return data_dict


def main():
    # data_root = Path.home().joinpath('Documents', 'testdata')
    # preprocessed_path = data_root.joinpath(
    #     'HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0/preprocessed/')
    # tag = 'HD015_11.30.2023'

    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    preprocessed_data_root = Path.home().joinpath('Documents', 'preprocessed_ephys')

    session_name = 'CT011_20250624'  #'CT009_current_20250302'
    # raw data
    run = 0
    gate = 0
    probe = 0
    trigger = 0
    opts = 'catgt'

    recording_path = raw_data_root.joinpath('{}/run{}_g{}'.format(session_name, run, gate))  # for raw data
    # imec_file_ap = recording_path.joinpath('run{0}_g{1}_imec{2}/run{0}_g{1}_t{3}.imec{2}.ap.bin'.format(run, gate, probe, trigger))  # for raw data
    # ni_file = raw_data_root.joinpath('{0}/run{1}_g{2}/run{1}_g{2}_t{3}.nidq.bin'.format(session_name, run, gate, trigger))  # for raw data
    # catgt processed data
    # recording_path = processed_data_root.joinpath('{0}_filter-gfix/catgt_run{1}_g{2}'.format(session_name, run, gate))
    # imec_file_ap = recording_path.joinpath('run{0}_g{1}_imec{2}/run{}_g{}.imec0.ap.bin')
    recording_path = preprocessed_data_root.joinpath('{}_{}/catgt_run{}_g{}'.format(session_name, opts, run, gate))
    imec0_path = recording_path.joinpath('run{0}_g{1}_imec0'.format(run, gate))
    imec1_path = recording_path.joinpath('run{0}_g{1}_imec1'.format(run, gate))

    ap_imec0 = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec0.ap.bin'.format(run, gate))
    lfp_imec0 = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec0.lf.bin'.format(run, gate))
    ap_imec1 = recording_path.joinpath('run{0}_g{1}_imec1/run{0}_g{1}_tcat.imec1.ap.bin'.format(run, gate))
    lfp_imec1 = recording_path.joinpath('run{0}_g{1}_imec1/run{0}_g{1}_tcat.imec1.lf.bin'.format(run, gate))
    # spikeinterface zarr data
    # recording_path = processed_data_root.joinpath('{}_Tartifacts/clean_ap.zarr'.format(session_name))
    # recording = pio.load_zarr_data(recording_path)
    # tag = session_name + '_Tartifacts'
    if opts:
        tag = session_name + '_' + opts
    else:
        tag = session_name

    # save_path = processed_data_root.joinpath('CT009_current_20250302_resaved-gfix')
    # recordings = pio.load_sg22111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111lx_data(recording_path)  # spikeinterface loading, not binary
    # ic(recordings)

    recordings0 = pio.load_sglx_data(imec0_path)  # spikeinterface loading, not binary
    bad_channel_ids0, channel_quality0 = si.detect_bad_channels(recordings0[1])
    recordings1 = pio.load_sglx_data(imec0_path)  # spikeinterface loading, not binary
    bad_channel_ids1, channel_quality1 = si.detect_bad_channels(recordings1[1])

    channel_ids = ["imec0.ap#AP{}".format(i) for i in range(5)]
    # w = si.plot_traces(recording, channel_ids=channel_ids, time_range=(360, 365))
    # w = si.plot_traces(recordings[0], channel_ids=channel_ids, time_range=(360, 361))
    # plt.show()
    # traces = recording_spikeglx.get_traces(start_frame=None, end_frame=None, return_scaled=False)

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

    # data_dict = IBL_metrics(preprocessed_path, params, tag)
    # si_checks(preprocessed_path)

    # plot_IBL_metrics(fname, date, tag)

    # save all the processing
    fname = '{}_preprocessed_inspection'.format(tag)
    # pio.save_inspection_data(data_dict, fname)

    # viz.plot_IBL_metrics(data_dict=data_dict, tag=tag)
    # viz.plot_sample_data()
    # t1 = 6 * 60
    # t2 = t1 + 5
    # sample_rate = recording.get_time_info()['sampling_frequency']
    # t1_ix = int(t1 * sample_rate)
    # t2_ix = int(t2 * sample_rate)
    #
    # traces = recording.get_traces(start_frame=t1_ix, end_frame=t2_ix).T
    # stats_dict = get_session_stats_chunks(recording, tag=tag)
    # comparison = np.abs(stats_dict['all'])[:, t1_ix:t2_ix]

    # all_events, offsets, onsets = ge.sync_for_demonstration(imec_file_ap, ni_file, debounce=0.0002)
    # all_events = all_events[(all_events > t1_ix) & (all_events < t2_ix)] - t1_ix
    # offsets = offsets[(offsets > t1_ix) & (offsets < t2_ix)] - t1_ix
    # onsets = onsets[(onsets > t1_ix) & (onsets < t2_ix)] - t1_ix

    # viz.plot_sample_data(traces, sample_freq=sample_rate, tag=tag + '_onsets',
    #                      processing_step='preprocessed',
    #                      event_times=onsets, comparison_data=comparison, plot_step=20)


if __name__ == '__main__':
    main()