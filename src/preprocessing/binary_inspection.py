import numpy as np
import seaborn as sns
from pathlib import Path
import preprocess_io as pio
from time import perf_counter
sns.set_theme(style='white')
import rms
import psd
import threshold_detection as td
from src.visualization import viz_preprocessing as viz
from dataclasses import dataclass
import get_events as ge
import sync_lines as sync
from icecream import ic
import matplotlib.pyplot as plt
import time
from icecream import ic
import src.SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX as readSGLX
import src.SGLXMetaToCoords.SGLXMetaToCoords as coordsSGLX

unprocessed_path = Path('../../data/unprocessed/')
preprocessed_path = Path('../../data/preprocessed/')
processed_path = Path('../../data/processed/')
figure_path = Path('../../reports/figures/')


@dataclass()
class InspectionParams:
    window_size: float = 1
    skip: float = 300  # seconds between RMS windows to skip
    ap_srate: int = 30000
    lfp_srate: int = 2500
    nperseg: int = 1024  # number of samples per segment for PSD calculation

    run_reduced_rms: bool = True
    run_full_rms: bool = True
    run_PSD: bool = False
    lfp_is_present: bool = True

    threshold: int = 10
    run_threshold_detection: bool = False


def run_inspection(params: InspectionParams, ap_data: np.ndarray=None, lfp_data: np.ndarray=None, tag: str='', data_dict=None) -> dict:
    if data_dict is None:
        data_dict = {}

    # ap_data = ap_data.astype(np.float32)
    if params.run_reduced_rms:
        print('processing reduced RMS')
        st = perf_counter()
        if ap_data:
            # data_dict = rms.np_windowed_rms(recording=ap_data, sample_rate=params.ap_srate, tag='AP_reduced',
            #                                 rms_window=params.window, skip_window=params.skip, data_dict=data_dict )
            data_dict = rms.np_windowed_rms(recording=ap_data, sample_rate=params.ap_srate, tag='AP_reduced',
                                            window_size=params.window_size, skip_window=params.skip, data_dict=data_dict)

        if lfp_data:
            # data_dict = rms.np_windowed_rms(recording=lfp_data, sample_rate=params.lfp_srate, tag='LFP_reduced',
            #                                 rms_window=params.window, skip_window=params.skip, data_dict=data_dict)
            data_dict = rms.np_windowed_rms(recording=lfp_data, sample_rate=params.lfp_srate, tag='LFP_reduced',
                                            window_size=params.window_size, skip_window=params.skip, data_dict=data_dict)
        print('done processing reduced RMS in {} seconds'.format(perf_counter() - st))

    if params.run_full_rms:
        print('processing full RMS')
        st = perf_counter()
        if ap_data:
            data_dict = rms.np_windowed_rms(recording=ap_data, sample_rate=params.ap_srate, tag='AP_full',
                                            window_size=params.window, skip_window=params.window, data_dict=data_dict)

        if lfp_data:
            data_dict = rms.np_windowed_rms(recording=lfp_data, sample_rate=params.lfp_srate, tag='LFP_full',
                                            window_size=params.window, skip_window=params.window, data_dict=data_dict)
        print('done processing full RMS in {} seconds'.format(perf_counter() - st))

    if params.run_PSD:
        print("Calculating PSDs...")
        st = perf_counter()
        data_dict = psd.np_channelwise_PSD(ap_data, params.ap_srate, tag='AP', nperseg=params.nperseg,
                                           data_dict=data_dict)  # run this on the destriped AP data
        print('done processing AP PSD in {} seconds'.format(perf_counter() - st))
        if params.lfp_is_present:
            st = perf_counter()
            data_dict = psd.np_channelwise_PSD(lfp_data, params.lfp_srate, tag='LFP', nperseg=1024, data_dict=data_dict)
        print('done processing LFP PSD in {} seconds'.format(perf_counter() - st))

    if params.run_threshold_detection:
        data_dict = td.np_channelwise_threshold_detection(ap_data, params.threshold, tag='{:.1f}'.format(params.threshold),
                                                          t_session=ap_data.shape[1] / params.ap_srate,
                                                          data_dict=data_dict)

    fname = '{}_inspection'.format(tag)
    pio.save_inspection_data(data_dict, fname)
    return data_dict


def get_session_stats(ap_data: np.ndarray, threshold_std=0, apply_gain=True) -> dict:
    data_dict = {}
    means = np.mean(ap_data, axis=0)
    medians = np.median(ap_data, axis=0)
    stds = np.std(ap_data, axis=0)
    if threshold_std:
        stds[stds < threshold_std] = 0
    maxs = np.max(np.abs(ap_data), axis=0)
    features = np.array([means, medians, stds, maxs])

    data_dict['mean'] = means
    data_dict['median'] = medians
    data_dict['std'] = stds
    data_dict['max'] = maxs
    data_dict['all'] = features
    return data_dict


def get_session_stats_chunks(ap_data: np.ndarray, metadata: dict, apply_gain=True, save=True) -> dict:
    ix_st = 0
    ix_end = ap_data.shape[1]
    sample_rate = int(readSGLX.SampRate(metadata))

    data_dict = {}
    ix1 = 0
    ix2 = 0
    means, medians, stds, maxs = [], [], [], []
    ic('Calculating session stats...')
    while ix2 < ix_end:
        ix1 = ix2
        ix2 = np.min([ix1 + sample_rate, ix_end])
        if ix2 % (10 * 60 * sample_rate) == 0:
            ic('now processing until timestep {}'.format(ix2 / sample_rate))

        chunk = ap_data[:, ix1:ix2]
        if apply_gain:
            chunk = pio.correct_binary_gain(chunk, metadata, chanList=list(range(384)))
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


def run_lick_inspection():
    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    # set data paths
    # data_root = Path.home().joinpath('Documents', 'testdata')
    # rec_root = data_root / 'catGT_out/catgt_HD015_11302023_g0/HD015_11302023_g0_imec0'
    # rec_ap_bin = rec_root / 'HD015_11302023_g0_tcat.imec0.ap.bin'
    # rec_lfp_bin = rec_root / 'HD015_11302023_g0_tcat.imec0.lf.bin'
    # tag = 'HD015_11302023_catGT'

    # rec_root = data_root / 'HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0'
    # rec_ap_bin = rec_root / 'HD015_11302023_g0_t0.imec0.ap.bin'
    # rec_lfp_bin = rec_root / 'HD015_11302023_g0_t0.imec0.lf.bin'
    # tag = 'HD015_11302023_unprocessed'

    # data_root = Path.home().joinpath('Documents', 'processed_ephys')  # either 'ephys_transfer' or 'processed_ephys'
    # rec_root = data_root / 'CT009_current_20250302_filter-gfix/catgt_run1_g0/run1_g0_imec0'
    # rec_ap_bin = rec_root / 'run1_g0_tcat.imec0.ap.bin'
    # tag = 'CT009_current_20250302_CatGT-filter-gfix'

    session_name = 'CT011_20250624'
    run = 0
    gate = 0
    opts = 'catgt'
    # recording_path = raw_data_root.joinpath('{}/run1_g0'.format(session_name))  # for raw data
    # imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')  # for raw data
    recording_path = processed_data_root.joinpath('{}_{}/catgt_run{}_g{}'.format(session_name, opts, run, gate))  # for filtered data
    ap_binary = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec0.ap.bin'.format(run, gate))
    lfp_binary = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec0.lf.bin'.format(run, gate))
    ap_binary1 = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec1.ap.bin'.format(run, gate))
    lfp_binary1 = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec1.lf.bin'.format(run, gate))

    ni_file = raw_data_root.joinpath('{}/run0_g0/run0_g0_t0.nidq.bin'.format(session_name))
    if opts:
        tag = session_name + '_' + opts
    else:
        tag = session_name

    # read in data, assuming it's neuropixel data
    ap_data, ap_meta, ap_srate, ap_shape = pio.read_binary(ap_binary)
    lfp_data, lfp_meta, lfp_srate, lfp_shape = pio.read_binary(lfp_binary)
    # ap_data, ap_meta, ap_srate, ap_shape = read_binary(ibl_bin)

    # ap_data = ap_data[:384].astype(np.float32)  # remove sync line during inspection
    ap_data = ap_data[:384]  # remove sync line during inspection
    ap_shape = (384, ap_shape[1])
    lfp_data = lfp_data[:384]
    lfp_shape = (384, lfp_shape[1])

    ### set parameters ###
    params = InspectionParams()
    params.window = 1  # seconds
    params.skip = 300  # this should be 300, using 30 for testing
    params.nperseg = 1024  # use 512, 1024, or 2048
    params.ap_srate = 30000
    params.lfp_srate = 2500

    params.run_reduced_rms = True
    params.run_full_rms = False
    params.run_PSD = False
    params.run_threshold_detection = False
    params.lfp_is_present = False

    # Lsync, Rsync = ge.sync_for_demonstration(imec_file_ap, ni_file, debounce=0.0002)
    all_events, offsets, onsets = ge.sync_for_demonstration(ap_binary, ni_file, debounce=0.0002)
    # ix_sync_dict, t_sync_dict = ge.sync_behavior(ap_binary, ni_file, debounce=0.0002)

    ### optionally, load a previous inspection ###
    # data_dict = pio.load_inspection_data('2023-12-15', 'HD015_11302023')

    ### run inspection ###
    data_dict = run_inspection(ap_data, lfp_data, params, tag)
    # data_dict = run_inspection(ap_data, params, tag)

    fname = '{}_inspection'.format(tag)
    # pio.save_inspection_data(data_dict, fname)  # included in run_inspection
    # viz.plot_IBL_metrics(data_dict=data_dict, tag=tag)

    # t1_ix = 0
    # t2_ix = ap_shape[1]
    t1 = 5 * 60
    t2 = t1 + 1
    # t1 = 0
    # t2 = 10
    t1_ix = int(t1 * ap_srate)
    t2_ix = int(t2 * ap_srate)

    # Lsync = Lsync[(Lsync > t1_ix) & (Lsync < t2_ix)] - t1_ix
    # Rsync = Rsync[(Rsync > t1_ix) & (Rsync < t2_ix)] - t1_ix
    all_events = all_events[(all_events > t1_ix) & (all_events < t2_ix)] - t1_ix
    offsets = offsets[(offsets > t1_ix) & (offsets < t2_ix)] - t1_ix
    onsets = onsets[(onsets > t1_ix) & (onsets < t2_ix)] - t1_ix

    # event_ix = np.concatenate((Lsync, Rsync))
    # ic(Lsync / ap_srate, Rsync / ap_srate)
    # plt.show()

    # viz.plot_sample_data(ap_data, t1_ix, t2_ix, ap_srate, tag, processing_step='preprocessed')
    tag += '_onsets'
    viz.plot_sample_data(ap_data, t1_ix, t2_ix, ap_srate, tag, processing_step='preprocessed', plot_step=15)
    # viz.plot_onset_offset(ap_data, event_times=[onsets, offsets], t1_ix=t1_ix, t2_ix=t2_ix, sample_freq=ap_srate, tag=tag, processing_step='preprocessed',
    #                       plot_step=15)

    # data_mmap, metadata, samplerate, shape = pio.read_binary(imec_file_ap)
    # get_session_stats_chunks(ap_data, ap_meta, threshold_std=25)

    # selectData = ap_data[:384, t1_ix:t2_ix]
    # selectData = pio.correct_binary_gain(selectData, ap_meta, chanList=np.arange(384))
    tst = perf_counter()
    # sess_stats_dict = get_session_stats(selectData, metadata, threshold_std=25)
    # sess_stats_dict = get_session_stats_chunks(ap_data, ap_meta)

    date = '2025-03-17'
    sess_stats_dict = pio.load_inspection_data(date, 'session_stats')

    t_compute_stats = perf_counter() - tst
    ic(t_compute_stats)

    std_threshold = 0
    sess_stats_dict['std'][sess_stats_dict['std'] < std_threshold] = 0
    sess_stats_dict['all'][2] = sess_stats_dict['std']

    # viz.plot_stats(sess_stats_dict, tag=tag, processing_step='preprocessed')
    # viz.plot_sample_data(ap_data, t1_ix, t2_ix, sample_freq=ap_srate, tag=tag + '_onsets', processing_step='preprocessed',
    #                      event_times=onsets, comparison_data=np.abs(sess_stats_dict['all']), plot_step=20)


def main():
    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    session_name = 'CT011_20250624'
    run = 0
    gate = 0
    opts = 'catgt'
    # for raw data
    # recording_path = raw_data_root.joinpath('{}/run1_g0'.format(session_name))
    # imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')
    # for filtered data
    recording_path = processed_data_root.joinpath('{}_{}/catgt_run{}_g{}'.format(session_name, opts, run, gate))
    ap_imec0 = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec0.ap.bin'.format(run, gate))
    lfp_imec0 = recording_path.joinpath('run{0}_g{1}_imec0/run{0}_g{1}_tcat.imec0.lf.bin'.format(run, gate))
    ap_imec1 = recording_path.joinpath('run{0}_g{1}_imec1/run{0}_g{1}_tcat.imec1.ap.bin'.format(run, gate))
    lfp_imec1 = recording_path.joinpath('run{0}_g{1}_imec1/run{0}_g{1}_tcat.imec1.lf.bin'.format(run, gate))

    nidaq = raw_data_root.joinpath('{}/run0_g0/run0_g0_t0.nidq.bin'.format(session_name))
    if opts:
        tag = session_name + '_' + opts
    else:
        tag = session_name

    # read in data, assuming it's neuropixel data
    ap0_data, ap0_meta, ap_srate, ap_shape = pio.read_binary(ap_imec0)
    lfp0_data, lfp0_meta, lfp_srate, lfp_shape = pio.read_binary(lfp_imec0)
    ap1_data, ap1_meta, _, _ = pio.read_binary(ap_imec1)
    lfp1_data, lfp1_meta, _, _ = pio.read_binary(lfp_imec1)

    n_shank, shank_width, shank_pitch, shank_ind, x, y, connected = coordsSGLX.geomMapToGeom(ap0_meta)
    coords = np.stack((x,y), axis=1)
    geometric_sort_imec0 = np.argsort(coords[:,1])  # sort by y coordinate; for now, sites are on one shank, so this is sufficient
    sorted_coords_imec0 = coords[geometric_sort_imec0]

    n_shank, shank_width, shank_pitch, shank_ind, x, y, connected = coordsSGLX.geomMapToGeom(ap1_meta)
    coords = np.stack((x, y), axis=1)
    geometric_sort_imec1 = np.argsort(coords[:, 1])  # sort by y coordinate; for now, sites are on one shank, so this is sufficient
    sorted_coords_imec1 = coords[geometric_sort_imec1]

    # ap_data = ap_data[:384].astype(np.float32)  # remove sync line during inspection
    ap0_data = ap0_data[:384]  # remove sync line during inspection
    ap1_data = ap1_data[:384]
    ap_shape = (384, ap_shape[1])
    lfp0_data = lfp0_data[:384]
    lfp1_data = lfp1_data[:384]
    lfp_shape = (384, lfp_shape[1])
    session_length = ap0_data.shape[1] / ap_srate

    ### set parameters ###
    params = InspectionParams()
    params.window = 1  # seconds
    params.skip = 300  # this should be 300, using 30 for testing
    params.nperseg = 1024  # use 512, 1024, or 2048
    params.ap_srate = 30000
    params.lfp_srate = 2500

    params.run_reduced_rms = True
    params.run_full_rms = False
    params.run_PSD = False
    params.run_threshold_detection = False
    params.lfp_is_present = False

    ix_sync_dict, t_sync_dict = ge.sync_behavior(ap_imec0, nidaq, debounce=0.0002)

    ### optionally, load a previous inspection ###
    # data_dict = pio.load_inspection_data('2025-06-27', 'CT011_20250624_catgt')

    ### run inspection ###
    # data_dict_imec0 = run_inspection(ap0_data, lfp0_data, params, tag)
    # data_dict_imec0 = run_inspection(ap_data, lfp_data, params, tag)

    data_dict = {}

    """
    Root mean square (RMS) inspection. Per the IBL white paper, 
    - vertical stripes indicate electrical noise. 
    - horizontal lines show reference channels and noisy channels that should be excluded
    """
    ### Reduced RMS ###
    # data_dict = rms.np_windowed_rms(recording=ap0_data, sample_rate=params.ap_srate, tag='AP0_reduced',
    #                                 window_size=params.window_size, skip_window=params.skip, data_dict=data_dict,
    #                                 metadata=ap0_meta, chanlist=np.arange(384))
    # xticks = dict(ticks=np.arange(data_dict['AP0_reduced_rms'].shape[1]),
    #               labels=data_dict['AP0_reduced_rms_times'].astype(int))
    # yticks = dict(ticks=np.arange(384), labels=geometric_sort_imec0)
    # viz.plot_heatmap(data_dict['AP0_reduced_rms'][geometric_sort_imec0], xticks=xticks, #yticks=yticks,
    #              title=tag + ' AP0 reduced RMS')
    #
    # data_dict = rms.np_windowed_rms(recording=lfp0_data, sample_rate=params.lfp_srate, tag='LFP0_reduced',
    #                                 window_size=params.window_size, skip_window=params.skip, data_dict=data_dict,
    #                                 metadata=lfp0_meta, chanlist=np.arange(384))
    # xticks = dict(ticks=np.arange(data_dict['LFP0_reduced_rms'].shape[1]),
    #               labels=data_dict['LFP0_reduced_rms_times'].astype(int))
    # yticks = dict(ticks=np.arange(384), labels=geometric_sort_imec0)
    # viz.plot_heatmap(data_dict['LFP0_reduced_rms'][geometric_sort_imec0], xticks=xticks, #yticks=yticks,
    #              title=tag + ' LFP0 reduced RMS')

    # data_dict = rms.np_windowed_rms(recording=ap1_data, sample_rate=params.ap_srate, tag='AP1_reduced',
    #                                 window_size=params.window_size, skip_window=params.skip, data_dict=data_dict,
    #                                 metadata=ap1_meta, chanlist=np.arange(384))
    # xticks = dict(ticks=np.arange(data_dict['AP1_reduced_rms'].shape[1]),
    #               labels=data_dict['AP1_reduced_rms_times'].astype(int))
    # yticks = dict(ticks=np.arange(384), labels=geometric_sort_imec1)
    # viz.plot_rms(data_dict['AP1_reduced_rms'][geometric_sort_imec1], xticks=xticks, #yticks=yticks,
    # title=tag + ' AP1 reduced RMS')

    # data_dict = rms.np_windowed_rms(recording=lfp1_data, sample_rate=params.lfp_srate, tag='LFP1_reduced',
    #                                 window_size=params.window_size, skip_window=params.skip, data_dict=data_dict,
    #                                 metadata=lfp1_meta, chanlist=np.arange(384))
    # xticks = dict(ticks=np.arange(data_dict['LFP1_reduced_rms'].shape[1]),
    #               labels=data_dict['LFP1_reduced_rms_times'].astype(int))
    # viz.plot_rms(data_dict['LFP1_reduced_rms'][geometric_sort_imec1], xticks=xticks,
    #              title=tag + ' LFP1 reduced RMS')

    ### Full RMS ###

    # data_dict = rms.np_windowed_rms(recording=ap0_data, sample_rate=params.ap_srate, tag='AP0_full',
    #                                 window_size=params.window_size, skip_window=params.window_size, data_dict=data_dict,
    #                                 metadata=ap0_meta, chanlist=np.arange(384))
    # xticks = np.arange(0, float(ap0_meta['fileTimeSecs']), 600)
    # xticks = dict(ticks=xticks, labels=xticks.astype(int))
    # viz.plot_rms(data_dict['AP0_full_rms']geometric_sort_imec0], xticks=xticks, title=tag + ' AP0 full RMS')

    # data_dict = rms.np_windowed_rms(recording=lfp0_data, sample_rate=params.lfp_srate, tag='LFP0_full',
    #                                 window_size=params.window_size, skip_window=params.window_size, data_dict=data_dict,
    #                                 metadata=ap0_meta, chanlist=np.arange(384))
    # xticks = np.arange(0, float(ap0_meta['fileTimeSecs']), 600)
    # xticks = dict(ticks=xticks, labels=xticks.astype(int))
    # viz.plot_rms(data_dict['LFP0_full_rms'][geometric_sort_imec0], xticks=xticks, title=tag + ' LFP0 full RMS')

    # data_dict = rms.np_windowed_rms(recording=ap1_data, sample_rate=params.ap_srate, tag='AP1_full',
    #                                 window_size=params.window_size, skip_window=params.window_size, data_dict=data_dict,
    #                                 metadata=ap1_meta, chanlist=np.arange(384))
    # xticks = np.arange(0, float(ap1_meta['fileTimeSecs']), 600)
    # xticks = dict(ticks=xticks, labels=xticks.astype(int))
    # viz.plot_rms(data_dict['AP1_full_rms'][geometric_sort_imec1], xticks=xticks, title=tag + ' AP1 full RMS')

    # data_dict = rms.np_windowed_rms(recording=lfp1_data, sample_rate=params.lfp_srate, tag='LFP1_full',
    #                                 window_size=params.window_size, skip_window=params.window_size, data_dict=data_dict,
    #                                 metadata=ap1_meta, chanlist=np.arange(384))
    # xticks = np.arange(0, float(ap1_meta['fileTimeSecs']), 600)
    # xticks = dict(ticks=xticks, labels=xticks.astype(int))
    # viz.plot_rms(data_dict['LFP1_full_rms'][geometric_sort_imec1], xticks=xticks, title=tag + ' LFP1 full RMS')


    ### destriping inspection; really not necessary ###
    # tstart = 0
    # tend = int(tstart + 10 * params.ap_srate / 1e3)  # milliseconds
    # viz.plot_heatmap(ap0_data[:, tstart:tend], title=tag + ' AP0 destripe')

    ### PSD inspection ###
    print("Calculating PSDs...")
    # st = perf_counter()
    # data_dict = psd.np_channelwise_PSD(ap0_data, params.ap_srate, tag='AP0', nperseg=params.nperseg,
    #                                    data_dict=data_dict)  # run this on the destriped AP data
    # viz.plot_PSD(freq=data_dict['AP0_freq'], psd=data_dict['AP0_psd'], title='{} AP0 PSD'.format(tag))
    # print('done processing AP0 PSD in {} seconds'.format(perf_counter() - st))

    st = perf_counter()
    data_dict = psd.np_channelwise_PSD(lfp0_data, params.lfp_srate, tag='LFP0', nperseg=params.nperseg,
                                       data_dict=data_dict)
    viz.plot_PSD(freq=data_dict['LFP0_freq'], psd=data_dict['LFP0_psd'][geometric_sort_imec0],
                 title='{} LFP0 PSD'.format(tag))
    print('done processing LFP0 PSD in {} seconds'.format(perf_counter() - st))

    fname = '{}_inspection'.format(tag)
    pio.save_inspection_data(data_dict, fname)  # included in run_inspection
    # viz.plot_IBL_metrics(data_dict=data_dict, tag=tag)


if __name__ == '__main__':
    main()
