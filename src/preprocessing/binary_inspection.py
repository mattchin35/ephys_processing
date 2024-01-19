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


processed_path = Path('../../data/processed/')
figure_path = Path('../../reports/figures/')


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


def run_inspection(ap_data: np.ndarray, lfp_data: np.ndarray, params: InspectionParams, tag: str, data_dict=None) -> dict:
    if data_dict is None:
        data_dict = {}

    if params.run_reduced_rms:
        print('processing reduced RMS')
        st = perf_counter()
        data_dict = rms.np_windowed_rms(recording=ap_data, sample_rate=params.ap_srate, tag='AP_reduced',
                                        rms_window=params.window, skip_window=params.skip, data_dict=data_dict)
        data_dict = rms.np_windowed_rms(recording=lfp_data, sample_rate=params.lfp_srate, tag='LFP_reduced',
                                        rms_window=params.window, skip_window=params.skip, data_dict=data_dict)
        print('done processing reduced RMS in {} seconds'.format(perf_counter() - st))

    if params.run_full_rms:
        print('processing full RMS')
        st = perf_counter()
        data_dict = rms.np_windowed_rms(recording=ap_data, sample_rate=params.ap_srate, tag='AP_full',
                                        rms_window=params.window, skip_window=params.window, data_dict=data_dict)
        data_dict = rms.np_windowed_rms(recording=lfp_data, sample_rate=params.lfp_srate, tag='LFP_full',
                                        rms_window=params.window, skip_window=params.window, data_dict=data_dict)
        print('done processing full RMS in {} seconds'.format(perf_counter() - st))

    if params.run_PSD:
        print("Calculating PSDs...")
        st = perf_counter()
        data_dict = psd.np_channelwise_PSD(ap_data, params.ap_srate, tag='AP', nperseg=params.nperseg,
                                           data_dict=data_dict)  # run this on the destriped AP data
        print('done processing AP PSD in {} seconds'.format(perf_counter() - st))

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


def main():
    # set data paths
    data_root = Path.home().joinpath('Documents', 'testdata')
    # rec_root = data_root / 'catGT_out/catgt_HD015_11302023_g0/HD015_11302023_g0_imec0'
    # rec_ap_bin = rec_root / 'HD015_11302023_g0_tcat.imec0.ap.bin'
    # rec_lfp_bin = rec_root / 'HD015_11302023_g0_tcat.imec0.lf.bin'
    # tag = 'HD015_11302023_catGT'

    rec_root = data_root / 'HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0'
    rec_ap_bin = rec_root / 'HD015_11302023_g0_t0.imec0.ap.bin'
    rec_lfp_bin = rec_root / 'HD015_11302023_g0_t0.imec0.lf.bin'
    tag = 'HD015_11302023_unprocessed'

    # read in data, assuming it's neuropixel data
    # ap_data, ap_meta, ap_srate, ap_shape = read_binary(ibl_bin)
    ap_data, ap_meta, ap_srate, ap_shape = pio.read_binary(rec_ap_bin)
    lfp_data, lfp_meta, lfp_srate, lfp_shape = pio.read_binary(rec_lfp_bin)
    ap_data = ap_data[:384]
    ap_shape = (384, ap_shape[1])
    lfp_data = lfp_data[:384]
    lfp_shape = (384, lfp_shape[1])

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

    ### optionally, load a previous inspection ###
    # data_dict = pio.load_inspection_data('2023-12-15', 'HD015_11302023')

    ### run inspection ###
    data_dict = run_inspection(ap_data, lfp_data, params, tag)

    fname = '{}_inspection'.format(tag)
    pio.save_inspection_data(data_dict, fname)

    viz.plot_IBL_metrics(data_dict=data_dict, tag=tag)


if __name__ == '__main__':
    main()

