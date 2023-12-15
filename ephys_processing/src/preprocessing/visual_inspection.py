import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import src.DemoReadSGLXData.readSGLX as sglx # will use readMeta, SampRate, makeMemMapRaw, ExtractDigital, GainCorrectIM, GainCorrectNI
from typing import Tuple
from datetime import date


def read_metadata(binary_file: Path) -> Tuple[dict, int, Tuple[int,int]]:
    metadata = sglx.readMeta(binary_file)
    sample_rate = int(sglx.SampRate(metadata))
    n_chan = int(metadata['nSavedChans'])
    n_samples = int(int(metadata['fileSizeBytes']) / (2 * n_chan))
    return metadata, sample_rate, (n_chan, n_samples)


def rms(x: np.ndarray, axis: int=None):
    if axis is None:
        return np.sqrt(np.mean(np.square(x)))  # flatten the whole array
    else:
        return np.sqrt(np.mean(np.square(x), axis=axis))  # flatten along the specified axis


def get_windowed_rms(data: np.ndarray, sample_rate: int, rms_window: int=1, skip_window: int=300) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Root-mean-squared of the data, taking window_size segments every nskip seconds.
    :param data: data array
    :param srate: sample rate
    :param nsamples: number of samples in the data array
    :param rms_window: number of seconds in each snippet
    :param nskip: number of seconds to skip between samples
    :return: reduced data array
    """
    ix = 0
    snippets = []
    nsamples = data.shape[1]
    times = []
    while ix < nsamples:
        times.append(ix / sample_rate)
        window_rms = rms(data[:, ix:min(ix + sample_rate * rms_window, nsamples)], axis=1)
        snippets.append(window_rms)
        ix += sample_rate * skip_window
    return np.stack(snippets, axis=1), np.array(times)


def get_rms_quantiles(rms_data: np.ndarray, axis: int=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the median, 10th percentile, and 90th percentile of the RMS data for each channel.
    """
    median = np.median(rms_data, axis=axis)
    q1 = np.quantile(rms_data, 0.1, axis=axis)
    q9 = np.quantile(rms_data, 0.9, axis=axis)
    return median, q1, q9


def save_rms():
    pass


def save_fig(fname: str, fig: plt.Figure):
    pass


def plot_rms(rms_data: np.ndarray, xticks=None, yticks=None, title: str=''):
    f, ax = plt.subplots()
    sns.heatmap(data=rms_data, ax=ax, cbar=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    if title:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(np.arange(rms_data.shape[1]), xticks.astype(int))


def main():
    # set data paths
    data_root = Path.home().joinpath('Documents', 'testdata')
    ibl_bin = data_root.joinpath('IBL_testdata', 'imec_385_100s.ap.bin')
    rec_ap_bin = data_root.joinpath(
        'HD015_11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0/HD015_11.30.2023_g0_t0.imec0.ap.bin')
    rec_lfp_bin = data_root.joinpath(
        'HD015_11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0/HD015_11.30.2023_g0_t0.imec0.lf.bin')

    # set save paths
    processed_path = Path('../../data/processed/{}/preprocessing'.format(date.today().isoformat()))
    figures_path = Path('../../reports/figures/{}/preprocessing'.format(date.today().isoformat()))
    # processed_path.mkdir(parents=True, exist_ok=True)
    # figures_path.mkdir(parents=True, exist_ok=True)

    # read metadata and sample rates
    ap_meta, ap_srate, ap_shape = read_metadata(rec_ap_bin)
    lfp_meta, lfp_srate, lfp_shape = read_metadata(rec_lfp_bin)
    session_length = float(ap_meta['fileTimeSecs'])
    print("Session length is {} seconds, or {} minutes and {} seconds".format(session_length,
                                                                              session_length // 60,
                                                                              session_length % 60))

    ap_data = sglx.makeMemMapRaw(rec_ap_bin, ap_meta)
    lfp_data = sglx.makeMemMapRaw(rec_lfp_bin, lfp_meta)

    # RMS inspection
    window = 1
    skip = 30
    reduced_ap_rms, ap_windowed_times = get_windowed_rms(ap_data, ap_srate, rms_window=window, skip_window=skip)
    reduced_lfp_rms, lfp_windowed_times = get_windowed_rms(lfp_data, lfp_srate, rms_window=window, skip_window=skip)

    full_ap_rms, ap_rms_times = get_windowed_rms(ap_data, ap_srate, rms_window=window, skip_window=window)
    full_lfp_rms, lfp_rms_times = get_windowed_rms(lfp_data, lfp_srate, rms_window=window, skip_window=window)

    reduced_ap_median, reduced_ap_q1, reduced_ap_q9 = get_rms_quantiles(reduced_ap_rms)
    reduced_lfp_median, reduced_lfp_q1, reduced_lfp_q9 = get_rms_quantiles(reduced_lfp_rms)
    full_ap_median, full_ap_q1, full_ap_q9 = get_rms_quantiles(full_ap_rms)
    full_lfp_median, full_lfp_q1, full_lfp_q9 = get_rms_quantiles(full_lfp_rms)

    # convert to dB if desired
    reduced_ap_db = 20 * np.log10(reduced_ap_rms / reduced_ap_median)
    reduced_lfp_db = 20 * np.log10(reduced_lfp_rms / reduced_lfp_median)
    full_ap_db = 20 * np.log10(full_ap_rms / full_ap_median)
    full_lfp_db = 20 * np.log10(full_lfp_rms / full_lfp_median)

    # plot_rms(reduced_ap_rms[:384], xticks=ap_windowed_times, title='AP reduced RMS')
    # plot_rms(reduced_lfp_rms[:384], xticks=lfp_windowed_times, title='LFP reduced RMS')
    plot_rms(full_ap_rms[:384], title='AP full RMS')
    plot_rms(full_lfp_rms[:384], title='LFP full RMS')

    plot_rms(full_ap_db[:384], title='AP full RMS dB')
    plot_rms(full_lfp_db[:384], title='LFP full RMS dB')

    # PSD inspection

    # print("Calculating PSDs...")
    # freq_ap, psd_ap = sp.signal.welch(ap_data, ap_srate, nperseg=1024)  # pick 1024 or 2048 for nperseg; they are arbitrary but good enough powers of 2
    # freq_lfp, psd_lfp = sp.signal.welch(lfp_data, lfp_srate, nperseg=1024)

    # print("Plotting PSDs...")
    # # plt.semilogy(freq_ap, psd_ap)
    # plt.semilogy(freq_lfp, psd_lfp)
    # plt.ylim([0.5e-3, 1])
    # plt.xlabel('frequency [Hz]')
    # plt.ylabel('PSD [V**2/Hz]')
    plt.show()


if __name__ == '__main__':
    main()

