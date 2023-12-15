import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
import src.DemoReadSGLXData.readSGLX as sglx # will use readMeta, SampRate, makeMemMapRaw, ExtractDigital, GainCorrectIM, GainCorrectNI
from typing import Tuple, Dict, Any
import datetime as dt
import pickle as pkl
sns.set_theme(style='white')


processed_path = Path('../../data/processed/')
figure_path = Path('../../reports/figures/')


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


def save_data(data: Dict[str, Any], fname: str, note: str='') -> None:
    date = str(dt.date.today().isoformat())
    save_dir = processed_path / 'preprocessing' / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (fname + '.pkl')
    with open(p, 'wb') as f:
        pkl.dump(data, f)
    print("[***] Data saved in {}".format(p.name))

    if note:
        with open(save_dir / 'readme.txt', 'w') as f:
            f.write(note)


def load_inspection_data(date: str, fname: str) -> Dict[str, Any]:
    save_dir = processed_path / 'preprocessing' / date
    p = save_dir / (fname + '.pkl')
    with open(p, 'rb') as f:
        data = pkl.load(f)
    return data


def savefig(figure: plt.Figure, name: str, dpi=300) -> None:
    date = str(dt.date.today().isoformat())
    save_dir = figure_path / 'preprocessing' / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (name + '.png')
    figure.savefig(p, dpi=dpi)


def plot_rms(rms_data: np.ndarray, xticks=None, yticks=None, title: str=''):
    f, ax = plt.subplots()
    sns.heatmap(data=rms_data, ax=ax, cbar=True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    if title:
        ax.set_title(title)
    if xticks is not None:
        ax.set_xticks(np.arange(rms_data.shape[1]), xticks.astype(int))
    if yticks is not None:
        ax.set_yticks(np.arange(rms_data.shape[0]), yticks.astype(int))

    plt.tight_layout()
    savefig(f, title)
    plt.close('all')


def process_full_rms(data: np.ndarray, sample_rate, tag: str, rms_window: int=1, data_dict: Dict[str, Any]=None) -> dict:
    # RMS inspection
    full_rms, full_rms_times = get_windowed_rms(data, sample_rate=sample_rate, rms_window=rms_window,
                                                skip_window=rms_window)
    full_median, full_q1, full_q9 = get_rms_quantiles(full_rms)
    # convert rms to dB
    full_db = 20 * np.log10(full_rms / full_median)

    plot_rms(full_rms, title='{} full RMS'.format(tag))
    plot_rms(full_db, title='{} full RMS dB'.format(tag))

    if data_dict is None:
        data_dict = {}
    data_dict['{}_full_rms'.format(tag)] = full_rms
    data_dict['{}_full_db'.format(tag)] = full_db
    data_dict['{}_full_median'.format(tag)] = full_median
    data_dict['{}_full_q1'.format(tag)] = full_q1
    data_dict['{}_full_q9'.format(tag)] = full_q9
    data_dict['{}_full_rms_times'.format(tag)] = full_rms_times
    return data_dict


def process_reduced_rms(data: np.ndarray, sample_rate, tag: str, rms_window: int=1, skip_window: int=300,
                        data_dict: Dict[str, Any]=None) -> dict:
    # RMS inspection
    reduced_rms, windowed_times = get_windowed_rms(data, sample_rate=sample_rate, rms_window=rms_window,
                                                   skip_window=skip_window)
    reduced_median, reduced_q1, reduced_q9 = get_rms_quantiles(reduced_rms)

    # convert rms to dB
    reduced_db = 20 * np.log10(reduced_rms / reduced_median)

    plot_rms(reduced_rms, xticks=windowed_times, title='{} reduced RMS'.format(tag))
    plot_rms(reduced_db, xticks=windowed_times, title='{} reduced RMS dB'.format(tag))

    if data_dict is None:
        data_dict = {}
    data_dict['{}_reduced_rms'.format(tag)] = reduced_rms
    data_dict['{}_reduced_db'.format(tag)] = reduced_db
    data_dict['{}_reduced_median'.format(tag)] = reduced_median
    data_dict['{}_reduced_q1'.format(tag)] = reduced_q1
    data_dict['{}_reduced_q9'.format(tag)] = reduced_q9
    data_dict['{}_reduced_rms_times'.format(tag)] = windowed_times
    return data_dict


def read_binary(bin_file: Path):
    metadata, samplerate, shape = read_metadata(bin_file)
    session_length = float(metadata['fileTimeSecs'])
    print("Session length is {} seconds, or {} minutes and {} seconds".format(session_length,
                                                                              session_length // 60,
                                                                              session_length % 60))
    data = sglx.makeMemMapRaw(bin_file, metadata)
    return data, metadata, samplerate, shape


def plot_PSD(freq: np.ndarray, psd: np.ndarray, title: str=''):
    f, ax = plt.subplots()
    sns.heatmap(data=psd, cbar=True, norm=LogNorm())
    ax.set_xticks(np.arange(0, psd.shape[1], 100), freq[::100].astype(int))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Channel')
    ax.set_title(title)
    plt.tight_layout()

    plt.tight_layout()
    savefig(f, title)
    plt.close('all')


def process_PSD(data: np.ndarray, sample_rate: int, tag: str, nperseg=1024, data_dict: Dict[str, Any]=None) -> Dict[str, Any]:
    freq, psd = sp.signal.welch(data, sample_rate, nperseg=nperseg, axis=1)  # choose npersg to be a power of 2; 1024 or 2048 are good choices just because
    if data_dict is None:
        data_dict = {}
    data_dict['{}_freq'.format(tag)] = freq
    data_dict['{}_psd'.format(tag)] = psd
    plot_PSD(freq, psd, title='{} PSD'.format(tag))
    return data_dict


def main():
    # set data paths
    data_root = Path.home().joinpath('Documents', 'testdata')
    ibl_bin = data_root.joinpath('IBL_testdata', 'imec_385_100s.ap.bin')
    rec_ap_bin = data_root.joinpath(
        'HD015_11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0/HD015_11.30.2023_g0_t0.imec0.ap.bin')
    rec_lfp_bin = data_root.joinpath(
        'HD015_11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0/HD015_11.30.2023_g0_t0.imec0.lf.bin')

    rec_tag = 'HD015_11.30.2023'
    ibl_tag = 'IBL_testdata'

    tag = rec_tag

    # read in data
    # ap_data, ap_meta, ap_srate, ap_shape = read_binary(ibl_bin)
    ap_data, ap_meta, ap_srate, ap_shape = read_binary(rec_ap_bin)
    lfp_data, lfp_meta, lfp_srate, lfp_shape = read_binary(rec_lfp_bin)

    # RMS inspection
    window = 1
    skip = 30

    data_dict = load_inspection_data('2023-12-15', 'HD015_11.30.2023')
    # data_dict = {}

    reduced_rms = True
    full_rms = True
    if reduced_rms:
        data_dict = process_reduced_rms(ap_data, ap_srate, tag='AP', rms_window=window, skip_window=skip)
        data_dict = process_reduced_rms(lfp_data, lfp_srate, tag='LFP', rms_window=window, skip_window=skip, data_dict=data_dict)

    if full_rms:
        data_dict = process_full_rms(ap_data, ap_srate, tag='AP', rms_window=window, data_dict=data_dict)
        data_dict = process_full_rms(lfp_data, lfp_srate, tag='LFP', rms_window=window, data_dict=data_dict)

    # PSD inspection
    PSD = False
    if PSD:
        print("Calculating PSDs...")
        # data_dict = process_PSD(ap_data, ap_srate, tag='AP', nperseg=1024, data_dict=data_dict)
        data_dict = process_PSD(lfp_data, lfp_srate, tag='LFP', nperseg=1024, data_dict=data_dict)

    save_data(data_dict, '{}'.format(tag))


if __name__ == '__main__':
    main()

