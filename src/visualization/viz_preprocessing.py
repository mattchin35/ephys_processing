import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any
import datetime as dt
import src.visualization.save as save
sns.set_theme(style='white')


def plot_PSD(freq: np.ndarray, psd: np.ndarray, title: str=''):
    f, ax = plt.subplots()
    sns.heatmap(data=psd, cbar=True, norm=LogNorm())
    ax.set_xticks(np.arange(0, psd.shape[1], 100), freq[::100].astype(int))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Channel')
    ax.set_title(title)
    plt.tight_layout()

    plt.tight_layout()
    save.savefig(f,processing_step='preprocessing', name=title)
    plt.close('all')


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
    save.savefig(f, 'preprocessing', title)
    plt.close('all')


def plot_IBL_metrics_NP1(data_dict: Dict[str, Any], tag: str) -> None:
    keys = ['AP_full_rms', 'AP_full_db', 'LFP_full_rms', 'LFP_full_db',
            'AP_reduced_rms', 'AP_reduced_db', 'LFP_reduced_rms', 'LFP_reduced_db']
    for key in keys:
        if key in data_dict.keys():
            dtype, size, _ = key.split('_')
            if size == 'reduced':
                xticks = data_dict['AP_reduced_rms_times']
            else:
                xticks = None

            plot_rms(data_dict[key], xticks=xticks, title=tag + ' ' + key.replace('_', ' '))


def plot_IBL_metrics(data_dict: Dict[str, Any], tag: str) -> None:
    keys = ['AP_full_rms', 'AP_full_db', 'AP_reduced_rms', 'AP_reduced_db']
    for key in keys:
        if key in data_dict.keys():
            dtype, size, _ = key.split('_')
            if size == 'reduced':
                xticks = data_dict['AP_reduced_rms_times']
            else:
                xticks = None

            plot_rms(data_dict[key], xticks=xticks, title=tag + ' ' + key.replace('_', ' '))


def plot_sample_data(data: np.ndarray, t1_ix: int, t2_ix: int, sample_freq: float, tag: str, processing_step: str) -> None:
    f, ax = plt.subplots()
    subdata = data[:5, t1_ix:t2_ix]
    subdata = sp.stats.zscore(subdata, axis=1)
    for i in range(subdata.shape[0]):
        ax.plot(subdata[i] + i * 10, 'k', linewidth=0.5)
        # ax.plot(subdata[i], 'k')

    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    ax.set_title(tag)
    xlabels = np.linspace(t1_ix / sample_freq, t2_ix / sample_freq, 5)
    xlabels = np.around(xlabels, decimals=2)
    ax.set_xticks(ticks=np.linspace(0, t2_ix-t1_ix, 5), labels=xlabels)
    plt.xlim(0, t2_ix-t1_ix)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticklabels([])
    # ax.set_yticks(np.arange(0, nchannels, 10))
    plt.tight_layout()
    save.savefig(f, processing_step=processing_step, name='{}_sample_data'.format(tag))
    plt.close('all')

    print("Saved sample data plot as {}_sample_data.png".format(tag))
