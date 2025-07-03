import dataclasses

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
    xticks = np.sort(np.concatenate([np.arange(0, psd.shape[1], 100), [60/freq[-1] * psd.shape[1]]]))
    xticklabels = np.sort(np.concatenate([freq[::100],[60]])).astype(int)

    # ax.set_xticks(np.arange(0, psd.shape[1], 100), freq[::100].astype(int))
    # ax.set_xticks(np.arange(0, psd.shape[1], 100), xticks.astype(int))
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_yticks([])
    ax.set_ylabel('Channels')
    ax.set_title(title)
    plt.tight_layout()
    ax.invert_yaxis()

    save.savefig(f,processing_step='preprocessing', name=title)
    plt.close('all')


def plot_heatmap(data: np.ndarray, xticks=None, yticks=None, title: str=''):
    f, ax = plt.subplots()
    ax = sns.heatmap(data=data, ax=ax, cbar=True, cbar_kws={'label': 'RMS (uV)'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel')
    if title:
        ax.set_title(title)
    if xticks is not None:
        # ax.set_xticks(np.arange(rms_data.shape[1]), xticks.astype(int))
        # ax.set_xticks(xticks, xticks.astype(int))
        ax.set_xticks(xticks['ticks'], xticks['labels'])
        plt.xticks(fontsize=10, rotation=90)
    if yticks is not None:
        # ax.set_yticks(np.arange(rms_data.shape[0]), yticks.astype(int))
        ax.set_yticks(yticks['ticks'], yticks['labels'])
    else:
        ax.set_yticks([])

    ax.invert_yaxis()
    plt.tight_layout()
    # plt.show()
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


def plot_sample_data(data: np.ndarray, t1_ix: int = 0, t2_ix: int = None, sample_freq: float = 30000, tag: str = '',
                     processing_step: str = 'undefined_step', event_times: list[np.ndarray] = None, comparison_data: np.ndarray=None, plot_step=10) -> None:
    if t2_ix is None:
        t2_ix = data.shape[1]

    f, ax = plt.subplots()
    subdata = data[:5, t1_ix:t2_ix]
    subdata = sp.stats.zscore(subdata, axis=1)
    for i in range(subdata.shape[0]):
        ax.plot(subdata[i] + i * plot_step, 'k', linewidth=0.5)
        # ax.plot(subdata[i], 'k')

    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    # ax.set_title(tag)
    # ax.set_title('Sample electrophysiology traces')
    xlabels = np.linspace(t1_ix / sample_freq, t2_ix / sample_freq, 5)
    xlabels = np.around(xlabels, decimals=2)
    ax.set_xticks(ticks=np.linspace(0, t2_ix-t1_ix, 5), labels=xlabels)
    plt.xlim(0, t2_ix-t1_ix)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticklabels([])

    if event_times is not None:
        for event_group in event_times:
            for t in event_group:
                # ax.axvline(t, color='r', linestyle='--', linewidth=1, alpha=0.7)
                ax.axvline(t, linestyle='--', linewidth=1, alpha=0.7)

    if comparison_data is not None:
        if comparison_data.ndim == 1:
            comparison_data = comparison_data[t1_ix:t2_ix]
            comparison_data = sp.stats.zscore(comparison_data)
            ax.plot(comparison_data + subdata.shape[0] * 10, 'r', linewidth=0.5)
        else:
            comparison_data = comparison_data[:, t1_ix:t2_ix]
            comparison_data = sp.stats.zscore(comparison_data, axis=1)
            for i in range(comparison_data.shape[0]):
                ax.plot(comparison_data[i] + (subdata.shape[0] + i) * plot_step, 'r', linewidth=0.5)

    # ax.set_yticks(np.arange(0, nchannels, 10))
    plt.tight_layout()
    save.savefig(f, processing_step=processing_step, name='{}_sample_data'.format(tag))
    plt.close('all')

    print("Saved sample data plot as {}_sample_data.png".format(tag))


def plot_onset_offset(data: np.ndarray, event_times: list[np.ndarray], t1_ix: int = 0, t2_ix: int = None, sample_freq: float = 30000, tag: str = '',
                     processing_step: str = 'undefined_step', comparison_data: np.ndarray=None, plot_step=10) -> None:
    if t2_ix is None:
        t2_ix = data.shape[1]

    f, ax = plt.subplots()
    subdata = data[:5, t1_ix:t2_ix]
    subdata = sp.stats.zscore(subdata, axis=1)
    for i in range(subdata.shape[0]):
        ax.plot(subdata[i] + i * plot_step, 'k', linewidth=0.5)
        # ax.plot(subdata[i], 'k')

    ax.set_xlabel('Time')
    ax.set_ylabel('Voltage')
    # ax.set_title(tag)
    xlabels = np.linspace(t1_ix / sample_freq, t2_ix / sample_freq, 5)
    xlabels = np.around(xlabels, decimals=2)
    ax.set_xticks(ticks=np.linspace(0, t2_ix-t1_ix, 5), labels=xlabels)
    plt.xlim(0, t2_ix-t1_ix)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticklabels([])

    for t in event_times[0]:
        ax.axvline(t, color='g', linestyle='--', linewidth=1, alpha=0.7)
    for t in event_times[1]:
        ax.axvline(t, color='r', linestyle='--', linewidth=1, alpha=0.7)

    if comparison_data is not None:
        if comparison_data.ndim == 1:
            comparison_data = comparison_data[t1_ix:t2_ix]
            comparison_data = sp.stats.zscore(comparison_data)
            ax.plot(comparison_data + subdata.shape[0] * 10, 'r', linewidth=0.5)
        else:
            comparison_data = comparison_data[:, t1_ix:t2_ix]
            comparison_data = sp.stats.zscore(comparison_data, axis=1)
            for i in range(comparison_data.shape[0]):
                ax.plot(comparison_data[i] + (subdata.shape[0] + i) * plot_step, 'r', linewidth=0.5)

    # ax.set_yticks(np.arange(0, nchannels, 10))
    plt.tight_layout()
    save.savefig(f, processing_step=processing_step, name='{}_sample_data'.format(tag))
    plt.close('all')

    print("Saved sample data plot as {}_sample_data.png".format(tag))


def plot_stats(stats_dict: dict, tag: str, processing_step: str) -> None:
    f, ax = plt.subplots()
    plt.scatter(stats_dict['median'], stats_dict['std'], s=1)
    plt.title('Median vs. Std')
    plt.tight_layout()
    save.savefig(f, processing_step=processing_step, name='{}_std-median'.format(tag))

    f, ax = plt.subplots()
    plt.hist(stats_dict['std'], bins=20)
    plt.title('Std histogram')
    plt.tight_layout()
    save.savefig(f, processing_step=processing_step, name='{}_std-hist'.format(tag))

    plt.close('all')

