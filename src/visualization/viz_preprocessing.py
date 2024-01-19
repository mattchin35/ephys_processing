import numpy as np
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
    save.savefig(f, title)
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


def plot_IBL_metrics(data_dict: Dict[str, Any], tag: str) -> None:
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


