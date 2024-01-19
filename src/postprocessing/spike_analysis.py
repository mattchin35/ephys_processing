import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from icecream import ic
import pandas as pd
import src.visualization.save as save
import seaborn as sns
sns.set_theme(style='white')
from src.visualization import save
from typing import Tuple, Dict, List, Union

np.random.seed(0)


def load_data(data_folder: Path) -> Tuple[np.ndarray, np.ndarray]:
    spike_times = np.squeeze(np.load(data_folder.joinpath('spike_times.npy')))  # what are these times in reference to? load the session length as well
    spike_clusters = np.squeeze(np.load(data_folder.joinpath('spike_clusters.npy')))
    return spike_times, spike_clusters


def simulate_data(T: int = 1000, nspikes: int = 1000, nclusters: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    spike_times = np.random.randint(0, T, nspikes)
    spike_clusters = np.random.randint(0, nclusters, nspikes)
    return spike_times, spike_clusters


def bin_spikes(spike_clusters: np.ndarray, spike_times: np.ndarray, T: int = 1000, bin_size: int = 100) -> pd.DataFrame:
    df = pd.DataFrame({'spike_times': spike_times, 'spike_clusters': spike_clusters})
    bin_edges = np.arange(0, T + bin_size, bin_size)
    binned_spikes = pd.crosstab(index=df['spike_clusters'], columns=pd.cut(df['spike_times'], bin_edges))
    # binned_spikes = df.pivot_table(index=df['spike_clusters'], columns=pd.cut(df['spike_times'], bin_edges), aggfunc='size')
    return binned_spikes


def plot_raster(groups: Dict) -> None:
    # Recall that you're about to plot units, not channels or locations
    # will need to relate these to the channel map
    f, ax = plt.subplots()
    plt.eventplot(list(groups.values()), orientation='horizontal', color='k')
    plt.tight_layout()
    save.savefig(f, 'analysis', 'test_raster')
    plt.close('all')


def plot_binned_spikes(spikes: pd.DataFrame) -> None:
    f, ax = plt.subplots()
    [plt.plot(spikes.values[i]) for i in range(len(spikes))]
    plt.tight_layout()
    save.savefig(f, 'analysis', 'test_binned_spikes')
    plt.close('all')


def main():
    # data_folder = Path.home().joinpath('Documents', 'testdata', 'HD015_11302023', 'kilosort3')
    data_folder = Path.home().joinpath('Documents/testdata/HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0/kilosort3')
    # spike_times, spike_clusters = load_data(data_folder)

    spike_times, spike_clusters = simulate_data()
    clustered_spikes = pd.DataFrame({'spike_times': spike_times, 'spike_clusters': spike_clusters}).groupby('spike_clusters')
    clustered_spikes = clustered_spikes.groups
    binned_spikes = bin_spikes(spike_clusters, spike_times)
    cluster_spikesums = binned_spikes.sum(axis=1)

    ic(spike_times.shape)
    ic(spike_clusters.shape)
    ic(cluster_spikesums)

    plot_raster(clustered_spikes)
    plot_binned_spikes(binned_spikes)

    # now plot the binned spikes


if __name__ == '__main__':
    main()
