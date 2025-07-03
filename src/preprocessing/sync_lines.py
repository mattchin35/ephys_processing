import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pathlib import Path


class DAQ:
    def __init__(self, sync_lines: dict, sample_rate: float):
        self.sync_lines = sync_lines
        self.sample_rate = sample_rate
        self.interp_fn = {}

    def sync(self, sync_from: np.ndarray, sync_to: np.ndarray, desc: str = ''):
        interp_fn = sp.interpolate.interp1d(sync_from, sync_to, kind='linear', fill_value='extrapolate')
        self.interp_fn[desc] = interp_fn

    def sync_data(self, description: str, data):
        sync_fn = self.interp_fn[description]
        sync_data = sync_fn(data)
        return sync_data

    def ix_to_time(self, ix: np.ndarray):
        return ix / self.sample_rate

    def time_to_ix(self, time: np.ndarray):
        return time * self.sample_rate
