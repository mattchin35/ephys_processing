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
import src.SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX as sglx

unprocessed_path = Path('../../data/unprocessed/')
preprocessed_path = Path('../../data/preprocessed/')
processed_path = Path('../../data/processed/')

raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

session_name = 'CT009_current_20250302'
# recording_path = raw_data_root.joinpath('{}/run1_g0'.format(session_name))  # for raw data
# imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')  # for raw data
recording_path = processed_data_root.joinpath('{}_filter-gfix/catgt_run1_g0'.format(session_name))  # for filtered data
imec_file_ap = recording_path.joinpath('run1_g0_imec0/run1_g0_tcat.imec0.ap.bin')  # for filtered data
ni_file = raw_data_root.joinpath('{}/run1_g0/run1_g0_t0.nidq.bin'.format(session_name))
tag = session_name

# read in data, assuming it's neuropixel data
ap_data, ap_meta, ap_srate, ap_shape = pio.read_binary(imec_file_ap)

