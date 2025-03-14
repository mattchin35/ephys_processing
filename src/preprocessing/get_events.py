import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Iterable
from pathlib import Path
import src.SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX as readSGLX
# will use readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital, GainCorrectIM, GainCorrectNI
from scipy.signal import butter, lfilter
from icecream import ic
from functools import partial
import spikeinterface.full as si
import rms
import subset_helpers as sh


def get_event_ix(data: np.ndarray, threshold: float = 0.5) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect simple threshold crossings in the data.
    :param data: data array
    :param threshold: threshold value
    :return: threshold crossing array
    """
    crossings = np.diff(data > threshold, prepend=False)
    crossing_ix = np.nonzero(crossings)[0]
    positive_crossings = crossing_ix[data[crossing_ix] > threshold]
    negative_crossings = crossing_ix[data[crossing_ix] < threshold]
    return crossing_ix, positive_crossings, negative_crossings


def run_file(binaryFilePath: str, digitalWord: int, digitalLines: list[int]) -> [np.ndarray, np.ndarray, np.ndarray]:
    meta = readSGLX.readMeta(binaryFilePath)
    sampleRate = readSGLX.SampRate(meta)
    nChan = int(meta['nSavedChans'])
    nSamples = int(int(meta['fileSizeBytes']) / (2 * nChan))

    firstSamp = 0
    lastSamp = nSamples - 1  # sample ix is 0-indexed but ExtracDigital is inclusive

    # firstSamp = int(sRate * tStart)
    # lastSamp = int(sampleRate * 7) - 1
    rawData = readSGLX.makeMemMapRaw(binaryFilePath, meta)
    digArray = readSGLX.ExtractDigital(rawData, firstSamp, lastSamp, digitalWord, digitalLines, meta)

    # plot the digital data
    # plt.plot(np.arange(lastSamp-firstSamp+1), np.squeeze(digArray))
    # plt.show()

    # get the event indices
    event_ix, pos_ix, neg_ix = get_event_ix(np.squeeze(digArray))
    return event_ix, pos_ix, neg_ix


def main_unprocessed():
    # set data paths
    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    # run imec file
    recording_path = raw_data_root.joinpath('CT009_current_20250301/run1_g0')
    imec_file = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]

    imec_ix, imec_pos, imec_neg = run_file(binaryFilePath=imec_file, digitalWord=imec_word, digitalLines=imec_line)
    ic(imec_ix, imec_pos, imec_neg)

    # run ni file
    ni_file = recording_path.joinpath('run1_g0_t0.nidq.bin')
    ni_word = 0
    ni_lines = [0,1,2,3]

    ni_sync_ix, ni_sync_pos, ni_sync_neg = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[0])
    ni_flip_ix, ni_flip_pos, ni_flip_neg = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[1])
    ni_L_ix, ni_L_pos, ni_L_neg = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[2])
    ni_R_ix, ni_R_pos, ni_R_neg = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[3])


    # recording_path = processed_data_root.joinpath('CT009_current_20250302_filter-gfix/catgt_run1_g0/run1_g0_imec0')
    # save_path = processed_data_root.joinpath('CT009_current_20250302_hipassfilter-spatialfilter')


if __name__ == '__main__':
    main_unprocessed()
