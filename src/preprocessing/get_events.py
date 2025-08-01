import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Iterable
from pathlib import Path
import src.SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX as readSGLX
# will use readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital, GainCorrectIM, GainCorrectNI
from icecream import ic
from functools import partial
import spikeinterface.full as si
import sync_lines as sync
from icecream import ic
import scipy as sp


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


def get_events(data: np.ndarray, sample_rate: float, threshold: float = 0.5, debounce=.0002):
    crossings = np.diff(data > threshold, prepend=False)
    crossing_ix = np.nonzero(crossings)[0]
    if len(crossing_ix) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    positive_crossings_ix = crossing_ix[data[crossing_ix] > threshold]
    negative_crossings_ix = crossing_ix[data[crossing_ix] < threshold]
    # ic(positive_crossings_ix / sample_rate, negative_crossings_ix / sample_rate)
    pos_ix, pos_t = get_positive_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    neg_ix, neg_t = get_negative_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    return pos_ix, pos_t, neg_ix, neg_t


def get_positive_crossings(pos_ix: np.ndarray, neg_ix: np.ndarray, sample_rate: float, debounce: float) -> [np.ndarray, np.ndarray]:
    neg_ix = neg_ix[neg_ix > pos_ix[0]]
    pos_t, neg_t = pos_ix / sample_rate, neg_ix / sample_rate
    for p, n in zip(pos_t, neg_t):
        if n - p < debounce:
            pos_ix = pos_ix[pos_t != p]
            pos_t = pos_t[pos_t != p]
            neg_ix = neg_ix[neg_t != n]
            neg_t = neg_t[neg_t != n]
    return pos_ix, pos_t


def get_negative_crossings(pos_ix: np.ndarray, neg_ix: np.ndarray, sample_rate: float, debounce=float) -> [np.ndarray, np.ndarray]:
    pos_ix = pos_ix[pos_ix > neg_ix[0]]
    pos_t, neg_t = pos_ix / sample_rate, neg_ix / sample_rate
    for p, n in zip(pos_t, neg_t):
        if p - n < debounce:
            pos_ix = pos_ix[pos_t != p]
            pos_t = pos_t[pos_t != p]
            neg_ix = neg_ix[neg_t != n]
            neg_t = neg_t[neg_t != n]
    return neg_ix, neg_t


def run_file(binaryFilePath: str, digitalWord: int, digitalLines: list[int], tag: str=None, debounce=.0002, plot=False) -> [np.ndarray, np.ndarray, np.ndarray]:
    meta = readSGLX.readMeta(binaryFilePath)
    sampleRate = readSGLX.SampRate(meta)
    nChan = int(meta['nSavedChans'])
    nSamples = int(int(meta['fileSizeBytes']) / (2 * nChan))

    firstSamp = 0
    lastSamp = nSamples - 1  # sample ix is 0-indexed but ExtracDigital is inclusive

    # firstSamp = int(sampleRate * 6*60)
    # lastSamp = int(sampleRate * (6*60 + 10)) - 1
    rawData = readSGLX.makeMemMapRaw(binaryFilePath, meta)
    digArray = readSGLX.ExtractDigital(rawData, firstSamp, lastSamp, digitalWord, digitalLines, meta)
    digArray = np.squeeze(digArray)

    pos_ix, pos_t, neg_ix, neg_t = get_events(digArray, sampleRate, debounce=debounce)
    event_ix, event_t = np.sort(np.concatenate((pos_ix, neg_ix))), np.sort(np.concatenate((pos_t, neg_t)))

    if plot:
        plotFirstSamp = int((6*60+4) * sampleRate)
        plotLastSamp = int(sampleRate * 1) + plotFirstSamp -1
        plt.plot(np.arange(plotLastSamp-plotFirstSamp+1), digArray[plotFirstSamp:plotLastSamp+1])
        plt.xticks(np.linspace(0, plotLastSamp-plotFirstSamp, 6), np.round(np.linspace(plotFirstSamp/sampleRate, plotLastSamp/sampleRate, 6), 2))
        events_window = event_ix[(event_ix > plotFirstSamp) & (event_ix < plotLastSamp)]
        ic(events_window / sampleRate)
        if tag:
            plt.title(tag)
        plt.show()

    return (event_ix, event_t), (pos_ix, pos_t), (neg_ix, neg_t), sampleRate


def sync_for_demonstration(imec_file: Path, ni_file: Path, debounce=.0002) -> [np.ndarray, np.ndarray]:
    """
    Synchronize one probe and NI file. Extracts sync lines, flipper, and left/right events.
    :param imec_file:
    :param ni_file:
    :param debounce:
    :return:
    """
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]
    imec_events, imec_pos, imec_neg, imec_srate = run_file(binaryFilePath=imec_file, digitalWord=imec_word, digitalLines=imec_line, tag='imec')

    # run ni file
    ni_word = 0
    ni_ephys, ni_ephys_pos, ni_ephys_neg, ni_srate = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[0], tag='ni_ephys')
    ni_flip, ni_flip_pos, ni_flip_neg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[1], tag='ni_flip')
    ni_Levents, ni_Lpos, ni_Lneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[2], tag='ni_L', plot=False, debounce=debounce)
    ni_Revents, ni_Rpos, ni_Rneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[3], tag='ni_R', plot=False, debounce=debounce)

    imec_synclines = {'imec_ix': imec_pos[0], 'imec_t': imec_pos[1]}
    imec = sync.DAQ(sync_lines=imec_synclines, sample_rate=imec_srate)

    ni_synclines = {'ephys_ix': ni_ephys_pos[0], 'ephys_t': ni_ephys_pos[1], 'flipper_ix': ni_flip_pos[0], 'flipper_t': ni_flip_pos[1]}
    ni = sync.DAQ(sync_lines=ni_synclines, sample_rate=ni_srate)

    ni.sync(sync_from=ni.sync_lines['ephys_ix'], sync_to=imec.sync_lines['imec_ix'], desc='ephys_ix')
    ni.sync(sync_from=ni.sync_lines['ephys_t'], sync_to=imec.sync_lines['imec_t'], desc='ephys_t')

    # ni.sync_data(description='flipper_ix', data=ni_flip[0])
    Lsync = ni.sync_data(description='ephys_ix', data=ni_Levents[0])
    Rsync = ni.sync_data(description='ephys_ix', data=ni_Revents[0])
    all_events = np.sort(np.concatenate((Lsync, Rsync)))
    Lsync = ni.sync_data(description='ephys_ix', data=ni_Lpos[0])
    Rsync = ni.sync_data(description='ephys_ix', data=ni_Rpos[0])
    all_offsets = np.sort(np.concatenate((Lsync, Rsync)))
    Lsync = ni.sync_data(description='ephys_ix', data=ni_Lneg[0])
    Rsync = ni.sync_data(description='ephys_ix', data=ni_Rneg[0])
    all_onsets = np.sort(np.concatenate((Lsync, Rsync)))
    return all_events, all_offsets, all_onsets


def sync_behavior(imec_file: Path, ni_file: Path, debounce=.0002) -> [np.ndarray, np.ndarray]:
    """
    Synchronize NI file to a probe. Extracts sync lines, flipper, and left/right events.
    :param imec_file:
    :param ni_file:
    :param debounce:
    :return:
    """
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]
    imec_events, imec_pos, imec_neg, imec_srate = run_file(binaryFilePath=imec_file, digitalWord=imec_word,
                                                           digitalLines=imec_line, tag='imec')

    # run ni file
    ni_word = 0
    ni_ephys, ni_ephys_pos, ni_ephys_neg, ni_srate = run_file(binaryFilePath=ni_file, digitalWord=ni_word,
                                                              digitalLines=[0], tag='ni_ephys')
    ni_flip, ni_flip_pos, ni_flip_neg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[1],
                                                    tag='ni_flip')
    ni_Levents, ni_Lpos, ni_Lneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[2],
                                               tag='ni_L', plot=False, debounce=debounce)
    ni_Revents, ni_Rpos, ni_Rneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[3],
                                               tag='ni_R', plot=False, debounce=debounce)

    imec_synclines = {'imec_ix': imec_pos[0], 'imec_t': imec_pos[1]}
    imec = sync.DAQ(sync_lines=imec_synclines, sample_rate=imec_srate)

    ni_synclines = {'ephys_ix': ni_ephys_pos[0], 'ephys_t': ni_ephys_pos[1], 'flipper_ix': ni_flip_pos[0],
                    'flipper_t': ni_flip_pos[1]}
    ni = sync.DAQ(sync_lines=ni_synclines, sample_rate=ni_srate)

    ni.sync(sync_from=ni.sync_lines['ephys_ix'], sync_to=imec.sync_lines['imec_ix'], desc='ephys_ix')
    ni.sync(sync_from=ni.sync_lines['ephys_t'], sync_to=imec.sync_lines['imec_t'], desc='ephys_t')

    # synchronize flipper events
    flipper_ix = ni.sync_data(description='ephys_ix', data=ni_flip_pos[0])
    flipper_t = ni.ix_to_time(flipper_ix)

    # synchronize left/right events
    Lonset_ix = ni.sync_data(description='ephys_ix', data=ni_Lneg[0])
    Ronset_ix = ni.sync_data(description='ephys_ix', data=ni_Rneg[0])
    Loffset_ix = ni.sync_data(description='ephys_ix', data=ni_Lpos[0])
    Roffset_ix = ni.sync_data(description='ephys_ix', data=ni_Rpos[0])

    Lonset_t = ni.ix_to_time(Lonset_ix)
    Ronset_t = ni.ix_to_time(Ronset_ix)
    Loffset_t = ni.ix_to_time(Loffset_ix)
    Roffset_t = ni.ix_to_time(Roffset_ix)

    ix_sync_dict = dict(flipper_ix=flipper_ix, L_onset_ix=Lonset_ix, R_onset_ix=Ronset_ix,
                        L_offset_ix=Loffset_ix, R_offset_ix=Roffset_ix)
    t_sync_dict = dict(flipper_t=flipper_t, L_onset_t=Lonset_t, R_onset_t=Ronset_t,
                       L_offset_t=Loffset_t, R_offset_t=Roffset_t)

    return ix_sync_dict, t_sync_dict#, imec, ni


def get_events2(data: np.ndarray, sample_rate: float, threshold: float = 0.5, debounce=.0002):
    crossings = np.diff(data > threshold, prepend=False)
    crossing_ix = np.nonzero(crossings)[0]
    crossing_t = crossing_ix / sample_rate
    if len(crossing_ix) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    positive_crossings_ix = crossing_ix[data[crossing_ix] > threshold]
    negative_crossings_ix = crossing_ix[data[crossing_ix] < threshold]
    # ic(positive_crossings_ix / sample_rate, negative_crossings_ix / sample_rate)
    pos_ix, pos_t = get_positive_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    neg_ix, neg_t = get_negative_crossings(positive_crossings_ix, negative_crossings_ix, sample_rate, debounce)
    events = dict(crossing_ix=crossing_ix, crossing_t=crossing_t,
                  pos_ix=pos_ix, pos_t=pos_t,
                  neg_ix=neg_ix, neg_t=neg_t)
    return events


def decode_flipper(flipper_signal: np.ndarray, sample_rate: float=25000, barcode_present=True):
    nbits = 32
    wrapper_bit_time = 10  # milliseconds
    barcode_bit_time = 30  # milliseconds

    wrapper_time = 3 * wrapper_bit_time  # Off-On-Off
    barcode_time = nbits * barcode_bit_time  # 32 bits
    total_barcode_time = barcode_time + 2 * wrapper_time

    # Tolerance conversions
    tolerance = .2 # % tolerance - so for a duration of 10 and 10% tolerance, 9-11 is acceptable
    min_wrap_duration = wrapper_bit_time - wrapper_bit_time * tolerance
    max_wrap_duration = wrapper_bit_time + wrapper_bit_time * tolerance
    min_bar_duration = barcode_bit_time - barcode_bit_time * tolerance
    max_bar_duration = barcode_bit_time + barcode_bit_time * tolerance
    # sample_conversion = 1000 / expected_sample_rate  # Convert sampling rate to msec

    events = get_events(flipper_signal, sample_rate)
    # 1. remove noise events (anything under 2 ms)
    noise_ix = []
    i = 0
    while i < len(events['crossing_ix']) - 1:
        if (events['crossing_t'][i+1] - events['crossing_t'][i]) < 2:
            noise_ix.append(i)
            noise_ix.append(i+1)
            i += 2

    positive_crossings_ix = crossing_ix[flipper_signal[crossing_ix] > .5]
    negative_crossings_ix = crossing_ix[flipper_signal[crossing_ix] < .5]

    # 2. determine the wrapper event times
    wrapper_count = 0
    wrapper_ix = []
    wrapper_t = []
    pulse_time_match = min_wrap_duration < events['crossing_t'][1:] - events['crossing_t'][:-1] < max_wrap_duration
    pulse_shape_match = [flipper_signal[ix_st >.5] and flipper_signal[ix_end > .5]
                         for ix_st, ix_end in zip(events['crossing_ix'][:-1], events['crossing_ix'][1:])]
    for ix, (time, shape) in enumerate(zip(pulse_time_match, pulse_shape_match)):
        if time and shape:
            wrapper_ix.append(ix)
            wrapper_t.append(events['crossing_t'][ix])
    assert len(wrapper_ix) == 4, "File is missing barcodes or barcode wrappers"

    barcode_st = [wrapper_t[0] + max_wrap_duration, wrapper_t[2] + max_wrap_duration]
    barcode_end = [wrapper_t[1] - max_wrap_duration, wrapper_t[3] - max_wrap_duration]

    signals_barcodes = []
    for start, end in zip(barcode_st, barcode_end):
        oncode = on_times[
            np.where(
                np.logical_and(on_times > start_time,
                               on_times < start_time + total_barcode_duration)
            )[0]
        ]
        offcode = off_times[
            np.where(
                np.logical_and(off_times > start_time,
                               off_times < start_time + total_barcode_duration)
            )[0]
        ]
        curr_time = offcode[0] + ind_wrap_duration  # Jumps ahead to start of barcode
        bits = np.zeros((nbits,))
        interbit_ON = False  # Changes to "True" during multiple ON bars

        for bit in range(0, nbits):
            next_on = np.where(oncode >= (curr_time - ind_bar_duration * global_tolerance))[0]
            next_off = np.where(offcode >= (curr_time - ind_bar_duration * global_tolerance))[0]

            if next_on.size > 1:  # Don't include the ending wrapper
                next_on = oncode[next_on[0]]
            else:
                next_on = start_time + inter_barcode_interval

            if next_off.size > 1:  # Don't include the ending wrapper
                next_off = offcode[next_off[0]]
            else:
                next_off = start_time + inter_barcode_interval

            # Recalculate min/max bar duration around curr_time
            min_bar_duration = curr_time - ind_bar_duration * global_tolerance
            max_bar_duration = curr_time + ind_bar_duration * global_tolerance

            if min_bar_duration <= next_on <= max_bar_duration:
                bits[bit] = 1
                interbit_ON = True
            elif min_bar_duration <= next_off <= max_bar_duration:
                interbit_ON = False
            elif interbit_ON == True:
                bits[bit] = 1

            curr_time += ind_bar_duration

        barcode = 0

        for bit in range(0, nbits):  # least sig left
            barcode += bits[bit] * pow(2, bit)

        signals_barcodes.append(barcode)


    # 3. determine and decode the barcode events
    # 4. obtain flipper timestamps for syncing

    if barcode_present:
        pass


def main():
    # set data paths
    raw_data_root = Path.home().joinpath('Documents', 'ephys_transfer')
    processed_data_root = Path.home().joinpath('Documents', 'processed_ephys')

    # run imec file
    recording_path = raw_data_root.joinpath('CT009_current_20250302/run1_g0')
    imec_file = recording_path.joinpath('run1_g0_imec0/run1_g0_t0.imec0.ap.bin')
    imec_word = 0
    imec_line = [6]  # [0, 1, 6]

    imec_events, imec_pos, imec_neg, imec_srate = run_file(binaryFilePath=imec_file, digitalWord=imec_word, digitalLines=imec_line, tag='imec')
    ic(imec_events[1])

    # run ni file
    ni_file = recording_path.joinpath('run1_g0_t0.nidq.bin')
    ni_word = 0
    ni_lines = [0,1,2,3]

    ni_ephys, ni_ephys_pos, ni_ephys_neg, ni_srate = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[0], tag='ni_ephys')
    ni_flip, ni_flip_pos, ni_flip_neg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[1], tag='ni_flip')
    ni_Levents, ni_Lpos, ni_Lneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[2], tag='ni_L')
    ni_Revents, ni_Rpos, ni_Rneg, _ = run_file(binaryFilePath=ni_file, digitalWord=ni_word, digitalLines=[3], tag='ni_R')
    ic(ni_Lneg[1])
    ic(ni_Rneg[1])

    imec_synclines = {'imec_ix': imec_pos[0], 'imec_t': imec_pos[1]}
    imec = sync.DAQ(sync_lines=imec_synclines, sample_rate=imec_srate)

    ni_synclines = {'ephys_ix': ni_ephys_pos[0], 'ephys_t': ni_ephys_pos[1], 'flipper_ix': ni_flip_pos[0], 'flipper_t': ni_flip_pos[1]}
    ni = sync.DAQ(sync_lines=ni_synclines, sample_rate=ni_srate)

    ni.sync(sync_from=ni.sync_lines['ephys_ix'], sync_to=imec.sync_lines['imec_ix'], desc='ephys_ix')
    ni.sync(sync_from=ni.sync_lines['ephys_t'], sync_to=imec.sync_lines['imec_t'], desc='ephys_t')
    Lsync = ni.sync_data(description='ephys_ix', data=ni_Levents[0])
    Rsync = ni.sync_data(description='ephys_ix', data=ni_Revents[0])

    # ic(imec_pos, ni)
    ic(Lsync.astype(np.int32), Rsync)


if __name__ == '__main__':
    main()
