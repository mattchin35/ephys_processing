import mountainsort5 as ms5
from mountainsort5.util import load_binary_recording, save_binary_recording
import spikeinterface.full as si
import get_mountainsort_params as ms5_params
from pathlib import Path
import os
import shutil

# Constants
scheme = 2
save_rec = False
sort = True
extract_waveforms = True
delete_cached_recording = False  # delete the cached recording after sorting and extracting waveforms

# Path definitions
run = 0
gate = 0
probe = 0
session_name = 'CT011_20250624_catgt'

# work PC
imec_path = '/home/matt/Documents/preprocessed_ephys/{2}/catgt_run{0}_g{1}/run{0}_g{1}_imec{3}'.format(run, gate, session_name, probe)

# HPC
tmp_recording_path = imec_path + '/spikeinterface_preprocessed'
sorter_path = imec_path + '/mountainsort5'
waveforms_path = sorter_path

# waveforms_path = '/home/matt/Documents/sorted_ephys/mtnsort_waveforms/{0}/imec{1}'.format(session_name, probe)

# Preprocess recording and save
if save_rec:
    # Preprocess
    recording = si.read_spikeglx(imec_path, stream_id="imec0.ap", load_sync_channel=False)

    bad_channel_ids, channel_labels = si.detect_bad_channels(recording)
    recording = recording.remove_channels(bad_channel_ids)
    print("Bad channels: ", bad_channel_ids)

    # motion correction is one line of code
    rec_motion_corrected = si.correct_motion(recording=recording, preset="nonrigid_accurate")

    # "whiten" the recording prior to spike sorting
    rec_whitened = si.whiten(rec_motion_corrected, dtype="float32")

    # Save recording
    print("    + Preprocessing complete...")
    save_binary_recording(recording, tmp_recording_path, n_jobs=64)
    del recording
    print("    + Recording saved...")


# Load cached recording no-matter-what
cached_recording = load_binary_recording(tmp_recording_path)
assert isinstance(cached_recording, si.BaseRecording)
print("    + Cached recording loaded...")

if sort:
    sorting_params = {}
    ms5_params.make_mountainsort_params(settings='low_memory')  # 'default', 'low_memory', 'preliminary'
    # use low_memory or preliminary on work PC to avoid memory crashes, default on server

    # sorting_params["max_num_snippets_per_training_batch"] = 500  # default 1000, 500 for speed/less memory
    # sorting_params["snippet_mask_radius"] = 60  # default 60, 30 for speed/less memory
    # sorting_params["phase1_npca_per_channel"] = 3
    # sorting_params["phase1_npca_per_subdivision"] = 10  # default 10, 3 for speed/less memory
    # sorting_params["classifier_npca"] = 10  # default 10, 3 for speed/less memory
    # sorting_params["detect_channel_radius"] = 60  # default 60, 30 for speed/less memory
    # sorting_params["phase1_detect_channel_radius"] = 60  # default 60, 30 for speed/less memory
    # sorting_params["training_recording_sampling_mode"] = "uniform"
    # sorting_params["training_duration_sec"] = 350  # default 350, 150 for speed/less memory
    # sorting_params["phase1_detect_threshold"] = 6.5  # default 5.5, 6.0-7.0 for speed/less memory
    # sorting_params["detect_threshold"] = 6.0  # default 5.25, 6.0-7.0 for speed/less memory
    # sorting_params["snippet_T1"] = 15
    # sorting_params["snippet_T2"] = 40  # default 40, 35 for speed/less memory
    # sorting_params["detect_sign"] = 0  # default 0 (positve and negative spikes), -1 for speed/less memory (negative spikes only)
    # sorting_params["phase1_detect_time_radius_msec"] = 0.5
    # sorting_params["detect_time_radius_msec"] = 0.5
    # sorting_params["classification_chunk_sec"] = 100

    if scheme == 2:
        sorting = ms5.sorting_scheme2(
            recording=cached_recording,
            sorting_parameters=ms5.Scheme2SortingParameters(**sorting_params),
        )
        assert isinstance(sorting, si.BaseSorting)

    elif scheme == 3:
        sorting = ms5.sorting_scheme3(
            recording=cached_recording,
            sorting_parameters=ms5.Scheme3SortingParameters(
                block_sorting_parameters=ms5.Scheme2SortingParameters(**sorting_params),
                block_duration_sec=60 * 5,
            ),
        )
        assert isinstance(sorting, si.BaseSorting)
    else:
        raise ValueError(f"Scheme not supported: {scheme}")

    print("    + Sorting completed...")
    # max_num_snippets_per_training_batch=1000
    sorting.save_to_folder(sorter_path)

    print("    + Sorting saved...")

    if delete_cached_recording:
        shutil.rmtree(tmp_recording_path)


# Extract waveforms
if extract_waveforms:
    sorting = si.read_numpy_sorting_folder(sorter_path)
    print("    + Sorting loaded...")
    assert isinstance(sorting, si.BaseSorting)

    job_kwargs = dict(n_jobs=64, chunk_duration="5s", progress_bar=True)
    cached_recording.annotate(is_filtered=True)

    waveforms = si.extract_waveforms(
        cached_recording,
        sorting,
        folder=waveforms_path,
        max_spikes_per_unit=1500,
        dtype="float32",
        ms_before=0.5,
        ms_after=1.5,
    )

    locations = si.compute_unit_locations(waveforms)

# Load the results
waveforms = si.load_waveforms(waveforms_path)
sorting = waveforms.sorting
noise = si.compute_noise_levels(waveforms)
print("    + Waveforms extracted...")

sorting_analyzer = si.SortingAnalyzer(sorting, cached_recording)
# sorting_analyzer = si.SortingAnalyzer(sorting, cached_recording)

# the waveforms are sparse so it is faster to export to phy
sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)

# some computations are done before to control all options
sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
_ = sorting_analyzer.compute('spike_amplitudes')
_ = sorting_analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")

# the export process is fast because everything is pre-computed
si.export_to_phy(sorting_analyzer=sorting_analyzer, output_folder='path/to/phy_folder')
