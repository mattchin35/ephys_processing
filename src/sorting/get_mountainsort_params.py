
def make_mountainsort_params(settings: str='default') -> dict:
    assert settings in ['default', 'low_memory', 'preliminary']
    sorting_params = {}

    if settings == 'default':
        sorting_params["max_num_snippets_per_training_batch"] = 1000
        sorting_params["snippet_mask_radius"] = 60
        sorting_params["phase1_npca_per_channel"] = 3
        sorting_params["phase1_npca_per_subdivision"] = 10
        sorting_params["classifier_npca"] = 10
        sorting_params["detect_channel_radius"] = 60
        sorting_params["phase1_detect_channel_radius"] = 60
        sorting_params["training_recording_sampling_mode"] = "uniform"
        sorting_params["training_duration_sec"] = 350
        sorting_params["phase1_detect_threshold"] = 5.5
        sorting_params["detect_threshold"] = 5.25
        sorting_params["snippet_T1"] = 15
        sorting_params["snippet_T2"] = 40
        sorting_params["detect_sign"] = 0
        sorting_params["phase1_detect_time_radius_msec"] = 0.5
        sorting_params["detect_time_radius_msec"] = 0.5
        sorting_params["classification_chunk_sec"] = 100

    elif settings == 'low_memory':
        sorting_params["max_num_snippets_per_training_batch"] = 500  # default 1000, 500 for speed/less memory
        sorting_params["snippet_mask_radius"] = 60  # default 60, 30 for speed/less memory
        sorting_params["phase1_npca_per_channel"] = 3
        sorting_params["phase1_npca_per_subdivision"] = 10  # default 10, 3 for speed/less memory
        sorting_params["classifier_npca"] = 10  # default 10, 3 for speed/less memory
        sorting_params["detect_channel_radius"] = 60  # default 60, 30 for speed/less memory
        sorting_params["phase1_detect_channel_radius"] = 60  # default 60, 30 for speed/less memory
        sorting_params["training_recording_sampling_mode"] = "uniform"
        sorting_params["training_duration_sec"] = 350  # default 350, 150 for speed/less memory
        sorting_params["phase1_detect_threshold"] = 6.5  # default 5.5, 6.0-7.0 for speed/less memory
        sorting_params["detect_threshold"] = 6.0  # default 5.25, 6.0-7.0 for speed/less memory
        sorting_params["snippet_T1"] = 15
        sorting_params["snippet_T2"] = 40  # default 40, 35 for speed/less memory
        sorting_params["detect_sign"] = 0  # default 0 (positve and negative spikes), -1 for speed/less memory (negative spikes only)
        sorting_params["phase1_detect_time_radius_msec"] = 0.5
        sorting_params["detect_time_radius_msec"] = 0.5
        sorting_params["classification_chunk_sec"] = 100

    elif settings == 'preliminary':
        sorting_params["max_num_snippets_per_training_batch"] = 500
        sorting_params["snippet_mask_radius"] = 30
        sorting_params["phase1_npca_per_channel"] = 3
        sorting_params["phase1_npca_per_subdivision"] = 3
        sorting_params["classifier_npca"] = 3
        sorting_params["detect_channel_radius"] = 30
        sorting_params["phase1_detect_channel_radius"] = 30
        sorting_params["training_recording_sampling_mode"] = "uniform"
        sorting_params["training_duration_sec"] = 150
        sorting_params["phase1_detect_threshold"] = 7
        sorting_params["detect_threshold"] = 7
        sorting_params["snippet_T1"] = 15
        sorting_params["snippet_T2"] = 35
        sorting_params["detect_sign"] = -1
        sorting_params["phase1_detect_time_radius_msec"] = 0.5
        sorting_params["detect_time_radius_msec"] = 0.5
        sorting_params["classification_chunk_sec"] = 100

    return settings