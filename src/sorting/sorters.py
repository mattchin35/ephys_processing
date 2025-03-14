from pathlib import Path
from icecream import ic
import src.preprocessing.preprocess_io as pio
import spikeinterface.full as si
import spikeinterface.sorters as ss
import src.preprocessing.preprocessing as pp


def prep_spyking_params():
    spyking1_params = ss.get_default_sorter_params('spykingcircus')
    # ic(ss.get_default_sorter_params('spykingcircus'))
    # ic(ss.get_sorter_params_description('spykingcircus'))
    spyking1_params['apply_preprocessing'] = False
    spyking1_params['detect_sign'] = -1
    spyking1_params['adjacency_radius'] = 100
    spyking1_params['detect_threshold'] = 6
    spyking1_params['template_width_ms'] = 3
    spyking1_params['filter'] = True
    spyking1_params['merge_spikes'] = True
    spyking1_params['auto_merge'] = 0.75
    spyking1_params['num_workers'] = None
    spyking1_params['whitening_max_elts'] = 1000
    spyking1_params['clustering_max_elts'] = 10000
    return spyking1_params


def prep_KS2_params():
    # ic(KS2_params)
    KS2_params = ss.get_default_sorter_params('kilosort2')
    return KS2_params


def prep_KS2_5_params(kilosort2_5_path: str, default_params: bool=False):
    ss.Kilosort2_5Sorter.set_kilosort2_5_path(kilosort2_5_path)
    KS25_params = ss.get_default_sorter_params('kilosort2_5')
    if default_params:
        return KS25_params

    KS25_params['AUCsplit'] = 0.9  # default 0.8
    KS25_params['freq_min'] = 300  # default 150
    KS25_params['car'] = True  # default True
    KS25_params['detect_threshold'] = 6  # default 6
    KS25_params['do_correction'] = True  # default True
    KS25_params['minFR'] = 0.1  # default 0.1
    KS25_params['minfr_goodchannels'] = 0.1  # default 0.2
    KS25_params['nblocks'] = 5  # default 5
    KS25_params['preclust_threshold'] = 8  # this is ThPre in Matlab
    KS25_params['progress_bar'] = False  # default True
    KS25_params['projection_threshold'] = [10, 4]
    KS25_params['save_rez_to_mat'] = True  # default False
    KS25_params['sig'] = 20  # default 20
    KS25_params['sigmaMask'] = 30  # default 30
    KS25_params['skip_kilosort_preprocessing'] = True  # default False
    return KS25_params


def prep_KS3_params(kilosort3_path: str, default_params: bool=False) -> dict:
    ss.Kilosort3Sorter.set_kilosort3_path(kilosort3_path)
    KS3_params = ss.get_default_sorter_params('kilosort3')
    if default_params:
        return KS3_params

    # ic(KS3_params)
    KS3_params['AUCsplit'] = 0.8  # default 0.8
    KS3_params['freq_min'] = 300  # default 300
    KS3_params['car'] = True  # default True
    KS3_params['detect_threshold'] = 6  # default 6
    KS3_params['do_correction'] = True  # default True
    KS3_params['minFR'] = 0.2  # default 0.2
    KS3_params['minfr_goodchannels'] = 0.2  # default 0.2
    KS3_params['nblocks'] = 5  # default 5
    KS3_params['preclust_threshold'] = 8  # this is ThPre in Matlab
    KS3_params['progress_bar'] = False  # default True
    KS3_params['projection_threshold'] = [9, 9]  # default [9, 9]
    KS3_params['save_rez_to_mat'] = True  # default False
    KS3_params['sig'] = 20  # default 20
    KS3_params['sigmaMask'] = 30  # default 30
    KS3_params['skip_kilosort_preprocessing'] = False  # default False
    KS3_params['scaleproc'] = 200  # default 200
    return KS3_params


def prep_pykilosort_params() -> dict:
    pykilosort_params = ss.get_default_sorter_params('pykilosort')
    pykilosort_params['Th'] = [10, 4]
    pykilosort_params['ThPre'] = 8
    pykilosort_params['do_whitening'] = True
    pykilosort_params['fshigh'] = 300.0
    pykilosort_params['fslow'] = None
    pykilosort_params['gain'] = 1
    pykilosort_params['genericSpkTh'] = 8.0
    pykilosort_params['keep_good_only'] = False
    pykilosort_params['lam'] = 10
    pykilosort_params['loc_range'] = [5, 4]
    pykilosort_params['long_range'] = [30, 6]
    pykilosort_params['minFR'] = 0.02
    pykilosort_params['minfr_goodchannels'] = 0.1
    pykilosort_params['momentum'] = [20, 400]
    pykilosort_params['nblocks'] = 5
    pykilosort_params['perform_drift_registration'] = False  # default False, probably want True
    pykilosort_params['preprocessing_function'] = 'kilosort2'  # dunno what the other options here are
    pykilosort_params['progress_bar'] = False  # default True
    pykilosort_params['save_temp_files'] = True
    pykilosort_params['sig'] = 1
    pykilosort_params['sig_datashift'] = 20.0
    pykilosort_params['sigmaMask'] = 30
    pykilosort_params['spkTh'] = -6
    return pykilosort_params


def prep_params(KS2_5_path: str, KS3_path: str) -> dict:
    spyking_params = prep_spyking_params()
    KS2_params = prep_KS2_params()
    KS25_params = prep_KS2_5_params(KS2_5_path)
    KS3_params = prep_KS3_params(KS3_path)
    pykilosort_params = prep_pykilosort_params()
    param_dict = dict(spykingcircus=spyking_params, kilosort2=KS2_params,
                      kilosort2_5=KS25_params, kilosort3=KS3_params, pykilosort=pykilosort_params)
    return param_dict


# lab computer
# data_root = Path.home().joinpath('Documents', 'testdata', 'HD015_11302023')
# preprocessed_path = data_root.joinpath('HD015_11302023_g0/HD015_11302023_g0_imec0/preprocessed/clean_ap.zarr')
# output_folder = data_root.joinpath('ks3_output')
# recording = pio.load_preprocessed_data(preprocessed_path)
# KS25_path = Path().home().joinpath('Documents/HPC_kilosort/kilosort_debugging/Kilosort-2.5')
# KS3_path = Path().home().joinpath('Documents/HPC_kilosort/kilosort_debugging/Kilosort-3')

# computing cluster
data_path = Path.home().joinpath('raw_data/11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0')
recordings = pp.load_sglx_data(data_path)
recording = recordings[0]  # this is the AP data
KS25_path = Path().home().joinpath('kilosort/kilosort_debugging/Kilosort-2.5')
KS3_path = Path().home().joinpath('kilosort/kilosort_debugging/Kilosort-3')

sorter = 'kilosort3'
output_folder = data_path.joinpath('si_{}'.format(sorter))
param_dict = prep_params(KS25_path, KS3_path)

# at home
# preprocessed_path = Path("C:/Users/mattc/Dropbox (EinsteinMed)/phd_data/11302023/HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0/preprocessed/")

# run the sorter - this is enough to finish postprocessing in phy
sorting = ss.run_sorter(sorter_name=sorter, recording=recording, output_folder=output_folder, **param_dict[sorter])
ic(sorting)
# load a sorting back in
sorting_KS = si.read_kilosort(folder_path="kilosort-folder")

# save the sorter and features
extractor = si.extract_waveforms(recording, sorting, folder=output_folder / 'waveforms', sparse=True,
                                 ms_before=1, ms_after=2, max_spikes_per_unit=500)
_ = si.compute_noise_levels(extractor)
_ = si.compute_correlograms(extractor)
_ = si.compute_unit_locations(extractor)
_ = si.compute_template_similarity(extractor)
job_kwargs = dict(n_jobs=40, chunk_duration='1s', progress_bar=False)
_ = si.compute_principal_components(extractor, n_components=2, mode='by_channel_global')
# _ = si.compute_principal_components(extractor, n_components=2, mode='by_channel_local', **job_kwargs)  # may be better for separated recording sections/tetrodes
_ = si.compute_spike_amplitudes(extractor, **job_kwargs)
si.export_to_phy(waveform_extractor=extractor, output_folder=output_folder / 'phy')

ic(extractor)
ic(extractor.folder)

# extractor = si.load_extractor()
# extractor = si.load_waveforms(data_path / 'waveforms_{}'.format(sorter))

# compute quality metrics and curate; metrics produces a pandas dataframe
metrics = si.compute_quality_metrics(extractor, metric_names=['firing_rate', 'presence_ratio', 'snr',
                                                              'isi_violation', 'amplitude_cutoff'])

ic(metrics)

qm_params = si.get_default_qm_params()
ic(qm_params)

amplitude_cutoff_thresh = 0.1
isi_violations_ratio_thresh = 1
presence_ratio_thresh = 0.9

our_query = f"(amplitude_cutoff < {amplitude_cutoff_thresh}) & (isi_violations_ratio < {isi_violations_ratio_thresh}) & (presence_ratio > {presence_ratio_thresh})"
print(our_query)

keep_units = metrics.query(our_query)
keep_unit_ids = keep_units.index.values
ic(keep_unit_ids)

# save the curated units
we_clean = extractor.select_units(keep_unit_ids, new_folder=output_folder / 'waveforms_clean')
ic(we_clean)

# export spike sorting report to a folder
si.export_report(we_clean, output_folder / 'report', format='png')

# push to web-based viewer?
si.plot_sorting_summary(we_clean, backend='sortingview')

# compare sorters
# comp_pair = si.compare_two_sorters(sorting1=sorting_TDC, sorting2=sorting_SC2)
# comp_multi = si.compare_multiple_sorters(sorting_list=[sorting_TDC, sorting_SC2, sorting_KS2],
#                                          name_list=['ks2.5', 'ks3'])

