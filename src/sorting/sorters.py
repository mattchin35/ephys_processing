from pathlib import Path
from icecream import ic
import src.preprocessing.preprocess_io as pio
import spikeinterface.sorters as ss
import src.preprocessing.preprocessing as pp

# ic(ss.available_sorters())
# ic(ss.installed_sorters())

def prep_spyking_params():
    # ic("SpykingCircus params:")
    spyking1_params = ss.get_default_sorter_params('spykingcircus')
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


def prep_KS2_5_params(kilosort2_5_path, default_params=False):
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


def prep_KS3_params(kilosort3_path, default_params=False):
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


def prep_pykilosort_params():
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


# lab computer
# data_root = Path.home().joinpath('Documents', 'testdata', 'HD015_11302023')
# preprocessed_path = data_root.joinpath('HD015_11302023_g0/HD015_11302023_g0_imec0/preprocessed/clean_ap.zarr')
# output_folder = data_root.joinpath('ks3_output')
# recording = pio.load_preprocessed_data(preprocessed_path)
# KS25_path = Path().home().joinpath('Documents/HPC_kilosort/kilosort_debugging/Kilosort-2.5')
# KS3_path = Path().home().joinpath('Documents/HPC_kilosort/kilosort_debugging/Kilosort-3')

# computing cluster
data_path = Path.home().joinpath('raw_data/11.30.2023/HD015_11.30.2023_g0/HD015_11.30.2023_g0_imec0')
output_folder = data_path.joinpath('si_kilosort3')
recordings = pp.load_sglx_data(data_path)
recording = recordings[0]  # this is the AP data
KS25_path = Path().home().joinpath('kilosort/kilosort_debugging/Kilosort-2.5')
KS3_path = Path().home().joinpath('kilosort/kilosort_debugging/Kilosort-3')
# ic(recording)

# at home
# preprocessed_path = Path("C:/Users/mattc/Dropbox (EinsteinMed)/phd_data/11302023/HD015_11302023/HD015_11302023_g0/HD015_11302023_g0_imec0/preprocessed/")

# tag = 'HD015_11302023_preprocessed'

# set parameters
spyking_params = prep_spyking_params()
KS2_params = prep_KS2_params()
KS25_params = prep_KS2_5_params(KS25_path)
KS3_params = prep_KS3_params(KS3_path)


# sorting = ss.run_sorter(sorter_name="spykingcircus", recording=recording, output_folder=output_folder,
#                             **spyking1_params)
sorting = ss.run_sorter(sorter_name="kilosort3", recording=recording, output_folder=output_folder, **KS3_params)
# ic(sorting)
