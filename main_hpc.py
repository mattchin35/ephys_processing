import argparse
import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path.home() / 'ephys_processing'
PREPROCESSING_ROOT = PROJECT_ROOT / 'src' / 'preprocessing'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PREPROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_ROOT))

import src.preprocessing.preprocess_io as pio
import src.SGLXMetaToCoords.SGLXMetaToCoords as coordsSGLX
from src.preprocessing.binary_inspection import InspectionParams, process_stream_inspection


# Hardcoded session configuration. Adjust these values directly for the HPC job.
SESSION_NAME = 'CT014_20251221_latentInference'
SESSION_ROOT = Path.home() / 'contextProjectData/CT014' / SESSION_NAME
RAW_DATA_ROOT = SESSION_ROOT / 'ephys' / 'raw'
CATGT_DATA_ROOT = SESSION_ROOT / 'ephys' / 'catgt'
FIGURE_PATH = SESSION_ROOT / 'figures'
PROCESSED_PATH = SESSION_ROOT / 'processed'
RUN = 0
GATE = 0
OPTS = 'catgt'
TIME_TOKEN = 'tcat'


@dataclass(frozen=True)
class ProbePaths:
    probe_index: int
    probe_dir: Path
    ap_binary: Path
    lfp_binary: Path | None
    ap_meta: Path


def build_default_params(run_psd: bool = True) -> InspectionParams:
    params = InspectionParams()
    params.window_size = 1
    params.reduced_skip = 300
    params.full_skip = 1
    params.nperseg = 1024
    params.ap_srate = 30000
    params.lfp_srate = 2500
    params.run_reduced_rms = True
    params.run_full_rms = True
    params.run_PSD = run_psd
    params.run_threshold_detection = False
    params.lfp_is_present = True
    return params


def build_session_tag() -> str:
    if OPTS:
        return '{}_{}'.format(SESSION_NAME, OPTS)
    return SESSION_NAME


def build_recording_path() -> Path:
    if OPTS == 'catgt':
        return CATGT_DATA_ROOT / 'catgt_run{}_g{}'.format(RUN, GATE)
    return RAW_DATA_ROOT / 'run{}_g{}'.format(RUN, GATE)


def discover_probe_paths(recording_path: Path) -> list[ProbePaths]:
    probe_paths = []
    probe_dirs = sorted(recording_path.glob('run{}_g{}_imec*'.format(RUN, GATE)))
    for probe_dir in probe_dirs:
        match = re.search(r'imec(\d+)$', probe_dir.name)
        if match is None:
            continue

        probe_index = int(match.group(1))
        ap_binary = probe_dir / 'run{0}_g{1}_{2}.imec{3}.ap.bin'.format(RUN, GATE, TIME_TOKEN, probe_index)
        lfp_binary = probe_dir / 'run{0}_g{1}_{2}.imec{3}.lf.bin'.format(RUN, GATE, TIME_TOKEN, probe_index)
        ap_meta = pio.get_meta_path(ap_binary)

        if not ap_binary.exists():
            raise FileNotFoundError('AP binary not found: {}'.format(ap_binary))

        probe_paths.append(
            ProbePaths(
                probe_index=probe_index,
                probe_dir=probe_dir,
                ap_binary=ap_binary,
                lfp_binary=lfp_binary if lfp_binary.exists() else None,
                ap_meta=ap_meta,
            )
        )

    if not probe_paths:
        raise FileNotFoundError('No imec probe folders found in {}'.format(recording_path))

    return probe_paths


def get_geometric_sort(metadata: dict) -> np.ndarray:
    if 'snsGeomMap' in metadata:
        _, _, _, _, x_coord, y_coord, _ = coordsSGLX.geomMapToGeom(metadata)
    else:
        _, _, _, _, x_coord, y_coord, _ = coordsSGLX.shankMapToGeom(metadata)

    coords = np.stack((x_coord, y_coord), axis=1)
    return np.argsort(coords[:, 1])


def build_chanmap_path(ap_meta: Path) -> Path:
    return ap_meta.parent / '{}_chanMap.mat'.format(ap_meta.stem)


def run_probe_inspection(
    probe_paths: ProbePaths,
    session_tag: str,
    params: InspectionParams,
    figure_path: Path,
    processed_path: Path,
) -> list[dict]:
    stream_summaries = []

    print('Loading AP binary for imec{}: {}'.format(probe_paths.probe_index, probe_paths.ap_binary))
    ap_data, ap_meta, ap_srate, _ = pio.read_binary(probe_paths.ap_binary)
    ap_data = ap_data[:384]
    geometric_sort = get_geometric_sort(ap_meta)

    stream_summaries.append(
        process_stream_inspection(
            recording=ap_data,
            metadata=ap_meta,
            sample_rate=ap_srate,
            session_tag=session_tag,
            stream_tag='AP{}'.format(probe_paths.probe_index),
            params=params,
            geometric_sort=geometric_sort,
            figure_path=figure_path,
            processed_path=processed_path,
        )
    )
    del ap_data
    gc.collect()

    if probe_paths.lfp_binary is not None and params.lfp_is_present:
        print('Loading LFP binary for imec{}: {}'.format(probe_paths.probe_index, probe_paths.lfp_binary))
        lfp_data, lfp_meta, lfp_srate, _ = pio.read_binary(probe_paths.lfp_binary)
        lfp_data = lfp_data[:384]
        stream_summaries.append(
            process_stream_inspection(
                recording=lfp_data,
                metadata=lfp_meta,
                sample_rate=lfp_srate,
                session_tag=session_tag,
                stream_tag='LFP{}'.format(probe_paths.probe_index),
                params=params,
                geometric_sort=geometric_sort,
                figure_path=figure_path,
                processed_path=processed_path,
                run_psd=True,
            )
        )
        del lfp_data
        gc.collect()

    return stream_summaries


def write_probe_chanmap(probe_paths: ProbePaths) -> Path:
    chanmap_path = build_chanmap_path(probe_paths.ap_meta)
    print('Writing channel map for imec{} to {}'.format(probe_paths.probe_index, chanmap_path))
    coordsSGLX.MetaToCoords(
        metaFullPath=probe_paths.ap_meta,
        outType=1,
        destFullPath=chanmap_path,
        showPlot=False,
    )
    return chanmap_path


def run_session_preprocessing(skip_inspection: bool = False, skip_chanmap: bool = False, skip_psd: bool = False) -> dict:
    recording_path = build_recording_path()
    if not recording_path.exists():
        raise FileNotFoundError('Recording path does not exist: {}'.format(recording_path))

    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    session_tag = build_session_tag()
    params = build_default_params(run_psd=not skip_psd)
    probe_paths = discover_probe_paths(recording_path)

    summary = {
        'session_name': SESSION_NAME,
        'session_root': str(SESSION_ROOT),
        'recording_path': str(recording_path),
        'session_tag': session_tag,
        'skip_inspection': skip_inspection,
        'skip_chanmap': skip_chanmap,
        'skip_psd': skip_psd,
        'probes': [],
    }

    for probe in probe_paths:
        probe_summary = {
            'probe_index': probe.probe_index,
            'probe_dir': str(probe.probe_dir),
            'ap_binary': str(probe.ap_binary),
            'lfp_binary': str(probe.lfp_binary) if probe.lfp_binary is not None else None,
            'ap_meta': str(probe.ap_meta),
        }

        if not skip_chanmap:
            probe_summary['chanmap_path'] = str(write_probe_chanmap(probe))

        if not skip_inspection:
            probe_summary['streams'] = run_probe_inspection(
                probe_paths=probe,
                session_tag=session_tag,
                params=params,
                figure_path=FIGURE_PATH,
                processed_path=PROCESSED_PATH,
            )

        summary['probes'].append(probe_summary)

    if not skip_inspection:
        pio.save_inspection_data(summary, '{}_inspection_index'.format(session_tag), PROCESSED_PATH)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run session-level ephys preprocessing on the HPC.')
    parser.add_argument('--skip-inspection', action='store_true', help='Skip binary inspection outputs.')
    parser.add_argument('--skip-chanmap', action='store_true', help='Skip channel map generation.')
    parser.add_argument('--skip-psd', action='store_true', help='Skip PSD generation during inspection.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_session_preprocessing(
        skip_inspection=args.skip_inspection,
        skip_chanmap=args.skip_chanmap,
        skip_psd=args.skip_psd,
    )
    print('Finished preprocessing session {}'.format(summary['session_name']))


if __name__ == '__main__':
    main()
