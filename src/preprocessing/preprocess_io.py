import numpy as np
from pathlib import Path
import src.SpikeGLX_Datafile_Tools.Python.DemoReadSGLXData.readSGLX as sglx # will use readMeta, SampRate, makeMemMapRaw, ExtractDigital, GainCorrectIM, GainCorrectNI
import pickle as pkl
from typing import Dict, Any, List, Tuple
from icecream import ic
import spikeinterface.full as si
import datetime as dt


unprocessed_path = Path('../../data/unprocessed/')
preprocessed_path = Path('../../data/preprocessed/')


def read_metadata(binary_file: Path) -> Tuple[dict, int, Tuple[int,int]]:
    metadata = sglx.readMeta(binary_file)
    sample_rate = int(sglx.SampRate(metadata))
    n_chan = int(metadata['nSavedChans'])
    n_samples = int(int(metadata['fileSizeBytes']) / (2 * n_chan))
    return metadata, sample_rate, (n_chan, n_samples)


def read_binary(bin_file: Path, allow_write: bool = False, correct_gain=False) -> Tuple[np.ndarray, dict, int, Tuple[int,int]]:
    metadata, samplerate, shape = read_metadata(bin_file)
    session_length = float(metadata['fileTimeSecs'])
    print("Session length is {} seconds, or {} minutes and {} seconds".format(session_length,
                                                                              session_length // 60,
                                                                              session_length % 60))
    data = sglx.makeMemMapRaw(bin_file, metadata, allowWrite=allow_write)
    if correct_gain:
        chanList = list(range(shape[0]))
        correct_binary_gain(data, metadata, chanList, datatype='analog')

    return data, metadata, samplerate, shape


def correct_binary_gain(signal: np.ndarray, metadata: dict, chanList: list) -> np.ndarray:
    """Convert raw binary data to physical units (e.g., uV or mV) based on metadata. Only applies to analog data!"""
    if metadata['typeThis'] == 'imec':
        # apply gain correction and convert to uV
        convData = 1e6 * sglx.GainCorrectIM(signal, chanList, metadata)
    elif metadata['typeThis'] == 'nidq':
        MN, MA, XA, DW = sglx.ChannelCountsNI(metadata)
        # print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
        # apply gain correction and convert to mV
        convData = 1e3 * sglx.GainCorrectNI(signal, chanList, metadata)
    elif metadata['typeThis'] == 'obx':
        # Gain correct is just conversion to volts
        convData = 1e3 * sglx.GainCorrectOBX(signal, chanList, metadata)
    else:
        raise ValueError("Unknown metadata type: {}".format(metadata['typeThis']))
    return convData


def load_sglx_data(spikeglx_folder: Path) -> List[si.SpikeGLXRecordingExtractor]:
    stream_names, stream_ids = si.get_neo_streams('spikeglx', spikeglx_folder)
    ic(stream_names)
    recordings = [si.read_spikeglx(spikeglx_folder, stream_name=name, load_sync_channel=False) for name in stream_names]
    return recordings


def load_zarr_data(path: Path) -> Tuple[si.ZarrRecordingExtractor]:
    rec = si.read_zarr(path)
    return rec


def load_channel_quality_ids(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        channel_quality_ids = pkl.load(f)
    return channel_quality_ids


def save_preprocessed_data(recording: si.SpikeGLXRecordingExtractor, channel_quality_ids: Tuple[np.ndarray, np.ndarray],
                      save_path: Path, **kwargs) -> None:

    recording.save(folder=save_path, **kwargs)
    with open(save_path, 'wb') as f:
        pkl.dump(channel_quality_ids, f)


def save_inspection_data(data: Dict[str, Any], fname: str, note: str = '') -> None:
    date = str(dt.date.today().isoformat())
    save_dir = preprocessed_path / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (fname + '.pkl')
    with open(p, 'wb') as f:
        pkl.dump(data, f)
    print("[***] Data saved in {}".format(p.name))

    if note:
        with open(save_dir / 'readme.txt', 'w') as f:
            f.write(note)


def load_inspection_data(date: str, fname: str) -> Dict[str, Any]:
    save_dir = preprocessed_path / date
    p = save_dir / (fname + '.pkl')
    with open(p, 'rb') as f:
        data = pkl.load(f)
    return data


def get_meta_path(binFullPath: Path) -> Path:
    """
    Get the path to the metadata file in a SpikeGLX folder.
    """
    metaName = binFullPath.stem + ".meta"
    metaPath = Path(binFullPath.parent / metaName)
    if metaPath.exists():
        return metaPath
    else:
        raise FileNotFoundError("Metadata file {} not found in {}".format(metaName, binFullPath.parent))