from pathlib import Path

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw

raw_root = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260727_alternating_latent/ephys/raw")
out_root = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT026/CT026_20260727_alternating_latent/ephys/derived")
experiment = "2026-07-27_14-37-43"

print(
    se.OpenEphysBinaryRecordingExtractor.get_available_experiments(
        raw_root
    )
)

stream_names, stream_ids = (
    se.OpenEphysBinaryRecordingExtractor.get_streams(
        raw_root,
        experiment_names=[experiment],
    )
)

for name, stream_id in zip(stream_names, stream_ids):
    print(stream_id, name)