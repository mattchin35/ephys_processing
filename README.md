Ephys Processing
==============================

Analysis pipeline for ephys data. Supports SGLX; now implementing OpenEphys support based in spikeinterface.

Open Ephys preprocessing
------------

Use **`run_openephys_preprocessing.py`** in the project root on both the
workstation and the cluster. It replaces `preprocess_oe_hpc.py`; the old
`main_oe_hpc.py` launcher reference was stale.

1. Edit the session configuration at the top of the Python file: `SESSION_NAME`,
   `SUBJECT_ID`, `SESSION_ROOT`, `EXPERIMENT_NAME`, and `STREAM_NAME`.
   Set `SESSION_ROOT` to the session's actual path on the machine running it.
   All processing settings also live in that file; the Slurm wrapper contains
   no session or processing configuration.
2. With uv installed, run these commands from the project root on each machine
   to prepare its own Python 3.12 environment and check imports without processing data:

   ```bash
   uv sync --locked
   uv run --locked run_openephys_preprocessing.py --help
   ```

   Keep `pyproject.toml`, `uv.lock`, and `.python-version` with the checkout.
   Create the cluster environment there rather than copying the workstation's
   `.venv`. Prepare it before submitting jobs, especially if compute nodes
   cannot download packages. uv must be available on the batch job's PATH;
   conda activation is no longer used by this launcher.
3. On the workstation, execute the Python entry point directly:

   ```bash
   uv run --locked run_openephys_preprocessing.py
   ```

   You can also run that file in the IDE using the project's `.venv` interpreter.
4. On the cluster, the wrapper assumes the checkout is `~/ephys_processing`.
   From that checkout, submit:

   ```bash
   sbatch src/shell_scripts/hpc_openephys_preprocessing.sh
   ```

   Review its scheduler resources, email, and log path for your cluster account;
   the log directory must already exist. Its only executable command invokes
   the same Python entry point through uv.

**Dry run is enabled by default.** Leave `DRY_RUN = True` in the Python
script to validate the selected session and print the plan without creating
processing outputs or reading signal samples. Set `DRY_RUN = False` to run
processing; the workstation and Slurm commands stay the same. Validation
checks input/probe metadata, planned output locations, existing output files,
and required inspection inputs. It does not test filtering, signal quality,
sampled AP ranges, available disk space, or guarantee future write access.
`--help` only checks imports/argument parsing; run without it for validation.

Both commands default to `--stage all`: preprocessing followed by binary
inspection. To run either stage separately, append `--stage preprocess` or
`--stage inspect` to the workstation or `sbatch` command. Inspection requires
the existing AP and LFP outputs. A dry run of `all` treats enabled preprocessing
products as future inspection inputs; disabled products must already exist.

Raw data are read from `SESSION_ROOT/ephys/raw`. Outputs are saved under
`SESSION_ROOT/ephys/derived/<sanitized-stream-name>/`: probe layout,
`chanMap.mat`, LFP binary and metadata, AP binary and metadata, and channel
quality JSON/CSV. Inspection writes to `SESSION_ROOT/figures` and
`SESSION_ROOT/processed`. Preprocessing does not skip existing results based
on parameters; check the output location before rerunning.

The files in `src/preprocessing` are implementation modules:
`preprocess_openephys.py` orchestrates the shared processing and
`ap_preprocessing_openephys.py` supplies AP preprocessing. Their `main()`
functions are standalone examples with separate hardcoded sessions; use the
root entry point above for routine runs. This workflow prepares data and
inspection outputs; it does not run spike sorting.

Project Organization
------------

    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── preprocessing           <- All the important code is in here right now.
    │   │   └── binary_inspection.py - Inspect binary recordings with simple visualizations.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
