# BPTI dynamics

BPTI (Bovine Pancreatic Trypsin Inhibitor) is a model system to study protein dynamics transiting between multiple metastable states (Shaw, et. al., 2010, https://www.science.org/doi/abs/10.1126/science.1187409). The PDB files of 5 metastable states are available at https://www.science.org/doi/10.1126/science.1187409 and the MD trajectories can be obtained from the authors.


## Step 1: survey trajectory data and precompute reference MD information
- Download and decompress the raw trajectories files, each from `.tar.xz` as a separate sub-folder under `/path/to/traj_root/`
    - We only need alpha-carbon trajectories (`*_c-alpha_*`) for analysis
- Survey trajectory files and prepare the metadata csv

```bash
python3 -m datasets.bpti.prepare \
    --traj-root /path/to/bpti/traj_root/ \
    --output-root /path/to/bpti
```

It will create following files under `/path/to/bpti/`:

- `metadata.csv`: metadata info for each protein
- `processed/`: folder contains extracted info for each trajectory
- `fullmd_ref_value/`: folder contains precomputed stats of full MD trajectories

## Step 2: run TICA analysis

```bash
python3 -m datasets.bpti.tica_fit \
    --dataset-root /path/to/bpti
```

It will create `tica` folder under `/path/to/bpti/` with TICA model info.


## Step 3: analyze the results

```python

from src.analysis import bpti

results = bpti.eval_bpti(
    result_root={
        'exp1': '/path/to/generated/bpti/',
        'exp2': '/path/to/generated/bpti/',
        ...
    },
    metadata_csv_path='/path/to/bpti/metadata.csv',
    ref_root='/path/to/bpti/fullmd_ref_value/',
    num_samples=1000,
    tmscore_exec='/path/to/bin/TMscore',
)


# Print out the table results
results['report_tab']
```