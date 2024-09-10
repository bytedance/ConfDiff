# Fast-folding proteins dynamics

Fast-folding protein dataset contain 12 small proteins studied in How fast-folding proteins fold, Lindorff-Larsen et. al. (https://www.science.org/doi/10.1126/science.1208351). The dataset can be obtained by the original authors. 


## Step 1: survey trajectory data and precompute reference MD information
- Download and decompress the raw trajectories files, each from `.tar.xz` as a separate sub-folder under `/path/to/traj_root/`
    - We only need alpha-carbon trajectories (`*_c-alpha_*`) for analysis
- Survey trajectory files and prepare the metadata csv

```bash
python3 -m datasets.fastfold.prepare \
    --traj-root /path/to/fastfold/traj_root/ \
    --output-root /path/to/fastfold/ \
    --num-workers 6
```

It will create following files under `/path/to/fastfold/`:

- `metadata.csv`: metadata info for each protein
- `processed/`: folder contains extracted info for each trajectory
- `fullmd_ref_value/`: folder contains precomputed stats of full MD trajectories

## Step 2: run TICA analysis

```bash
python3 -m datasets.fastfold.tica_fit \
    --dataset-root /path/to/fastfold/
```

It will create `tica` folder under `/path/to/fastfold/` with TICA model info.


## Step 3: analyze the results

```python

from src.analysis import fastfold

results = fastfold.eval_fastfold(
    result_root={
        'exp1': '/path/to/generated/fastfold/',
        'exp2': '/path/to/generated/fastfold/',
        ...
    },
    metadata_csv_path='/path/to/fastfold/metadata.csv',
    ref_root='/path/to/fastfold/fullmd_ref_value/',
    num_samples=1000, # number of samples generated
    n_proc=12,      # number of parallel workers for evaluation
    tmscore_exec='/path/to/bin/TMscore', # provide the executable path if the env variable TMSCORE is not set
)


# Print out the table results
results['report_tab']
```