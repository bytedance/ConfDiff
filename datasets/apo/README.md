# Apo: ligand-induced dual structure prediction benchmark

## Step 1: download data and tools

- A folder contains MMCIF files downloaded from RCSB is required to generate ground truth PDB structures. See `rcsb` module for details.
- A list of apo-holo targets `revision1_86_plus_5.csv` is provided by EigenFold (https://github.com/bjing2016/EigenFold/blob/master/splits/revision1_86_plus_5.csv), or from the original repo of Saldaño et al, 2022, Bioinformatics (https://gitlab.com/sbgunq/publications/af2confdiv-oct2021/-/blob/main/data/26-01-2022/revision1_86_plus_5.csv)



## Step 2: prepare metadata csv

```bash
python3 -m datasets.apo.make_csv \
    --apo_orig_csv_path /path/to/revision1_86_plus_5.csv \
    --mmcif_dir /path/to/mmcif_files \
    --output_csv_path /path/to/apo_metadata.csv \
    --output_pdb_dir /path/to/apo/pdb/  \
    --tmscore_exec /path/to/TMscore \
    --num_workers 8
```

## Step 3: analyze generated samples

```python
# In jupyter notebook

from src.analysis import apo

results = apo.eval_apo(
    result_root={
        'exp1': '/path/to/generated/apo/',
        'exp2': '/path/to/generated/apo/',
        ...
    },
    metadata_csv_path='/path/to/apo_metadata.csv',
    ref_root='/path/to/ground_truth/apo/pdb',
    num_samples=20, # number of samples generated
    n_proc=32,      # number of parallel workers for evaluation
    tmscore_exec='/path/to/bin/TMscore', # provide the executable path if the env variable TMSCORE is not set
)

# Print out the table results
results['report_tab']
```