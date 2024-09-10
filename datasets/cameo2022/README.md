# CAMEO2022: single structure prediction benchmark


## Step 1: download data and tools

- A folder contains MMCIF files downloaded from RCSB is required to generate ground truth PDB structures. See `rcsb` module for details.
- `cameo2022_orig.csv` is available at https://github.com/bjing2016/EigenFold/blob/master/splits/cameo2022_orig.csv, or can be downloaded from CAMEO website (https://www.cameo3d.org/modeling/targets/1-year/?to_date=2022-12-31).
- Following tools are required for protein structural comparison.
  - TM-score: https://zhanggroup.org/TM-score/
  - lddt: https://swissmodel.expasy.org/lddt/downloads/


## Step 2: prepare metadata csv file
```bash
python3 -m datasets.cameo2022.make_csv \
    --cameo2022_orig_csv_path /path/to/cameo2022_original.csv \
    --mmcif_dir /path/to/mmcif_files \
    --output_csv_path /path/to/cameo2022_metadata.csv \
    --output_pdb_dir /path/to/pdb/ \ # ground truth PDB files for evaluation
    --num_workers 8
```


## Step 3: analyze generated samples

```python
# In jupyter notebook

from src.analysis import cameo2022

results = cameo2022.eval_cameo2022(
    result_root={
        'exp1': '/path/to/generated/cameo2022/',
        'exp2': '/path/to/generated/cameo2022/',
        ...
    },
    metadata_csv_path='/path/to/cameo2022_metadata.csv',
    ref_root='/path/to/ground_truth/cameo2022/pdb',
    num_samples=5,  # number of samples generated
    n_proc=32,      # number of parallel workers for evaluation
    tmscore_exec='/path/to/bin/TMscore', # provide the executable path if the env variable TMSCORE is not set
    lddt_exec='/path/to/bin/lddt'        # provide the executable path if the env variable LDDT is not set
)

# Print out table results
results['report_tab']
```
