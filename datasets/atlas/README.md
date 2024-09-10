# ATLAS MD dataset

Atlas dataset contains 1390 diverse proteins with their MD trajectories (3 replicates of 100 ns, each). 
See Meersche, et. al. (https://academic.oup.com/nar/article/52/D1/D384/7438909?login=false) and database website (https://www.dsimb.inserm.fr/ATLAS/index.html) for details.


## Step 1: Download MD trajectories and prepare metadata csv
- Download `protein` level MD trajectories from the server.
```bash

# Download the Atlas-test trajectories for evaluation.

bash ./download_all.sh \
    ./2022_06_13_ATLAS_test_only.txt \
    /path/to/traj_root/ \
    32 # number of parallel downloaders


# Download all Atlas trajectories for training
cd datasets/atlas
bash ./download_all.sh \
    ./2022_06_13_ATLAS_pdb.txt \
    /path/to/traj_root/ \
    32 # number of parallel downloaders
```

- Download metadata csv
    - We followed the AlphaFlow's data splits and the metadata csv files for each split are available at https://github.com/bjing2016/alphaflow/tree/master/splits


## Step 2: Preparing the PDB files for ATLAS training [optional]
```bash

python3 -m datasets.atlas.unpack_pdb_from_traj \
    --traj-root /path/to/traj_root/ \
    --output-root /path/to/pdb_root/ \
    --n-proc 32 # number of parallel workers

```


## Step 3: Analyze the results

1. Run the evaluate script

```bash

python3 -m datasets.atlas.analysis_aflow \
    --result-dir /path/to/atlas_test/ \
    --atlas-dir /path/to/traj_root/ \
    --n-proc 4 # number of parallel workers. Roughly 32 threads for 1 worker.
```
- It will generate an `aflow_analysis.pkl` file under `/path/to/atlas_test/`

2. Report metrics

```python

from src.analysis import atlas

results = atlas.report_analysis(
    result_root={
        'exp1': '/path/to/generated/atlas/',
        'exp2': '/path/to/generated/atlas/',
        ...
    }
)

# Print out the table results
results['report_tab'][atlas.REPORT_TAB_COLS]

```

