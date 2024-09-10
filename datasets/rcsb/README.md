# RCSB PDB dataset


## Step 1: download PDB MMCIF files:

```bash
# Download compressed cif.gz data from RCSB server. It will take around 70GB of the storage
bash datasets/rcsb/download_pdb_mmcif.sh /path/to/mmcif_data
```

This will download compressed MMCIF files to `/path/to/mmcif_data/{pdbid[1:3]}/{pdbid}.mmcif.gz`


## Step 2: process MMCIF files

```python
python3 -m datasets.rcsb.process_mmcif_metadata \
    --mmcif_dir /path/to/mmcif_data \
    --cluster_dir /path/to/cluster_data \
    --output_csv_path /path/to/rcsb_metadata.csv \
    --output_pdb_dir /path/to/pdb_files \
    --num_workers num_parallel_workers 
```

This step generates following data for model training:

1. `rcsb_metadata.csv` file containing the metadatax info
2. a folder of single-chain PDB files. It will take around 95GB of the storage.

