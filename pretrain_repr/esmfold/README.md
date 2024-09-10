# ESMFold representation

We use the weights `esmfold_3B_v1.pt` (https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt) to generate sequence representation of ESMFold model. See https://github.com/facebookresearch/esm for details.


## Generate representation for sequences in metadata.csv

- `metadata.csv` should contains two columns: `chain_name` the name of the protein and `seqres` the protein sequence.

```bash

# Example: generate node and edge representations with recycle=3
python3 -m pretrain_repr.esmfold.make_esm_repr.py \
    --input-csv-path /path/to/metadata.csv \
    --output-dir /path/to/esm_repr/ \
    --esm-ckpt-fpath /path/to/esmfold_3B_v1.pt \
    --num-recycles 3 \
    --batch-size 1 # batch_size to run inference

```
- An index file `seqres_to_index.recycle0/3.csv` will be generated under `/path/to/esm_repr/`


#### Multi-GPU script: we also provide a script to run inference with multiple GPUs in parallel:


```bash
#!/bin/bash

PREFIX=/path/to/confdiff/pretrain_repr/esmfold
NUM_GPU=8
INPUT_CSV_PATH=/path/to/metadata.csv
CKPT_PATH=/path/to/esmfold_3B_v1.pt
NUM_RECYCLES=3
OUTPUT=/path/to/esm_repr/recycle3/
BATCH_SIZE=1

for GPU_ID in $(seq 0 $(($NUM_GPU-1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 $PREFIX/make_esm_repr.py \
        --input-csv-path $INPUT_CSV_PATH \
        --output-dir $OUTPUT \
        --esm-ckpt-fpath $CKPT_PATH \
        --num-recycles $NUM_RECYCLES \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_GPU \
        --worker-id $GPU_ID &
done

wait

# concat all index files
cat $OUTPUT/seqres_to_index.worker*.csv >> $OUTPUT/seqres_to_index.recycle$NUM_RECYCLES.csv
rm $OUTPUT/seqres_to_index.worker*.csv

```